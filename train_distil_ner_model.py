import argparse
from functools import partial
from pathlib import Path

import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from transformers import AutoTokenizer, BertForTokenClassification
from transformers import (AutoModelForTokenClassification,
                          TrainingArguments,
                          Trainer)

from transformers import DataCollatorForTokenClassification
from transformers import TrainerCallback
import numpy as np
import json
import pandas as pd
from distillation.losses import KLDivLoss, MSEHiddenStatesLoss
from distillation.student_init.google_students_models import get_student_models, all_google_students
from const import device
from utils import dict_to_device, dotdict, set_seed
from utils.common import file_size

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


def tokenize_and_align_labels(examples,
                              tokenizer,
                              text_name,
                              label_name,
                              label_list,
                              is_split_into_words,
                              label_all_tokens=True):
    tokenized_inputs = tokenizer(examples[text_name], truncation=True, is_split_into_words=is_split_into_words)

    labels = []
    for i, label in enumerate(examples[label_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def train_iter(
        student,
        teacher,
        batch,
        logits_criterion,
        hiddens_criterion,
        weights,
        optimizer,
        mapping_optimizer,
        metric_fn,
        label_list
):
    student.train()
    teacher.eval()

    with torch.no_grad():
        teacher_outputs = teacher(**batch, output_hidden_states=True, return_dict=True)
    student_outputs = student(**batch, output_hidden_states=True, return_dict=True)

    predictions = student_outputs["logits"].argmax(-1).detach().cpu().numpy()
    labels = batch["labels"]
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    metric_fn.add_batch(predictions=true_predictions, references=true_labels)

    task_loss = student_outputs["loss"]
    logits_loss = logits_criterion(student_outputs["logits"], teacher_outputs["logits"])

    t_hiddens = teacher_outputs["hidden_states"]
    t_hiddens = tuple([t_h[:, 0] for t_h in t_hiddens])
    s_hiddens = student_outputs["hidden_states"]
    s_hiddens = tuple([s_h[:, 0] for s_h in s_hiddens])
    hiddens_loss = hiddens_criterion(s_hiddens, t_hiddens)

    final_loss = weights[0] * task_loss + weights[1] * logits_loss + weights[2] * hiddens_loss

    optimizer.zero_grad()
    mapping_optimizer.zero_grad()
    final_loss.backward()
    optimizer.step()
    # scheduler.step()
    mapping_optimizer.step()
    return {
        "final_loss": final_loss.item(),
        "task_loss": task_loss.item(),
        "logits_loss": logits_loss.item(),
        "hidden_state_loss": hiddens_loss.item(),
    }


@torch.no_grad()
def val_iter(student, batch, metric_fn, label_list):
    student.eval()
    student_outputs = student(**batch, output_hidden_states=False, return_dict=True)
    predictions = student_outputs["logits"].argmax(-1).detach().cpu().numpy()
    labels = batch["labels"]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    metric_fn.add_batch(predictions=true_predictions, references=true_labels)
    return {
        "logits_loss": student_outputs["logits"]
    }


def main(args):
    # base_dir, student_model_name, temperature, batch_size = 16, num_epochs = 5

    temperature = args.temperature
    student_model_name = args.student_model_name
    output_dir = args.output_dir
    num_epochs = args.num_train_epochs
    root_dir = str(Path(output_dir) / f"{student_model_name}_T-{int(temperature)}")
    Path(root_dir).mkdir(exist_ok=True, parents=True)

    writer = SummaryWriter()
    teacher_model_name = args.teacher_model_name
    label_name = "ner_tags"
    text_name = "tokens"
    datasets = load_dataset("conllpp")
    metric_fn = load_metric("seqeval")
    label_list = datasets["train"].features[f"ner_tags"].feature.names
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    tokenize_and_align = partial(tokenize_and_align_labels,
                                 tokenizer=tokenizer,
                                 text_name=text_name,
                                 label_name=label_name,
                                 label_list=label_list,
                                 is_split_into_words=True,
                                 label_all_tokens=True)
    datasets = datasets.map(tokenize_and_align, batched=True)
    datasets = datasets.remove_columns(['chunk_tags', 'id', 'tokens', 'pos_tags', 'ner_tags'])
    data_collator = DataCollatorForTokenClassification(tokenizer)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.train_batch_size, shuffle=True,
                            collate_fn=data_collator),
        "valid": DataLoader(datasets["test"], batch_size=args.val_batch_size, collate_fn=data_collator),
    }
    teacher_model = AutoModelForTokenClassification.from_pretrained(teacher_model_name, num_labels=len(label_list)).to(
        device)
    student_model = BertForTokenClassification.from_pretrained(student_model_name, num_labels=len(label_list)).to(
        device)
    student_model.config.label2id = teacher_model.config.label2id
    student_model.config.id2label = teacher_model.config.id2label

    kl_div_loss = KLDivLoss(temperature=temperature)
    mse_hiddens_loss = MSEHiddenStatesLoss(
        normalize=True,
        need_mapping=True,
        teacher_hidden_state_dim=teacher_model.config.hidden_size,
        student_hidden_state_dim=student_model.config.hidden_size,
        num_layers=student_model.config.num_hidden_layers,
    ).to(device)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
    mapping_optimizer = torch.optim.Adam(mse_hiddens_loss.parameters(), lr=1e-3)
    pbar_epochs = trange(num_epochs, leave=False)
    pbar_epochs.set_description("Epoch: ")
    best_accuracy = 0
    for epoch in pbar_epochs:
        for loader_key, loader in loaders.items():
            pbar_loader = tqdm(loader, leave=False)
            for batch in pbar_loader:
                batch = dict_to_device(batch, device)
                if loader_key.startswith("train"):
                    metrics = train_iter(
                        student=student_model,
                        teacher=teacher_model,
                        batch=batch,
                        logits_criterion=kl_div_loss,
                        hiddens_criterion=mse_hiddens_loss,
                        weights=[0.6, 0.3, 0.1],
                        optimizer=optimizer,
                        mapping_optimizer=mapping_optimizer,
                        metric_fn=metric_fn,
                        label_list=label_list
                    )
                else:
                    metrics = val_iter(student=student_model,
                                       batch=batch,
                                       metric_fn=metric_fn,
                                       label_list=label_list)
                try:
                    log_str = " ".join([f"{key}: {met:.3f}" for key, met in metrics.items()])
                    pbar_loader.set_description(log_str)
                    for key, met in metrics.items():
                        writer.add_scalar(f"{loader_key}/{key}", met, epoch)
                except:
                    pass
            metric_values = metric_fn.compute()
            accuracy = metric_values["overall_accuracy"]
            f1 = metric_values["overall_f1"]
            if not loader_key.startswith("train"):
                if accuracy > best_accuracy:
                    torch.save(student_model.state_dict(), str(Path(root_dir) / "best_student.pth"))
                    best_accuracy = accuracy
            print(f"{loader_key} accuracy: {accuracy}")
            writer.add_scalar(f"{loader_key}/accuracy", accuracy, epoch)
            writer.add_scalar(f"{loader_key}/f1", f1, epoch)

    training_args = TrainingArguments(
        f"test-ner",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01
    )

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric_fn.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    class WiterCallback(TrainerCallback):

        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                with open(str(Path(root_dir) / 'valid_result.json'), 'w') as fp:
                    json.dump(logs, fp)

    student_model = BertForTokenClassification.from_pretrained(student_model_name, num_labels=len(label_list)).to(
        device)
    student_model.load_state_dict(torch.load("best_student.pth"))

    # TODO : Trainer is only for eval. part.  - fix it, removing hf trainer and combining with catalyst HFRunner
    trainer = Trainer(
        student_model,
        training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[WiterCallback()]
    )
    trainer.evaluate()
    predictions, labels, _ = trainer.predict(datasets["test"])
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric_fn.compute(predictions=true_predictions, references=true_labels)
    pd.DataFrame(results).to_json(str(Path(root_dir) / "test_result.json"))

    train_info = {
        "train_batch_size": args.train_batch_size,
        "val_batch_size": args.val_batch_size,
        "num_epochs": num_epochs,
        "temperature": temperature,
        "teacher_model_name": teacher_model_name,
        "student_model_name": student_model_name,
        "model_size": file_size("best_student.pth")}

    with open(str(Path(root_dir) / "train_info.json"), "w") as fp:
        json.dump(train_info, fp)

    writer.flush()
    writer.close()


if __name__ == '__main__':
    hidden_size, num_layers = 256, 6
    student_model_name = get_student_models(hidden_size=hidden_size, num_layers=num_layers)
    teacher_model_name = "imvladikon/bert-large-cased-finetuned-conll03-english"
    teacher_model_name = "bert-base-uncased"

    parser = argparse.ArgumentParser(description='Fine-tuning bert')
    parser.add_argument("--student_model_name", default=student_model_name, type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            all_google_students()))
    parser.add_argument("--student_config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--teacher_model_name", default=teacher_model_name, type=str, required=False)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--one_cycle_train", default=True, action='store_true', required=False)
    parser.add_argument("--train_size", default=-1, type=int, required=False)
    parser.add_argument("--val_size", default=-1, type=int, required=False)
    parser.add_argument("--tokenizer_name",
                        default="bert-base-uncased",
                        type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name",
                        required=False)
    parser.add_argument("--train_batch_size", default=24, type=int, required=False)
    parser.add_argument("--val_batch_size", default=12, type=int, required=False)
    parser.add_argument("--n_threads", default=4, type=int, required=False)
    parser.add_argument("--warmup_linear", default=0.1, type=float, required=False)
    parser.add_argument("--optimize_on_cpu", default=True, type=bool, required=False)
    parser.add_argument("--loss_scale", default=128, type=int, required=False)
    parser.add_argument("--use_wandb", action='store_true', required=False)
    parser.add_argument("--wandb_api_token", default='', type=str, required=False)
    parser.add_argument("--wandb_notes", default='', type=str, required=False)
    parser.add_argument("--wandb_project", default='', type=str, required=False)
    parser.add_argument("--wandb_entity", default='', type=str, required=False)
    parser.add_argument("--wandb_group", default='', type=str, required=False)
    parser.add_argument("--wandb_name", default='', type=str, required=False)
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--temperature", default=1, type=float, required=False)
    parser.add_argument("--kl_div_loss_weight", default=0.2, type=float, required=False)
    parser.add_argument("--mse_loss_weight", default=0.1, type=float, required=False)
    parser.add_argument("--task_loss_weight", default=0.5, type=float, required=False)
    parser.add_argument("--emd_loss_weight", default=0.2, type=float, required=False)
    parser.add_argument("--threshold", default=0.5, type=float, required=False)
    parser.add_argument("--output_dir", default=".", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--calculate_per_class",
                        action='store_true',
                        help="Calculate metrics per class")
    args = parser.parse_args()
    args = vars(args)
    args = dotdict(args)
    set_seed(args.seed)
    main(args)
