import os
from data import load_dataset, load_metric
import torch
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from transformers import AutoTokenizer, BertForTokenClassification
from transformers import (AutoModelForTokenClassification,
                          TrainingArguments,
                          Trainer)

from transformers import DataCollatorForTokenClassification
from transformers import TrainerCallback
import numpy as np
import ujson as json
import pandas as pd
from compressors.distillation.losses import KLDivLoss, MSEHiddenStatesLoss

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

label_list = None

def tokenize_and_align_labels(examples, tokenizer, text_name, label_name, is_split_into_words, label_all_tokens=True):
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


def convert_bytes(num):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def file_size(file_path):
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)


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
def val_iter(student, batch, metric_fn):
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


def dict_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def distill(base_dir, student_model_name, temperature, batch_size=16, num_epochs=5):
    root_dir = str(base_dir/f"{student_model_name}_T-{int(temperature)}")
    # !mkdir -p {root_dir}
    import os
    os.chdir(root_dir)
    global label_list
    writer = SummaryWriter()
    # task = "ner"
    teacher_model_name = "imvladikon/bert-large-cased-finetuned-conll03-english"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_name = "ner_tags"
    text_name = "tokens"
    datasets = load_dataset("conllpp")
    metric_fn = load_metric("seqeval")
    label_list = datasets["train"].features[f"ner_tags"].feature.names
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    tokenize_and_align = lambda examples: tokenize_and_align_labels(examples, tokenizer=tokenizer, text_name=text_name,
                                                                    label_name=label_name, is_split_into_words=True,
                                                                    label_all_tokens=True)
    datasets = datasets.map(tokenize_and_align, batched=True)
    datasets = datasets.remove_columns(['chunk_tags', 'id', 'tokens', 'pos_tags', 'ner_tags'])
    data_collator = DataCollatorForTokenClassification(tokenizer)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=data_collator),
        "valid": DataLoader(datasets["test"], batch_size=batch_size, collate_fn=data_collator),
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
                    )
                else:
                    metrics = val_iter(student=student_model, batch=batch, metric_fn=metric_fn)
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
                    torch.save(student_model.state_dict(), "best_student.pth")
                    best_accuracy = accuracy
            print(f"{loader_key} accuracy: {accuracy}")
            writer.add_scalar(f"{loader_key}/accuracy", accuracy, epoch)
            writer.add_scalar(f"{loader_key}/f1", f1, epoch)

    args = TrainingArguments(
        f"test-ner",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
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
                with open('valid_result.json', 'w') as fp:
                    json.dump(logs, fp)


    student_model = BertForTokenClassification.from_pretrained(student_model_name, num_labels=len(label_list)).to(device)
    student_model.load_state_dict(torch.load("best_student.pth"))

    trainer = Trainer(
        student_model,
        args,
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
    pd.DataFrame(results).to_json("test_result.json")

    train_info = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "temperature": temperature,
        "teacher_model_name": teacher_model_name,
        "student_model_name": student_model_name,
        "model_size": file_size("best_student.pth")}
    with open("train_info.json", "w") as fp:
        json.dump(train_info, fp)


    writer.flush()
    writer.close()


if __name__ == '__main__':
    # distill()
    pass