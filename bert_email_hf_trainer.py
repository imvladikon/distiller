import argparse
from typing import Callable, Dict

import torch
import torchmetrics
from transformers import TrainerCallback, EvalPrediction, TrainerState, TrainerControl
import datasets
from config.google_students_models import get_student_models
from metrics.multiclasseval import Multiclasseval
from const import *
from utils import dotdict, set_seed
from utils.dataloader import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from modeling.bert_multilabel_classification import BertForMultiLabelSequenceClassification

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
import numpy as np

# def compute_metrics(p):
#     predictions, labels = p
#
#     # Remove ignored index (special tokens)
#     true_predictions = [
#         [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#     true_labels = [
#         [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#
#     results = metric.compute(predictions=true_predictions, references=true_labels)
#     return {
#         "precision": results["overall_precision"],
#         "recall": results["overall_recall"],
#         "f1": results["overall_f1"],
#         "accuracy": results["overall_accuracy"],
#     }

acc_metric = datasets.load_metric('accuracy')
f1_metric = datasets.load_metric('f1')
prec_metric = datasets.load_metric('precision')
rec_metric = datasets.load_metric("recall")


def compute_metrics(eval_pred: EvalPrediction) -> Callable[[EvalPrediction], Dict]:
    logits, labels = eval_pred
    labels = torch.LongTensor(labels)
    predictions = torch.FloatTensor(logits)

    acc_metric = torchmetrics.Accuracy(threshold=0.5, num_classes=len(label2id), average="micro")
    f1_metric = torchmetrics.F1(threshold=0.5, num_classes=len(label2id), average="micro")
    prec_metric = torchmetrics.Precision(threshold=0.5, num_classes=len(label2id), average="micro")
    rec_metric = torchmetrics.Recall(threshold=0.5, num_classes=len(label2id), average="micro")

    metrics = {
        'accuracy': acc_metric(predictions, labels),

        'f1': f1_metric(predictions, labels),

        'precision': prec_metric(predictions, labels),

        'recall': rec_metric(predictions, labels)
    }
    return metrics


class MyCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass



def main(args):
    class PrinterCallback(TrainerCallback):

        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                print(logs)

    model_checkpoint = "bert-large-uncased"
    batch_size = 16
    label_list = labels
    metric = Multiclasseval()
    metric.threshold = 0.5
    metric.num_classes = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    teacher_model = BertForMultiLabelSequenceClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
    teacher_model.config.label2id = label2id
    teacher_model.config.id2label = id2label

    hidden_size, num_layers = 256, 6
    student_model_name = get_student_models(hidden_size=hidden_size, num_layers=num_layers)
    student_model = BertForMultiLabelSequenceClassification.from_pretrained(
        student_model_name, num_labels=len(label_list)
    )
    student_model.config.label2id = teacher_model.config.label2id
    student_model.config.id2label = teacher_model.config.id2label

    model = torch.nn.ModuleDict({"teacher": teacher_model, "student": student_model})

    training_args = TrainingArguments(
        f"test",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        # logging_dir="logs"
    )

    ds = load_dataset(
        train_filename=args.train_filename,
        val_filename=args.val_filename,
        tokenizer=tokenizer,
        max_seq_length=512,
        train_format_with_proba=args.train_format_with_proba,
        train_size=args.train_size,
        val_size=args.val_size
    )

    trainer = Trainer(
        teacher_model,
        training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        # data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[PrinterCallback()]
    )
    trainer.train()
    trainer.evaluate()

    # predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])
    # predictions = np.argmax(predictions, axis=2)
    #
    # # Remove ignored index (special tokens)
    # true_predictions = [
    #     [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]
    # true_labels = [
    #     [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]
    #
    # results = metric.compute(predictions=true_predictions, references=true_labels)



def main_train(args):
    class PrinterCallback(TrainerCallback):

        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                print(logs)

    model_checkpoint = "bert-base-uncased"
    batch_size = 16
    label_list = labels
    metric = Multiclasseval()
    metric.threshold = 0.5
    metric.num_classes = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = BertForMultiLabelSequenceClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

    training_args = TrainingArguments(
        f"test",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        # logging_dir="logs"
    )

    ds = load_dataset(
        train_filename=args.train_filename,
        val_filename=args.val_filename,
        tokenizer=tokenizer,
        max_seq_length=512,
        train_format_with_proba=args.train_format_with_proba,
        train_size=args.train_size,
        val_size=args.val_size
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        # data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[PrinterCallback()]
    )
    trainer.train()
    trainer.evaluate()



    # predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])
    # predictions = np.argmax(predictions, axis=2)
    #
    # # Remove ignored index (special tokens)
    # true_predictions = [
    #     [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]
    # true_labels = [
    #     [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]
    #
    # results = metric.compute(predictions=true_predictions, references=true_labels)


if __name__ == "__main__":
    import wandb

    hidden_size, num_layers = 256, 6
    student_model_name = get_student_models(hidden_size=hidden_size, num_layers=num_layers)
    teacher_model_name = ROOT_DIR / 'models' / 'tuned' / 'tuned_bertreply'

    parser = argparse.ArgumentParser(description='Fine-tuning bert')
    parser.add_argument("--student_model_name", default=student_model_name, type=str, required=False)
    parser.add_argument("--teacher_model_name", default=teacher_model_name, type=str, required=False)
    parser.add_argument("--do_train", default=True, type=bool, required=False)
    parser.add_argument("--do_eval", default=True, type=bool, required=False)
    parser.add_argument("--train_filename", default=str(ROOT_DIR / "data" / "0" / "train_weak_label_bin_email_id.csv"),
                        type=str, required=False)
    parser.add_argument("--val_filename", default=str(ROOT_DIR / "data" / "0" / "test.csv"), type=str, required=False)
    parser.add_argument("--one_cycle_train", default=True, type=bool, required=False)
    parser.add_argument("--train_format_with_proba", default=False, type=bool, required=False)
    parser.add_argument("--train_size", default=-1, type=int, required=False)
    parser.add_argument("--val_size", default=-1, type=int, required=False)
    parser.add_argument("--data_dir", default=str(ROOT_DIR / 'data'), type=str, required=False)
    parser.add_argument("--task_name", default='email_reject', type=str, required=False)
    parser.add_argument("--bert_tokenizer", default="bert-base-uncased", type=str, required=False)
    parser.add_argument("--output_model_dir", default=str(ROOT_DIR / 'models' / 'distill'), type=str, required=False)
    parser.add_argument("--data_output_dir", default=(ROOT_DIR / 'data' / 'class' / 'output'), type=str, required=False)
    parser.add_argument("--max_seq_length", default=512, type=int, required=False)
    parser.add_argument("--do_lower_case", default=True, type=bool, required=False)
    parser.add_argument("--train_batch_size", default=24, type=int, required=False)
    parser.add_argument("--val_batch_size", default=12, type=int, required=False)
    parser.add_argument("--n_threads", default=4, type=int, required=False)
    parser.add_argument("--learning_rate", default=3e-5, required=False)
    parser.add_argument("--num_train_epochs", default=5, type=int, required=False)
    parser.add_argument("--warmup_linear", default=0.1, type=float, required=False)
    parser.add_argument("--no_cuda", default=False, type=bool, required=False)
    parser.add_argument("--local_rank", default=-1, type=int, required=False)
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, required=False)
    parser.add_argument("--optimize_on_cpu", default=True, type=bool, required=False)
    parser.add_argument("--fp16", default=False, type=bool, required=False)
    parser.add_argument("--loss_scale", default=128, type=int, required=False)
    parser.add_argument("--labels_list", default=labels, type=list, required=False)
    parser.add_argument("--use_wandb", default=True, type=bool, required=False)
    parser.add_argument("--wandb_token", default='028b28de5b2e1acd20824041eaf39c98d5ca1eab', type=str, required=False)
    parser.add_argument("--temperature", default=1, type=float, required=False)
    parser.add_argument("--kl_div_loss_weight", default=0.2, type=float, required=False)
    parser.add_argument("--mse_loss_weight", default=0.3, type=float, required=False)
    parser.add_argument("--task_loss_weight", default=0.5, type=float, required=False)
    parser.add_argument("--threshold", default=0.5, type=float, required=False)
    parser.add_argument("--additional_data", default=str(ROOT_DIR / 'data' / 'unlabeled_data'), type=str,
                        required=False)

    args = parser.parse_args()
    args = vars(args)
    args = dotdict(args)
    set_seed(args.seed)

    os.environ["WANDB_API_KEY"] = args.wandb_token

    main_train(args)
