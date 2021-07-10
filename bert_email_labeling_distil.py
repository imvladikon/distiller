import argparse
from pathlib import Path

import matplotlib
from catalyst.callbacks import ControlFlowCallback, OptimizerCallback
from catalyst.callbacks.metric import LoaderMetricCallback
from catalyst.loggers import WandbLogger
from datasets import load_metric
from transformers import AutoTokenizer

from config.google_students_models import get_student_models
from modeling.bert_multilabel_classification import BertForMultiLabelSequenceClassification

from const import labels, device, ROOT_DIR
from utils import set_seed, dotdict
from utils.dataloader import load_dataset, datasets_as_loaders

matplotlib.use("agg")

import logging
import torch

from modeling.gong import bert_seq_classification as bsc

import pandas as pd
import wandb

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 100)

from compressors.distillation.callbacks import (
    HiddenStatesSelectCallback,
    KLDivCallback,
    LambdaPreprocessCallback,
    MetricAggregationCallback,
    MSEHiddenStatesCallback,
)
from compressors.distillation.runners import HFDistilRunner
from compressors.metrics.hf_metric import HFMetric


def main(args):
    if 'n_threads' in args:
        torch.set_num_threads(args['n_threads'])
        logger.info(f"Setting #threads to {args['n_threads']}")

    logger.info(f"device: {device}")
    logger.info(f"numbers of gpu: {torch.cuda.device_count()}")
    logger.info(f'teacher model: {str(args.teacher_model_name)}')
    logger.info(f'student model: {str(args.student_model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer)

    label_list = labels
    teacher_model = BertForMultiLabelSequenceClassification.from_pretrained(args.teacher_model_name,
                                                                            num_labels=len(label_list))
    teacher_model.to(device)

    ds = load_dataset(
        train_filename=args.train_filename,
        val_filename=args.val_filename,
        tokenizer=tokenizer,
        max_seq_length=512,
        train_format_with_proba=args.train_format_with_proba,
        train_size=args.train_size,
        val_size=args.val_size
    )
    if args.additional_data:
        import glob
        import os
        from datasets import concatenate_datasets

        for f in glob.glob(f"{args.additional_data}/*.csv"):
            if not os.path.exists(f): continue
            f = str(ROOT_DIR / "data/0/train_weak_label_bin_email_id.csv")
            new_ds = load_dataset(
                train_filename=f,
                tokenizer=tokenizer,
                max_seq_length=512,
                train_format_with_proba=True,
                threshold=args.threshold
            )
            ds["train"] = concatenate_datasets([ds["train"], new_ds["train"]])

    loaders = datasets_as_loaders(ds, batch_size=args.train_batch_size, val_batch_size=args.val_batch_size)

    student_model = bsc.BertForMultiLabelSequenceClassification.from_pretrained(
        args.student_model_name, num_labels=len(label_list)
    )

    student_model.config.label2id = teacher_model.config.label2id
    student_model.config.id2label = teacher_model.config.id2label

    ############### Distillation ##################

    num_teacher_layers = teacher_model.config.num_hidden_layers + 1
    num_student_layers = student_model.config.num_hidden_layers + 1

    map_layers = {
        2: [1, 3],
        4: [1, 3, 5, 7],
        6: [1, 3, 5, 7, 9, 11],
        8: [1, 2, 3, 5, 7, 9, 11, 13],
        10: [1, 2, 3, 4, 5, 6, 7, 9, 11, 13],
    }
    if num_student_layers < num_teacher_layers and num_student_layers in map_layers:
        slct_callback = ControlFlowCallback(
            HiddenStatesSelectCallback(hiddens_key="t_hidden_states", layers=map_layers[num_student_layers]),
            loaders="train",
        )

    lambda_hiddens_callback = ControlFlowCallback(
        LambdaPreprocessCallback(
            lambda s_hiddens, t_hiddens: (
                [c_s[:, 0] for c_s in s_hiddens],
                [t_s[:, 0] for t_s in t_hiddens],  # tooks only CLS token
            )
        ),
        loaders="train",
    )

    mse_hiddens = ControlFlowCallback(MSEHiddenStatesCallback(
        normalize=True,
        need_mapping=True,
        teacher_hidden_state_dim=teacher_model.config.hidden_size,
        student_hidden_state_dim=student_model.config.hidden_size,
        num_layers=student_model.config.num_hidden_layers,
        device=device
    ), loaders="train")

    kl_div = ControlFlowCallback(KLDivCallback(temperature=args.temperature), loaders="train")

    loss_weights = {
        "kl_div_loss": args.kl_div_loss_weight,
        "mse_loss": args.mse_loss_weight,
        "task_loss": args.task_loss_weight
    }
    aggregator = ControlFlowCallback(
        MetricAggregationCallback(
            prefix="loss",
            metrics=loss_weights,
            mode="weighted_sum",
        ),
        loaders="train",
    )

    runner = HFDistilRunner()

    teacher_model.config.output_hidden_states = True
    student_model.config.output_hidden_states = True

    metric = load_metric(str(ROOT_DIR / 'metrics' / 'multiclasseval.py'),
                         threshold=args.threshold,
                         num_classes=len(label_list))
    metric.threshold = args.threshold
    metric.num_classes = len(label_list)

    # regression is setting to True, for avoiding of calculating logits.argmax(-1) in HFMetric
    metric_callback = LoaderMetricCallback(
        metric=HFMetric(metric=metric,
                        regression=True),
        input_key="s_logits", target_key="labels",
    )

    if args.use_wandb:

        import os

        if args.wandb_token:
            os.environ["WANDB_API_KEY"] = args.wandb_token
            del args["wandb_token"]

    callbacks = [
        # metric_callback,
        lambda_hiddens_callback,
        mse_hiddens,
        kl_div,
        aggregator,
        OptimizerCallback(metric_key="loss"),
    ]
    if num_student_layers < num_teacher_layers and num_student_layers in map_layers:
        callbacks = [
            # metric_callback,
            slct_callback,
            *callbacks
        ]
    callbacks = [
        metric_callback,
        *callbacks
    ]
    wandb_logger = None
    if args.use_wandb:
        t_model_name = os.path.basename(teacher_model_name) if os.path.isabs(teacher_model_name) else teacher_model_name
        s_model_name = os.path.basename(student_model_name) if os.path.isabs(student_model_name) else student_model_name
        s_model_name = f"{s_model_name}_T-{args.temperature}"
        wandb_logger = WandbLogger(project="distill_bert",
                                   name=f"distill_t_{t_model_name}_s_{s_model_name}")
        run = wandb_logger.run
        run.config.update({
            **dict(args),
            "student_hidden_size": student_model.config.hidden_size,
            "student_num_hidden_layers": student_model.config.num_hidden_layers,
            "student_num_attention_heads": student_model.config.num_attention_heads,
            **loss_weights
        })
        run.tags = [t_model_name, s_model_name]
        wandb.watch(student_model)

    # callbacks = [WandbLogger(project="catalyst", name='Example'), logging_params = {params}]
    if args.use_wandb:
        runner.train(
            model=torch.nn.ModuleDict({"teacher": teacher_model, "student": student_model}),
            loaders=loaders,
            optimizer=torch.optim.Adam(student_model.parameters(), lr=args.learning_rate),
            callbacks=callbacks,
            num_epochs=args.num_train_epochs,
            valid_metric="accuracy",
            minimize_valid_metric=False,
            valid_loader="valid",
            verbose=True,
            loggers={"wandb_logger": wandb_logger}
        )
    else:
        runner.train(
            model=torch.nn.ModuleDict({"teacher": teacher_model, "student": student_model}),
            loaders=loaders,
            optimizer=torch.optim.Adam(student_model.parameters(), lr=args.learning_rate),
            callbacks=callbacks,
            num_epochs=args.num_train_epochs,
            valid_metric="accuracy",
            minimize_valid_metric=False,
            valid_loader="valid",
            verbose=True
        )

    output_model_dir = Path(args.output_model_dir) / s_model_name
    output_model_dir.mkdir(parents=True, exist_ok=True)
    student_model.save_pretrained(output_model_dir)
    if args.use_wandb:
        wandb.save(output_model_dir)


if __name__ == "__main__":
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
    parser.add_argument("--train_format_with_proba", default=True, type=bool, required=False)
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
    parser.add_argument("--use_wandb", default=False, type=bool, required=False)
    parser.add_argument("--wandb_token", default='', type=str, required=False)
    parser.add_argument("--temperature", default=1, type=int, required=False)
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
    main(args)
