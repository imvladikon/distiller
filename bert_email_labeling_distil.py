import argparse
from pathlib import Path

import matplotlib
from catalyst.callbacks import ControlFlowCallback, OptimizerCallback
from catalyst.callbacks.metric import LoaderMetricCallback
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
    logger.info(f'Using model: {str(args.bert_model_dir)}')

    tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer)

    label_list = labels
    teacher_model = BertForMultiLabelSequenceClassification.from_pretrained(args.teacher_model_name,
                                                                            num_labels=len(label_list))
    teacher_model.to(device)

    ds = load_dataset(
        train_filename=str(ROOT_DIR / "data" / "0" / "train.csv"),
        val_filename=str(ROOT_DIR / "data" / "0" / "test.csv"),
        tokenizer=tokenizer,
        max_seq_length=512,
    )
    loaders = datasets_as_loaders(ds, batch_size=args.train_batch_size, val_batch_size=args.val_batch_size)

    metric_callback = LoaderMetricCallback(
        metric=HFMetric(metric=load_metric("accuracy")), input_key="logits", target_key="labels",
    )

    ############### Distillation ##################

    slct_callback = ControlFlowCallback(
        HiddenStatesSelectCallback(hiddens_key="t_hidden_states", layers=[1, 3]), loaders="train",
    )
    # should len(layers)>=len(student)

    lambda_hiddens_callback = ControlFlowCallback(
        LambdaPreprocessCallback(
            lambda s_hiddens, t_hiddens: (
                [c_s[:, 0] for c_s in s_hiddens],
                [t_s[:, 0] for t_s in t_hiddens],  # tooks only CLS token
            )
        ),
        loaders="train",
    )

    student_model = bsc.BertForMultiLabelSequenceClassification.from_pretrained(
        args.student_model_name, num_labels=len(label_list)
    )

    student_model.config.label2id = teacher_model.config.label2id
    student_model.config.id2label = teacher_model.config.id2label

    mse_hiddens = ControlFlowCallback(MSEHiddenStatesCallback(
        normalize=True,
        need_mapping=True,
        teacher_hidden_state_dim=teacher_model.config.hidden_size,
        student_hidden_state_dim=student_model.config.hidden_size,
        num_layers=student_model.config.num_hidden_layers,
    ), loaders="train")

    kl_div = ControlFlowCallback(KLDivCallback(temperature=4), loaders="train")

    aggregator = ControlFlowCallback(
        MetricAggregationCallback(
            prefix="loss",
            metrics={"kl_div_loss": 0.2, "mse_loss": 0.2, "task_loss": 0.6},
            mode="weighted_sum",
        ),
        loaders="train",
    )

    runner = HFDistilRunner()

    teacher_model.config.output_hidden_states = True
    student_model.config.output_hidden_states = True

    metric_callback = LoaderMetricCallback(
        metric=HFMetric(metric=load_metric("accuracy")), input_key="s_logits", target_key="labels",
    )
    # load_metric("f1")
    if args.use_wandb:
        wandb.watch(student_model, log_freq=100)

    runner.train(
        model=torch.nn.ModuleDict({"teacher": teacher_model, "student": student_model}),
        loaders=loaders,
        optimizer=torch.optim.Adam(student_model.parameters(), lr=1e-4),
        callbacks=[
            # metric_callback,
            slct_callback,
            lambda_hiddens_callback,
            mse_hiddens,
            kl_div,
            aggregator,
            OptimizerCallback(metric_key="loss"),
        ],
        num_epochs=args.num_train_epochs,
        valid_metric="accuracy",
        minimize_valid_metric=False,
        valid_loader="valid",
        verbose=True,
        # use_wandb=args.use_wandb
    )

    # TODO : add callback wandb
    #     for batch_idx, (data, target) in enumerate(train_loader):
    #         output = model(data)
    #         loss = F.nll_loss(output, target)
    #         loss.backward()
    #         optimizer.step()
    #         if batch_idx % args.log_interval == 0:
    #             wandb.log({"loss": loss})

    output_model_dir = Path(args.output_model_dir) / student_model_name
    output_model_dir.mkdir(parents=True, exist_ok=True)
    student_model.save_pretrained(output_model_dir)


if __name__ == "__main__":
    hidden_size, num_layers = 256, 4
    student_model_name = get_student_models(hidden_size=hidden_size, num_layers=num_layers)

    parser = argparse.ArgumentParser(description='Fine-tuning bert')
    parser.add_argument("-i", "--student_model_name", default=student_model_name, type=str, required=False)
    parser.add_argument("-s", "--teacher_model_name", default='bert-base-uncased', type=str, required=False)
    parser.add_argument("--do_train", default=True, type=bool, required=False)
    parser.add_argument("--do_eval", default=True, type=bool, required=False)
    parser.add_argument("--one_cycle_train", default=True, type=bool, required=False)
    parser.add_argument("--train_format_with_proba", default=False, type=bool, required=False)
    parser.add_argument("--train_size", default=-1, type=int, required=False)
    parser.add_argument("--val_size", default=-1, type=int, required=False)
    parser.add_argument("--data_dir", default=str(ROOT_DIR / 'data'), type=str, required=False)
    parser.add_argument("--task_name", default='email_reject', type=str, required=False)
    parser.add_argument("--bert_tokenizer", default="bert-base-uncased", type=str, required=False)
    parser.add_argument("--bert_model_dir", default='bert-base-uncased', type=str, required=False)
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
    args = parser.parse_args()
    args = vars(args)
    # TODO: log args
    if args["use_wandb"]:
        wandb.init(config=args)
    args = dotdict(args)
    set_seed(args.seed)
    main(args)