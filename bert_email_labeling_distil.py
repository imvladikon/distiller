import matplotlib
from catalyst.callbacks import ControlFlowCallback, OptimizerCallback
from catalyst.callbacks.metric import LoaderMetricCallback
from datasets import load_metric
from transformers import AutoTokenizer
from modeling.bert_multilabel_classification import BertForMultiLabelSequenceClassification

from const import labels, device, ROOT_DIR
from utils import set_seed, dotdict
from utils.dataloader import load_dataset, datasets_as_loaders

matplotlib.use("agg")

import logging
import torch

from modeling.gong import bert_seq_classification as bsc

import pandas as pd

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

    logger.info(f"device: {device} \t #number of gpu: {args.n_gpu}")
    logger.info(f'Using model: {str(args.bert_model_dir)}')

    tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer)

    label_list = labels
    num_labels = len(label_list)

    teacher_model = BertForMultiLabelSequenceClassification.from_pretrained(args.teacher_model_name,
                                                                                num_labels=len(label_list))
    teacher_model.to(device)

    ds = load_dataset(
        train_filename=str(ROOT_DIR / "data" / "0" / "train.csv"),
        val_filename=str(ROOT_DIR / "data" / "0" / "test.csv"),
        tokenizer=tokenizer,
        max_seq_length=512,
    )
    loaders = datasets_as_loaders(ds, batch_size=args.batch_size)

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

    # mse_hiddens = ControlFlowCallback(MSEHiddenStatesCallback(), loaders="train")

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
        num_epochs=args.num_epochs,
        valid_metric="accuracy",
        minimize_valid_metric=False,
        valid_loader="valid",
        verbose=True
    )

    student_model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    # TODO: add argparse
    from config import default

    args = default.args
    args["teacher_model_name"] = ROOT_DIR / 'models' / 'tuned' / 'tuned_bertreply'
    args["student_model_name"] = 'google/bert_uncased_L-2_H-128_A-2'
    args["num_epochs"] = 5
    args["batch_size"] = 1
    args = dotdict(args)
    set_seed(args.seed)
    label_list = labels

    main(args)
