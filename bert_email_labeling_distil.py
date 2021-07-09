import pickle

from bert_email_labeling import TRAIN_LABELS, DATA_PATH, chekpoint_path, BERT_PRETRAINED_PATH, BERT_FINETUNED_PATH, \
    CLAS_DATA_PATH

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, load_metric

from catalyst.callbacks import ControlFlowCallback, OptimizerCallback
from catalyst.callbacks.metric import LoaderMetricCallback
import matplotlib

matplotlib.use("agg")

import logging
import os
import torch

from modeling.gong import bert_seq_classification as bsc
from modeling.gong.bert_seq_classification import InputExample
from misc import dotdict
from collections import Counter

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_curve, classification_report, roc_auc_score, average_precision_score

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
from compressors.runners.hf_runner import HFRunner


def main(args):
    # device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpu = torch.cuda.device_count()

    # if args.do_train:
    #     logger.info('Training')
    #     logger.info(f"device: {device} \t #gpu: {n_gpu}")
    #     logger.info(f'Using model: {str(args.bert_model_dir)}')
    #
    #     bsc.set_seed(args.seed)
    #     tokenizer = bsc.BertTokenizer.from_pretrained(args.bert_tokenizer)
    #     processor = bsc.BinaryLabelTextProcessor(TRAIN_LABELS)
    #
    #     label_list = processor.get_labels()
    #
    #     if args.train_format_with_proba:
    #         data_df = pd.read_csv(os.path.join(args.data_dir, args.input_fname))
    #         lbls = ['proba_' + el.lower() for el in TRAIN_LABELS]
    #         train_examples = data_df.apply(
    #             lambda x: InputExample(guid=hash(x['document_id']),
    #                                    text_a=x['document_text'],
    #                                    labels=x[lbls].tolist()),
    #             axis=1).tolist()
    #     else:
    #         train_examples = processor.get_train_examples(args, size=args.train_size)
    #
    #     train_features = bsc.convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)

    processor = bsc.BinaryLabelTextProcessor(TRAIN_LABELS)

    tokenizer = bsc.BertTokenizer.from_pretrained(args.bert_tokenizer)

    label_list = processor.get_labels()
    num_labels = len(label_list)

    teacher_model = bsc.BertForMultiLabelSequenceClassification.from_pretrained(args.teacher_model_name,
                                                                                num_labels=len(label_list))
    teacher_model.to(device)

    if os.path.exists("train_features.pkl"):
        train_features = pickle.load(open("train_features.pkl", "rb"))
    else:
        label_list = processor.get_labels()

        if args.train_format_with_proba:
            data_df = pd.read_csv(os.path.join(args.data_dir, args.input_fname))
            lbls = ['proba_' + el.lower() for el in TRAIN_LABELS]
            train_examples = data_df.apply(
                lambda x: InputExample(guid=hash(x['document_id']),
                                       text_a=x['document_text'],
                                       labels=x[lbls].tolist()),
                axis=1).tolist()
        else:
            train_examples = processor.get_train_examples(args, size=args.train_size)

        train_features = bsc.convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)


    def to_dict(record):
        return {
            "input_ids": record.input_ids,
            "token_type_ids": record.segment_ids,
            "attention_mask": record.input_mask,
            "labels": record.label_ids
        }
        # return {a: getattr(record, a) for a in ["input_ids", "input_mask", "label_ids", "segment_ids"]}

    # "input_ids", "token_type_ids", "attention_mask", "labels"

    train_df = pd.DataFrame([to_dict(record) for record in train_features])

    from sklearn.model_selection import train_test_split
    from datasets import Dataset, DatasetDict

    args.eval_fname = 'test.csv'
    eval_examples = processor.get_dev_examples(args)
    eval_features = bsc.convert_examples_to_features(
        eval_examples,
        label_list,
        args.max_seq_length,
        tokenizer,
        cut_long=True)

    val_df = pd.DataFrame([to_dict(record) for record in eval_features])

    datasets = DatasetDict(
        dict(
            train=Dataset.from_pandas(train_df),
            test=Dataset.from_pandas(val_df)
            # test=Dataset.from_pandas(test_df)
        )
    )

    # datasets = datasets.map(
    #     lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=128),
    #     batched=True,
    # )
    # datasets = datasets.map(lambda e: {"labels": e["label"]}, batched=True)
    datasets.set_format(
        type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    )


    loaders = {
        "train": DataLoader(datasets["train"], batch_size=1, shuffle=True),
        "valid": DataLoader(datasets["test"], batch_size=1),
    }
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
    args = {
        "do_train": True,
        "do_eval": True,
        "one_cycle_train": True,
        "train_format_with_proba": False,
        # "input_fname": 'train_sel+cur+aug.csv',
        "train_size": -1,
        "val_size": -1,
        "data_dir": DATA_PATH,
        "task_name": "email_reject",
        "bert_tokenizer": Path(chekpoint_path),
        "bert_model_dir": BERT_PRETRAINED_PATH,
        "output_model_dir": BERT_FINETUNED_PATH,
        "data_output_dir": CLAS_DATA_PATH,
        "max_seq_length": 512,
        "do_lower_case": True,
        "train_batch_size": 24,
        "eval_batch_size": 12,
        "n_threads": 4,
        "learning_rate": 3e-5,
        "num_train_epochs": 3,
        "warmup_linear": False,
        "warmup_proportion": 0.1,
        "no_cuda": False,
        "local_rank": -1,
        "seed": 888,
        "gradient_accumulation_steps": 1,
        "optimize_on_cpu": True,
        "fp16": False,
        "loss_scale": 128,
        "teacher_model_name": "bert-base-uncased",
        "student_model_name": "google/bert_uncased_L-2_H-128_A-2",
        "output_dir": ".",
        "num_epochs": 5
    }
    args = dotdict.dotdict(args)

    main(args)
