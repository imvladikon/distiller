import os
from typing import Dict, Callable
import logging

import numpy as np
import pandas as pd
import transformers
from datasets import DatasetDict, Dataset

logger = logging.getLogger(__name__)


def read_as_dataset(
        filename: str,
        map_label_columns: Dict,
        text_column: str,
        data_size: int = -1,
        transform_label_fn: Callable = None,
        processing_data_fn: Callable = None,
        print_label_dist: bool = True
) -> DatasetDict:
    label_list = list(map_label_columns.keys())
    label_cols = list(map_label_columns.values())
    id2label = {k: v for k, v in enumerate(label_list)}
    label2id = {v: k for k, v in id2label.items()}
    logger.info("LOOKING AT {}".format(filename))
    if data_size != -1:
        df = pd.read_csv(filename).sample(data_size).reset_index(drop=True)
    else:
        df = pd.read_csv(filename).reset_index(drop=True)
    if processing_data_fn is not None:
        df = processing_data_fn(df, text_column)
    label_column = "labels"
    if print_label_dist:
        freq_map = {
            col: list(zip(*np.unique(df[label_cols[0]].values, return_counts=True)))
            for col in label_cols
        }
        print(pd.DataFrame(freq_map).T)

    df[label_column] = list(df[label_cols].values)
    if transform_label_fn is not None:
        df[label_column] = df[label_column].map(transform_label_fn)
    ds = Dataset.from_pandas(df[[text_column, label_column]])
    return ds


def read_data(
        train_filename: str,
        map_label_columns: Dict,
        text_column: str,
        val_filename: str = None,
        max_seq_length: int = 512,
        tokenizer: transformers.AutoTokenizer = None,
        train_size: int = -1,
        val_size: int = -1,
        transform_label_fn: Callable = None,
        processing_data_fn: Callable = None,
        val_split_size: float = 0,
        print_label_dist: bool = True
):
    train_ds = read_as_dataset(train_filename,
                               map_label_columns,
                               text_column,
                               train_size,
                               transform_label_fn,
                               processing_data_fn,
                               print_label_dist=print_label_dist)
    if val_filename and os.path.exists(val_filename):
        val_ds = read_as_dataset(val_filename,
                                 map_label_columns,
                                 text_column,
                                 val_size,
                                 transform_label_fn,
                                 processing_data_fn,
                                 print_label_dist=print_label_dist)

        datasets = DatasetDict(
            dict(
                train=train_ds,
                test=val_ds
                # test=Dataset.from_pandas(test_df)
            )
        )
    elif val_split_size > 0:
        datasets = DatasetDict(
            train_ds.train_test_split(test_size=val_split_size)
        )
    else:
        datasets = DatasetDict(
            dict(
                train=train_ds
            )
        )

    if tokenizer is not None:
        cols = datasets["train"].column_names
        cols.remove("labels")

        datasets = datasets.map(
            lambda e: tokenizer(e[text_column], truncation=True, padding="max_length", max_length=max_seq_length),
            batched=True,
            remove_columns=cols
        )

        datasets.set_format(
            type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"]
        )
    return datasets
