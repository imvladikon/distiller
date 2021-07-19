import ast
import logging
import os
from abc import ABC
from collections import Counter, OrderedDict
from typing import List, Optional, Callable, Dict

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset
from datasets import DatasetDict
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class DataProcessor(ABC):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir, data_file_name, size=-1):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, document_id=None, text_a="", labels=None, guid=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.document_id = document_id
        self.guid = hash(document_id) if guid is None else guid
        self.text_a = text_a
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, guid, input_ids, input_mask, segment_ids, label_ids):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class BinaryLabelTextProcessor(DataProcessor):
    def __init__(self, labels=None):
        if labels is None:
            labels = [1]
        self.labels = labels

    def get_train_examples(self, args, size=-1):
        """
        The train.csv should be in the following format:
        document_id: a unique number as document id
        document_text: the text related to this document (remember that the tokens are truncated according to max_seq_length)
        label: an integer corresponding with the label of the document.

        For example:

            document_id, document_text, label
            1, this is a good thing, 0
            2, I really hate this, 1
            3, This is the shit!, 0

        :param args:
        :param size:
        :return:
        """
        if 'train_fname' not in args:
            args['train_fname'] = 'train.csv'
        logger.info("LOOKING AT {}".format(os.path.join(args.data_dir, args.train_fname)))
        data_df = pd.read_csv(os.path.join(args.data_dir, args.train_fname), dtype=str)
        if size == -1:
            return self._create_examples(data_df)
        else:
            return self._create_examples(data_df.sample(size))

    def get_dev_examples(self, args, size=-1):
        """See base class."""
        if 'eval_fname' in args:
            input_fname = args["eval_fname"]
        else:
            input_fname = args["train_fname"]
        logger.info(f"Reading {os.path.join(args['data_dir'], input_fname)}")
        data_df = pd.read_csv(os.path.join(args['data_dir'], input_fname), dtype=str)
        if size == -1:
            return self._create_examples(data_df)
        else:
            return self._create_examples(data_df.sample(size))

    def get_test_examples(self, data_dir, data_file_name, size=-1):
        data_df = pd.read_csv(os.path.join(data_dir, data_file_name), dtype=str)
        #         data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
        if size == -1:
            return self._create_examples(data_df)
        else:
            return self._create_examples(data_df.sample(size))

    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples(self, df):
        """Creates examples for the training and dev sets."""
        logger.info(f'Read {df.shape[0]} entries')
        df['document_text'] = df.document_text.str.lower().str.replace('(https?|ftp|file|zoom|aws|mailto)\:\S+', ' ')
        # if 'subject' in df.columns:
        #     df['document_text'] = df['subject'] + ' ' + df['document_text']        # if 'subject' in df.columns:
        #     df['document_text'] = df['subject'] + ' ' + df['document_text']

        if 'label_text' in df.columns:
            df['label_text'] = df['label_text'].fillna('{}')
            df['label_text'] = df['label_text'].str.lower().apply(ast.literal_eval).apply(list)
        else:
            df['label_text'] = 'no_label'
            df['label_text'] = df['label_text'].apply(lambda x: [x])

        gdf = df.groupby('document_id').agg({'label_text': sum, 'document_text': 'first', }).reset_index()
        gdf['label_text'] = gdf['label_text'].apply(set)
        c = Counter()
        _ = [c.update(el) for el in df["label_text"] for x in el]
        logger.info(f'Labels {c}\n')
        logger.info(f'Processing {gdf.shape[0]} entries')

        res = gdf.apply(
            lambda x: InputExample(document_id=str(x['document_id']),
                                   text_a=x['document_text'],
                                   labels=list(x['label_text'])),
            axis=1).tolist()
        return res


def _create_examples(df):
    """
    Creates examples for the training and dev sets.
    """

    logger.info(f'Read {df.shape[0]} entries')
    df['document_text'] = df.document_text.str.lower().str.replace('(https?|ftp|file|zoom|aws|mailto)\:\S+', ' ')
    # if 'subject' in df.columns:
    #     df['document_text'] = df['subject'] + ' ' + df['document_text']        # if 'subject' in df.columns:
    #     df['document_text'] = df['subject'] + ' ' + df['document_text']

    if 'label_text' in df.columns:
        df['label_text'] = df['label_text'].fillna('{}')
        df['label_text'] = df['label_text'].str.lower().apply(ast.literal_eval).apply(list)
    else:
        df['label_text'] = 'no_label'
        df['label_text'] = df['label_text'].apply(lambda x: [x])

    gdf = df.groupby('document_id').agg({'label_text': sum, 'document_text': 'first', }).reset_index()
    gdf['label_text'] = gdf['label_text'].apply(set)
    c = Counter()
    _ = [c.update(el) for el in df["label_text"] for x in el]
    logger.info(f'Labels {c}\n')
    logger.info(f'Processing {gdf.shape[0]} entries')

    res = gdf.apply(
        lambda x: InputExample(document_id=str(x['document_id']),
                               text_a=x['document_text'],
                               labels=list(x['label_text'])),
        axis=1).tolist()
    return res


def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 cut_long=True,
                                 is_multilabel=True,
                                 is_multilabel_with_proba=False,
                                 threshold=0.5):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label.lower(): i for i, label in enumerate(label_list)}

    features = []
    for ex_index, example in enumerate(examples):
        try:
            if len(example.text_a) > 50000:
                logger.warning(f'Skipping email that is longer than 50k characters')
                raise
            tokens_a = tokenizer.tokenize(example.text_a)
        except:
            logger.warning(f'Failed to tokenize email_id= {example.document_id}')
            tokens_a = [tokenizer.unk_token]

        unk_count = tokens_a.count([tokenizer.unk_token])
        if unk_count > 10:
            logger.warning(
                f'Found {unk_count} {tokenizer.unk_token} in  [{example.document_id}]: \"{example.text_a[:500]}\"')

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        OVERLAP = 30  # Tokens
        while len(tokens_a) > 0:
            cut_position = max_seq_length - 2
            tokens = tokens_a[:cut_position]
            if not cut_long:
                if cut_position > len(tokens_a):
                    cut_position -= OVERLAP
                if len(tokens_a) < max_seq_length + 2 * OVERLAP:
                    cut_position -= OVERLAP

            tokens_a = tokens_a[cut_position:]

            # Account for [CLS] and [SEP] with "-2"
            tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if is_multilabel:  # multi label
                labels_ids = [int(l in example.labels) for l in
                              label_map]  # [1.0 if example.labels==lbl else 0.0 for lbl in label_map]  # this should be used for multi-label
            elif is_multilabel_with_proba:
                labels_ids = [int(proba > threshold) for proba in example.labels]
            else:  # multi class
                labels_ids = label_map[example.labels[0].lower()] if example.labels[
                                                                         0].lower() in label_map else 0  # this should be used for multi-class

            features.append(
                InputFeatures(guid=example.guid,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=labels_ids))

            if cut_long:
                break

    return features


def load_dataset(
        train_filename: str,
        val_filename: str = None,
        max_seq_length: int = 512,
        tokenizer: transformers.AutoTokenizer = None,
        train_format_with_proba: bool = False,
        text_column: str = "",
        label_column: str = "",
        train_size: int = -1,
        val_size: int = -1,
        columns: Optional[List] = None,
        threshold: float = 0.5
) -> DatasetDict:
    # TODO : add column names (text_column , label_column)
    # TODO : add calculating max_seq_length from dataset if it's None

    # train_df = pd.read_csv("data/0/train.csv").reset_index(drop=True)
    # test_df = pd.read_csv("data/0/test.csv").reset_index(drop=True)
    # train_df["label_text"] = train_df.label_text.map(literal_eval)
    # test_df["label_text"] = test_df.label_text.map(literal_eval)

    if columns is None:
        columns = ["input_ids", "token_type_ids", "attention_mask", "labels"]

    processor = BinaryLabelTextProcessor(labels)
    label_list = processor.get_labels()

    logger.info("LOOKING AT {}".format(train_filename))
    if train_format_with_proba:
        data_df = pd.read_csv(train_filename).reset_index(drop=True)
        lbls = ['proba_' + el.lower() for el in label_list]
        train_examples = data_df.apply(
            lambda x: InputExample(guid=hash(x['document_id']),
                                   text_a=x['document_text'],
                                   labels=x[lbls].tolist()),
            axis=1).tolist()
    else:
        data_df = pd.read_csv(train_filename, dtype=str)
        if train_size == -1:
            train_examples = _create_examples(data_df)
        else:
            train_examples = _create_examples(data_df.sample(train_size))

    if not train_format_with_proba:
        train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer)
    else:
        logger.info("proba threshold {}".format(threshold))
        train_features = convert_examples_to_features(train_examples,
                                                      label_list,
                                                      max_seq_length,
                                                      tokenizer,
                                                      is_multilabel=False,
                                                      is_multilabel_with_proba=True,
                                                      threshold=threshold)

    def to_dict(record):
        return {
            "input_ids": record.input_ids,
            "token_type_ids": record.segment_ids,
            "attention_mask": record.input_mask,
            "labels": record.label_ids,
            # "guid": record.guid
        }
        # return {a: getattr(record, a) for a in ["input_ids", "input_mask", "label_ids", "segment_ids"]}

    train_df = pd.DataFrame([to_dict(record) for record in train_features])

    if val_filename and os.path.exists(val_filename):
        logger.info(f"Reading {val_filename}")
        val_df = pd.read_csv(val_filename, dtype=str)
        if val_size == -1:
            eval_examples = _create_examples(val_df)
        else:
            eval_examples = _create_examples(val_df.sample(val_size))

        eval_features = convert_examples_to_features(
            eval_examples,
            label_list,
            max_seq_length,
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
    else:
        datasets = DatasetDict(
            dict(
                train=Dataset.from_pandas(train_df),
            )
        )

    # datasets = datasets.map(
    #     lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=128),
    #     batched=True,
    # )
    # datasets = datasets.map(lambda e: {"labels": e["label"]}, batched=True)

    datasets.set_format(
        type="torch", columns=columns
    )

    return datasets


def datasets_as_loaders(ds: DatasetDict, batch_size: int, val_batch_size: int = None) -> 'OrderedDict':
    if val_batch_size is None:
        val_batch_size = batch_size

    # TODO : add collate_fn=
    loaders = OrderedDict({
        "train": DataLoader(ds["train"], batch_size=batch_size, shuffle=True),
        "valid": DataLoader(ds["test"], batch_size=val_batch_size),
    })
    return loaders


if __name__ == '__main__':
    from transformers import BertTokenizerFast
    from const import ROOT_DIR

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    ds = load_dataset(
        train_filename=str(ROOT_DIR / "data" / "0" / "train.csv"),
        val_filename=str(ROOT_DIR / "data" / "0" / "test.csv"),
        tokenizer=tokenizer,
        max_seq_length=512,
    )
    print(len(ds["train"]))

    loaders = datasets_as_loaders(ds, batch_size=64)

    print(len(loaders["train"]))
