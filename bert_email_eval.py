import argparse

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from const import *
from metrics.multiclasseval import Multiclasseval
from modeling.bert_multilabel_classification import BertForMultiLabelSequenceClassification
from pipelines.text_multiclassification_pipeline import TextClassificationPipeline
from utils import dotdict
from utils.dataloader import load_dataset


def main(args):
    try:
        tqdm._instances.clear()
    except:
        pass

    text_column = args.text_column
    label_column_prefix = args.label_column_prefix
    label_list = labels

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    df = pd.read_csv(args.data_path).reset_index(drop=False)
    df[text_column] = df[text_column].fillna("")

    model = BertForMultiLabelSequenceClassification.from_pretrained(args.model_name)
    model.config.label2id = label2id
    model.config.id2label = id2label

    if device == "cpu":
        pipe = TextClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            is_multilabel=True)
    else:
        pipe = TextClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            is_multilabel=True,
            device="0")

    predictions = []
    target = []

    # TODO: add skipping record without annotations
    df[[label_column_prefix.format(c) for c in label_list]] = df[
        [label_column_prefix.format(c) for c in label_list]].replace(-1, 0)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row[text_column]
        target.append([row[label_column_prefix.format(c)] for c in label_list])
        p = {r["label"]: r["score"] for r in pipe(text, truncation=True, padding="max_length", max_length=512)[0]}
        p = [p[c] for c in label_list]
        predictions.append(p)

    metric = Multiclasseval()
    metric.threshold = args.threshold
    metric.num_classes = len(label_list)

    scores = metric.compute(predictions=predictions, references=target)
    print(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval bert')
    parser.add_argument("--data_path", default=str(ROOT_DIR / "data" / "0" / "test.csv"), type=str, required=False)
    parser.add_argument("--text_column", default="document_text", type=str, required=False)
    parser.add_argument("--label_column_prefix", default="was_label_{}", type=str, required=False)
    parser.add_argument("--model_name", default=str(ROOT_DIR / "models" / "tuned" / "tuned_bertreply"), type=str,
                        required=False)
    parser.add_argument("--tokenizer_name", default="bert-base-uncased",
                        type=str, required=False)
    parser.add_argument("--threshold", default=0.5,
                        type=float, required=False)

    args = parser.parse_args()
    args = vars(args)
    args = dotdict(args)
    main(args)
