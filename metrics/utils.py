from typing import Callable, Dict

import torch
from transformers import EvalPrediction
from datasets import Metric


def compute_multilabel_metrics(eval_pred: EvalPrediction,
                               metric: Metric) -> Callable[[EvalPrediction], Dict]:
    logits, labels = eval_pred
    labels = torch.LongTensor(labels)
    predictions = torch.FloatTensor(logits)

    scores = metric.compute(predictions=predictions,
                            references=labels)

    return scores
