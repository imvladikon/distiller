from typing import List, Optional, Union

import torch
import torchmetrics

import datasets

_CITATION = """\
"""

_DESCRIPTION = """\
"""

_KWARGS_DESCRIPTION = """
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Multiclasseval(datasets.Metric):

    # TODO: bug with buffer writer , fix it. currently need to use monkey patching
    # def __init__(self, *args, **kwargs):
    #     super(Multiclasseval, self).__init__(args, kwargs)
    #
    #     self.threshold = kwargs.get("threshold", 0.5)
    #     self.num_classes = kwargs.get("num_classes", None)

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="",
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float32", id="label"), id="sequence"),
                    "references": datasets.Sequence(datasets.Value("int32", id="label"), id="sequence"),
                }
            ),
            codebase_urls=[""],
            reference_urls=[""],
        )

    def _compute(
            self,
            predictions,
            references,
            suffix: bool = False,
            scheme: Optional[str] = None,
            mode: Optional[str] = None,
            sample_weight: Optional[List[int]] = None,
            zero_division: Union[str, int] = "warn",
            **metric_init_kwargs
    ):
        threshold = 0.5 if self.threshold is None else self.threshold
        num_classes = len(references[0]) if self.num_classes is None else self.num_classes

        predictions = torch.FloatTensor(predictions)
        target = torch.LongTensor(references)

        accuracy_samples = torchmetrics.Accuracy(threshold=threshold, multiclass=True)(predictions,
                                                                                       target)
        f1_samples = torchmetrics.F1(threshold=threshold, multiclass=True, average="samples",
                                     mdmc_average="samplewise")(
            predictions,
            target)

        precision_samples = torchmetrics.Precision(threshold=threshold, multiclass=True, average="samples",
                                                   mdmc_average="samplewise")(
            predictions,
            target)

        recall_samples = torchmetrics.Recall(threshold=threshold, multiclass=True, average="samples",
                                             mdmc_average="samplewise")(
            predictions,
            target)

        f1_micro = torchmetrics.F1(threshold=threshold, average="micro", num_classes=num_classes)(
            predictions,
            target)

        precision_micro = torchmetrics.Precision(threshold=threshold, average="micro", num_classes=num_classes)(
            predictions,
            target)

        recall_micro = torchmetrics.Recall(threshold=threshold, average="micro", num_classes=num_classes)(
            predictions,
            target)

        f1_macro = torchmetrics.F1(threshold=threshold, average="macro", num_classes=num_classes)(
            predictions,
            target)

        precision_macro = torchmetrics.Precision(threshold=threshold, average="macro", num_classes=num_classes)(
            predictions,
            target)

        recall_macro = torchmetrics.Recall(threshold=threshold, average="macro", num_classes=num_classes)(
            predictions,
            target)

        matthews_corrcoef = torchmetrics.MatthewsCorrcoef(num_classes=num_classes, threshold=threshold)(
            predictions,
            target)

        # scores = {
        #     type_name: {
        #         "precision": score["precision"],
        #         "recall": score["recall"],
        #         "f1": score["f1-score"],
        #         "number": score["support"],
        #     }
        #     for type_name, score in report.items()
        # }
        scores = {}

        scores["precision_micro"] = precision_micro
        scores["recall_micro"] = recall_micro
        scores["f1_micro"] = f1_micro

        scores["precision_macro"] = precision_macro
        scores["recall_macro"] = recall_macro
        scores["f1_macro"] = f1_macro

        scores["overall_precision"] = precision_samples
        scores["overall_recall"] = recall_samples
        scores["overall_f1"] = f1_samples
        scores["overall_accuracy"] = accuracy_samples
        scores["matthews_corrcoef"] = matthews_corrcoef

        try:
            # TODO: fix no positive cases
            aucroc_macro = torchmetrics.AUROC(num_classes=num_classes, average="macro")(
                predictions,
                target)

            aucroc_micro = torchmetrics.AUROC(num_classes=num_classes, average="micro")(
                predictions,
                target)
        except:
            aucroc_macro = 0
            aucroc_micro = 0


        scores["aucroc_macro"] = aucroc_macro
        scores["aucroc_micro"] = aucroc_micro

        # TODO: add scores per class
        # predictions = torch.FloatTensor(preds)
        # target = torch.LongTensor(gt)
        # num_classes = 8
        # accuracy_per_class = {}
        # for i in range(num_classes):
        #     predictions_cls = predictions[i, :]
        #     target_cls = target[i, :]
        #     accuracy_per_class[i] = torchmetrics.Accuracy(threshold=threshold)(predictions_cls,
        #                                                                        target_cls)

        return scores

Multiclasseval.threshold = 0.5
Multiclasseval.num_classes = None

if __name__ == '__main__':
    # from datasets import load_metric
    # from const import ROOT_DIR

    # metric = load_metric(str(ROOT_DIR / 'metrics' / 'multiclasseval.py'))

    metric = Multiclasseval()
    metric.threshold = 0.8
    metric.num_classes = 4


    predictions = [[0.3, 0.7, 0.9, 0.1], [0.9, 0, 1, 0], [0, 0, 0, 1]]
    target = [[1, 0, 0, 1], [1, 1, 1, 0], [0, 0, 0, 1]]
    scores = metric.compute(predictions=predictions, references=target)
    print(scores)

    predictions = [[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    target = [[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    scores = metric.compute(predictions=predictions, references=target)
    print(scores)
