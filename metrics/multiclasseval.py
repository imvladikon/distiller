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
    # TODO: add class weights
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

        f2_samples = torchmetrics.FBeta(beta=2, threshold=threshold, multiclass=True, average="samples",
                                        mdmc_average="samplewise")(
            predictions,
            target)
        f2_micro = torchmetrics.FBeta(beta=2, threshold=threshold, average="micro", num_classes=num_classes)(
            predictions,
            target)
        f2_macro = torchmetrics.FBeta(beta=2, threshold=threshold, average="macro", num_classes=num_classes)(
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
        scores["f2_micro"] = f2_micro

        scores["precision_macro"] = precision_macro
        scores["recall_macro"] = recall_macro
        scores["f1_macro"] = f1_macro
        scores["f2_macro"] = f2_macro

        scores["overall_precision"] = precision_samples
        scores["overall_recall"] = recall_samples
        scores["overall_f1"] = f1_samples
        scores["overall_f2"] = f2_samples
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

        if self.calculate_per_class:
            for i in range(num_classes):
                predictions_cls = predictions[:, i]
                target_cls = target[:, i]
                class_name = self.labels[i] if self.labels else f"class_{i}"
                scores[f"accuracy_{class_name}"] = torchmetrics.Accuracy(threshold=threshold)(predictions_cls,
                                                                                              target_cls)
                scores[f"f1_{class_name}"] = torchmetrics.F1(threshold=threshold)(
                    predictions_cls,
                    target_cls)

        return scores


Multiclasseval.threshold = 0.5
Multiclasseval.num_classes = None
Multiclasseval.calculate_per_class = False
Multiclasseval.labels = None


def generate_plausible_inputs_multilabel(num_classes, num_batches, batch_size):
    correct_targets = torch.randint(high=num_classes, size=(num_batches, batch_size))
    preds = torch.rand(num_batches, batch_size, num_classes)
    targets = torch.zeros_like(preds, dtype=torch.long)
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            targets[i, j, correct_targets[i, j]] = 1
    preds += torch.rand(num_batches, batch_size, num_classes) * targets / 3

    preds = preds / preds.sum(dim=2, keepdim=True)

    return preds, targets


if __name__ == '__main__':
    # from datasets import load_metric
    # from const import ROOT_DIR

    # metric = load_metric(str(ROOT_DIR / 'metrics' / 'multiclasseval.py'))

    metric = Multiclasseval()
    metric.threshold = 0.8
    metric.num_classes = 4

    predictions = [[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    target = [[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    scores = metric.compute(predictions=predictions, references=target)
    print(scores)

    metric = Multiclasseval()
    metric.threshold = 0.5
    metric.num_classes = 8

    batch_preds, batch_targets = generate_plausible_inputs_multilabel(num_classes=8, num_batches=4, batch_size=16)

    for predictions, target in zip(batch_preds, batch_targets):
        metric.add_batch(predictions=predictions, references=target)
    scores = metric.compute()
    print(scores)

    from sklearn.metrics import multilabel_confusion_matrix, classification_report
    import numpy as np

    predictions = np.array([[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    target = np.array([[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    # print(multilabel_confusion_matrix(target, predictions, samplewise=True))

    threshold = 0.8
    metric = Multiclasseval()
    metric.threshold = threshold
    metric.num_classes = 4

    predictions = (np.array([[0.3, 0.7, 0.9, 0.1], [0.9, 0, 1, 0], [0, 0, 0, 1]]) > threshold).astype("int")
    target = np.array([[1, 0, 0, 1], [1, 1, 1, 0], [0, 0, 0, 1]])

    scores = metric.compute(predictions=predictions, references=target)
    print(scores)

    print(classification_report(target, predictions, target_names=["1", "2", "3", "4"]))

    metric = Multiclasseval()
    metric.threshold = threshold
    metric.num_classes = 4
    metric.calculate_per_class = True
    metric.labels = ["class1", "class2", "class3", "class4"]

    predictions = (np.array([[0.3, 0.7, 0.9, 0.1], [0.9, 0, 1, 0], [0, 0, 0, 1]]) > threshold).astype("int")
    target = np.array([[1, 0, 0, 1], [1, 1, 1, 0], [0, 0, 0, 1]])

    scores = metric.compute(predictions=predictions, references=target)
    print(scores)

    predictions = np.array([[1, 0, 0, 1], [1, 1, 1, 0], [0, 0, 0, 1]])
    target = np.array([[1, 0, 0, 1], [1, 1, 1, 0], [0, 0, 0, 1]])

    scores = metric.compute(predictions=predictions, references=target)
    print(scores)
