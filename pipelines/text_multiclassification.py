import numpy as np
from transformers import Pipeline


class TextClassificationPipeline(Pipeline):

    def __init__(self, return_all_scores: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.return_all_scores = return_all_scores

    def __call__(self, *args, **kwargs):
        outputs = super().__call__(*args, **kwargs)

        if self.model.config.num_labels == 1:
            scores = 1.0 / (1.0 + np.exp(-outputs))
        else:
            scores = np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True)
        if self.return_all_scores:
            return [
                [{"label": self.model.config.id2label[i], "score": score.item()} for i, score in enumerate(item)]
                for item in scores
            ]
        else:
            return [
                {"label": self.model.config.id2label[item.argmax()], "score": item.max().item()} for item in scores
            ]
