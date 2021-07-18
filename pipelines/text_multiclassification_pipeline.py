import numpy as np
from transformers import Pipeline
from transformers.tokenization_utils_base import TruncationStrategy


class TextClassificationPipeline(Pipeline):

    def __init__(self, return_all_scores: bool = False,
                 is_multilabel: bool = True,
                 return_only_logits: bool = False,
                 threshold: float = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.return_all_scores = return_all_scores
        self.is_multilabel = is_multilabel
        self.return_only_logits = return_only_logits
        self.threshold = threshold

    def _parse_and_tokenize(
            self, inputs, padding=True, add_special_tokens=True, truncation=TruncationStrategy.DO_NOT_TRUNCATE, **kwargs
    ):
        """
        Parse arguments and tokenize
        """
        # Parse arguments
        inputs = self.tokenizer(
            inputs,
            add_special_tokens=add_special_tokens,
            return_tensors=self.framework,
            padding=padding,
            truncation=truncation,
            **kwargs
        )

        return inputs

    def __call__(self, *args, **kwargs):
        outputs = super().__call__(*args, **kwargs)

        if self.model.config.num_labels == 1:
            scores = 1.0 / (1.0 + np.exp(-outputs))
        elif self.is_multilabel:
            # assume that bert multi label classifier do sigmoid
            scores = outputs
        else:
            scores = np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True)

        if self.return_all_scores or self.is_multilabel:
            if not self.return_only_logits:
                return [
                    [{"label": self.model.config.id2label[i], "score": score.item()} for i, score in enumerate(item)]
                    for item in scores
                ]
            else:
                results = np.array([
                        [score[i] for i in self.model.config.id2label.keys()]
                        for score in scores
                    ])
                if self.threshold is not None:
                    results = (results>self.threshold).astype("int")
                return results
        else:
            return [
                {"label": self.model.config.id2label[item.argmax()], "score": item.max().item()} for item in scores
            ]


if __name__ == '__main__':
    from transformers import AutoTokenizer
    from pprint import pprint
    from modeling.bert_multilabel_classification import BertForMultiLabelSequenceClassification

    labels = [
        "reject",
        "scheduling",
        "pricing",
        "pi",
        "ooo",
        "payments",
        "contract",
        "reply"
    ]

    id2label = {k: v for k, v in enumerate(labels)}
    label2id = {v: k for k, v in id2label.items()}

    model_checkpoint = "../models/tuned/tuned_bertreply"
    model = BertForMultiLabelSequenceClassification.from_pretrained(model_checkpoint, num_labels=len(labels))
    model.config.label2id = label2id
    model.config.id2label = id2label

    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
        is_multilabel=True,
        return_only_logits=True,
        threshold=0.5
    )
    message = """Nice to meet you. I can still me tomorrow at 2pm for 30 minutes. I have another call at 2:30."""
    pprint(pipe([message,message], truncation=True, padding="max_length", max_length=512))
