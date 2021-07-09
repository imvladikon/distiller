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


if __name__ == '__main__':
    from transformers import AutoTokenizer
    from pprint import pprint
    from modeling.bert_multilabel_classification import BertForMultiLabelSequenceClassification


    labels = ['contract',
              'ooo',
              'payments',
              'pi',
              'pricing',
              'reject',
              'reply',
              'scheduling']

    id2label = {k: v for k, v in enumerate(labels)}
    label2id = {v: k for k, v in id2label.items()}

    model_checkpoint = "../models/tuned/tuned_final_model_8_tag_case_sensitive"
    model = BertForMultiLabelSequenceClassification.from_pretrained(model_checkpoint, num_labels=len(labels))
    model.config.label2id = label2id
    model.config.id2label = id2label

    pipe = TextClassificationPipeline(
        model=model,
        tokenizer= AutoTokenizer.from_pretrained("bert-base-uncased"),
        return_all_scores=True)
    message = """Nice to meet you. I can still me tomorrow at 2pm for 30 minutes. I have another call at 2:30."""
    pprint(pipe(message))
