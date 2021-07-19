from typing import Dict

import numpy as np
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from transformers import BertConfig, BertModel, BertPreTrainedModel

from config.google_students_models import get_student_models

AVAILABLE_REDUCE_METHODS = {
    "PCA": PCA,
    "TruncatedSVD": TruncatedSVD
}


class StudentFactory:

    def __init__(self,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads,
                 teacher_model=None,
                 reduce_word_embeddings_method=None,
                 init_layers_from_teacher=False):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.teacher_model = teacher_model
        self.reduce_word_embeddings_method = reduce_word_embeddings_method
        self.init_layers_from_teacher = init_layers_from_teacher

    def produce(self, *args, **kwargs):
        bert_class: BertModel = kwargs.get("bert_class", BertModel)
        config = BertConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
        )
        model = bert_class(config)
        if self.reduce_word_embeddings_method in AVAILABLE_REDUCE_METHODS:
            tokenizer = kwargs.get("tokenizer", AutoTokenizer.from_pretrained("bert-base-uncased"))
            word_embedding_matrix = self.reduce_word_embeddings(tokenizer=tokenizer)
            model_prefix = model.base_model_prefix
            base_model = model
            if hasattr(model, model_prefix):
                base_model = getattr(model, model_prefix)
            base_model.embeddings.word_embeddings.weight.data.copy_(torch.from_numpy(word_embedding_matrix))
            # TODO: add fp16
        if self.init_layers_from_teacher and self.teacher_model.config.hidden_size == model.config.hidden_size:
            mapping_layers = kwargs.get("mapping_layers", {
                2: [0, 11],
                4: [0, 1, 10, 11],
                6: [0, 1, 4, 9, 10, 11],
                8: [0, 1, 3, 4, 8, 9, 10, 11],
                10: [0, 1, 2, 3, 6, 7, 8, 9, 10, 11],
                12: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            })

            self.copy_layers_from_teacher(model, mapping_layers)
        return model

    def from_pretrained_google(self, *args, **kwargs):
        bert_class: BertModel = kwargs.get("bert_class", BertModel)
        model_name = get_student_models(hidden_size=self.hidden_size,
                                        num_layers=self.num_hidden_layers)
        return bert_class.from_pretrained(model_name)

    def reduce_word_embeddings(self,
                               tokenizer: "AutoTokenizer"):
        model = self.teacher_model
        model_prefix = model.base_model_prefix
        if hasattr(model, model_prefix):
            model = getattr(model, model_prefix)
        if hasattr(model, "embeddings"):
            word_embedding_matrix = model.embeddings.word_embeddings.weight.cpu().detach().numpy()
        else:
            word_embedding_matrix = np.random.uniform(size=(len(tokenizer.get_vocab()), self.hidden_size))
        if word_embedding_matrix.shape[1] > self.hidden_size:
            reducer = AVAILABLE_REDUCE_METHODS.get(self.reduce_word_embeddings_method, PCA)(
                n_components=self.hidden_size)
            word_embedding_matrix = reducer.fit_transform(word_embedding_matrix)
        return word_embedding_matrix

    def copy_layers_from_teacher(self, student_model, mapping_layers):
        teacher_model = self.teacher_model
        assert teacher_model.config.hidden_size == student_model.config.hidden_size
        layers_to_select = mapping_layers.get(self.num_hidden_layers)
        for n_student_layer, n_teacher_layer in enumerate(layers_to_select):
            # hiddens [in_features, hid_size]
            student_model.bert.encoder.layer[n_student_layer].intermediate.dense.weight.data.copy_(
                teacher_model.bert.encoder.layer[n_teacher_layer].intermediate.dense.weight)
            student_model.bert.encoder.layer[n_student_layer].intermediate.dense.bias.data.copy_(
                teacher_model.bert.encoder.layer[n_teacher_layer].intermediate.dense.bias.data)

            # atentions [hid_size, hid_size]
            student_model.bert.encoder.layer[n_student_layer].attention.output.dense.weight.data.copy_(
                teacher_model.bert.encoder.layer[n_teacher_layer].attention.output.dense.weight)
            student_model.bert.encoder.layer[n_student_layer].attention.output.dense.bias.data.copy_(
                teacher_model.bert.encoder.layer[n_teacher_layer].attention.output.dense.bias.data)

            # outputs [hid_size, in_features]
            student_model.bert.encoder.layer[n_student_layer].output.dense.weight.data.copy_(
                teacher_model.bert.encoder.layer[n_teacher_layer].output.dense.weight)
            student_model.bert.encoder.layer[n_student_layer].output.dense.bias.data.copy_(
                teacher_model.bert.encoder.layer[n_teacher_layer].output.dense.bias.data)


if __name__ == '__main__':
    from modeling.bert_multilabel_classification import BertForMultiLabelSequenceClassification
    from transformers import AutoTokenizer

    teacher_model = BertForMultiLabelSequenceClassification.from_pretrained("bert-base-uncased")
    student_factory = StudentFactory(hidden_size=256,
                                     num_hidden_layers=6,
                                     num_attention_heads=4,
                                     teacher_model=teacher_model,
                                     reduce_word_embeddings_method="PCA")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    student_model = student_factory.produce(tokenizer=tokenizer,
                                            bert_class=BertForMultiLabelSequenceClassification)
    student_model = student_factory.from_pretrained_google(bert_class=BertForMultiLabelSequenceClassification)

    student_factory = StudentFactory(hidden_size=768,
                                     num_hidden_layers=6,
                                     num_attention_heads=4,
                                     teacher_model=teacher_model,
                                     reduce_word_embeddings_method="PCA",
                                     init_layers_from_teacher=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    student_model = student_factory.produce(tokenizer=tokenizer,
                                            bert_class=BertForMultiLabelSequenceClassification)
