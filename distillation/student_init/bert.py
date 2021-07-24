from typing import List

import numpy as np
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from transformers import BertConfig, AutoTokenizer

from transformers import BertModel

from distillation.student_init.google_students_models import get_student_models

AVAILABLE_REDUCE_METHODS = {
    "PCA": PCA,
    "TruncatedSVD": TruncatedSVD
}

DEFAULT_LAYER_MAPPING = {
    2: [0, 11],
    4: [0, 1, 10, 11],
    6: [0, 1, 4, 9, 10, 11],
    8: [0, 1, 3, 4, 8, 9, 10, 11],
    10: [0, 1, 2, 3, 6, 7, 8, 9, 10, 11],
    12: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
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

    def produce(self, num_labels = 2, *args, **kwargs):
        bert_class: BertModel = kwargs.get("bert_class", BertModel)
        config = BertConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_labels=num_labels
        )
        model = bert_class(config)
        if self.init_layers_from_teacher and self.teacher_model.config.hidden_size == model.config.hidden_size:
            mapping_layers = kwargs.get("mapping_layers", DEFAULT_LAYER_MAPPING)
            model = self.copy_layers(student_model=model,
                                     layers_to_transfer=mapping_layers.get(model.config.num_hidden_layers))
        if self.reduce_word_embeddings_method in AVAILABLE_REDUCE_METHODS:
            tokenizer = kwargs.get("tokenizer", AutoTokenizer.from_pretrained("bert-base-uncased"))
            word_embedding_matrix = self.reduce_word_embeddings(tokenizer=tokenizer)
            device = model.device
            model_prefix = model.base_model_prefix
            base_model = model
            if hasattr(model, model_prefix):
                base_model = getattr(model, model_prefix)
            base_model.embeddings.word_embeddings.weight.data.copy_(torch.from_numpy(word_embedding_matrix).to(device))
            # TODO: add fp16
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

    def copy_layers(
            self,
            student_model,
            layers_to_transfer: List[int] = None,
            encoder_name="encoder",
    ):
        teacher_model = self.teacher_model
        teacher_hidden_size = teacher_model.config.hidden_size
        student_hidden_size = student_model.config.hidden_size
        if teacher_hidden_size != student_hidden_size:
            raise Exception("Teacher and student hidden size should be the same")
        teacher_layers_num = teacher_model.config.num_hidden_layers
        student_layers_num = student_model.config.num_hidden_layers

        if layers_to_transfer is None:
            layers_to_transfer = list(
                range(teacher_layers_num - student_layers_num, teacher_layers_num)
            )

        prefix_teacher = list(teacher_model.state_dict().keys())[0].split(".")[0]
        prefix_student = list(student_model.state_dict().keys())[0].split(".")[0]

        state_dict = teacher_model.state_dict()
        compressed_sd = student_model.state_dict()

        # extract embeddings
        for w in ["word_embeddings", "position_embeddings"]:
            compressed_sd[f"{prefix_student}.embeddings.{w}.weight"] = state_dict[
                f"{prefix_teacher}.embeddings.{w}.weight"
            ]
        for w in ["weight", "bias"]:
            compressed_sd[f"{prefix_student}.embeddings.LayerNorm.{w}"] = state_dict[
                f"{prefix_teacher}.embeddings.LayerNorm.{w}"
            ]

        # extract encoder
        for std_idx, teacher_idx in enumerate(layers_to_transfer):
            for w in ["weight", "bias"]:
                # atentions [hid_size, hid_size]
                compressed_sd[
                    f"{prefix_student}.{encoder_name}.layer.{std_idx}.attention.self.query.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.{encoder_name}.layer.{teacher_idx}.attention.self.query.{w}"  # noqa: E501
                ]
                compressed_sd[
                    f"{prefix_student}.{encoder_name}.layer.{std_idx}.attention.self.key.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.{encoder_name}.layer.{teacher_idx}.attention.self.key.{w}"  # noqa: E501
                ]
                compressed_sd[
                    f"{prefix_student}.{encoder_name}.layer.{std_idx}.attention.self.value.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.{encoder_name}.layer.{teacher_idx}.attention.self.value.{w}"  # noqa: E501
                ]

                compressed_sd[
                    f"{prefix_student}.{encoder_name}.layer.{std_idx}.attention.output.dense.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.{encoder_name}.layer.{teacher_idx}.attention.output.dense.{w}"  # noqa: E501
                ]
                compressed_sd[
                    f"{prefix_student}.{encoder_name}.layer.{std_idx}.attention.output.LayerNorm.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.{encoder_name}.layer.{teacher_idx}.attention.output.LayerNorm.{w}"  # noqa: E501
                ]
                # hiddens [in_features, hid_size]
                compressed_sd[
                    f"{prefix_student}.{encoder_name}.layer.{std_idx}.intermediate.dense.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.{encoder_name}.layer.{teacher_idx}.intermediate.dense.{w}"  # noqa: E501
                ]
                # outputs [hid_size, in_features]
                compressed_sd[
                    f"{prefix_student}.{encoder_name}.layer.{std_idx}.output.dense.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.{encoder_name}.layer.{teacher_idx}.output.dense.{w}"  # noqa: E501
                ]
                compressed_sd[
                    f"{prefix_student}.{encoder_name}.layer.{std_idx}.output.LayerNorm.{w}"  # noqa: E501
                ] = state_dict[
                    f"{prefix_teacher}.{encoder_name}.layer.{teacher_idx}.output.LayerNorm.{w}"  # noqa: E501
                ]

        # extract vocab
        if "cls.predictions.decoder.weight" in state_dict:
            compressed_sd["cls.predictions.decoder.weight"] = state_dict["cls.predictions.decoder.weight"]
        if "cls.predictions.bias" in state_dict:
            compressed_sd["cls.predictions.bias"] = state_dict["cls.predictions.bias"]

        for w in ["weight", "bias"]:
            if f"cls.predictions.transform.dense.{w}" in state_dict:
                compressed_sd[f"vocab_transform.{w}"] = state_dict[f"cls.predictions.transform.dense.{w}"]
            if f"cls.predictions.transform.LayerNorm.{w}" in state_dict:
                compressed_sd[f"vocab_layer_norm.{w}"] = state_dict[
                    f"cls.predictions.transform.LayerNorm.{w}"
                ]
        student_model.load_state_dict(compressed_sd)
        return student_model


if __name__ == '__main__':
    from modeling.bert_multilabel_classification import BertForMultiLabelSequenceClassification
    from transformers import AutoTokenizer

    teacher_model = BertForMultiLabelSequenceClassification.from_pretrained("bert-base-uncased", num_labels=8)
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
