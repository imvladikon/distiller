import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from transformers import BertConfig, BertModel, BertPreTrainedModel


def create_student_model(hidden_size,
                         num_hidden_layers,
                         num_attention_heads):
    # TODO: add weights init from teacher

    config = BertConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
    )
    model = BertModel(config)
    return model


AVAILABLE_REDUCE_METHODS = {
    "PCA": PCA,
    "TruncatedSVD": TruncatedSVD
}


def reduce_word_embeddings(model: BertPreTrainedModel,
                               tokenizer: "AutoTokenizer",
                               hidden_size: int,
                               method: str = "PCA"):
    """
    get word embedding matrix from teacher
    Args:
        model:
        tokenizer:
        hidden_size:

    Returns:

    """
    model_prefix = model.base_model_prefix
    if hasattr(model, model_prefix):
        model = getattr(model, model_prefix)
    if hasattr(model, "embeddings"):
        word_embedding_matrix = model.embeddings.word_embeddings.weight.cpu().detach().numpy()
    else:
        word_embedding_matrix = np.random.uniform(size=(len(tokenizer.get_vocab()), hidden_size))
    if word_embedding_matrix.shape[1] > hidden_size:
        reducer = AVAILABLE_REDUCE_METHODS.get(method, PCA)(n_components=hidden_size)
        word_embedding_matrix = reducer.fit_transform(word_embedding_matrix)
    return word_embedding_matrix


if __name__ == '__main__':
    from modeling.bert_multilabel_classification import BertForMultiLabelSequenceClassification
    from transformers import AutoTokenizer

    teacher_model = BertForMultiLabelSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    word_embedding_matrix = reduce_word_embeddings(teacher_model, tokenizer, hidden_size=256, method="PCA")