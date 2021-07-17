from transformers import BertConfig, BertModel


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
