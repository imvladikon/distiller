
"""
Well-Read Students Learn Better:
On the Importance of Pre-training Compact Models
https://arxiv.org/abs/1908.08962
https://github.com/google-research/bert/
"""

# 2/128 (BERT-Tiny)
# 4/256 (BERT-Mini)
# 4/512 (BERT-Small)
# 8/512 (BERT-Medium)
# 12/768 (BERT-Base)


num_hidden_num_attention_heads_mapping = {
    128: 2,
    256: 4,
    512: 8,
    768: 12
}


def get_student_models(hidden_size, num_layers):
    assert hidden_size in [128, 256, 512, 768]
    assert num_layers in [2, 4, 6, 8, 10, 12]

    num_attention_heads = num_hidden_num_attention_heads_mapping[hidden_size]
    return f"google/bert_uncased_L-{num_layers}_H-{hidden_size}_A-{num_attention_heads}"

def all_google_students():
    for hidden_size in [128, 256, 512, 768]:
        for num_layers in [2, 4, 6, 8, 10, 12]:
            yield get_student_models(hidden_size, num_layers)


if __name__ == '__main__':
    from transformers import AutoModel

    hidden_size, num_layers = 256, 4
    num_attention_heads = num_hidden_num_attention_heads_mapping[hidden_size]

    student_model_name = get_student_models(hidden_size=hidden_size, num_layers=num_layers)

    model = AutoModel.from_pretrained(student_model_name)

    assert model.config.hidden_size, hidden_size
    assert model.config.num_hidden_layers, num_layers
    assert model.config.num_attention_heads, num_attention_heads