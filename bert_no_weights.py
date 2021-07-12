from transformers import BertModel, BertConfig

if __name__ == '__main__':
    config = BertConfig(
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=6,
    )
    model = BertModel(config)
    print(model)
