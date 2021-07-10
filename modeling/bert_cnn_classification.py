import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from torch.nn import functional as F, CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput


class BertForClassificationCNN(BertPreTrainedModel):

    def __init__(self,
                 config: BertConfig,
                 hidden_size=128,
                 is_multilabel: bool = True,
                 regression: bool = False,
                 *args,
                 **kwargs) -> None:

        super(BertForClassificationCNN, self).__init__(config, *args, **kwargs)

        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.conv1 = nn.Conv2d(1, hidden_size, (3, config.hidden_size))
        self.conv2 = nn.Conv2d(1, hidden_size, (4, config.hidden_size))
        self.conv3 = nn.Conv2d(1, hidden_size, (5, config.hidden_size))

        self.classifier = nn.Linear(hidden_size * 3, config.num_labels)

        self.init_weights()

        self.is_multilabel = is_multilabel
        self.regression = regression


    def forward(self,
                    input_ids: torch.LongTensor = None,
                    attention_mask: torch.FloatTensor = None,
                    token_type_ids: torch.LongTensor = None,
                    position_ids: torch.LongTensor = None,
                    head_mask: torch.FloatTensor = None,
                    inputs_embeds: torch.FloatTensor = None,
                    labels: torch.FloatTensor = None,
                    output_attentions: bool = None,
                    output_hidden_states: bool = None,
                    return_dict: bool = None,
                    *args,
                    **kwargs) -> SequenceClassifierOutput:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            *args,
            **kwargs
        )

        sequence_output = outputs.last_hidden_state if return_dict else outputs[0]

        out = self.dropout(sequence_output).unsqueeze(1)
        c1 = torch.relu(self.conv1(out).squeeze(3))
        p1 = F.max_pool1d(c1, c1.size(2)).squeeze(2)

        c2 = torch.relu(self.conv2(out).squeeze(3))
        p2 = F.max_pool1d(c2, c2.size(2)).squeeze(2)

        c3 = torch.relu(self.conv3(out).squeeze(3))
        p3 = F.max_pool1d(c3, c3.size(2)).squeeze(2)

        pool = self.dropout(torch.cat((p1, p2, p3), 1))

        logits = self.classifier(pool)

        loss = None
        if labels is not None:
            if self.is_multilabel:
                # loss_fct = BCEWithLogitsLoss()
                loss_fct = nn.MultiLabelSoftMarginLoss()

                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, ))

        if not self.regression:
            logits = torch.sigmoid(logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self, unfreeze=None):
        if unfreeze is None:
            unfreeze = ['10', '11', 'pooler']

        self.freeze_bert_encoder()

        for name, param in self.bert.named_parameters():
            if any(t in name for t in unfreeze):
                # logger.info(f'Unfreezing {name}')
                param.requires_grad = True

if __name__ == '__main__':
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

    model = BertForClassificationCNN.from_pretrained("bert-base-uncased", num_labels=len(labels))
    model.config.label2id = label2id
    model.config.id2label = id2label

    attention_mask = torch.LongTensor([[1] * 512])
    input_ids = torch.randint(100, 1000, size=(1, 512)).long()
    labels = torch.FloatTensor([[0, 0, 0, 1, 0, 0, 0, 0]])
    assert labels.shape[-1] == len(label2id)
    token_types_ids = torch.LongTensor([[0] * 512])

    print(model(input_ids=input_ids, attention_mask=attention_mask, labels=labels))

