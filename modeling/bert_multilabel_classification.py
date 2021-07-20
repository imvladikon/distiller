from collections import OrderedDict
from typing import Optional

import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.configuration_bert import BertConfig
import logging

logger = logging.getLogger(__name__)


class FocalLossLogits(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLossLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss()(inputs, targets)
        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return torch.mean(f_loss)


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):

    def __init__(self,
                 config: BertConfig,
                 is_multilabel: bool = True,
                 regression: bool = False,
                 loss_fct: Optional["Module"] = None,
                 *args,
                 **kwargs) -> None:

        super().__init__(config, *args, **kwargs)

        self.num_labels = config.num_labels
        if self.num_labels == 1:
            raise ValueError("num_labels must be greater than 1 for multi-label classification problems")

        self.bert = BertModel(config)

        # self.backbone = nn.Sequential(OrderedDict([
        #     ('dropout', torch.nn.Dropout(config.hidden_dropout_prob)),
        #     ('classifier', torch.nn.Linear(config.hidden_size, self.num_labels))
        # ]))

        self.middle_layer_size = config.hidden_size // 10
        self.backbone = nn.Sequential(OrderedDict([
            ('dropout0', torch.nn.Dropout(config.hidden_dropout_prob)),
            ('fc1', torch.nn.Linear(config.hidden_size, self.middle_layer_size)),
            ('dropout1', torch.nn.Dropout(config.hidden_dropout_prob)),
            ('classifier', torch.nn.Linear(self.middle_layer_size, self.num_labels))
        ]))

        self.init_weights()
        self.is_multilabel = is_multilabel
        self.regression = regression
        self.loss_fct = loss_fct

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
        )

        pooled_output = outputs[1]
        logits = self.backbone(pooled_output)

        loss = None
        if labels is not None:
            if self.regression:

                loss_fct = nn.MSELoss() if self.loss_fct is None else self.loss_fct
                loss = loss_fct(logits, labels)

            elif self.is_multilabel:

                loss_fct = nn.MultiLabelSoftMarginLoss() if self.loss_fct is None else self.loss_fct
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))

            else:

                loss_fct = nn.CrossEntropyLoss() if self.loss_fct is None else self.loss_fct
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
                logger.info(f'Unfreezing {name}')
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

    model = BertForMultiLabelSequenceClassification.from_pretrained("bert-base-uncased",
                                                                    num_labels=len(label2id),
                                                                    loss_fct=FocalLossLogits())
    model.config.label2id = label2id
    model.config.id2label = id2label

    attention_mask = torch.LongTensor([[1] * 512])
    input_ids = torch.randint(100, 1000, size=(1, 512)).long()
    labels = torch.FloatTensor([[0, 0, 0, 1, 0, 0, 0, 0]])
    assert labels.shape[-1] == len(label2id)
    token_types_ids = torch.LongTensor([[0] * 512])

    print(model(input_ids=input_ids, attention_mask=attention_mask, labels=labels))
