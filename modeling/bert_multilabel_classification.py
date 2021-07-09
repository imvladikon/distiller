import torch
from torch import nn
from torch.nn import MSELoss, BCEWithLogitsLoss
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

from const import labels, label2id, id2label


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, regression=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.regression = regression
        if self.num_labels == 1:
            raise ValueError("num_labels must be greater than 1 for multi-label classification problems")

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_labels)`, `optional`):
            Labels for computing the multi-label sequence classification/regression loss
        """
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

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.regression:
                loss_fct = MSELoss()
                loss = loss_fct(logits, labels)
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

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

if __name__ == '__main__':
    model = BertForMultiLabelSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(labels))
    model.config.label2id = label2id
    model.config.id2label = id2label
