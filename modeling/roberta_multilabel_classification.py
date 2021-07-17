import logging
import torch
from torch import nn
from transformers import RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger(__name__)

class RobertaForMultiLabelSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config, regression=False, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.init_weights()
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

        outputs = self.roberta(
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

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.regression:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.MultiLabelSoftMarginLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))

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
        for param in self.roberta.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self, unfreeze=['10', '11', 'pooler']):
        self.freeze_bert_encoder()
        for name, param in self.roberta.named_parameters():
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

    model = RobertaForMultiLabelSequenceClassification.from_pretrained("roberta-base", num_labels=len(label2id))
    model.config.label2id = label2id
    model.config.id2label = id2label

    attention_mask = torch.LongTensor([[1] * 512])
    input_ids = torch.randint(100, 1000, size=(1, 512)).long()
    labels = torch.FloatTensor([[0, 0, 0, 1, 0, 0, 0, 0]])
    assert labels.shape[-1] == len(label2id)
    token_types_ids = torch.LongTensor([[0] * 512])

    print(model(input_ids=input_ids, attention_mask=attention_mask, labels=labels))