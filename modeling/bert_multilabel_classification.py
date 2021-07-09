import torch
import transformers
from torch import nn
from torch.nn import MSELoss, BCEWithLogitsLoss
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.configuration_bert import BertConfig


class HFBertForMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config: BertConfig, regression: bool = False, *args, **kwargs) -> None:

        """
        based on huggingface model

        Args:
            config:
            regression:
            *args:
            **kwargs:
        """


        super().__init__(config, *args, **kwargs)
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
            **kwargs
    ):
        r"""
        input_ids - (batch_size, sequence_length)

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
            *args,
            **kwargs
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


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self,
                 config: BertConfig,
                 is_multilabel: bool = True,
                 regression: bool = False,
                 *args,
                 **kwargs) -> None:

        super().__init__(config, *args, **kwargs)

        self.num_labels = config.num_labels
        self.bert = BertModel(config)

        self.middle_layer_size = config.hidden_size // 10

        self.dropout1 = torch.nn.Dropout(config.hidden_dropout_prob)
        self.fc1 = torch.nn.Linear(config.hidden_size, self.middle_layer_size)

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.middle_layer_size, self.num_labels)

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
        )

        pooled_output = outputs[1]

        # comment out for 1 layer head
        pooled_output = self.dropout1(pooled_output)
        pooled_output = self.fc1(pooled_output)

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

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

    model = BertForMultiLabelSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(labels))
    model.config.label2id = label2id
    model.config.id2label = id2label

    attention_mask = torch.LongTensor([[1] * 512])
    input_ids = torch.randint(100, 1000, size=(1, 512)).long()
    labels = torch.FloatTensor([[0, 0, 0, 1, 0, 0, 0, 0]])
    assert labels.shape[-1] == len(label2id)
    token_types_ids = torch.LongTensor([[0] * 512])

    print(model(input_ids=input_ids, attention_mask=attention_mask, labels=labels))
