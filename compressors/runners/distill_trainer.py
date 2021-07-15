from typing import Optional, Callable, Dict, Tuple, List

import torch
from transformers import Trainer, TrainingArguments, DataCollator, PreTrainedTokenizerBase, TrainerCallback, \
    TrainerState, TrainerControl
from torch import nn
from transformers.trainer_callback import CallbackHandler

from compressors.distillation.data.label_smoothing import probability_shift
from compressors.utils import set_requires_grad


def on_compute_loss_begin(self,
                          args: TrainingArguments,
                          state: TrainerState,
                          control: TrainerControl,
                          **kwargs):
    """
    Event called at the compute_loss of the :class:`~transformers.Trainer`.
    """
    pass


def on_compute_loss_end(self,
                        args: TrainingArguments,
                        state: TrainerState,
                        control: TrainerControl,
                        **kwargs):
    """
    Event called at the compute_loss of the :class:`~transformers.Trainer`.
    """
    pass


TrainerCallback.on_compute_loss_begin = on_compute_loss_begin
TrainerCallback.on_compute_loss_end = on_compute_loss_end


def on_compute_loss_begin_handler(self,
                                  args: TrainingArguments,
                                  state: TrainerState,
                                  control: TrainerControl,
                                  **kwargs):
    return self.call_event("on_compute_loss_begin", args, state, control, **kwargs)


def on_compute_loss_end_handler(self,
                                args: TrainingArguments,
                                state: TrainerState,
                                control: TrainerControl,
                                **kwargs):
    return self.call_event("on_compute_loss_begin", args, state, control, **kwargs)


CallbackHandler.on_compute_loss_begin = on_compute_loss_begin_handler
CallbackHandler.on_compute_loss_end = on_compute_loss_end_handler


# self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

class DistllTrainer(Trainer):

    def __init__(
            self,
            model,
            teacher_model,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional["Dataset"] = None,
            eval_dataset: Optional["Dataset"] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], "PreTrainedModel"] = None,
            compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            output_hidden_states: bool = True,
            apply_probability_shift: bool = False,
            **kwargs
    ):
        super().__init__(model,
                         args,
                         data_collator,
                         train_dataset,
                         eval_dataset,
                         tokenizer,
                         model_init,
                         compute_metrics,
                         callbacks,
                         optimizers,
                         **kwargs)

        self.output_hidden_states = output_hidden_states
        self.apply_probability_shift = apply_probability_shift
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        set_requires_grad(self.teacher_model, False)

    def compute_loss(self, model, batch, return_outputs=False):
        # batch - Dict['attention_mask', 'input_ids', 'labels', 'token_type_ids']

        student = model
        teacher = self.teacher_model
        if self.is_in_train:
            t_outputs = teacher(
                **batch,
                output_hidden_states=self.output_hidden_states,
                return_dict=True,
            )
        s_outputs = student(
            **batch,
            output_hidden_states=self.output_hidden_states,
            return_dict=True,
        )
        batch["s_logits"] = s_outputs["logits"]
        if self.is_in_train:
            batch["t_logits"] = t_outputs["logits"]
            if self.apply_probability_shift:
                batch["t_logits"] = probability_shift(
                    logits=batch["t_logits"],
                    labels=batch["targets"]
                )
        if self.output_hidden_states:
            batch["s_hidden_states"] = s_outputs["hidden_states"]
            if self.is_in_train:
                batch["t_hidden_states"] = t_outputs["hidden_states"]

        batch["task_loss"] = s_outputs["loss"]
        self.control = self.callback_handler.on_compute_loss_begin(self.args, self.state, self.control, batch=batch)
        #TODO : update metrics
        loss = batch["task_loss"] + batch["kl_div_loss"] + batch["mse_loss"]
        self.log({
                  "task_loss":batch["task_loss"].item(),
                  "kl_div_loss":batch["kl_div_loss"].item(),
                  "mse_loss":batch["mse_loss"].item(),
                  "loss":loss.item()
                  })
        return (loss, s_outputs) if return_outputs else loss

        ### Multiclass trainer
        # labels = batch.pop("labels")
        # outputs = model(**batch, return_dict=True)
        # logits = outputs.logits
        #
        # loss_fct = nn.BCEWithLogitsLoss()
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels),
        #                 labels.float().view(-1, self.model.config.num_labels))
        # return (loss, outputs) if return_outputs else loss

        # outputs = self.model(**batch, return_dict=True)
        # self.batch_metrics["loss"] = outputs["loss"]
        # self.batch["logits"] = outputs["logits"]
