import collections
import logging
from typing import Optional, Callable, Dict, Tuple, List

import transformers.trainer
import torch
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, DataCollator, PreTrainedTokenizerBase, TrainerCallback, \
    TrainerState, TrainerControl
from torch import nn
from transformers.trainer_callback import CallbackHandler

from compressors.distillation.data.label_smoothing import probability_shift
from compressors.utils import set_requires_grad

"""
Overriding trainers functionality from hugginface
"""

TrainerState.batch_metrics = dict()


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
    return self.call_event("on_compute_loss_end", args, state, control, **kwargs)


CallbackHandler.on_compute_loss_begin = on_compute_loss_begin_handler
CallbackHandler.on_compute_loss_end = on_compute_loss_end_handler



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
        self.state.batch_metrics.update({
            "task_loss": batch["task_loss"],
            "kl_div_loss": batch["kl_div_loss"],
            "mse_loss": batch["mse_loss"],
        })
        self.control = self.callback_handler.on_compute_loss_end(self.args, self.state, self.control, batch=batch)
        loss = self.state.batch_metrics.get("loss", batch["task_loss"])
        if not "loss" in self.state.batch_metrics:
            # log.WARN()
            print("incorrect loss, check aggregation callback. only task loss is used")
        return (loss, s_outputs) if return_outputs else loss


class ProgressCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that displays the progress of training or evaluation.
    """

    def __init__(self):
        self.training_bar: tqdm = None
        self.prediction_bar: tqdm = None
        self.current_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            loader_key = "train"
            self.training_bar = tqdm(total=state.max_steps,
                                     # desc=f"{self.current_step}/{state.max_steps}"
                                     #      f" * Epoch ({loader_key})"
                                     )
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            batch_metrics = {k: float(v) for k, v in state.batch_metrics.items()}
            self.training_bar.set_postfix(
                **{
                    k: "{:3.3f}".format(v) if v > 1e-3 else "{:1.3e}".format(v)
                    for k, v in sorted(batch_metrics.items())
                }
            )
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step
            # self.tqdm.update()

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_local_process_zero and isinstance(eval_dataloader.dataset, collections.abc.Sized):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(total=len(eval_dataloader), leave=self.training_bar is None)
            self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)
            self.training_bar.write(str(logs))

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar.close()
            self.training_bar = None


transformers.trainer.DEFAULT_PROGRESS_CALLBACK = ProgressCallback
