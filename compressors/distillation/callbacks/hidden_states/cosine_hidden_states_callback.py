from typing import Dict

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from compressors.distillation.callbacks.order import CallbackOrder
from compressors.distillation.losses import CosineHiddenStateLoss


class CosineHiddenStatesCallback(TrainerCallback):
    """
    Cosine loss for difference between hidden states of teacher and student model.

    Args:
        output_key: name for loss. Defaults to cosine_loss.
        last_only: If set to True takes only last hidden state.
                Usually cosine loss applied in this way. Defaults to True.
    """

    def __init__(
            self,
            output_key: str = "cosine_loss",
            last_only: bool = True,
            need_mapping: bool = False,
            teacher_hidden_state_dim: int = None,
            student_hidden_state_dim: int = None,
    ):
        """
        Cosine loss for difference between hidden states of teacher and student model.

        Args:
             output_key: name for loss. Defaults to cosine_loss.
             last_only: If set to True takes only last hidden state.
                Usually cosine loss applied in this way. Defaults to True.
        """
        # super().__init__(order=CallbackOrder.Metric)
        self.output_key = output_key
        self.last_only = last_only
        self.criterion = CosineHiddenStateLoss(
            need_mapping=need_mapping,
            teacher_hidden_state_dim=teacher_hidden_state_dim,
            student_hidden_state_dim=student_hidden_state_dim,
        )

    def on_compute_loss_begin(self,
                    args: TrainingArguments,
                    state: TrainerState,
                    control: TrainerControl,
                    batch: Dict[str, float] = None,
                    **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        s_hiddens = batch["s_hidden_states"]
        t_hiddens = batch["t_hidden_states"]
        if self.last_only:
            s_hiddens = s_hiddens[-1]
            t_hiddens = t_hiddens[-1]
        batch[self.output_key] = self.criterion(s_hiddens, t_hiddens)


__all__ = ["CosineHiddenStatesCallback"]
