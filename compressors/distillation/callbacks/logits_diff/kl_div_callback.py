from typing import Dict

from catalyst.core import Callback
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from compressors.distillation.callbacks.order import CallbackOrder
from compressors.distillation.losses import KLDivLoss


class KLDivCallback(TrainerCallback):
    def __init__(
            self,
            output_key: str = "kl_div_loss",
            temperature: float = 1.0,
            student_logits_key: str = "s_logits",
            teacher_logits_key: str = "t_logits",
    ):
        # super().__init__(order=CallbackOrder.Metric)
        self.output_key = output_key
        self.criterion = KLDivLoss(temperature=temperature)
        self.teacher_logits_key = teacher_logits_key
        self.student_logits_key = student_logits_key

    def on_compute_loss_begin(self,
                              args: TrainingArguments,
                              state: TrainerState,
                              control: TrainerControl,
                              batch: Dict[str, float] = None,
                              **kwargs):
        batch[self.output_key] = self.criterion(
            s_logits=batch[self.student_logits_key],
            t_logits=batch[self.teacher_logits_key],
        )


__all__ = ["KLDivCallback"]
