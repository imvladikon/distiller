from typing import Dict

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from distillation.losses import MSEHiddenStatesLoss


class MSEHiddenStatesCallback(TrainerCallback):
    """
    MSE loss aka Hint loss for difference between hidden
    states of teacher and student model.

    Args:
        output_key: name for loss. Defaults to mse_loss.
    """

    def __init__(
            self,
            output_key: str = "mse_loss",
            normalize: bool = False,
            need_mapping: bool = False,
            teacher_hidden_state_dim: int = None,
            student_hidden_state_dim: int = None,
            num_layers: int = None,
            device=None
    ):
        """
        MSE loss aka Hint loss for difference between hidden
        states of teacher and student model.

        Args:
            output_key: name for loss. Defaults to mse_loss.
        """
        # super().__init__(order=CallbackOrder.Metric)
        self.output_key = output_key
        self.criterion = MSEHiddenStatesLoss(
            normalize=normalize,
            need_mapping=need_mapping,
            teacher_hidden_state_dim=teacher_hidden_state_dim,
            student_hidden_state_dim=student_hidden_state_dim,
            num_layers=num_layers,
        )
        if device is not None:
            self.criterion.to(device)

    def on_compute_loss_begin(self,
                              args: TrainingArguments,
                              state: TrainerState,
                              control: TrainerControl,
                              batch: Dict[str, float] = None,
                              **kwargs):
        batch[self.output_key] = self.criterion(
            s_hidden_states=batch["s_hidden_states"],
            t_hidden_states=batch["t_hidden_states"],
        )


__all__ = ["MSEHiddenStatesCallback"]
