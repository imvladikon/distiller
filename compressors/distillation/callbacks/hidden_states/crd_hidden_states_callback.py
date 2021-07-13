from typing import Dict

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from compressors.distillation.callbacks.order import CallbackOrder
from compressors.distillation.losses import CRDLoss


class CRDHiddenStatesCallback(TrainerCallback):
    """
    CONTRASTIVE REPRESENTATION DISTILLATION for difference between hidden states of teacher and student model.

    Args:
        output_key: name for loss. Defaults to cosine_loss.
        last_only: If set to True takes only last hidden state.
                Usually cosine loss applied in this way. Defaults to True.
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        n_data: int,
        output_key: str = "crd_loss",
        last_only: bool = True,
        feature_dim: int = 128,
        nce_k: int = 16384,
        nce_t: float = 0.07,
        nce_m: float = 0.5,
    ):
        """
        Cosine loss for difference between hidden states of teacher and student model.

        Args:
             output_key: name for loss. Defaults to cosine_loss.
             last_only: If set to True takes only last hidden state.
                Usually cosine loss applied in this way. Defaults to True.
        """
        super().__init__(order=CallbackOrder.Metric)
        self.output_key = output_key
        self.last_only = last_only
        self.criterion = CRDLoss(
            student_dim=student_dim,
            teacher_dim=teacher_dim,
            n_data=n_data,
            feature_dim=feature_dim,
            nce_k=nce_k,
            nce_t=nce_t,
            nce_m=nce_m,
        )

    def on_step_end(self,
                    args: TrainingArguments,
                    state: TrainerState,
                    control: TrainerControl,
                    metrics: Dict[str, float] = None,
                    **kwargs):

        s_hiddens = metrics["s_hidden_states"]
        t_hiddens = metrics["t_hidden_states"]
        if self.last_only:
            s_hiddens = s_hiddens[-1]
            t_hiddens = t_hiddens[-1]
        metrics[self.output_key] = self.criterion(s_hiddens, t_hiddens)





__all__ = ["CRDHiddenStatesCallback"]
