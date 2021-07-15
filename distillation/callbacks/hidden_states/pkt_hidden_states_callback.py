from typing import Dict

from transformers import TrainerCallback, TrainingArguments, TrainerControl, TrainerState

from distillation.losses import pkt_loss


class PKTHiddenStatesCallback(TrainerCallback):
    """
    Probabilistic Knowlewdge Transfer loss for difference between hidden states
    of teacher and student model.
    Proposed in https://arxiv.org/abs/1803.10837.

    Args:
        output_key: name for loss. Defaults to mse_loss.
        last_only: If set to True takes only last hidden state.
                Usually pkt loss applied in this way. Defaults to True.
    """

    def __init__(self, output_key: str = "pkt_loss", last_only: bool = True):
        """
        Probabilistic Knowlewdge Transfer loss for difference between hidden states
        of teacher and student model.
        Proposed in https://arxiv.org/abs/1803.10837.

        Args:
            output_key: name for loss. Defaults to pkt_loss.
            last_only: If set to True takes only last hidden state.
                Usually pkt loss applied in this way. Defaults to True.
        """
        # super().__init__(order=CallbackOrder.Metric)
        self.output_key = output_key
        self.last_only = last_only

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
        metrics[self.output_key] = pkt_loss(
            s_hidden_states=s_hiddens, t_hidden_states=t_hiddens,
        )


__all__ = ["PKTHiddenStatesCallback"]
