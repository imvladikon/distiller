from typing import List, Union, Dict

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from distillation.hidden_states import hidden_states_select


class HiddenStatesSelectCallback(TrainerCallback):
    def __init__(self, layers: Union[int, List[int]], hiddens_key: str = "t_hidden_states"):
        """

        Args:
            layers:
            hiddens_key:
        """
        super().__init__(order=CallbackOrder.hiddens_slct)
        self.layers = layers
        self.hiddens_key = hiddens_key

    def on_step_end(self,
                    args: TrainingArguments,
                    state: TrainerState,
                    control: TrainerControl,
                    metrics: Dict[str, float] = None,
                    **kwargs):
        metrics[self.hiddens_key] = hidden_states_select(
            hidden_states=metrics[self.hiddens_key], layers=self.layers
        )  # runner.batch


__all__ = ["HiddenStatesSelectCallback"]
