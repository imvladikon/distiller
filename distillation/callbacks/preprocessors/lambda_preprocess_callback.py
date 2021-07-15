from typing import Callable, List, Union

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class LambdaPreprocessCallback(TrainerCallback):
    """Filters output with your lambda function. Inplace analog of ``LambdaWrapper``.

        Args:
            lambda_fn (Callable): Function to apply.
            keys_to_apply (Union[List[str], str], optional): Keys in batch dict to apply function.
                Defaults to ["s_hidden_states", "t_hidden_states"].

        Raises:
            TypeError: When keys_to_apply is not str or list.
    """

    def __init__(
        self,
        lambda_fn: Callable,
        keys_to_apply: Union[List[str], str] = ["s_hidden_states", "t_hidden_states"],
    ):
        """Filters output with your lambda function.

        Args:
            lambda_fn (Callable): Function to apply.
            keys_to_apply (Union[List[str], str], optional): Keys in batch dict to apply function.
                Defaults to ["s_hidden_states", "t_hidden_states"].

        Raises:
            TypeError: When keys_to_apply is not str or list.
        """
        # super().__init__(order=CallbackOrder.HiddensSlct)
        if not isinstance(keys_to_apply, (list, str)):
            raise TypeError("keys to apply should be str or list of str.")
        self.keys_to_apply = keys_to_apply
        self.lambda_fn = lambda_fn

    def on_compute_loss_begin(self,
                    args: TrainingArguments,
                    state: TrainerState,
                    control: TrainerControl,
                    batch=None,
                    **kwargs):

        if isinstance(self.keys_to_apply, list): # ['s_hidden_states', 't_hidden_states']
            fn_inp = [batch[key] for key in self.keys_to_apply]
            fn_output = self.lambda_fn(*fn_inp)
            if isinstance(fn_output, tuple):
                for idx, key in enumerate(self.keys_to_apply):
                    batch[key] = fn_output[idx]
            elif isinstance(fn_output, dict):
                for outp_k, outp_v in fn_output.items():
                    batch[outp_k] = outp_v
            else:
                raise Exception(
                    "If keys_to_apply is list, then function output should be tuple or dict."
                )
        elif isinstance(self.keys_to_apply, str):
            batch[self.keys_to_apply] = self.lambda_fn(self.keys_to_apply)


__all__ = ["LambdaPreprocessCallback"]
