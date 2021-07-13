from typing import Dict

from transformers import TrainingArguments, TrainerState, TrainerControl, TrainerCallback

from compressors.distillation.losses import AttentionLoss


class AttentionHiddenStatesCallback(TrainerCallback):
    """
    A class for objects that will inspect the state of the training loop at some events and take some decisions. At
    each of those events the following arguments are available:

    Args:
        args (:class:`~transformers.TrainingArguments`):
            The training arguments used to instantiate the :class:`~transformers.Trainer`.
        state (:class:`~transformers.TrainerState`):
            The current state of the :class:`~transformers.Trainer`.
        control (:class:`~transformers.TrainerControl`):
            The object that is returned to the :class:`~transformers.Trainer` and can be used to make some decisions.
        model (:class:`~transformers.PreTrainedModel` or :obj:`torch.nn.Module`):
            The model being trained.
        tokenizer (:class:`~transformers.PreTrainedTokenizer`):
            The tokenizer used for encoding the data.
        optimizer (:obj:`torch.optim.Optimizer`):
            The optimizer used for the training steps.
        lr_scheduler (:obj:`torch.optim.lr_scheduler.LambdaLR`):
            The scheduler used for setting the learning rate.
        train_dataloader (:obj:`torch.utils.data.dataloader.DataLoader`, `optional`):
            The current dataloader used for training.
        eval_dataloader (:obj:`torch.utils.data.dataloader.DataLoader`, `optional`):
            The current dataloader used for training.
        metrics (:obj:`Dict[str, float]`):
            The metrics computed by the last evaluation phase.

            Those are only accessible in the event :obj:`on_evaluate`.
        logs  (:obj:`Dict[str, float]`):
            The values to log.

            Those are only accessible in the event :obj:`on_log`.

    The :obj:`control` object is the only one that can be changed by the callback, in which case the event that changes
    it should return the modified version.

    The argument :obj:`args`, :obj:`state` and :obj:`control` are positionals for all events, all the others are
    grouped in :obj:`kwargs`. You can unpack the ones you need in the signature of the event using them. As an example,
    see the code of the simple :class:`~transformer.PrinterCallback`.

    Example::

        class PrinterCallback(TrainerCallback):

            def on_log(self, args, state, control, logs=None, **kwargs):
                _ = logs.pop("total_flos", None)
                if state.is_local_process_zero:
                    print(logs)
    """

    def __init__(self, output_key: str = "attention_loss", exclude_first_and_last: bool = True, p: int = 2, ):
        # super().__init__(order=CallbackOrder.Metric)
        self.output_key = output_key
        self.exclude_first_and_last = exclude_first_and_last
        self.criterion = AttentionLoss(p=p)


    def on_step_end(self,
                    args: TrainingArguments,
                    state: TrainerState,
                    control: TrainerControl,
                    metrics: Dict[str, float] = None,
                    **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        s_hiddens = metrics["s_hidden_states"]
        t_hiddens = metrics["t_hidden_states"]
        if self.exclude_first_and_last:
            s_hiddens = s_hiddens[1:-1]
            t_hiddens = t_hiddens[1:-1]
        metrics[self.output_key] = self.criterion(s_hiddens, t_hiddens)

__all__ = ["AttentionHiddenStatesCallback"]