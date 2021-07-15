import os
from typing import Optional, List

from transformers import is_torch_tpu_available
from transformers.integrations import WandbCallback, logger


class WandbCallbackCustomized(WandbCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `Weight and Biases <https://www.wandb.com/>`__.
    """

    def __init__(self,
                 project: Optional[str] = None,
                 name: Optional[str] = None,
                 entity: Optional[str] = None,
                 notes: Optional[str] = None,
                 tags: Optional[List] = None,
                 **config_kwargs) -> None:

        super().__init__()
        project = project if project is not None else os.environ.get("WANDB_PROJECT", "huggingface")
        entity = entity if entity is not None else os.environ.get("WANDB_ENTITY", "huggingface")
        self.init_args = {
            "project": project,
            "name": name,
            "entity": entity,
            "notes": notes
        }
        self.tags = tags
        self.config_kwargs = config_kwargs

    def setup(self, args, state, model, **kwargs):
        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_sanitized_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            trial_name = state.trial_name
            init_args = self.init_args
            if trial_name is not None:
                run_name = trial_name
                init_args["group"] = args.run_name
            else:
                run_name = args.run_name
            if not init_args.get("name", ""):
                init_args["name"] = run_name

            if self._wandb.run is None:
                self._wandb.run = self._wandb.init(
                    **init_args,
                )
            if self.tags is not None:
                self._wandb.run.tags = self.tags
            # add config parameters (run may have been created manually)
            self._wandb.config.update(combined_dict, allow_val_change=True)
            if self.config_kwargs is not None:
                self._wandb.config.update(self.config_kwargs, allow_val_change=True)
            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, args.logging_steps)
                )
