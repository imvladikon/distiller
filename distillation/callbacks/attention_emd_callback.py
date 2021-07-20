import numpy as np
from catalyst.core import Callback

from distillation.callbacks.order import CallbackOrder
from distillation.losses._attention_emd_loss import AttentionEmdLoss


class AttentionEmdCallback(Callback):
    """
    MSE loss aka Hint loss for difference between hidden
    states of teacher and student model.

    Args:
        output_key: name for loss. Defaults to mse_loss.
    """

    def __init__(
            self,
            att_student_weight,
            att_teacher_weight,
            rep_student_weight,
            rep_teacher_weight,
            output_key: str = "emd_loss",
            use_att: bool = True,
            update_weight: bool = True,
            use_rep: bool = True,
            embedding_emd: bool = True,
            separate: bool = False,
            add_softmax: bool = True,
            temperature: float = 1.,
            device=None
    ):
        """

        Attention EMD loss
        https://arxiv.org/abs/2010.06133
        BERT-EMD: Many-to-Many Layer Mapping for BERT Compression with Earth Mover's Distance

        Args:
            output_key: name for loss. Defaults to mse_loss.
        """
        super().__init__(order=CallbackOrder.Metric)
        self.output_key = output_key

        self.criterion = AttentionEmdLoss(att_student_weight=att_student_weight,
                                          att_teacher_weight=att_teacher_weight,
                                          rep_student_weight=rep_student_weight,
                                          rep_teacher_weight=rep_teacher_weight,
                                          device=device,
                                          args=dict(use_att=use_att,
                                                    update_weight=update_weight,
                                                    use_rep=use_rep,
                                                    embedding_emd=embedding_emd,
                                                    separate=separate,
                                                    add_softmax=add_softmax),
                                          temperature=temperature)
        if device is not None:
            self.criterion.to(device)

    def on_batch_end(self, runner):
        runner.batch_metrics[self.output_key] = self.criterion(
            s_hidden_states=runner.batch["s_hidden_states"],
            s_attentions=runner.batch["s_attentions"],
            t_hidden_states=runner.batch["t_hidden_states"],
            t_attentions=runner.batch["t_attentions"]
        )

    @staticmethod
    def create_from_configs(teacher_config,
                            student_config,
                            use_att: bool = True,
                            update_weight: bool = True,
                            use_rep: bool = False,
                            embedding_emd: bool = False,
                            separate: bool = False,
                            add_softmax: bool = True,
                            temperature: float = 1.,
                            device=None):
        att_student_weight = np.ones(student_config.num_hidden_layers) / student_config.num_hidden_layers
        rep_student_weight = np.ones(student_config.num_hidden_layers) / student_config.num_hidden_layers
        att_teacher_weight = np.ones(teacher_config.num_hidden_layers) / teacher_config.num_hidden_layers
        rep_teacher_weight = np.ones(teacher_config.num_hidden_layers) / teacher_config.num_hidden_layers
        return AttentionEmdCallback(att_student_weight,
                                    att_teacher_weight,
                                    rep_student_weight,
                                    rep_teacher_weight,
                                    use_att=use_att,
                                    update_weight=update_weight,
                                    use_rep=use_rep,
                                    embedding_emd=embedding_emd,
                                    separate=separate,
                                    add_softmax=add_softmax,
                                    temperature=temperature,
                                    device=device)


__all__ = ["AttentionEmdCallback"]
