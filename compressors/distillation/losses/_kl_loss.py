"""Module with loss functions for knoweledge distillation.

"""
import torch
from torch import FloatTensor, nn
from torch.nn import functional as F

from compressors.distillation.schedulers.temperature_schedulers import ConstantTemperatureScheduler


class KLDivLoss(nn.Module):
    def __init__(self,
                 temperature: float = 1.0,
                 scheduler: nn.Module = None):
        super(KLDivLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.KLDivLoss()
        self.scheduler = scheduler
        if scheduler is None:
            self.scheduler = ConstantTemperatureScheduler(temperature=temperature)
            self.reduction_fn = lambda x: x
        else:
            self.reduction_fn = torch.sum

    def forward(self, s_logits, t_logits):
        temperature = self.scheduler(s_logits,
                                     t_logits,
                                     temperature=self.temperature)

        return self.reduction_fn(self.criterion(
            F.log_softmax(s_logits / temperature, dim=1),
            F.softmax(t_logits / temperature, dim=1),
        ) * (temperature ** 2))


def kl_div_loss(
        s_logits: FloatTensor, t_logits: FloatTensor, temperature: float = 1.0
) -> FloatTensor:
    """KL-divergence loss
    https://arxiv.org/abs/1503.02531
    Distilling the Knowledge in a Neural Network
    Args:
        s_logits (FloatTensor): output for student model.
        t_logits (FloatTensor): output for teacher model.
        temperature (float, optional): Temperature for teacher distribution.
            Defaults to 1.

    Returns:
        FloatTensor: Divergence between student and teachers distribution.
    """
    loss_fn = nn.KLDivLoss()
    loss = loss_fn(
        F.log_softmax(s_logits / temperature, dim=1), F.softmax(t_logits / temperature, dim=1),
    ) * (temperature ** 2)
    return loss
