import torch
from torch import nn


class ConstantTemperatureScheduler(nn.Module):

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self,
                logits_S: torch.FloatTensor,
                logits_T: torch.FloatTensor,
                temperature: float = None):
        return self.temperature


class FlswTemperatureScheduler(nn.Module):
    """
    https://arxiv.org/abs/1911.07471
    Preparing Lessons: Improve Knowledge Distillation with Better Supervision
    Dynamic Temperature Distillation to avoid overly uncertain supervision from teacher model
    """

    def __init__(self,
                 beta: float,
                 gamma: float,
                 eps: float = 1e-4,
                 *args,
                 **kwargs):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    @torch.no_grad()
    def forward(self,
                logits_S: torch.FloatTensor,
                logits_T: torch.FloatTensor,
                temperature: float):
        v = logits_S.detach()
        t = logits_T.detach()
        v = v / (torch.norm(v, dim=-1, keepdim=True) + self.eps)
        t = t / (torch.norm(t, dim=-1, keepdim=True) + self.eps)
        w = torch.pow((1 - (v * t).sum(dim=-1)), self.gamma)
        tau = temperature + (w.mean() - w) * self.beta
        return tau.unsqueeze(-1)


class CwsmTemperatureScheduler(nn.Module):
    """
    https://arxiv.org/abs/1911.07471
    Preparing Lessons: Improve Knowledge Distillation with Better Supervision
    Dynamic Temperature Distillation to avoid overly uncertain supervision from teacher model
    """

    def __init__(self, beta: float, *args, **kwargs):
        super().__init__()
        self.beta = beta

    @torch.no_grad()
    def forward(self,
                logits_S: torch.FloatTensor,
                logits_T: torch.FloatTensor,
                temperature: float):
        v = logits_S.detach()
        v = torch.softmax(v, dim=-1)
        v_max = v.max(dim=-1)[0]
        w = 1 / (v_max + 1e-3)
        tau = temperature + (w.mean() - w) * self.beta
        return tau.unsqueeze(-1)