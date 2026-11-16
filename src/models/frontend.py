from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from core.config import P


def logit_initializer(n_mels: int, value: float) -> torch.Tensor:
    return torch.full((1, 1, n_mels, 1), value).logit()


def inverse_softplus_initializer(n_mels: int, value: float) -> torch.Tensor:
    return torch.full((1, 1, n_mels, 1), value).expm1().log()


class TrainablePCEN(nn.Module):
    def __init__(self, n_mels: int):
        super().__init__()
        # Sin restringir, una fila por banda; `forward` los lleva a su rango.
        self.s_raw = nn.Parameter(logit_initializer(n_mels, 0.025))
        self.alpha_raw = nn.Parameter(logit_initializer(n_mels, 0.98))
        self.delta_raw = nn.Parameter(inverse_softplus_initializer(n_mels, 2.0))
        self.r_raw = nn.Parameter(logit_initializer(n_mels, 0.5))

    def smoothed_energy(self, mel: torch.Tensor, smoothing: torch.Tensor) -> torch.Tensor:
        decay = (1 - smoothing).clamp_min(P.eps)
        log_decay = decay.log()
        log_smoothing = smoothing.clamp_min(P.eps).log()
        log_mel = mel.clamp_min(P.eps).log()
        frame = torch.arange(mel.shape[-1], device=mel.device, dtype=mel.dtype).view(1, 1, 1, -1)

        # M_k = (1-s)·M_{k-1} + s·x_k en forma cerrada, sumado en log para no desbordar.
        log_terms = torch.where(frame == 0, log_mel, log_smoothing + log_mel - frame * log_decay)
        return (frame * log_decay + torch.logcumsumexp(log_terms, dim=-1)).exp()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        smoothing = self.s_raw.sigmoid()
        gain_exponent = self.alpha_raw.sigmoid()
        bias = F.softplus(self.delta_raw)
        compression = self.r_raw.sigmoid()

        energy = P.eps + self.smoothed_energy(mel, smoothing)
        gain_controlled = mel / torch.pow(energy, gain_exponent)
        return torch.pow(gain_controlled + bias, compression) - torch.pow(bias, compression)


class LogMel(nn.Module):
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return torch.log(mel.clamp_min(0) + P.eps)


FRONTENDS: dict[str, Callable[[int], nn.Module]] = {
    "none": lambda n_mels: nn.Identity(),
    "logmel": lambda n_mels: LogMel(),
    "pcen": TrainablePCEN,
}


def build_frontend(name: str, n_mels: int) -> nn.Module:
    if name not in FRONTENDS:
        raise ValueError(f"unknown front-end {name!r}; known: {sorted(FRONTENDS)}")
    return FRONTENDS[name](n_mels)
