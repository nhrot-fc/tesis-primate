import torch
from torch import nn
from torch.nn import functional as F


def logit_initializer(n_mels: int, value: float) -> torch.Tensor:
    return torch.full((1, 1, n_mels, 1), value).logit()


def inverse_softplus_initializer(n_mels: int, value: float) -> torch.Tensor:
    return torch.full((1, 1, n_mels, 1), value).expm1().log()


class TrainablePCEN(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        smoothing_init: float = 0.025,
        gain_exponent_init: float = 0.98,
        bias_init: float = 2.0,
        compression_init: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        # Sin restringir, una fila por banda; `forward` los lleva a su rango.
        self.s_raw = nn.Parameter(logit_initializer(n_mels, smoothing_init))
        self.alpha_raw = nn.Parameter(logit_initializer(n_mels, gain_exponent_init))
        self.delta_raw = nn.Parameter(inverse_softplus_initializer(n_mels, bias_init))
        self.r_raw = nn.Parameter(logit_initializer(n_mels, compression_init))

    def smoothed_energy(self, mel: torch.Tensor, smoothing: torch.Tensor) -> torch.Tensor:
        decay = (1 - smoothing).clamp_min(self.eps)
        log_decay = decay.log()
        log_smoothing = smoothing.clamp_min(self.eps).log()
        log_mel = mel.clamp_min(self.eps).log()
        frame = torch.arange(mel.shape[-1], device=mel.device, dtype=mel.dtype).view(1, 1, 1, -1)

        # M_k = (1-s)·M_{k-1} + s·x_k en forma cerrada, sumado en log para no desbordar.
        log_terms = torch.where(frame == 0, log_mel, log_smoothing + log_mel - frame * log_decay)
        return (frame * log_decay + torch.logcumsumexp(log_terms, dim=-1)).exp()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        smoothing = self.s_raw.sigmoid()
        gain_exponent = self.alpha_raw.sigmoid()
        bias = F.softplus(self.delta_raw)
        compression = self.r_raw.sigmoid()

        energy = self.eps + self.smoothed_energy(mel, smoothing)
        gain_controlled = mel / torch.pow(energy, gain_exponent)
        return torch.pow(gain_controlled + bias, compression) - torch.pow(bias, compression)


class LogMelFrontend(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return torch.log(mel.clamp_min(0) + self.eps)
