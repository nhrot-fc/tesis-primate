import torch
from torch import nn


class TrainablePCEN(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        s_init: float = 0.025,
        alpha_init: float = 0.98,
        delta_init: float = 2.0,
        r_init: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps

        self.s = nn.Parameter(torch.full((1, 1, n_mels, 1), s_init))
        self.alpha = nn.Parameter(torch.full((1, 1, n_mels, 1), alpha_init))
        self.delta = nn.Parameter(torch.full((1, 1, n_mels, 1), delta_init))
        self.r = nn.Parameter(torch.full((1, 1, n_mels, 1), r_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.s.clamp(0.0, 1.0)[..., 0]  # (1, 1, n_mels), sin el eje temporal
        alpha = self.alpha.clamp(0.0, 1.0)
        delta = self.delta.clamp(min=0.0)
        r = self.r.clamp(min=self.eps, max=1.0)

        frames = x.unbind(-1)  # T tensores (B, 1, n_mels)
        smoothed = [frames[0]]
        for frame in frames[1:]:
            smoothed.append((1 - s) * smoothed[-1] + s * frame)
        M = torch.stack(smoothed, dim=-1)  # (B, 1, n_mels, T)

        agc = x / torch.pow(self.eps + M, alpha)
        return torch.pow(agc + delta, r) - torch.pow(delta, r)
