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
        s_clamped = torch.clamp(self.s, min=0.0, max=1.0)

        M_t = x[:, :, :, 0]
        M = [M_t]

        for t in range(1, x.shape[-1]):
            M_t = (1 - s_clamped) * M_t + s_clamped * x[:, :, :, t]
            M.append(M_t)

        M = torch.stack(M, dim=-1)  # (B, 1, F, T)

        smooth_denom = torch.pow(self.eps + M, self.alpha)
        pcen = torch.pow((x / smooth_denom) + self.delta, self.r) - torch.pow(self.delta, self.r)
        return pcen
