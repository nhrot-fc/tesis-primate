import torch
from torch import nn
from torch.nn import functional as F


def init_logit_tensor(n_mels: int, p: float) -> torch.Tensor:
    return torch.full((1, 1, n_mels, 1), p).logit()


def init_inverse_softplus(n_mels: int, y: float) -> torch.Tensor:
    return torch.full((1, 1, n_mels, 1), y).expm1().log()


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

        self.s_raw = nn.Parameter(init_logit_tensor(n_mels, s_init))
        self.alpha_raw = nn.Parameter(init_logit_tensor(n_mels, alpha_init))
        self.delta_raw = nn.Parameter(init_inverse_softplus(n_mels, delta_init))
        self.r_raw = nn.Parameter(init_logit_tensor(n_mels, r_init))

    def log_smoothing(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        a = (1 - s).clamp_min(self.eps)
        log_a = a.log()  # (1, 1, n_mels, 1), <= 0

        T = x.shape[-1]
        k = torch.arange(T, device=x.device, dtype=x.dtype).view(1, 1, 1, T)
        log_x = x.clamp_min(self.eps).log()
        log_s = s.clamp_min(self.eps).log()

        # z_k = log(coeficiente_k) + log(x_k) - k*log(a); coeficiente_0 = 1 (no `s`),
        # porque `M_0 = x_0` sin mezclar con el frame anterior.
        z_first_frame = log_x  # válido sólo en k=0, donde k*log_a = 0
        z_rest = log_s + log_x - k * log_a
        z = torch.where(k == 0, z_first_frame, z_rest)

        log_M = k * log_a + torch.logcumsumexp(z, dim=-1)
        return log_M.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.s_raw.sigmoid()
        alpha = self.alpha_raw.sigmoid()
        delta = F.softplus(self.delta_raw)
        r = self.r_raw.sigmoid()

        M = self.log_smoothing(x, s)
        agc = x / torch.pow(self.eps + M, alpha)
        return torch.pow(agc + delta, r) - torch.pow(delta, r)
