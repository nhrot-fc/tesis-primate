"""Backbone AST (Gong et al., 2021) y pirámide multiescala al estilo ViTDet (Li et al., 2022).

`ASTBackbone` envuelve el AST real preentrenado en AudioSet (HuggingFace
`transformers`: 12 capas, 768-dim, patches 16x16 con stride 10) en vez de
reimplementarlo. Cargar los pesos preentrenados es lo que aporta valor frente
a un transformer entrenado desde cero con ~8k ventanas.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import ASTModel

from core.config import P

AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"


class ASTBackbone(nn.Module):
    """AST preentrenado + proyección lineal a `embed_dim`.

    El checkpoint fue preentrenado con `num_mel_bins=128`, `max_length=1024` y
    patch embedding 16x16 con stride 10 en ambos ejes. Aquí `num_mel_bins` ya
    coincide (evita distorsionar el eje de frecuencia: ver notas de la Etapa 0
    sobre por qué mantener 128 mels), pero `max_length` casi nunca coincide con
    `n_frames` del clip, así que los embeddings de posición se interpolan
    bilinealmente sobre el eje temporal tras cargar el checkpoint — el mismo
    truco que usa el AST original al cambiar de duración de entrada.

    `time_stride` además reduce el stride temporal del patch embedding (10 ->
    5 por defecto): duplica la resolución de tokens en el tiempo (~93ms ->
    ~47ms por token con el `n_frames` de Etapa 0). Es el techo real de
    localización de eventos cortos (p.ej. sm/cc, ~81ms) que ninguna
    augmentation ni ajuste de umbral puede cruzar -- la pirámide sube a más
    posiciones pero por interpolación, la información sigue viniendo de acá.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_frames: int | None = None,
        time_stride: int = 5,
        checkpoint: str = AST_CHECKPOINT,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.model = ASTModel.from_pretrained(checkpoint)
        self._interpolate_time_pos_embed(
            n_frames if n_frames is not None else P.n_frames, time_stride
        )

        self.freeze = freeze
        if freeze:
            for param in self.model.parameters():
                param.requires_grad_(False)

        self.proj = nn.Linear(self.model.config.hidden_size, embed_dim)

    def _interpolate_time_pos_embed(self, n_frames: int, time_stride: int) -> None:
        config = self.model.config
        patch_size = (
            config.patch_size if isinstance(config.patch_size, int) else config.patch_size[0]
        )
        freq_out = (config.num_mel_bins - patch_size) // config.frequency_stride + 1
        # con el stride ORIGINAL del checkpoint (10): así fueron preentrenados estos
        # embeddings de posición, hay que leerlo antes de pisar `config.time_stride`.
        time_out_old = (config.max_length - patch_size) // config.time_stride + 1
        self.freq_out = freq_out
        self.time_out = (n_frames - patch_size) // time_stride + 1

        pos_embed = self.model.embeddings.position_embeddings  # (1, 2+freq_out*time_out_old, C)
        special, patches = pos_embed[:, :2], pos_embed[:, 2:]
        patches = patches.reshape(1, freq_out, time_out_old, -1).permute(0, 3, 1, 2)
        patches = F.interpolate(
            patches, size=(freq_out, self.time_out), mode="bilinear", align_corners=False
        )
        patches = patches.permute(0, 2, 3, 1).reshape(1, freq_out * self.time_out, -1)
        self.model.embeddings.position_embeddings = nn.Parameter(
            torch.cat([special, patches], dim=1)
        )

        self.model.embeddings.patch_embeddings.projection.stride = (
            config.frequency_stride,
            time_stride,
        )
        config.time_stride = time_stride
        config.max_length = n_frames

    def train(self, mode: bool = True) -> "ASTBackbone":
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 1, n_mels, n_frames) -> AST espera (B, n_frames, n_mels)
        input_values = x.squeeze(1).transpose(1, 2)
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            tokens = self.model(input_values=input_values).last_hidden_state[:, 2:]  # sin CLS+dist
        tokens = self.proj(tokens)
        return tokens.transpose(1, 2).unflatten(-1, (self.freq_out, self.time_out))


class MultiScalePyramid(nn.Module):
    def __init__(self, dim: int = 256, num_groups: int = 8) -> None:
        super().__init__()
        self.upsample_4x = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups, dim),
            nn.GELU(),
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
        )
        self.upsample_2x = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.identity = nn.Identity()
        self.downsample_2x = nn.Conv2d(dim, dim, kernel_size=2, stride=2)

    def forward(self, features: Tensor) -> list[Tensor]:
        return [
            self.upsample_4x(features),
            self.upsample_2x(features),
            self.identity(features),
            self.downsample_2x(features),
        ]
