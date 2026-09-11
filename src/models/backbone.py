import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from core.config import HF_DIR, P

if TYPE_CHECKING:
    from transformers import ASTModel

logger = logging.getLogger(__name__)

AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
# Paso temporal de los parches del AST (el checkpoint trae 10); con 331 cuadros deja 32 tokens
TIME_STRIDE = 10
# Pirámide {4x, 2x, 1x, 1/2x} sobre el mapa de tokens del AST, como en ViTDet.
N_LEVELS = 4


def local_ast_dir(checkpoint: str = AST_CHECKPOINT) -> Path:
    return HF_DIR / checkpoint.replace("/", "__")


def load_ast_model(checkpoint: str = AST_CHECKPOINT) -> "ASTModel":
    # `transformers` es el extra `detr`: sólo hace falta si se carga esta arquitectura.
    from transformers import ASTModel

    local_dir = local_ast_dir(checkpoint)
    if local_dir.is_dir():
        try:
            return ASTModel.from_pretrained(local_dir, local_files_only=True)
        except Exception:
            logger.warning("Copia local inutilizable en %s; se redescarga.", local_dir)

    logger.info("Descargando backbone AST '%s' desde HuggingFace...", checkpoint)
    model = ASTModel.from_pretrained(checkpoint)
    local_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(local_dir)
    logger.info("Backbone AST guardado en %s", local_dir)
    return model


class ASTBackbone(nn.Module):
    # Se afina entero, a la misma tasa que la cabeza
    def __init__(
        self,
        n_frames: int | None = None,
        time_stride: int = TIME_STRIDE,
        checkpoint: str = AST_CHECKPOINT,
    ) -> None:
        super().__init__()
        self.model = load_ast_model(checkpoint)
        self.n_frames = n_frames if n_frames is not None else P.n_frames
        self.time_stride = time_stride
        self.interpolate_position_embeddings(self.n_frames, time_stride)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def n_mels(self) -> int:
        return int(self.model.config.num_mel_bins)

    def interpolate_position_embeddings(self, n_frames: int, time_stride: int) -> None:
        config = self.model.config
        patch_size = (
            config.patch_size if isinstance(config.patch_size, int) else config.patch_size[0]
        )
        self.freq_out = (config.num_mel_bins - patch_size) // config.frequency_stride + 1
        # Leer el paso del checkpoint antes de pisarlo.
        pretrained_time_out = (config.max_length - patch_size) // config.time_stride + 1
        self.time_out = (n_frames - patch_size) // time_stride + 1

        position_embeddings = self.model.embeddings.position_embeddings
        special_tokens = position_embeddings[:, :2]
        patch_positions = position_embeddings[:, 2:]
        patch_positions = patch_positions.reshape(
            1, self.freq_out, pretrained_time_out, -1
        ).permute(0, 3, 1, 2)
        patch_positions = F.interpolate(
            patch_positions,
            size=(self.freq_out, self.time_out),
            mode="bilinear",
            align_corners=False,
        )
        patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(
            1, self.freq_out * self.time_out, -1
        )
        self.model.embeddings.position_embeddings = nn.Parameter(
            torch.cat([special_tokens, patch_positions], dim=1)
        )

        self.model.embeddings.patch_embeddings.projection.stride = (
            config.frequency_stride,
            time_stride,
        )
        config.time_stride = time_stride
        config.max_length = n_frames
        logger.info(
            "AST: %d x %d tokens | time_stride=%d", self.freq_out, self.time_out, time_stride
        )

    def forward(self, mel: Tensor) -> Tensor:
        input_values = mel.squeeze(1).transpose(1, 2)  # (B,1,n_mels,T) -> (B,T,n_mels)
        return self.model(input_values=input_values).last_hidden_state[:, 2:]


class MultiScalePyramid(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
                    nn.GroupNorm(num_groups, dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
                ),
                nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
                nn.Identity(),
                nn.Conv2d(dim, dim, kernel_size=2, stride=2),
            ]
        )
        assert len(self.blocks) == N_LEVELS

    @staticmethod
    def check_input_size(height: int, width: int) -> None:
        # Con una dimensión impar el nivel 1/2x pierde una fila y los niveles se desalinean.
        if height % 2 or width % 2:
            raise ValueError(
                f"la pirámide necesita dimensiones pares, no ({height}, {width}); ajustá "
                "`time_stride` para que `time_out` y `freq_out` lo sean."
            )

    def forward(self, features: Tensor) -> list[Tensor]:
        return [block(features) for block in self.blocks]
