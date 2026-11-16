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
TIME_STRIDE = 10  # con 331 cuadros deja 32 tokens
# Pirámide {4x, 2x, 1x, 1/2x} sobre el mapa de tokens del AST, como en ViTDet.
N_LEVELS = 4


def local_ast_dir() -> Path:
    return HF_DIR / AST_CHECKPOINT.replace("/", "__")


def load_ast_model() -> "ASTModel":
    from transformers import ASTModel

    local_dir = local_ast_dir()
    if local_dir.is_dir():
        try:
            return ASTModel.from_pretrained(local_dir, local_files_only=True)
        except Exception:
            logger.warning("Local copy in %s is unusable; downloading it again.", local_dir)

    logger.info("Downloading the AST backbone '%s' from HuggingFace...", AST_CHECKPOINT)
    model = ASTModel.from_pretrained(AST_CHECKPOINT)
    local_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(local_dir)
    logger.info("AST backbone saved to %s", local_dir)
    return model


class ASTBackbone(nn.Module):
    def __init__(
        self,
        n_frames: int | None = None,
        time_stride: int = TIME_STRIDE,
        frozen: bool = False,
    ) -> None:
        super().__init__()
        self.model = load_ast_model()
        self.n_frames = n_frames if n_frames is not None else P.n_frames
        self.time_stride = time_stride
        self.interpolate_position_embeddings(self.n_frames, time_stride)
        self.frozen = frozen
        if frozen:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "ASTBackbone":
        # Congelado = sin dropout: la misma salida en train y en val
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

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
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
                    nn.GroupNorm(8, dim),
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
        if height % 2 or width % 2:
            raise ValueError(
                f"the pyramid needs even sizes, not ({height}, {width}); set "
                "`time_stride` so that `time_out` and `freq_out` are even."
            )

    def forward(self, features: Tensor) -> list[Tensor]:
        return [block(features) for block in self.blocks]
