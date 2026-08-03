import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import ASTModel

from core.config import P, settings

logger = logging.getLogger(__name__)

AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"


def local_ast_dir(checkpoint: str = AST_CHECKPOINT) -> Path:
    return settings.checkpoints_dir / "hf" / checkpoint.replace("/", "__")


def load_ast_model(checkpoint: str = AST_CHECKPOINT) -> ASTModel:
    local_dir = local_ast_dir(checkpoint)
    if local_dir.is_dir():
        try:
            return ASTModel.from_pretrained(local_dir, local_files_only=True)
        except Exception:
            logger.warning("Copia local inutilizable en %s; se redescarga.", local_dir)

    logger.info("Descargando backbone AST '%s' desde HuggingFace...", checkpoint)
    token = settings.HF_TOKEN.get_secret_value() if settings.HF_TOKEN else None
    model = ASTModel.from_pretrained(checkpoint, token=token)
    local_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(local_dir)
    logger.info("Backbone AST guardado en %s", local_dir)
    return model


class ASTBackbone(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        n_frames: int | None = None,
        time_stride: int = 5,
        checkpoint: str = AST_CHECKPOINT,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.model = load_ast_model(checkpoint)
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
        # stride original del checkpoint (10): hay que leerlo antes de pisar config.time_stride
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
