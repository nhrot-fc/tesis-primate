import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import ASTModel

from core.config import P, settings

logger = logging.getLogger(__name__)

AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
# Pirámide {4x, 2x, 1x, 1/2x} sobre el mapa de tokens del AST, como en ViTDet.
N_LEVELS = 4


def local_ast_dir(checkpoint: str = AST_CHECKPOINT) -> Path:
    return settings.hf_dir / checkpoint.replace("/", "__")


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
    # Se afina entero y a la misma tasa que la cabeza: congelarlo, darle un LR propio o
    # subirle el dropout fueron ablaciones que no separaron o empeoraron (runs/comparacion_nms).
    def __init__(
        self,
        n_frames: int | None = None,
        time_stride: int = 10,
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
        # El paso del checkpoint hay que leerlo antes de pisar `config.time_stride`.
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
    def __init__(self, dim: int = 256, num_groups: int = 8) -> None:
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
        # Con una dimensión impar el nivel 1/2x tira la última fila o columna y pasa a
        # cubrir menos extensión física que los otros; `grid_sample` normaliza a [-1,1]
        # sobre el mapa entero y los niveles quedan desalineados sin dar error.
        if height % 2 or width % 2:
            raise ValueError(
                f"la pirámide necesita dimensiones pares, no ({height}, {width}); ajustá "
                "`time_stride` para que `time_out` y `freq_out` lo sean."
            )

    def forward(self, features: Tensor) -> list[Tensor]:
        return [block(features) for block in self.blocks]
