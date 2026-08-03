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
    """AST pre-entrenado -> tokens `(B, freq_out*time_out, hidden_size)`.

    Devuelve los tokens crudos (sin proyectar). Con `freeze=True` sus pesos no se
    actualizan, pero eso **no** alcanza para precomputarlos: la salida sólo es constante
    entre épocas si todo lo que va antes también lo es. Hoy no lo es —el PCEN de
    `ASTDeformableDETR` entrena—, así que cachear estos tokens daría features viejas.
    El gradiente sí atraviesa el backbone congelado para llegar a ese PCEN.
    """

    def __init__(
        self,
        n_frames: int | None = None,
        time_stride: int = 5,
        checkpoint: str = AST_CHECKPOINT,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.model = load_ast_model(checkpoint)
        self.n_frames = n_frames if n_frames is not None else P.n_frames
        self.time_stride = time_stride
        self._interpolate_time_pos_embed(self.n_frames, time_stride)

        self.freeze = freeze
        if freeze:
            for param in self.model.parameters():
                param.requires_grad_(False)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def n_mels(self) -> int:
        """Bandas mel que espera el checkpoint. Fijo: el pos-embed sólo se re-interpola
        en el eje temporal, así que el eje de frecuencia queda atado a AudioSet."""
        return int(self.model.config.num_mel_bins)

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
        # Sin `no_grad` aunque esté congelado: `requires_grad_(False)` sobre los pesos ya
        # evita que se acumulen gradientes en ellos, y si nada aguas arriba requiere
        # gradiente autograd no construye grafo igual. Un `no_grad` acá cortaría la cadena
        # hacia lo que sí es entrenable antes del backbone (el PCEN de `ASTDeformableDETR`).
        input_values = x.squeeze(1).transpose(1, 2)
        return self.model(input_values=input_values).last_hidden_state[:, 2:]  # sin CLS+dist


class MultiScalePyramid(nn.Module):
    """Niveles de resolución a partir del único mapa que entrega el AST.

    Con `n_levels=3` son 2x, 1x y 1/2x. El nivel 4x que había antes costaba 12 288
    posiciones (48x256) que se proyectan enteras en cada capa del decoder para
    muestrear 4 puntos por cabeza, y no aporta información nueva: sale del mismo
    mapa por deconvolución. En Deformable-DETR los niveles vienen de etapas distintas
    del backbone, que no es el caso acá.
    """

    def __init__(self, dim: int = 256, n_levels: int = 3, num_groups: int = 8) -> None:
        super().__init__()
        if not 2 <= n_levels <= 4:
            raise ValueError(f"n_levels debe estar entre 2 y 4, no {n_levels}")

        blocks: list[nn.Module] = []
        if n_levels == 4:
            blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
                    nn.GroupNorm(num_groups, dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
                )
            )
        blocks.append(nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2))
        blocks.append(nn.Identity())
        if n_levels >= 3:
            blocks.append(nn.Conv2d(dim, dim, kernel_size=2, stride=2))

        self.blocks = nn.ModuleList(blocks)
        self.n_levels = len(blocks)
        self.downsamples = n_levels >= 3

    def check_input_size(self, height: int, width: int) -> None:
        """El nivel 1/2x es `Conv2d(kernel=2, stride=2)`: con una dimensión impar tira la
        última fila/columna, así que ese nivel cubre menos extensión física que los otros.
        `grid_sample` mapea [-1,1] al mapa entero, sin enterarse, y los niveles quedan
        desalineados entre sí sin dar error."""
        if self.downsamples and (height % 2 or width % 2):
            raise ValueError(
                f"la pirámide con downsampling necesita dimensiones pares, no ({height}, {width}); "
                "ajustá `time_stride` (o `n_levels=2`) para que `time_out` y `freq_out` lo sean."
            )

    def forward(self, features: Tensor) -> list[Tensor]:
        return [block(features) for block in self.blocks]
