import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from core.config import P, settings

logger = logging.getLogger(__name__)

# Checkpoint autosupervisado (data2vec sobre AudioSet-2M) del repo oficial de EAT.
EAT_CHECKPOINT = "worstchan/EAT-base_epoch30_pretrain"

# Geometría del preentrenamiento: ViT-B/16 sobre mel de 128 bandas. El paso en frecuencia
# se queda en 16 (parches sin solape); el temporal es el que se barre, como en el AST.
PATCH_SIZE = 16
FREQ_STRIDE = 16
N_MELS = 128
EMBED_DIM = 768
DEPTH = 12
N_HEADS = 12
MLP_RATIO = 4.0
NORM_EPS = 1e-6


def local_eat_dir(checkpoint: str = EAT_CHECKPOINT) -> Path:
    return settings.checkpoints_dir / "hf" / checkpoint.replace("/", "__")


def load_eat_weights(checkpoint: str = EAT_CHECKPOINT) -> dict[str, Tensor]:
    """Pesos del EAT preentrenado, con copia local como caché (igual que el AST).

    El repo publica un `safetensors` plano --un ViT-B/16 con nombres de timm--, así que se
    lee directo en vez de pasar por `trust_remote_code`: `EATEncoder` reproduce ese módulo
    y las claves coinciden una a una.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = local_eat_dir(checkpoint) / "model.safetensors"
    if not path.is_file():
        logger.info("Descargando backbone EAT '%s' desde HuggingFace...", checkpoint)
        token = settings.HF_TOKEN.get_secret_value() if settings.HF_TOKEN else None
        hf_hub_download(checkpoint, "model.safetensors", local_dir=path.parent, token=token)
        logger.info("Backbone EAT guardado en %s", path)

    # `fixed_positional_encoder.positions` es sincos determinista sobre la rejilla del
    # preentrenamiento (768 x 8 parches); acá se regenera para la rejilla propia.
    return {
        key.removeprefix("model."): value
        for key, value in load_file(path).items()
        if "fixed_positional_encoder" not in key
    }


def sincos_1d(dim: int, positions: Tensor) -> Tensor:
    omega = 1.0 / 10000 ** (torch.arange(dim // 2, dtype=torch.float32) / (dim / 2))
    angles = positions.reshape(-1, 1) * omega
    return torch.cat([angles.sin(), angles.cos()], dim=1)


def sincos_2d(dim: int, rows: Tensor, cols: Tensor) -> Tensor:
    """Codificación posicional 2D fija del EAT, en orden por filas.

    La primera mitad de los canales codifica la columna y la segunda la fila: ése es el
    orden con el que se preentrenó, y cambiarlo invalida los pesos.
    """
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
    return torch.cat([sincos_1d(dim // 2, col_grid), sincos_1d(dim // 2, row_grid)], dim=1)


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int, patch_size: int, stride: tuple[int, int]) -> None:
        super().__init__()
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    """Bloque del EAT: el `AltBlock` de data2vec con `layer_norm_first=False`.

    No es el pre-norm habitual de un ViT: la normalización va después de cada residual y
    la del MLP se aplica sobre la suma. Se reproduce tal cual porque los pesos
    preentrenados asumen ese orden.
    """

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float, eps: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(x)
        residual = x = self.norm1(x)
        return self.norm2(residual + self.mlp(x))


class EATEncoder(nn.Module):
    def __init__(self, time_stride: int = PATCH_SIZE) -> None:
        super().__init__()
        self.local_encoder = PatchEmbed(EMBED_DIM, PATCH_SIZE, (time_stride, FREQ_STRIDE))
        self.extra_tokens = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))  # el CLS del EAT
        self.blocks = nn.ModuleList(
            Block(EMBED_DIM, N_HEADS, MLP_RATIO, NORM_EPS) for _ in range(DEPTH)
        )
        self.pre_norm = nn.LayerNorm(EMBED_DIM, eps=NORM_EPS)

    def forward(self, x: Tensor, pos_embed: Tensor) -> Tensor:
        # La posición se suma a los parches y recién después entra el CLS, que no la lleva.
        tokens = self.local_encoder(x) + pos_embed
        tokens = torch.cat([self.extra_tokens.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        tokens = self.pre_norm(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


class EATBackbone(nn.Module):
    """EAT (ViT-B/16 autosupervisado sobre AudioSet) como extractor de parches.

    Devuelve los tokens en el mismo orden que `ASTBackbone` --frecuencia primero--, así
    los dos entran igual a `DetectionHead`.
    """

    def __init__(
        self,
        n_frames: int | None = None,
        time_stride: int = PATCH_SIZE,
        checkpoint: str = EAT_CHECKPOINT,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.n_frames = n_frames if n_frames is not None else P.n_frames
        self.time_stride = time_stride
        self.freeze = freeze
        self.time_out = (self.n_frames - PATCH_SIZE) // time_stride + 1
        self.freq_out = (N_MELS - PATCH_SIZE) // FREQ_STRIDE + 1

        self.model = EATEncoder(time_stride=time_stride)
        self.model.load_state_dict(load_eat_weights(checkpoint))
        self.register_buffer("pos_embed", self._pos_embed(), persistent=False)

        if freeze:
            for name, param in self.model.named_parameters():
                param.requires_grad_(name.startswith("local_encoder.proj"))

        logger.info(
            "EAT: %d x %d tokens | time_stride=%d | %s",
            self.freq_out,
            self.time_out,
            time_stride,
            "congelado" if freeze else "fine-tune",
        )

    @property
    def hidden_size(self) -> int:
        return EMBED_DIM

    @property
    def n_mels(self) -> int:
        return N_MELS

    def _pos_embed(self) -> Tensor:
        """Sincos 2D sobre la rejilla de parches propia, en unidades de parche.

        El preentrenamiento usó paso 16 en los dos ejes, así que la posición de un token es
        su desplazamiento medido en parches: con paso `s` el token `t` arranca en `t*s/16`.
        Al ser una función continua no hay nada que interpolar, y alargar el clip es pedir
        más filas de la rejilla --que es cómo el paper salva la diferencia de duración
        entre AudioSet y sus grabaciones--.
        """
        rows = torch.arange(self.time_out, dtype=torch.float32) * (self.time_stride / PATCH_SIZE)
        cols = torch.arange(self.freq_out, dtype=torch.float32) * (FREQ_STRIDE / PATCH_SIZE)
        return sincos_2d(EMBED_DIM, rows, cols)[None]  # (1, time_out*freq_out, C)

    def train(self, mode: bool = True) -> "EATBackbone":
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 1, n_mels, n_frames). El EAT parchea sobre (tiempo, frecuencia), al revés
        # que el AST, así que la entrada va transpuesta y los tokens vuelven reordenados.
        if x.shape[-2:] != (N_MELS, self.n_frames):
            raise ValueError(
                f"el EAT se armó para mel de ({N_MELS}, {self.n_frames}) y llegó "
                f"{tuple(x.shape[-2:])}; la codificación posicional es de esa rejilla."
            )
        tokens = self.model(x.transpose(2, 3), self.pos_embed)[:, 1:]  # sin CLS
        return tokens.unflatten(1, (self.time_out, self.freq_out)).transpose(1, 2).flatten(1, 2)
