import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from models.criterion import Outputs
from utils.boxes import Detections


def mlp(dim: int, hidden: int, out: int, layers: int = 3) -> nn.Sequential:
    stack: list[nn.Module] = []
    width = dim
    for _ in range(layers - 1):
        stack += [nn.Linear(width, hidden), nn.ReLU(inplace=True)]
        width = hidden
    stack += [nn.Linear(width, out)]
    return nn.Sequential(*stack)


def inverse_sigmoid(coordinates: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    coordinates = coordinates.clamp(min=0, max=1)
    return torch.log(coordinates.clamp(min=eps) / (1 - coordinates).clamp(min=eps))


class DeformableAttention(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        n_heads: int = 8,
        n_points: int = 4,
        n_levels: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads, self.n_points, self.n_levels = n_heads, n_points, n_levels
        self.offsets = nn.Linear(dim, n_heads * n_levels * n_points * 2)
        self.weights = nn.Linear(dim, n_heads * n_levels * n_points)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.head_dim = dim // n_heads
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        # Init de Zhu et al. 2021: los offsets arrancan en una rejilla radial --cada cabeza
        # mira en una dirección, cada punto a un radio mayor-- en vez de ruido, para no
        # gastar épocas aprendiendo dónde muestrear.
        constant_(self.offsets.weight.data, 0.0)
        angles = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        directions = torch.stack([angles.cos(), angles.sin()], -1)
        directions = directions / directions.abs().max(-1, keepdim=True)[0]
        directions = directions.view(self.n_heads, 1, 1, 2).repeat(
            1, self.n_levels, self.n_points, 1
        )
        for point in range(self.n_points):
            directions[:, :, point, :] *= point + 1
        with torch.no_grad():
            self.offsets.bias = nn.Parameter(directions.reshape(-1))

        constant_(self.weights.weight.data, 0.0)
        constant_(self.weights.bias.data, 0.0)

        xavier_uniform_(self.value.weight.data)
        constant_(self.value.bias.data, 0.0)
        xavier_uniform_(self.out.weight.data)
        constant_(self.out.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        ref_boxes: torch.Tensor,
        value_maps: list[torch.Tensor],
    ) -> torch.Tensor:
        # query: (B,Q,C) | ref_boxes: (B,Q,4) cxcywh en [0,1] | value_maps: (B,C,H,W) por nivel
        batch_size, n_queries, channels = query.shape
        n_levels = len(value_maps)
        assert self.n_levels == n_levels, f"esperaba {self.n_levels} niveles, llegaron {n_levels}"

        offsets = self.offsets(query).view(
            batch_size, n_queries, self.n_heads, n_levels, self.n_points, 2
        )
        weights = self.weights(query).view(
            batch_size, n_queries, self.n_heads, n_levels * self.n_points
        )
        weights = F.softmax(weights, dim=-1).view(
            batch_size, n_queries, self.n_heads, n_levels, self.n_points
        )

        # Box refinement (Zhu et al. 2021): los offsets se miden en fracciones de la caja de
        # referencia y no del mapa de features. Sin esto, una query que sigue una llamada de
        # 2 s muestrea la misma vecindad de 4 píxeles que una de 50 ms.
        centers = ref_boxes[:, :, None, None, None, :2]
        sizes = ref_boxes[:, :, None, None, None, 2:]
        sample_points = centers + offsets / self.n_points * sizes * 0.5
        sample_points = 2 * sample_points - 1  # -> [-1,1], convención align_corners=False

        aggregated = query.new_zeros(batch_size, self.n_heads, self.head_dim, n_queries)
        for level, value_map in enumerate(value_maps):
            height, width = value_map.shape[-2:]

            values = self.value(value_map.flatten(2).transpose(1, 2))
            values = values.transpose(1, 2).reshape(
                batch_size * self.n_heads, self.head_dim, height, width
            )
            sample_grid = sample_points[:, :, :, level].permute(0, 2, 1, 3, 4)
            sample_grid = sample_grid.reshape(
                batch_size * self.n_heads, n_queries, self.n_points, 2
            )

            sampled = F.grid_sample(
                values, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            sampled = sampled.reshape(
                batch_size, self.n_heads, self.head_dim, n_queries, self.n_points
            )
            level_weights = weights[:, :, :, level].permute(0, 2, 1, 3).unsqueeze(2)
            aggregated = aggregated + (sampled * level_weights).sum(-1)

        aggregated = aggregated.permute(0, 3, 1, 2).reshape(batch_size, n_queries, channels)
        return self.dropout(self.out(aggregated))


class DeformableDecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        n_heads: int = 8,
        n_points: int = 4,
        n_levels: int = 4,
        ffn: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = DeformableAttention(dim, n_heads, n_points, n_levels, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn, dim),
            nn.Dropout(dropout),
        )
        self.n1, self.n2, self.n3 = (nn.LayerNorm(dim) for _ in range(3))

    def forward(
        self,
        queries: torch.Tensor,
        query_pos: torch.Tensor,
        ref_boxes: torch.Tensor,
        value_maps: list[torch.Tensor],
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # La posición se resuma en cada capa: si no, la identidad de cada query se diluye
        # en los residuales y todas terminan mirando lo mismo.
        located = queries + query_pos
        attended = self.self_attn(
            located, located, queries, attn_mask=attn_mask, need_weights=False
        )
        queries = self.n1(queries + attended[0])
        queries = self.n2(queries + self.cross_attn(queries + query_pos, ref_boxes, value_maps))
        return self.n3(queries + self.ffn(queries))


class DeformableDETR(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        n_queries: int = 50,
        n_classes: int = 1,
        n_decoder_layers: int = 6,
        n_heads: int = 8,
        n_points: int = 4,
        n_levels: int = 3,
    ):
        super().__init__()
        self.n_queries = n_queries

        self.query_embed = nn.Embedding(n_queries, dim)
        self.query_pos = nn.Embedding(n_queries, dim)
        self.ref_point_head = nn.Linear(dim, 2)

        self.layers = nn.ModuleList(
            DeformableDecoderLayer(dim, n_heads, n_points, n_levels)
            for _ in range(n_decoder_layers)
        )
        # Cabezas por capa, para el refinamiento iterativo de caja. +1 clase: el no-objeto.
        self.class_heads = nn.ModuleList(
            nn.Linear(dim, n_classes + 1) for _ in range(n_decoder_layers)
        )
        self.bbox_heads = nn.ModuleList(mlp(dim, dim, 4) for _ in range(n_decoder_layers))

    def forward(self, features: list[torch.Tensor]) -> Outputs:
        batch_size = features[0].shape[0]
        queries = self.query_embed.weight[None].expand(batch_size, -1, -1)
        query_pos = self.query_pos.weight[None].expand(batch_size, -1, -1)
        centers = self.ref_point_head(query_pos).sigmoid()
        reference_boxes = torch.cat([centers, torch.full_like(centers, 0.1)], dim=-1)

        per_layer: list[dict[str, torch.Tensor]] = []
        for index, layer in enumerate(self.layers):
            queries = layer(queries, query_pos, reference_boxes, features)
            # Cada capa predice un delta en espacio logit sobre la caja de la anterior, y
            # el `detach` corta el gradiente entre capas para estabilizar el refinamiento.
            box_delta = self.bbox_heads[index](queries)
            reference_boxes = (box_delta + inverse_sigmoid(reference_boxes)).sigmoid()
            per_layer.append(
                {"pred_logits": self.class_heads[index](queries), "pred_boxes": reference_boxes}
            )
            reference_boxes = reference_boxes.detach()

        outputs: Outputs = dict(per_layer[-1])
        outputs["aux_outputs"] = per_layer[:-1]
        return outputs


class DetectionHead(nn.Module):
    def __init__(
        self,
        detr: nn.Module,
        token_dim: int,
        freq_out: int,
        time_out: int,
        dim: int = 256,
        n_levels: int = 3,
    ):
        super().__init__()
        from models.backbone import MultiScalePyramid

        self.freq_out, self.time_out = freq_out, time_out
        self.proj = nn.Linear(token_dim, dim)
        self.pyramid = MultiScalePyramid(dim, n_levels=n_levels)
        self.pyramid.check_input_size(freq_out, time_out)
        self.detr = detr

    def pyramid_features(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        features = self.proj(tokens.to(self.proj.weight.dtype))  # las cacheadas llegan en fp16
        features = features.transpose(1, 2).unflatten(-1, (self.freq_out, self.time_out))
        return self.pyramid(features)

    def forward(self, tokens: torch.Tensor) -> Outputs:
        return self.detr(self.pyramid_features(tokens))


class ASTDeformableDETR(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        n_queries: int = 50,
        n_classes: int = 1,
        freeze: bool = True,
        n_frames: int | None = None,
        time_stride: int = 2,
        n_levels: int = 3,
        n_mels: int = 128,
        frontend: str = "pcen",
    ):
        super().__init__()
        from models.backbone import ASTBackbone
        from models.criterion import SetCriterion
        from models.pcen import LogMelFrontend, TrainablePCEN

        self.backbone = ASTBackbone(n_frames=n_frames, time_stride=time_stride, freeze=freeze)
        if n_mels != self.backbone.n_mels:
            raise ValueError(
                f"n_mels={n_mels} no coincide con las {self.backbone.n_mels} bandas del "
                "checkpoint del AST; el pos-embed sólo se re-interpola en el eje temporal."
            )
        if frontend not in ("pcen", "logmel"):
            raise ValueError(f"frontend desconocido: {frontend!r}; hay 'pcen' y 'logmel'")

        self.frontend = frontend
        # El atributo se llama `pcen` con las dos ramas: es la clave con la que los
        # checkpoints ya guardados nombran esta capa.
        self.pcen = TrainablePCEN(n_mels=n_mels) if frontend == "pcen" else LogMelFrontend()
        self.pcen_norm = nn.BatchNorm2d(1, affine=False)

        self.head = DetectionHead(
            DeformableDETR(dim, n_queries, n_classes, n_levels=n_levels),
            token_dim=self.backbone.hidden_size,
            freq_out=self.backbone.freq_out,
            time_out=self.backbone.time_out,
            dim=dim,
            n_levels=n_levels,
        )
        self.criterion = SetCriterion(n_classes=n_classes)

    def forward(
        self, mel: torch.Tensor, targets: list[dict[str, torch.Tensor]] | None = None
    ) -> Outputs | dict[str, torch.Tensor]:
        compressed = self.pcen_norm(self.pcen(mel)) / 2
        outputs = self.head(self.backbone(compressed))
        return outputs if targets is None else self.criterion(outputs, targets)


def detections_above_threshold(
    boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, score_threshold: float
) -> list[Detections]:
    detections = []
    for index in range(scores.shape[0]):
        above = scores[index] >= score_threshold  # `>=`, igual que en `evaluate`
        kept = scores[index][above]
        by_score = kept.argsort(descending=True)
        detections.append(
            Detections(
                boxes=boxes[index][above][by_score],
                scores=kept[by_score],
                labels=labels[index][above][by_score],
            )
        )
    return detections


def postprocess(outputs: Outputs, score_threshold: float = 0.5) -> list[Detections]:
    # No sirve `1 - p(no-objeto)` como score: una query indecisa da 1 - 1/(C+1), que crece
    # con el número de clases y pasa cualquier umbral.
    scores, labels = outputs["pred_logits"].softmax(-1)[..., :-1].max(-1)
    return detections_above_threshold(outputs["pred_boxes"], scores, labels, score_threshold)


@torch.no_grad()
def detect(
    model: nn.Module, images: torch.Tensor, score_threshold: float = 0.5
) -> list[Detections]:
    return postprocess(model(images), score_threshold)  # type: ignore[arg-type]
