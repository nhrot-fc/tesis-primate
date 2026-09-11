import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from core.config import SCORE_THRESHOLD
from models.backbone import N_LEVELS, TIME_STRIDE
from models.base import Detector
from models.criterion import Outputs
from models.frontend import build_frontend
from utils.boxes import Detections

PRIOR_PROB = 0.01
FRONTEND = "pcen"
DIM = 128
N_QUERIES = 100
N_DECODER_LAYERS = 6
N_HEADS = 8
N_POINTS = 4
FFN = 1024
DROPOUT = 0.1


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
        dim: int,
        n_heads: int = N_HEADS,
        n_points: int = N_POINTS,
        n_levels: int = N_LEVELS,
        dropout: float = DROPOUT,
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
        # Zhu et al. 2021: los offsets arrancan en una rejilla radial, no en ruido.
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

        # Los offsets se miden en fracciones de la caja de referencia, no del mapa.
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
        dim: int,
        n_heads: int = N_HEADS,
        n_points: int = N_POINTS,
        n_levels: int = N_LEVELS,
        ffn: int = FFN,
        dropout: float = DROPOUT,
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
        # La posición se resuma en cada capa para que las queries no se diluyan.
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
        dim: int,
        n_queries: int,
        n_classes: int,
        n_decoder_layers: int = N_DECODER_LAYERS,
        n_heads: int = N_HEADS,
        n_points: int = N_POINTS,
    ):
        super().__init__()
        self.n_queries = n_queries

        self.query_embed = nn.Embedding(n_queries, dim)
        self.query_pos = nn.Embedding(n_queries, dim)
        self.ref_point_head = nn.Linear(dim, 2)

        self.layers = nn.ModuleList(
            DeformableDecoderLayer(dim, n_heads, n_points, N_LEVELS)
            for _ in range(n_decoder_layers)
        )
        class_heads = [nn.Linear(dim, n_classes) for _ in range(n_decoder_layers)]
        for head in class_heads:
            constant_(head.bias, -math.log((1 - PRIOR_PROB) / PRIOR_PROB))
        self.class_heads = nn.ModuleList(class_heads)
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
            # Refina la caja anterior en espacio logit, sin gradiente entre capas.
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
        dim: int,
    ):
        super().__init__()
        from models.backbone import MultiScalePyramid

        self.freq_out, self.time_out = freq_out, time_out
        self.proj = nn.Linear(token_dim, dim)
        self.pyramid = MultiScalePyramid(dim)
        self.pyramid.check_input_size(freq_out, time_out)
        self.detr = detr

    def pyramid_features(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        features = self.proj(tokens.to(self.proj.weight.dtype))
        features = features.transpose(1, 2).unflatten(-1, (self.freq_out, self.time_out))
        return self.pyramid(features)

    def forward(self, tokens: torch.Tensor) -> Outputs:
        return self.detr(self.pyramid_features(tokens))


class ASTDeformableDETR(Detector):
    # Sin NMS: el matching húngaro ya es uno a uno
    nms_iou = None
    clip_grad = 0.1

    def __init__(
        self,
        n_classes: int,
        n_frames: int | None = None,
        time_stride: int = TIME_STRIDE,
        dim: int = DIM,
        n_queries: int = N_QUERIES,
        frontend: str = FRONTEND,
    ):
        super().__init__()
        from models.backbone import ASTBackbone
        from models.criterion import SetCriterion

        self.backbone = ASTBackbone(n_frames=n_frames, time_stride=time_stride)
        self.frontend = build_frontend(frontend, self.backbone.n_mels)
        # Media 0 y varianza 1 por lote, a la mitad: el rango con el que el AST se preentrenó.
        self.input_norm = nn.BatchNorm2d(1, affine=False)
        self.head = DetectionHead(
            DeformableDETR(dim, n_queries, n_classes),
            token_dim=self.backbone.hidden_size,
            freq_out=self.backbone.freq_out,
            time_out=self.backbone.time_out,
            dim=dim,
        )
        self.criterion = SetCriterion()

    def forward(
        self, mel: torch.Tensor, targets: list[dict[str, torch.Tensor]] | None = None
    ) -> Outputs | dict[str, torch.Tensor]:
        normalized = self.input_norm(self.frontend(mel)) / 2
        outputs = self.head(self.backbone(normalized))
        return outputs if targets is None else self.criterion(outputs, targets)

    @torch.no_grad()
    def detect(
        self, mel: torch.Tensor, score_threshold: float = SCORE_THRESHOLD
    ) -> list[Detections]:
        outputs: Outputs = self(mel)
        # Sigmoides independientes por clase: el score no mezcla "hay algo" con "qué es".
        scores, labels = outputs["pred_logits"].sigmoid().max(-1)
        detections = []
        for boxes, score, label in zip(outputs["pred_boxes"], scores, labels, strict=True):
            above = score >= score_threshold
            by_score = score[above].argsort(descending=True)
            detections.append(
                Detections(boxes[above][by_score], score[above][by_score], label[above][by_score])
            )
        return detections
