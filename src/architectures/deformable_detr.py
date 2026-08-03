"""Deformable DETR (versión de referencia en PyTorch puro para Etapa 0).

Estructura calcada de Zhu & Sato Fig. 2: multiscale features -> decoder con
object queries fijas -> refinamiento iterativo de caja + clase por capa. La
caja se predice como (cx, cy, w, h) = (time_center, freq_center, duration,
bandwidth).
"""

import math
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

Outputs = dict[str, Any]
# pred_logits, pred_boxes: Tensor; aux_outputs: list[dict[str, Tensor]]


def _mlp(dim: int, hidden: int, out: int, layers: int = 3) -> nn.Sequential:
    seq: list[nn.Module] = []
    d = dim
    for _ in range(layers - 1):
        seq += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
        d = hidden
    seq += [nn.Linear(d, out)]
    return nn.Sequential(*seq)


def _inv_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inversa de sigmoid: lleva coords en [0,1] al espacio logit para sumar offsets."""
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


class DeformableAttention(nn.Module):
    """Cada query predice K offsets de muestreo por nivel de la pirámide y
    combina esos puntos con pesos aprendidos. Aproxima Deformable-DETR [Zhu2021].
    """

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
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Init de Zhu et al. 2021: `offsets` arranca en una rejilla radial (cada
        cabeza mira en una dirección, cada punto a un radio mayor) en vez de ruido,
        para no gastar épocas aprendiendo dónde muestrear.
        """
        constant_(self.offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = grid / grid.abs().max(-1, keepdim=True)[0]
        grid = grid.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.offsets.bias = nn.Parameter(grid.reshape(-1))

        constant_(self.weights.weight.data, 0.0)
        constant_(self.weights.bias.data, 0.0)

        xavier_uniform_(self.value.weight.data)
        constant_(self.value.bias.data, 0.0)
        xavier_uniform_(self.out.weight.data)
        constant_(self.out.bias.data, 0.0)

    def forward(
        self, query: torch.Tensor, ref_boxes: torch.Tensor, value_maps: list[torch.Tensor]
    ) -> torch.Tensor:
        # query: (B,Q,C) | ref_boxes: (B,Q,4) cxcywh en [0,1] | value_maps: lista de
        # (B,C,H,W), una por nivel de la pirámide (mismo largo que `self.n_levels`)
        B, Q, C = query.shape
        L = len(value_maps)
        assert self.n_levels == L, f"esperaba {self.n_levels} niveles, llegaron {L}"

        offs = self.offsets(query).view(B, Q, self.n_heads, L, self.n_points, 2)
        attn = self.weights(query).view(B, Q, self.n_heads, L * self.n_points)
        attn = F.softmax(attn, dim=-1).view(B, Q, self.n_heads, L, self.n_points)

        # Zhu et al. 2021, ec. de box refinement: los offsets se miden en fracciones de
        # la caja de referencia, no del mapa de features. Sin esto una query que sigue
        # una llamada de 2 s muestrea la misma vecindad de 4 píxeles que una de 50 ms.
        ref_xy = ref_boxes[:, :, None, None, None, :2]  # (B,Q,1,1,1,2), contra offs (B,Q,h,L,P,2)
        ref_wh = ref_boxes[:, :, None, None, None, 2:]
        sample = ref_xy + offs / self.n_points * ref_wh * 0.5  # (B,Q,h,L,P,2) en [0,1]
        sample = 2 * sample - 1  # -> [-1,1], convención align_corners=False

        out = query.new_zeros(B, self.n_heads, self.head_dim, Q)
        for level, value_map in enumerate(value_maps):
            _, _, Hf, Wf = value_map.shape
            level_sample = sample[:, :, :, level]  # (B,Q,h,P,2)

            # value proyectado, separado por cabezas: (B*h, hd, Hf, Wf)
            val = self.value(value_map.flatten(2).transpose(1, 2))  # (B, HW, C)
            val = val.transpose(1, 2).reshape(B, self.n_heads, self.head_dim, Hf, Wf)
            val = val.reshape(B * self.n_heads, self.head_dim, Hf, Wf)

            # grid por cabeza: (B*h, Q, P, 2)
            grid = level_sample.permute(0, 2, 1, 3, 4).reshape(
                B * self.n_heads, Q, self.n_points, 2
            )

            sampled = F.grid_sample(
                val, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )  # (B*h, hd, Q, P)
            sampled = sampled.reshape(B, self.n_heads, self.head_dim, Q, self.n_points)

            # pesos de atención de este nivel, alineados a (B,h,1,Q,P) y sumados sobre P
            level_attn = attn[:, :, :, level].permute(0, 2, 1, 3).unsqueeze(2)  # (B,h,1,Q,P)
            out = out + (sampled * level_attn).sum(-1)

        out = out.permute(0, 3, 1, 2).reshape(B, Q, C)  # (B,Q,C)
        return self.dropout(self.out(out))


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
        q: torch.Tensor,
        query_pos: torch.Tensor,
        ref_boxes: torch.Tensor,
        value_maps: list[torch.Tensor],
    ) -> torch.Tensor:
        # `query_pos` se suma a query y key en cada capa (DETR): sin eso la identidad
        # de cada query se diluye en los residuales después de la primera capa y todas
        # tienden a mirar lo mismo.
        qk = q + query_pos
        q = self.n1(q + self.self_attn(qk, qk, q)[0])
        q = self.n2(q + self.cross_attn(q + query_pos, ref_boxes, value_maps))
        q = self.n3(q + self.ffn(q))
        return q


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

        self.query_embed = nn.Embedding(n_queries, dim)  # contenido inicial de la query
        self.query_pos = nn.Embedding(n_queries, dim)  # identidad, se resuma en cada capa
        self.ref_point_head = nn.Linear(dim, 2)

        self.layers = nn.ModuleList(
            DeformableDecoderLayer(dim, n_heads, n_points, n_levels)
            for _ in range(n_decoder_layers)
        )
        # cabezas por capa (refinamiento iterativo de caja): +1 clase para "no-objeto"
        self.class_heads = nn.ModuleList(
            nn.Linear(dim, n_classes + 1) for _ in range(n_decoder_layers)
        )
        self.bbox_heads = nn.ModuleList(_mlp(dim, dim, 4) for _ in range(n_decoder_layers))

    def forward(self, features: list[torch.Tensor]) -> Outputs:
        B = features[0].shape[0]
        q = self.query_embed.weight[None].expand(B, -1, -1)  # (B,Q,C)
        query_pos = self.query_pos.weight[None].expand(B, -1, -1)
        ref = self.ref_point_head(query_pos).sigmoid()  # (B,Q,2) en [0,1]

        # w,h arrancan en un tamaño moderado; cada capa predice un delta sobre la
        # caja de la anterior (Deformable-DETR box refinement) en vez de recalcularla.
        ref_box = torch.cat([ref, torch.full_like(ref, 0.1)], dim=-1)  # (B,Q,4) cxcywh

        aux: list[dict[str, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            q = layer(q, query_pos, ref_box, features)
            delta = self.bbox_heads[i](q)  # (B,Q,4) offsets crudos en espacio logit
            ref_box = (delta + _inv_sigmoid(ref_box)).sigmoid()
            aux.append({"pred_logits": self.class_heads[i](q), "pred_boxes": ref_box})
            ref_box = (
                ref_box.detach()
            )  # estabilidad: la siguiente capa no retropropaga por esta caja

        out: Outputs = dict(aux[-1])
        out["aux_outputs"] = aux[:-1]
        return out


class DetectionHead(nn.Module):
    """Parte entrenable: proyección de los tokens del AST -> pirámide -> Deformable DETR.

    Vive separada del backbone para poder entrenarla contra features precomputadas
    cuando el AST está congelado (ver `ASTDeformableDETR.encode`).
    """

    def __init__(
        self,
        token_dim: int,
        freq_out: int,
        time_out: int,
        dim: int = 256,
        n_queries: int = 50,
        n_classes: int = 1,
        n_levels: int = 3,
    ):
        super().__init__()
        from architectures.backbone import MultiScalePyramid

        self.freq_out, self.time_out = freq_out, time_out
        self.proj = nn.Linear(token_dim, dim)
        self.pyramid = MultiScalePyramid(dim, n_levels=n_levels)
        self.detr = DeformableDETR(dim, n_queries, n_classes, n_levels=self.pyramid.n_levels)

    def forward(self, tokens: torch.Tensor) -> Outputs:
        # tokens: (B, freq_out*time_out, token_dim); las cacheadas llegan en fp16
        features = self.proj(tokens.to(self.proj.weight.dtype))
        features = features.transpose(1, 2).unflatten(-1, (self.freq_out, self.time_out))
        return self.detr(self.pyramid(features))


class ASTDeformableDETR(nn.Module):
    """Modelo completo: normalización -> backbone AST -> pirámide -> Deformable DETR."""

    mel_mean: torch.Tensor
    mel_std: torch.Tensor

    def __init__(
        self,
        dim: int = 256,
        n_queries: int = 50,
        n_classes: int = 1,
        freeze: bool = True,
        n_frames: int | None = None,
        time_stride: int = 5,
        n_levels: int = 3,
        mel_mean: float = 0.0,
        mel_std: float = 1.0,
        n_mels: int = 128,
    ):
        super().__init__()
        from architectures.backbone import ASTBackbone
        from architectures.trainable_pcen import TrainablePCEN

        self.pcen = TrainablePCEN(n_mels=n_mels)
        self.backbone = ASTBackbone(n_frames=n_frames, time_stride=time_stride, freeze=freeze)
        self.head = DetectionHead(
            token_dim=self.backbone.hidden_size,
            freq_out=self.backbone.freq_out,
            time_out=self.backbone.time_out,
            dim=dim,
            n_queries=n_queries,
            n_classes=n_classes,
            n_levels=n_levels,
        )
        # Estadísticas del log-mel del split de train. Van como buffers para que viajen
        # dentro del checkpoint: así inferencia no puede normalizar distinto que train.
        self.register_buffer("mel_mean", torch.tensor(float(mel_mean)))
        self.register_buffer("mel_std", torch.tensor(float(mel_std)))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Log-mel crudo -> escala del pre-entrenamiento del AST.

        El `ASTFeatureExtractor` de HuggingFace normaliza con `(x - mean) / (2*std)`,
        o sea deja std ~= 0.5. Estandarizar a std=1 le mete al backbone congelado el
        doble de escala de la que vio en AudioSet.
        """
        return (x - self.mel_mean) / (2 * self.mel_std)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Espectrograma -> tokens del AST. Determinista si el backbone está congelado."""
        return self.backbone(self.normalize(x))

    def forward(self, x: torch.Tensor) -> Outputs:
        return self.head(self.encode(x))


class Detections(NamedTuple):
    boxes: torch.Tensor  # (K, 4) cxcywh normalizado
    scores: torch.Tensor  # (K,)
    labels: torch.Tensor  # (K,) id de clase en `domain.species.LabelSet`


def predict_scores(outputs: Outputs) -> tuple[torch.Tensor, torch.Tensor]:
    """Score y clase por query -> (B, Q), (B, Q).

    Probabilidad de la clase más probable, descartando el último canal ("no-objeto"
    de `SetCriterion`). No usar `1 - p(no-objeto)` como score: una query indecisa
    (softmax uniforme) da 1 - 1/(C+1), que crece con el número de clases y pasa
    cualquier umbral.
    """
    prob = outputs["pred_logits"].softmax(-1)
    scores, labels = prob[..., :-1].max(-1)
    return scores, labels


def postprocess(outputs: Outputs, score_threshold: float = 0.5) -> list[Detections]:
    """Detecciones por imagen sobre el umbral, ordenadas por score descendente."""
    scores, labels = predict_scores(outputs)
    detections = []
    for index in range(scores.shape[0]):
        keep = scores[index] >= score_threshold  # `>=`, igual que en `evaluate`
        kept_scores = scores[index][keep]
        order = kept_scores.argsort(descending=True)
        detections.append(
            Detections(
                boxes=outputs["pred_boxes"][index][keep][order],
                scores=kept_scores[order],
                labels=labels[index][keep][order],
            )
        )
    return detections
