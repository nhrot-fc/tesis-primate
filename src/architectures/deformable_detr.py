"""Deformable DETR (versión de referencia en PyTorch puro para Etapa 0).

Estructura calcada de Zhu & Sato Fig. 2: multiscale features -> encoder ->
decoder con object queries fijas -> head (caja + clase). La caja se predice
como (time_center, duration, freq_center, bandwidth) = (cx, cy, w, h),
exactamente los 4 parámetros que el paper enumera.
"""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Cada query predice K offsets de muestreo sobre el mapa de features y
    combina esos puntos con pesos aprendidos. Aproxima Deformable-DETR [Zhu2021].
    """

    def __init__(self, dim: int = 256, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        self.n_heads, self.n_points = n_heads, n_points
        self.offsets = nn.Linear(dim, n_heads * n_points * 2)
        self.weights = nn.Linear(dim, n_heads * n_points)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.head_dim = dim // n_heads

    def forward(self, query, ref_points, value_map):
        # query: (B, Q, C) | ref_points: (B, Q, 2) en [0,1] | value_map: (B,C,H,W)
        B, Q, C = query.shape
        _, _, Hf, Wf = value_map.shape

        offs = self.offsets(query).view(B, Q, self.n_heads, self.n_points, 2)
        attn = self.weights(query).view(B, Q, self.n_heads, self.n_points)
        attn = F.softmax(attn, dim=-1)  # (B,Q,h,P)

        # puntos de muestreo = referencia + offset, normalizados a [-1,1]
        ref = ref_points[:, :, None, None, :]  # (B,Q,1,1,2)
        wh = torch.tensor([Wf, Hf], device=query.device, dtype=query.dtype)
        sample = ref + offs / wh  # (B,Q,h,P,2) en [0,1]
        sample = 2 * sample - 1  # -> [-1,1]

        # value proyectado, separado por cabezas: (B*h, hd, Hf, Wf)
        val = self.value(value_map.flatten(2).transpose(1, 2))  # (B, HW, C)
        val = val.transpose(1, 2).reshape(B, self.n_heads, self.head_dim, Hf, Wf)
        val = val.reshape(B * self.n_heads, self.head_dim, Hf, Wf)

        # grid por cabeza: (B*h, Q, P, 2)
        grid = sample.permute(0, 2, 1, 3, 4).reshape(B * self.n_heads, Q, self.n_points, 2)

        sampled = F.grid_sample(
            val, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )  # (B*h, hd, Q, P)
        sampled = sampled.reshape(B, self.n_heads, self.head_dim, Q, self.n_points)

        # pesos de atención alineados a (B,h,1,Q,P) y suma sobre P
        attn = attn.permute(0, 2, 1, 3).unsqueeze(2)  # (B,h,1,Q,P)
        out = (sampled * attn).sum(-1)  # (B,h,hd,Q)
        out = out.permute(0, 3, 1, 2).reshape(B, Q, C)  # (B,Q,C)
        return self.out(out)


class DeformableDecoderLayer(nn.Module):
    def __init__(self, dim=256, n_heads=8, n_points=4, ffn=1024):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.cross_attn = DeformableAttention(dim, n_heads, n_points)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.ReLU(True), nn.Linear(ffn, dim))
        self.n1, self.n2, self.n3 = (nn.LayerNorm(dim) for _ in range(3))

    def forward(self, q, ref_points, value_map):
        q = self.n1(q + self.self_attn(q, q, q)[0])
        q = self.n2(q + self.cross_attn(q, ref_points, value_map))
        q = self.n3(q + self.ffn(q))
        return q


class DeformableDETR(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        n_queries: int = 100,  # Zhu&Sato: 100 queries (máx ~35 eventos/clip)
        n_classes: int = 1,  # Etapa 0: una sola clase "call"
        n_decoder_layers: int = 6,
    ):
        super().__init__()
        self.n_queries = n_queries
        # queries aprendidas + su punto de referencia inicial
        self.query_embed = nn.Embedding(n_queries, dim)
        self.ref_point_head = nn.Linear(dim, 2)  # ref (cx,cy) por query

        self.layers = nn.ModuleList(DeformableDecoderLayer(dim) for _ in range(n_decoder_layers))
        # cabezas: +1 clase para "no-objeto" (fondo)
        self.class_head = nn.Linear(dim, n_classes + 1)
        self.bbox_head = _mlp(dim, dim, 4)  # (cx,cy,w,h)

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        # fusionamos la pirámide en un mapa (suma tras alinear a la escala mayor)
        base = features[0]
        value_map = base
        for f in features[1:]:
            value_map = value_map + F.interpolate(
                f, size=base.shape[-2:], mode="bilinear", align_corners=False
            )

        B = value_map.shape[0]
        q = self.query_embed.weight[None].expand(B, -1, -1)  # (B,Q,C)
        ref = self.ref_point_head(q).sigmoid()  # (B,Q,2) en [0,1]

        for layer in self.layers:
            q = layer(q, ref, value_map)

        # caja = offset relativo a la referencia (Deformable-DETR box refinement):
        # cx,cy arrancan en su punto de referencia; el MLP solo aprende la
        # corrección fina + el tamaño. Ancla la salida al "dónde" y evita las
        # cajas plantilla que colapsan cuando la caja se predice de q en absoluto.
        delta = self.bbox_head(q)  # (B,Q,4) offsets crudos en espacio logit
        cxcy = delta[..., :2] + _inv_sigmoid(ref)  # ancla a la referencia
        wh = delta[..., 2:]
        pred_boxes = torch.cat([cxcy, wh], dim=-1).sigmoid()  # (B,Q,4) cxcywh en [0,1]

        return {
            "pred_logits": self.class_head(q),  # (B,Q,n_classes+1)
            "pred_boxes": pred_boxes,
        }


class ASTDeformableDETR(nn.Module):
    """Modelo completo: backbone AST -> pirámide -> Deformable DETR."""

    def __init__(self, dim=256, n_queries=100, n_classes=1):
        super().__init__()
        from architectures.backbone import ASTBackbone, MultiScalePyramid

        self.backbone = ASTBackbone(embed_dim=dim)
        self.pyramid = MultiScalePyramid(dim)
        self.detr = DeformableDETR(dim, n_queries, n_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        return self.detr(self.pyramid(features))


class Detections(NamedTuple):
    boxes: torch.Tensor  # (K, 4) cxcywh normalizado
    scores: torch.Tensor  # (K,)
    labels: torch.Tensor  # (K,) id de clase en `domain.species.LabelSet`


def predict_scores(outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Score y clase por query -> (B, Q), (B, Q).

    Probabilidad de la clase más probable, descartando el último canal (el
    "no-objeto" de `SetCriterion`). Las clases van en 0..C-1, los mismos ids que
    `domain.species.LabelSet`.

    No usar `1 - p(no-objeto)` como score: una query indecisa (softmax uniforme)
    da 1 - 1/(C+1), que crece con el número de clases (0.83 con C=5, 0.98 con
    C=41), pasa cualquier umbral y dispara un recall falso.
    """
    prob = outputs["pred_logits"].softmax(-1)
    scores, labels = prob[..., :-1].max(-1)
    return scores, labels


def postprocess(outputs: dict[str, torch.Tensor], score_threshold: float = 0.5) -> list[Detections]:
    """Detecciones por imagen sobre el umbral, ordenadas por score descendente."""
    scores, labels = predict_scores(outputs)
    detections = []
    for index in range(scores.shape[0]):
        keep = scores[index] > score_threshold
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
