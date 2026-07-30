"""Deformable DETR (versión de referencia en PyTorch puro para Etapa 0).

Estructura calcada de Zhu & Sato Fig. 2: multiscale features -> decoder con
object queries (fijas, o desde un score de objectness si `two_stage=True`) ->
refinamiento iterativo de caja + clase por capa. La caja se predice como
(cx, cy, w, h) = (time_center, freq_center, duration, bandwidth), exactamente
los 4 parámetros que el paper enumera.
"""

import math
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_


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
        temporal_bias_scale: float = 1.0,
    ):
        super().__init__()
        self.n_heads, self.n_points, self.n_levels = n_heads, n_points, n_levels
        self.offsets = nn.Linear(dim, n_heads * n_levels * n_points * 2)
        self.weights = nn.Linear(dim, n_heads * n_levels * n_points)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.head_dim = dim // n_heads
        self._reset_parameters(temporal_bias_scale)

    def _reset_parameters(self, temporal_bias_scale: float) -> None:
        """Init de Zhu et al. 2021 (`_reset_parameters` del repo de referencia): sin esto
        el modelo gasta decenas de épocas aprendiendo a muestrear en lugares sensatos.

        `offsets` arranca con peso cero y bias en una rejilla radial (cada cabeza mira
        en una dirección distinta, cada punto un radio mayor), así que en el primer
        forward ya muestrea posiciones útiles alrededor de la referencia en vez de ruido.
        `temporal_bias_scale` != 1.0 estira esa rejilla en el eje temporal (índice 0,
        ver convención de ejes en `DeformableDETR`) -- ablation aparte, no combinar con
        el resto de cambios en la misma corrida.
        """
        constant_(self.offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = grid / grid.abs().max(-1, keepdim=True)[0]
        grid = grid.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid[:, :, i, :] *= i + 1
        if temporal_bias_scale != 1.0:
            grid[..., 0] *= temporal_bias_scale
        with torch.no_grad():
            self.offsets.bias = nn.Parameter(grid.reshape(-1))

        constant_(self.weights.weight.data, 0.0)
        constant_(self.weights.bias.data, 0.0)

        xavier_uniform_(self.value.weight.data)
        constant_(self.value.bias.data, 0.0)
        xavier_uniform_(self.out.weight.data)
        constant_(self.out.bias.data, 0.0)

    def forward(
        self, query: torch.Tensor, ref_points: torch.Tensor, value_maps: list[torch.Tensor]
    ) -> torch.Tensor:
        # query: (B,Q,C) | ref_points: (B,Q,2) en [0,1] | value_maps: lista de (B,C,H,W),
        # una por nivel de la pirámide (mismo largo que `self.n_levels`)
        B, Q, C = query.shape
        L = len(value_maps)

        offs = self.offsets(query).view(B, Q, self.n_heads, L, self.n_points, 2)
        attn = self.weights(query).view(B, Q, self.n_heads, L * self.n_points)
        attn = F.softmax(attn, dim=-1).view(B, Q, self.n_heads, L, self.n_points)

        ref = ref_points[:, :, None, None, :]  # (B,Q,1,1,2)
        out = query.new_zeros(B, self.n_heads, self.head_dim, Q)
        for level, value_map in enumerate(value_maps):
            _, _, Hf, Wf = value_map.shape
            wh = torch.tensor([Wf, Hf], device=query.device, dtype=query.dtype)

            # puntos de muestreo = referencia + offset, normalizados a [-1,1]. El +0.5
            # antes de dividir por `wh` corrige la convención de `grid_sample`: sin él,
            # `ref`/`sample` en [0,1] se interpretan como si el píxel i estuviera en i/W
            # en vez de (i+0.5)/W, un sesgo sistemático de medio píxel.
            sample = ref + offs[:, :, :, level] / wh  # (B,Q,h,P,2) en [0,1]
            sample = 2 * (sample * wh + 0.5) / wh - 1  # -> [-1,1]

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
        self, q: torch.Tensor, ref_points: torch.Tensor, value_maps: list[torch.Tensor]
    ) -> torch.Tensor:
        q = self.n1(q + self.self_attn(q, q, q)[0])
        q = self.n2(q + self.cross_attn(q, ref_points, value_maps))
        q = self.n3(q + self.ffn(q))
        return q


class DeformableDETR(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        n_queries: int = 50,  # ~10 eventos/clip max: 100 (Zhu&Sato) sobraba para este dominio
        n_classes: int = 1,  # Etapa 0: una sola clase "call"
        n_decoder_layers: int = 6,
        n_heads: int = 8,
        n_points: int = 4,
        n_levels: int = 4,  # niveles que entrega `MultiScalePyramid`
        two_stage: bool = False,
    ):
        super().__init__()
        self.n_queries = n_queries
        self.two_stage = two_stage

        # queries aprendidas; su punto de referencia inicial sale de `ref_point_head`
        # (two_stage=False, DETR vanilla) o de un score de objectness sobre la pirámide
        # (two_stage=True, two-stage "ligero": solo la posición, no el contenido).
        self.query_embed = nn.Embedding(n_queries, dim)
        self.ref_point_head = None if two_stage else nn.Linear(dim, 2)
        self.enc_score = nn.Linear(dim, 1) if two_stage else None

        self.layers = nn.ModuleList(
            DeformableDecoderLayer(dim, n_heads, n_points, n_levels)
            for _ in range(n_decoder_layers)
        )
        # cabezas por capa (refinamiento iterativo de caja): +1 clase para "no-objeto"
        self.class_heads = nn.ModuleList(
            nn.Linear(dim, n_classes + 1) for _ in range(n_decoder_layers)
        )
        self.bbox_heads = nn.ModuleList(_mlp(dim, dim, 4) for _ in range(n_decoder_layers))

    def _initial_ref(
        self, query: torch.Tensor, features: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Punto de referencia (cx,cy) inicial por query, en [0,1], y opcionalmente un
        gate de contenido (`None` si no hay two-stage)."""
        if not self.two_stage:
            assert self.ref_point_head is not None
            return self.ref_point_head(query).sigmoid(), None

        # two-stage ligero: objectness sobre el nivel más fino (mejor resolución temporal,
        # el eje que más nos exige) en vez de una referencia aprendida fija por query.
        # `top_index` (usado para las coordenadas) no es diferenciable, así que sin más
        # `enc_score` nunca recibiría gradiente; `top_scores` sí lo es y gatea el
        # contenido de la query, lo que lo mantiene entrenable.
        assert self.enc_score is not None
        base = features[0]
        _, _, Hf, Wf = base.shape
        objectness = self.enc_score(base.flatten(2).transpose(1, 2)).squeeze(-1)  # (B, HW)
        top_scores, top_index = objectness.topk(self.n_queries, dim=1)  # (B, Q) cada uno
        cx = (top_index % Wf + 0.5) / Wf
        cy = (top_index // Wf + 0.5) / Hf
        ref = torch.stack([cx, cy], dim=-1)
        gate = top_scores.sigmoid().unsqueeze(-1)  # (B,Q,1)
        return ref, gate

    def forward(self, features: list[torch.Tensor]) -> dict[str, Any]:
        B = features[0].shape[0]
        q = self.query_embed.weight[None].expand(B, -1, -1)  # (B,Q,C)
        ref, gate = self._initial_ref(q, features)  # (B,Q,2) en [0,1]
        if gate is not None:
            q = q * gate

        # caja de referencia inicial: cx,cy desde `ref`, w,h arrancan en un tamaño
        # moderado y las capas los refinan iterativamente (Deformable-DETR box
        # refinement): cada capa predice un delta sobre la caja de la anterior en vez
        # de recalcularla desde cero, y el decoder puede converger sobre el evento en
        # vez de quedarse con la referencia inicial fija.
        ref_box = torch.cat([ref, torch.full_like(ref, 0.1)], dim=-1)  # (B,Q,4) cxcywh

        aux: list[dict[str, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            q = layer(q, ref_box[..., :2], features)
            delta = self.bbox_heads[i](q)  # (B,Q,4) offsets crudos en espacio logit
            ref_box = (delta + _inv_sigmoid(ref_box)).sigmoid()
            aux.append({"pred_logits": self.class_heads[i](q), "pred_boxes": ref_box})
            ref_box = (
                ref_box.detach()
            )  # estabilidad: la siguiente capa no retropropaga por esta caja

        out: dict[str, Any] = dict(aux[-1])
        out["aux_outputs"] = aux[:-1]
        return out


class ASTDeformableDETR(nn.Module):
    """Modelo completo: backbone AST -> pirámide -> Deformable DETR."""

    def __init__(
        self,
        dim: int = 256,
        n_queries: int = 50,
        n_classes: int = 1,
        freeze: bool = True,
        n_frames: int | None = None,
    ):
        super().__init__()
        from architectures.backbone import ASTBackbone, MultiScalePyramid

        self.backbone = ASTBackbone(embed_dim=dim, n_frames=n_frames, freeze=freeze)
        self.pyramid = MultiScalePyramid(dim)
        self.detr = DeformableDETR(dim, n_queries, n_classes)

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
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
