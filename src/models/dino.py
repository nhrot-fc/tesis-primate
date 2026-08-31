"""DINO (Zhang et al., ICLR 2023) sobre el backbone EAT.

Es el enfoque que la literatura reporta para detección de cajas 2D en espectrogramas
(Zhu & Sato, DCASE 2025): un AST autosupervisado --EAT-- como backbone y DINO como
detector. Cuatro cosas lo separan del Deformable-DETR de `models.deformable_detr`:

1. **Encoder deformable** sobre la pirámide, antes del decodificador.
2. **Selección mixta de queries**: las anclas iniciales salen de las mejores propuestas
   del encoder (dos etapas) y el contenido de la query sigue siendo aprendido.
3. **Denoising contrastivo (CDN)**: las cajas verdaderas entran con ruido como queries
   extra; las poco ruidosas tienen que reconstruir su caja y las muy ruidosas, decir que
   ahí no hay nada. Es lo que acelera la convergencia del emparejamiento húngaro.
4. **Look forward twice**: la caja de una capa deja pasar gradiente a la de la anterior,
   en vez de cortarlo con `detach`.

Además la clasificación es focal loss sobre logits sigmoides, sin canal de "no-objeto".
"""

import math

import torch
from torch import Tensor, nn
from torchvision.ops import box_convert

from models.criterion import HungarianMatcher, SetCriterion, Target
from models.deformable_detr import (
    DeformableAttention,
    DeformableDecoderLayer,
    DetectionHead,
    LogMelFrontend,
    Outputs,
    inverse_sigmoid,
    mlp,
)
from models.deformable_detr import postprocess as detr_postprocess
from utils.boxes import Detections

# Ruido del CDN, en las unidades del paper: la caja positiva mueve cada esquina hasta
# media caja y la negativa entre media y una caja entera.
DN_LABEL_NOISE = 0.5
DN_BOX_NOISE = 1.0
PRIOR_PROB = 0.01  # p(objeto) con la que arranca la cabeza de clase


def sine_embed(coords: Tensor, dim: int, temperature: float = 10000.0) -> Tensor:
    """Embedding sinusoidal de coordenadas en [0,1] (DAB-DETR).

    `coords` (..., K) -> (..., dim), con dim/K canales por coordenada. Es lo que convierte
    un ancla de cuatro números en la posición de la query, y se recalcula en cada capa: la
    query se mueve con su caja en vez de quedarse con la posición inicial.
    """
    per_coord = dim // coords.shape[-1]
    index = torch.arange(per_coord, device=coords.device, dtype=torch.float32)
    frequency = temperature ** (2 * (index // 2) / per_coord)
    angles = coords[..., None] * (2 * math.pi) / frequency  # (..., K, per_coord)
    return torch.stack([angles[..., 0::2].sin(), angles[..., 1::2].cos()], dim=-1).flatten(-3)


def grid_anchors(height: int, width: int, span: float, device: torch.device) -> Tensor:
    """Una caja por celda del mapa: centro de la celda y `span` celdas de lado.

    -> (height*width, 4) cxcywh normalizado, con x = tiempo (ancho) e y = frecuencia.
    """
    rows = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    columns = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    y, x = torch.meshgrid(rows, columns, indexing="ij")
    centers = torch.stack([x.flatten(), y.flatten()], dim=-1)
    sizes = centers.new_tensor([span / width, span / height]).expand_as(centers)
    return torch.cat([centers, sizes], dim=-1)


def flatten_levels(maps: list[Tensor]) -> tuple[Tensor, list[tuple[int, int]]]:
    """Pirámide -> una secuencia de tokens, con la forma de cada nivel para deshacerlo."""
    shapes = [(int(m.shape[-2]), int(m.shape[-1])) for m in maps]
    return torch.cat([m.flatten(2).transpose(1, 2) for m in maps], dim=1), shapes


def unflatten_levels(tokens: Tensor, shapes: list[tuple[int, int]]) -> list[Tensor]:
    sizes = [height * width for height, width in shapes]
    return [
        chunk.transpose(1, 2).unflatten(-1, shape)
        for chunk, shape in zip(tokens.split(sizes, dim=1), shapes, strict=True)
    ]


class DeformableEncoderLayer(nn.Module):
    """Cada posición del mapa atiende, deformablemente, a los cuatro niveles."""

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
        self.attn = DeformableAttention(dim, n_heads, n_points, n_levels, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn, dim),
            nn.Dropout(dropout),
        )
        self.n1, self.n2 = (nn.LayerNorm(dim) for _ in range(2))

    def forward(
        self, tokens: Tensor, pos: Tensor, ref_boxes: Tensor, shapes: list[tuple[int, int]]
    ) -> Tensor:
        maps = unflatten_levels(tokens, shapes)
        tokens = self.n1(tokens + self.attn(tokens + pos, ref_boxes, maps))
        return self.n2(tokens + self.ffn(tokens))


def noised_boxes(boxes: Tensor, scale: float, negative: bool) -> Tensor:
    """Ruido del CDN sobre cajas cxcywh normalizadas.

    Cada esquina se corre hasta media caja (positivo) o entre media y una caja entera
    (negativo). La negativa queda cerca pero mal: es la que enseña a rechazar el casi.
    """
    xyxy = box_convert(boxes, "cxcywh", "xyxy")
    half = torch.cat([boxes[:, 2:], boxes[:, 2:]], dim=-1) / 2
    sign = torch.randint(0, 2, xyxy.shape, device=boxes.device) * 2 - 1
    part = torch.rand_like(xyxy) + float(negative)
    return box_convert((xyxy + sign * part * half * scale).clamp(0.0, 1.0), "xyxy", "cxcywh")


def contrastive_denoise(
    targets: list[Target],
    n_classes: int,
    label_embed: nn.Embedding,
    n_queries: int,
    dn_queries: int,
    label_noise: float,
    box_noise: float,
) -> tuple[Tensor, Tensor, Tensor, list[tuple[Tensor, Tensor]]] | None:
    """Queries de denoising: contenido, anclas, máscara de atención y emparejamiento.

    Se arman `groups` copias de cada caja verdadera, cada una con su mitad positiva y su
    mitad negativa. El emparejamiento devuelto sólo cubre las positivas: las negativas y
    el relleno quedan sin emparejar y la focal las supervisa como fondo.
    """
    sizes = [len(target["labels"]) for target in targets]
    max_boxes = max(sizes)
    if max_boxes == 0 or dn_queries <= 0:
        return None

    device = label_embed.weight.device
    groups = max(1, dn_queries // (2 * max_boxes))
    block = 2 * max_boxes  # un grupo: positivas y después negativas
    n_dn = groups * block

    # Relleno: la clase extra de `label_embed` y una caja degenerada en la esquina, donde
    # nunca hay nada, para que supervisarla como fondo no enseñe nada falso.
    labels = torch.full((len(targets), groups, block), n_classes, dtype=torch.int64, device=device)
    boxes = torch.zeros(len(targets), groups, block, 4, device=device)
    boxes[..., 2:] = 1e-3

    for batch, target in enumerate(targets):
        count = sizes[batch]
        if count == 0:
            continue
        for half, negative in enumerate((False, True)):
            start = half * max_boxes
            boxes[batch, :, start : start + count] = torch.stack(
                [noised_boxes(target["boxes"], box_noise, negative) for _ in range(groups)]
            )
            noisy = target["labels"].to(torch.int64).expand(groups, count).clone()
            flip = torch.rand(noisy.shape, device=device) < label_noise / 2
            noisy[flip] = torch.randint(n_classes, (int(flip.sum()),), device=device)
            labels[batch, :, start : start + count] = noisy

    total = n_dn + n_queries
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    mask[n_dn:, :n_dn] = True  # las queries de matching no pueden ver las respuestas
    group_id = torch.arange(n_dn, device=device) // block
    mask[:n_dn, :n_dn] = group_id[:, None] != group_id[None, :]  # ni un grupo al otro

    indices = [
        (
            torch.cat([torch.arange(count, device=device) + g * block for g in range(groups)]),
            torch.arange(count, device=device).repeat(groups),
        )
        for count in sizes
    ]
    return label_embed(labels.flatten(1)), boxes.flatten(1, 2), mask, indices


class DINO(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        n_queries: int = 100,
        n_classes: int = 1,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        n_heads: int = 8,
        n_points: int = 4,
        n_levels: int = 4,
        ffn: int = 1024,
        dn_queries: int = 100,
        label_noise: float = DN_LABEL_NOISE,
        box_noise: float = DN_BOX_NOISE,
    ):
        super().__init__()
        self.dim, self.n_queries = dim, n_queries
        self.n_classes, self.n_points = n_classes, n_points
        self.dn_queries, self.label_noise, self.box_noise = dn_queries, label_noise, box_noise

        self.level_embed = nn.Parameter(torch.zeros(n_levels, dim))
        self.encoder = nn.ModuleList(
            DeformableEncoderLayer(dim, n_heads, n_points, n_levels, ffn)
            for _ in range(n_encoder_layers)
        )
        # Dos etapas: el encoder propone una caja por celda y las mejores son las anclas.
        self.enc_output = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.enc_class = nn.Linear(dim, n_classes)
        self.enc_bbox = mlp(dim, dim, 4)

        self.query_embed = nn.Embedding(n_queries, dim)  # selección mixta: contenido aprendido
        self.query_pos_head = mlp(2 * dim, dim, dim, layers=2)
        self.label_embed = nn.Embedding(n_classes + 1, dim)  # CDN; la última fila es relleno

        self.layers = nn.ModuleList(
            DeformableDecoderLayer(dim, n_heads, n_points, n_levels, ffn)
            for _ in range(n_decoder_layers)
        )
        # DINO comparte las cabezas entre capas, a diferencia del Deformable-DETR de acá.
        self.class_head = nn.Linear(dim, n_classes)
        self.bbox_head = mlp(dim, dim, 4)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Casi toda query es fondo: el sesgo arranca en p(objeto)=0.01 para que la focal
        # no dedique las primeras épocas a apagar todos los logits.
        bias = -math.log((1 - PRIOR_PROB) / PRIOR_PROB)
        for head in (self.class_head, self.enc_class):
            nn.init.constant_(head.bias, bias)
        # Las cajas arrancan siendo exactamente el ancla; el refinamiento se aprende.
        for last in (self.bbox_head[-1], self.enc_bbox[-1]):
            assert isinstance(last, nn.Linear)  # la salida de `mlp`
            nn.init.constant_(last.weight, 0.0)
            nn.init.constant_(last.bias, 0.0)
        nn.init.normal_(self.level_embed, std=0.02)

    def forward(self, features: list[Tensor], targets: list[Target] | None = None) -> Outputs:
        tokens, shapes = flatten_levels(features)
        batch, device = tokens.shape[0], tokens.device

        cells = torch.cat([grid_anchors(h, w, 1.0, device) for h, w in shapes])  # (S,4)
        # Referencia con la que muestrea el encoder: `n_points` celdas a cada lado, que es
        # el radio con el que `DeformableAttention` inicializa sus offsets.
        sampling = torch.cat([grid_anchors(h, w, 2.0 * self.n_points, device) for h, w in shapes])
        levels = torch.cat(
            [self.level_embed[level].expand(h * w, -1) for level, (h, w) in enumerate(shapes)]
        )
        pos = (sine_embed(cells[:, :2], self.dim) + levels)[None]

        memory = tokens
        reference = sampling[None].expand(batch, -1, -1)
        for layer in self.encoder:
            memory = layer(memory, pos, reference, shapes)

        # --- selección mixta de queries: anclas del encoder, contenido aprendido ---
        proposals = self.enc_output(memory)
        enc_logits = self.enc_class(proposals)
        enc_boxes = (self.enc_bbox(proposals) + inverse_sigmoid(cells)).sigmoid()
        best = enc_logits.max(-1).values.topk(self.n_queries, dim=1).indices  # (B,Q)
        interm = {
            "pred_logits": enc_logits.gather(1, best[..., None].expand(-1, -1, self.n_classes)),
            "pred_boxes": enc_boxes.gather(1, best[..., None].expand(-1, -1, 4)),
        }

        query = self.query_embed.weight[None].expand(batch, -1, -1)
        anchors = interm["pred_boxes"].detach()  # el decodificador no corrige al encoder
        attn_mask, dn_indices, n_dn = None, None, 0
        denoising = None
        if self.training and targets is not None:
            denoising = contrastive_denoise(
                targets,
                self.n_classes,
                self.label_embed,
                self.n_queries,
                self.dn_queries,
                self.label_noise,
                self.box_noise,
            )
        if denoising is not None:
            dn_query, dn_anchors, attn_mask, dn_indices = denoising
            query = torch.cat([dn_query, query], dim=1)
            anchors = torch.cat([dn_anchors, anchors], dim=1)
            n_dn = dn_query.shape[1]

        maps = unflatten_levels(memory, shapes)
        sampled, refined = anchors, anchors
        layers: list[dict[str, Tensor]] = []
        for layer in self.layers:
            query_pos = self.query_pos_head(sine_embed(sampled, 2 * self.dim))
            query = layer(query, query_pos, sampled, maps, attn_mask)
            boxes = (self.bbox_head(query) + inverse_sigmoid(refined)).sigmoid()
            layers.append({"pred_logits": self.class_head(query), "pred_boxes": boxes})
            # Look forward twice: la caja de la próxima capa se apoya en ésta con gradiente
            # (`refined`), pero el muestreo de la próxima capa no propaga por acá.
            refined, sampled = boxes, boxes.detach()

        return self._outputs(layers, interm, n_dn, dn_indices)

    @staticmethod
    def _outputs(
        layers: list[dict[str, Tensor]],
        interm: dict[str, Tensor],
        n_dn: int,
        dn_indices: list[tuple[Tensor, Tensor]] | None,
    ) -> Outputs:
        matching = [{key: value[:, n_dn:] for key, value in layer.items()} for layer in layers]
        out: Outputs = dict(matching[-1])
        # La salida del encoder se supervisa como una capa auxiliar más (DINO la llama
        # `interm`): sin eso la selección de anclas no recibe gradiente de clasificación.
        out["aux_outputs"] = matching[:-1] + [interm]
        if n_dn:
            out["dn_outputs"] = [
                {key: value[:, :n_dn] for key, value in layer.items()} for layer in layers
            ]
            out["dn_indices"] = dn_indices
        return out


class DINOHead(DetectionHead):
    """`DetectionHead` que además le pasa los targets al decodificador (para el CDN)."""

    def forward(self, tokens: Tensor, targets: list[Target] | None = None) -> Outputs:
        return self.detr(self.pyramid_features(tokens), targets)


def denoising_losses(
    criterion: SetCriterion, outputs: Outputs, targets: list[Target]
) -> dict[str, Tensor]:
    """Pérdidas del CDN, sumadas sobre las capas del decodificador.

    No hay húngaro: cada query ruidosa ya sabe de qué caja salió. Las negativas y el
    relleno no aparecen en `dn_indices` y la focal las supervisa como fondo, que es
    exactamente lo que el denoising contrastivo quiere enseñar.
    """
    layers: list[dict[str, Tensor]] = outputs.get("dn_outputs", [])
    if not layers:
        return {}

    losses: dict[str, Tensor] = {}
    for layer in layers:
        for key, value in criterion.losses(layer, targets, outputs["dn_indices"]).items():
            name = key.replace("loss_", "loss_dn_")
            losses[name] = value if name not in losses else losses[name] + value
    return losses


class EATDINO(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        n_queries: int = 100,
        n_classes: int = 1,
        freeze: bool = True,
        n_frames: int | None = None,
        time_stride: int = 16,
        n_levels: int = 4,
        n_mels: int = 128,
        frontend: str = "pcen",
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        dn_queries: int = 100,
    ):
        super().__init__()
        from models.eat import EATBackbone
        from models.pcen import TrainablePCEN

        self.backbone = EATBackbone(n_frames=n_frames, time_stride=time_stride, freeze=freeze)
        if n_mels != self.backbone.n_mels:
            raise ValueError(
                f"n_mels={n_mels} no coincide con las {self.backbone.n_mels} bandas con las "
                "que se preentrenó el EAT; la rejilla de parches en frecuencia es esa."
            )
        if frontend not in ("pcen", "logmel"):
            raise ValueError(f"frontend desconocido: {frontend!r}; hay 'pcen' y 'logmel'")

        # Misma entrada que `ASTDeformableDETR`, para que la comparación entre las dos
        # arquitecturas no mezcle el frontend con el backbone.
        self.frontend = frontend
        self.pcen = TrainablePCEN(n_mels=n_mels) if frontend == "pcen" else LogMelFrontend()
        self.pcen_norm = nn.BatchNorm2d(1, affine=False)

        self.head = DINOHead(
            DINO(
                dim,
                n_queries,
                n_classes,
                n_encoder_layers,
                n_decoder_layers,
                n_levels=n_levels,
                dn_queries=dn_queries,
            ),
            token_dim=self.backbone.hidden_size,
            freq_out=self.backbone.freq_out,
            time_out=self.backbone.time_out,
            dim=dim,
            n_levels=n_levels,
        )
        self.criterion = SetCriterion(
            n_classes=n_classes,
            matcher=HungarianMatcher(cost_class=2.0, focal=True),
            focal=True,
        )

    def forward(
        self, x: Tensor, targets: list[Target] | None = None
    ) -> Outputs | dict[str, Tensor]:
        x = self.pcen(x)
        x = self.pcen_norm(x) / 2
        outputs = self.head(self.backbone(x), targets)
        if targets is None:
            return outputs
        return self.criterion(outputs, targets) | denoising_losses(self.criterion, outputs, targets)


def predict_scores(outputs: Outputs) -> tuple[Tensor, Tensor]:
    """Score y clase por query -> (B, Q), (B, Q).

    Con focal loss no hay canal de "no-objeto": cada logit es un sigmoide independiente y
    el score de la query es el de su clase más probable.
    """
    scores, labels = outputs["pred_logits"].sigmoid().max(-1)
    return scores, labels


def postprocess(outputs: Outputs, score_threshold: float = 0.5) -> list[Detections]:
    return detr_postprocess(outputs, score_threshold, predict_scores)


@torch.no_grad()
def detect(model: nn.Module, images: Tensor, score_threshold: float = 0.5) -> list[Detections]:
    # En `train()` el decodificador tiene dropout activo y el CDN espera targets: detectar
    # siempre en `eval()`, como hace `models.faster_rcnn`.
    training = model.training
    model.eval()
    outputs = model(images)
    model.train(training)
    return postprocess(outputs, score_threshold)  # type: ignore[arg-type]
