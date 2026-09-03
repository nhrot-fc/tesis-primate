import math

import torch
from torch import Tensor, nn
from torchvision.ops import box_convert

from models.criterion import HungarianMatcher, SetCriterion, Target
from models.deformable_detr import (
    DeformableAttention,
    DeformableDecoderLayer,
    DetectionHead,
    Outputs,
    detections_above_threshold,
    inverse_sigmoid,
    mlp,
)
from utils.boxes import Detections

# Ruido del CDN, en las unidades del paper: la caja positiva mueve cada esquina hasta media
# caja y la negativa entre media y una caja entera.
DN_LABEL_NOISE = 0.5
DN_BOX_NOISE = 1.0
PRIOR_PROB = 0.01  # p(objeto) con la que arranca la cabeza de clase


def sine_embed(coords: Tensor, dim: int, temperature: float = 10000.0) -> Tensor:
    # (..., K) -> (..., dim), con dim/K canales por coordenada (DAB-DETR).
    per_coord = dim // coords.shape[-1]
    index = torch.arange(per_coord, device=coords.device, dtype=torch.float32)
    frequency = temperature ** (2 * (index // 2) / per_coord)
    angles = coords[..., None] * (2 * math.pi) / frequency
    return torch.stack([angles[..., 0::2].sin(), angles[..., 1::2].cos()], dim=-1).flatten(-3)


def grid_anchors(height: int, width: int, span: float, device: torch.device) -> Tensor:
    # -> (height*width, 4) cxcywh normalizado, con x = tiempo (ancho) e y = frecuencia.
    rows = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    columns = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    centers = torch.stack([column_grid.flatten(), row_grid.flatten()], dim=-1)
    sizes = centers.new_tensor([span / width, span / height]).expand_as(centers)
    return torch.cat([centers, sizes], dim=-1)


def flatten_levels(maps: list[Tensor]) -> tuple[Tensor, list[tuple[int, int]]]:
    shapes = [(int(level.shape[-2]), int(level.shape[-1])) for level in maps]
    return torch.cat([level.flatten(2).transpose(1, 2) for level in maps], dim=1), shapes


def unflatten_levels(tokens: Tensor, shapes: list[tuple[int, int]]) -> list[Tensor]:
    sizes = [height * width for height, width in shapes]
    return [
        chunk.transpose(1, 2).unflatten(-1, shape)
        for chunk, shape in zip(tokens.split(sizes, dim=1), shapes, strict=True)
    ]


class DeformableEncoderLayer(nn.Module):
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
        self, tokens: Tensor, positions: Tensor, ref_boxes: Tensor, shapes: list[tuple[int, int]]
    ) -> Tensor:
        maps = unflatten_levels(tokens, shapes)
        tokens = self.n1(tokens + self.attn(tokens + positions, ref_boxes, maps))
        return self.n2(tokens + self.ffn(tokens))


def noised_boxes(boxes: Tensor, scale: float, negative: bool) -> Tensor:
    corners = box_convert(boxes, "cxcywh", "xyxy")
    half_sizes = torch.cat([boxes[:, 2:], boxes[:, 2:]], dim=-1) / 2
    direction = torch.randint(0, 2, corners.shape, device=boxes.device) * 2 - 1
    magnitude = torch.rand_like(corners) + float(negative)
    moved = (corners + direction * magnitude * half_sizes * scale).clamp(0.0, 1.0)
    return box_convert(moved, "xyxy", "cxcywh")


def contrastive_denoise(
    targets: list[Target],
    n_classes: int,
    label_embed: nn.Embedding,
    n_queries: int,
    dn_queries: int,
    label_noise: float,
    box_noise: float,
) -> tuple[Tensor, Tensor, Tensor, list[tuple[Tensor, Tensor]]] | None:
    box_counts = [len(target["labels"]) for target in targets]
    max_boxes = max(box_counts)
    if max_boxes == 0 or dn_queries <= 0:
        return None

    device = label_embed.weight.device
    groups = max(1, dn_queries // (2 * max_boxes))
    group_size = 2 * max_boxes  # un grupo: positivas y después negativas
    n_denoising = groups * group_size

    # Relleno: la clase extra de `label_embed` y una caja degenerada donde nunca hay nada.
    labels = torch.full(
        (len(targets), groups, group_size), n_classes, dtype=torch.int64, device=device
    )
    boxes = torch.zeros(len(targets), groups, group_size, 4, device=device)
    boxes[..., 2:] = 1e-3

    for batch, target in enumerate(targets):
        count = box_counts[batch]
        if count == 0:
            continue
        true_labels = target["labels"].to(torch.int64).expand(groups, count)
        for half, negative in enumerate((False, True)):
            half_start = half * max_boxes
            # `noised_boxes` trabaja fila por fila: los `groups` grupos van en una sola
            # llamada, no en un bucle de ~15 kernels cada uno.
            noisy = noised_boxes(target["boxes"].repeat(groups, 1), box_noise, negative)
            boxes[batch, :, half_start : half_start + count] = noisy.view(groups, count, 4)

            corrupt = torch.rand((groups, count), device=device) < label_noise / 2
            # `where` y no indexado por máscara: ese `int(mask.sum())` sincroniza GPU->CPU.
            random_labels = torch.randint(n_classes, (groups, count), device=device)
            labels[batch, :, half_start : half_start + count] = torch.where(
                corrupt, random_labels, true_labels
            )

    total = n_denoising + n_queries
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    mask[n_denoising:, :n_denoising] = True  # las de matching no pueden ver las respuestas
    group_of = torch.arange(n_denoising, device=device) // group_size
    mask[:n_denoising, :n_denoising] = group_of[:, None] != group_of[None, :]

    # Sólo las positivas se emparejan; negativas y relleno los supervisa la focal como fondo.
    group_offsets = (torch.arange(groups, device=device) * group_size)[:, None]
    indices = [
        (
            (torch.arange(count, device=device) + group_offsets).flatten(),
            torch.arange(count, device=device).repeat(groups),
        )
        for count in box_counts
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
        self.class_head = nn.Linear(dim, n_classes)
        self.bbox_head = mlp(dim, dim, 4)
        self.grid_cache: dict[tuple, tuple[Tensor, Tensor, Tensor]] = {}
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        # Casi toda query es fondo: el sesgo arranca en p(objeto)=0.01 para que la focal no
        # dedique las primeras épocas a apagar todos los logits.
        bias = -math.log((1 - PRIOR_PROB) / PRIOR_PROB)
        for head in (self.class_head, self.enc_class):
            nn.init.constant_(head.bias, bias)
        # Las cajas arrancan siendo exactamente el ancla; el refinamiento se aprende.
        for last in (self.bbox_head[-1], self.enc_bbox[-1]):
            assert isinstance(last, nn.Linear)
            nn.init.constant_(last.weight, 0.0)
            nn.init.constant_(last.bias, 0.0)
        nn.init.normal_(self.level_embed, std=0.02)

    def grid_constants(
        self, shapes: list[tuple[int, int]], device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Dependen sólo de la forma de la pirámide, fija para un `time_stride` dado, así que
        # recalcularlas por paso son ~60 kernels que devuelven siempre lo mismo. `level_embed`
        # queda afuera: es un parámetro y tiene que reentrar al grafo en cada forward.
        key = (tuple(shapes), str(device))
        if key not in self.grid_cache:
            cells = torch.cat([grid_anchors(h, w, 1.0, device) for h, w in shapes])
            # `n_points` celdas de lado: el radio de los offsets de `DeformableAttention`.
            sampling = torch.cat(
                [grid_anchors(h, w, 2.0 * self.n_points, device) for h, w in shapes]
            )
            self.grid_cache[key] = (cells, sampling, sine_embed(cells[:, :2], self.dim))
        return self.grid_cache[key]

    def forward(self, features: list[Tensor], targets: list[Target] | None = None) -> Outputs:
        tokens, shapes = flatten_levels(features)
        batch_size, device = tokens.shape[0], tokens.device

        cell_boxes, sampling_boxes, cell_positions = self.grid_constants(shapes, device)
        level_embeddings = torch.cat(
            [self.level_embed[level].expand(h * w, -1) for level, (h, w) in enumerate(shapes)]
        )
        positions = (cell_positions + level_embeddings)[None]

        memory = tokens
        encoder_reference = sampling_boxes[None].expand(batch_size, -1, -1)
        for layer in self.encoder:
            memory = layer(memory, positions, encoder_reference, shapes)

        proposals = self.enc_output(memory)
        proposal_logits = self.enc_class(proposals)
        proposal_boxes = (self.enc_bbox(proposals) + inverse_sigmoid(cell_boxes)).sigmoid()
        top_queries = proposal_logits.max(-1).values.topk(self.n_queries, dim=1).indices
        encoder_outputs = {
            "pred_logits": proposal_logits.gather(
                1, top_queries[..., None].expand(-1, -1, self.n_classes)
            ),
            "pred_boxes": proposal_boxes.gather(1, top_queries[..., None].expand(-1, -1, 4)),
        }

        queries = self.query_embed.weight[None].expand(batch_size, -1, -1)
        anchor_boxes = encoder_outputs["pred_boxes"].detach()  # el decoder no corrige al encoder
        attn_mask, dn_indices, n_denoising = None, None, 0
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
            dn_content, dn_anchors, attn_mask, dn_indices = denoising
            queries = torch.cat([dn_content, queries], dim=1)
            anchor_boxes = torch.cat([dn_anchors, anchor_boxes], dim=1)
            n_denoising = dn_content.shape[1]

        memory_maps = unflatten_levels(memory, shapes)
        sampling_reference, refined_boxes = anchor_boxes, anchor_boxes
        per_layer: list[dict[str, Tensor]] = []
        for layer in self.layers:
            query_pos = self.query_pos_head(sine_embed(sampling_reference, 2 * self.dim))
            queries = layer(queries, query_pos, sampling_reference, memory_maps, attn_mask)
            boxes = (self.bbox_head(queries) + inverse_sigmoid(refined_boxes)).sigmoid()
            per_layer.append({"pred_logits": self.class_head(queries), "pred_boxes": boxes})
            # Look forward twice: la próxima capa se apoya en esta caja con gradiente
            # (`refined_boxes`), pero su muestreo no propaga por acá.
            refined_boxes, sampling_reference = boxes, boxes.detach()

        return self.split_outputs(per_layer, encoder_outputs, n_denoising, dn_indices)

    @staticmethod
    def split_outputs(
        per_layer: list[dict[str, Tensor]],
        encoder_outputs: dict[str, Tensor],
        n_denoising: int,
        dn_indices: list[tuple[Tensor, Tensor]] | None,
    ) -> Outputs:
        matching = [
            {key: value[:, n_denoising:] for key, value in layer.items()} for layer in per_layer
        ]
        outputs: Outputs = dict(matching[-1])
        # El encoder se supervisa como capa auxiliar: si no, su selección de anclas no
        # recibe gradiente de clasificación.
        outputs["aux_outputs"] = matching[:-1] + [encoder_outputs]
        if n_denoising:
            outputs["dn_outputs"] = [
                {key: value[:, :n_denoising] for key, value in layer.items()} for layer in per_layer
            ]
            outputs["dn_indices"] = dn_indices
        return outputs


class DINOHead(DetectionHead):
    def forward(self, tokens: Tensor, targets: list[Target] | None = None) -> Outputs:
        return self.detr(self.pyramid_features(tokens), targets)


def denoising_losses(
    criterion: SetCriterion, outputs: Outputs, targets: list[Target]
) -> dict[str, Tensor]:
    # Sin húngaro: cada query ruidosa ya sabe de qué caja salió (`dn_indices`).
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
        from models.pcen import LogMelFrontend, TrainablePCEN

        self.backbone = EATBackbone(n_frames=n_frames, time_stride=time_stride, freeze=freeze)
        if n_mels != self.backbone.n_mels:
            raise ValueError(
                f"n_mels={n_mels} no coincide con las {self.backbone.n_mels} bandas con las "
                "que se preentrenó el EAT; la rejilla de parches en frecuencia es esa."
            )
        if frontend not in ("pcen", "logmel"):
            raise ValueError(f"frontend desconocido: {frontend!r}; hay 'pcen' y 'logmel'")

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
        self, mel: Tensor, targets: list[Target] | None = None
    ) -> Outputs | dict[str, Tensor]:
        compressed = self.pcen_norm(self.pcen(mel)) / 2
        outputs = self.head(self.backbone(compressed), targets)
        if targets is None:
            return outputs
        return self.criterion(outputs, targets) | denoising_losses(self.criterion, outputs, targets)


def postprocess(outputs: Outputs, score_threshold: float = 0.5) -> list[Detections]:
    # Con focal no hay canal de no-objeto: cada logit es un sigmoide independiente y el
    # score de la query es el de su clase más probable.
    scores, labels = outputs["pred_logits"].sigmoid().max(-1)
    return detections_above_threshold(outputs["pred_boxes"], scores, labels, score_threshold)


@torch.no_grad()
def detect(model: nn.Module, images: Tensor, score_threshold: float = 0.5) -> list[Detections]:
    return postprocess(model(images), score_threshold)  # type: ignore[arg-type]
