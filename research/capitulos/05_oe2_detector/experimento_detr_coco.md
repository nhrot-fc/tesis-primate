# Experimento: Deformable DETR con pesos COCO sobre el AST

Corrida `runs/detr_coco`, lanzada el 2026-09-11 (`train.py --arch detr_coco`).

**Por qué.** En la comparación final Faster R-CNN gana al DETR propio. Zhu et al. (DCASE 2025)
obtienen lo contrario, pero allí todos los detectores parten de pesos COCO salvo la cabeza. Acá
Faster R-CNN carga `COCO_V1` completo y el `ASTDeformableDETR` entrena el transformer desde
cero: la comparación confunde arquitectura con inicialización. Este experimento iguala la
política.

**Qué es.** `models/coco_deformable_detr.py::CocoDeformableDETR`, registrado como
`coco_deformable_detr`. Toma `SenseTime/deformable-detr-with-box-refine-two-stage` (HF, COCO
46,2 AP) y carga tal cual encoder, decoder, `level_embed`, cabezas de caja y generador de
propuestas; reinicia sólo `input_proj` y las cabezas de clase. El backbone es el mismo
`ASTBackbone` + PCEN + `MultiScalePyramid` del repo en lugar del ResNet-50. Pérdida: el
`SetCriterion` del repo en las 6 capas del decoder más las propuestas del encoder con etiqueta
binaria. Misma receta que el preset `detr`.

| | `ast_deformable_detr` | `coco_deformable_detr` |
|---|---|---|
| Encoder deformable | no | 6 capas, COCO |
| Dimensión | 128 | 256 |
| Queries | 100 aprendidas | 100 propuestas del encoder (dos etapas) |
| Posición | embedding por query | seno 2D por nivel + `level_embed` |
| Transformer | desde cero | COCO |

**Costo.** 0,70 s/paso, 17,7 GB de VRAM, ~32 min por época + validación (~20 h).

**Después.** `dump_predictions.py --run detr_coco` y `compare_models.py` con los demás. Si no
supera a Faster R-CNN, la conclusión pasa de "el DETR perdió" a "perdió incluso con la receta
del paper: deformable + pirámide + pesos COCO", y lo que queda son datos, 25 clases, ventanas
de 3 s y AST a 16 kHz.
