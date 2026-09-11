# Documentación

El documento vivo es la tesis, [research/main.tex](../research/main.tex) (se compila con
`make` desde la raíz). Sus figuras las genera
[notebooks/figuras_documento.ipynb](../notebooks/figuras_documento.ipynb), una sección por
capítulo; ver [research/figures/README.md](../research/figures/README.md).

Lo demás son informes fechados: **cada uno describe el código tal como estaba ese día**. La
columna de estado dice qué dejó de ser cierto; el código manda.

| Documento | Fecha | Qué es | Estado frente al código actual |
|---|---|---|---|
| [informes/2026-08-02_advancements_report/](informes/2026-08-02_advancements_report/informe.md) (+ PDF) | 2026-08-02 | Informe de avance en inglés: problema, estado del arte, línea base en TensorFlow (`notebooks/TF_Experimental_Training.ipynb`), primer AST-DETR y sus resultados. Sus figuras (`fig/`) contrastan la curva de probabilidad de la línea base con las cajas del detector sobre las mismas grabaciones | **Histórico.** Protocolo distinto al actual: recall class-agnostic a IoU 0,5, AP a {0,25, 0,5, 0,75}; hoy la comparación es class-aware a IoU 0,3 con punto de operación elegido en val (`docs/protocolo_comparacion.md`, `src/evaluation/protocol.py`). El AST ya no va congelado y el DETR usa 100 consultas, no 64 |
| [informes/2026-08-21_RE1_dataset_curado/](informes/2026-08-21_RE1_dataset_curado/informe.md) (+ PDF, `re1_stats.json`) | 2026-08-21 | Protocolo de curación y ficha del conjunto derivado (OE1) | **Vigente en lo esencial** (sinónimos, pares válidos, `MIN_PAIR_COUNT`, particionado por grabación, ventaneo). Desactualizado: `MIN_BOX_SIZE` es ahora la duración mínima de una anotación (10 ms) y no 3 ms; el jitter usa ese mismo mínimo y no 0,02; hay aumento sobre el audio (`data/augment.py`) además del jitter; las grabaciones de aves (PteroSet, `birds__AV`) entran como fondo con `BACKGROUND_RATIO`; `lw/{tr,tj,tt,tf}` se unen en `lw/trino` y `sb/lpc` en `sb/ppc` (`JOINED_LABELS`). Las figuras `re1_*.png` que cita se perdieron con el cuaderno que las hacía; están en la lista de pendientes del notebook de figuras |
| [informes/2026-08-21_RE2_detector_pipeline/](informes/2026-08-21_RE2_detector_pipeline/informe.md) (+ PDF) | 2026-08-21 | Arquitectura, pérdida, pipeline de entrenamiento e inferencia del AST-DETR (OE2) | **Desactualizado en la receta**, vigente en la explicación de PCEN, pirámide, atención deformable y matching. Cambios: el AST se afina entero (`ASTBackbone`); `time_stride` 10 (no 2); 100 consultas (no 64); pirámide de 4 niveles; focal sigmoide por clase sin canal de no-objeto (`SetCriterion`); sin NMS para el DETR (`nms_iou = None`); 30 épocas con lote 8; la entrada admite `frontend` = `none` / `logmel` / `pcen` (`models/frontend.py`). Los scripts `eval.py`/`predict.py` y `checkpoints/` ya no existen: hoy son `dump_predictions.py`, `compare_models.py`, `inference/predictor.py` y `runs/` |
| [protocolo_comparacion.md](protocolo_comparacion.md) | 2026-09-05 | Por qué los tres detectores se comparan con el mismo caché, el mismo IoU 0,3 y el mismo tope de detecciones | **Vigente en las decisiones** (IoU 0,3, igualación de presupuesto, umbral por modelo). Desactualizado en la mecánica: ya no hay F-beta ni `fp_per_hour` ni `MAP_THRESHOLDS`; `best.pt` se elige por mAP@0,3 y el punto de operación lo fija `compare_models.py` (precisión mínima en val, IC por bootstrap de grabaciones, `operating_point.json`). `train_yolo.py` es `train.py --arch yolo` |
| [call_type_notes.md](call_type_notes.md) | 2026-08-21 | Notas de campo por especie y tipo de llamada | Vigente (referencia) |
| [presentations/](presentations/README.md) | 2026-09-05 | Presentación beamer de resultados | Sus figuras salieron del protocolo anterior; se regeneran desde el notebook de figuras |

## Pendiente de rehacer

- **RE2** merece una reescritura completa contra `src/models/deformable_detr.py`,
  `criterion.py` y `training/trainer.py`; la mitad de sus cifras cambió.
- **protocolo_comparacion.md**, sección "Punto de operación y métricas": reemplazar F-beta
  por el punto pareado de `evaluation/protocol.py`.
- Las figuras de RE1 (`re1_curation_funnel`, `re1_long_events`, `re1_split_composition`,
  `re1_boxes_per_class_split`, `re1_box_geometry`) y `unreachable_events` del informe de
  agosto: rehacerlas en `figuras_documento.ipynb` cuando se necesiten.
