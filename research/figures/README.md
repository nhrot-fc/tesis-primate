# Figuras de la tesis

Las figuras de `main.tex` **no se editan a mano**: las genera
`notebooks/figuras_documento.ipynb`, con una sección por capítulo del documento en el mismo
orden que `main.tex`. Los PNG no llevan título dentro: el nombre lo pone el `\caption`.

| Sección del notebook | Capítulo de `main.tex` | Figuras |
|---|---|---|
| 1. Generalidades | Problemática, árbol de problemas, metodología | `recorded_vs_annotated`, `problem_tree`, `crisp_dm` |
| 2. Marco referencial | Marco teórico y conceptual | `annotation_example`, `output_forms`, `signal_chain`, `stft_tradeoff`, `mel_axis`, `frontends`, `box_coordinates`, `iou_criterion` |
| 3. Curación del conjunto (OE1) | Análisis, protocolo, particionado, ventaneo | `annotations_per_pair`, `class_geometry`, `class_geometry_facets`, `nesting_phrase`, `call_gallery`, `curation_pipeline`, `split_by_recording`, `boxes_per_split`, `windowing`, `boxes_per_window`, `window_example` |
| 4. Detección y evaluación (OE2/OE3) | Arquitectura, pipeline, protocolo, resultados | `architecture`, `multiscale_pyramid`, `deformable_sampling`, `decoder_layer`, `hungarian_matching`, `training_pipeline`, `inference_pipeline`, `training_curves`, `comparison_map`, `comparison_paired`, `comparison_per_class`, `score_sweep` |

## `data/`: las series, aparte del PNG

Las figuras que son series de datos escriben además `figures/data/<nombre>.json`: sirve
para redibujarlas en pgfplots y para citar una cifra sin leerla del gráfico.

## Qué hace falta para regenerarlas

- Secciones 1–3: `data/cleaned/` (y `data/processed/` para `window_example` y `boxes_per_split`).
- Sección 4: las corridas en `runs/` (`metrics.jsonl` para las curvas de entrenamiento,
  `*_predictions.pt` y `runs/comparacion/*.json` para la comparación). Las figuras de
  resultados que necesitan un checkpoint se saltan con aviso si no está.

Cada celda del notebook es independiente después de la sección 0 (imports y estilo): se
puede correr sólo la sección que cambió.

## Imágenes que no genera el notebook

`image3.png` … `image9.png` vienen del informe de avance E3; `wbs_tree.png` y
`pucp-logo.png`, de la portada; `raven_table_preview.png` se captura a mano (es la prueba de
que el archivo exportado abre en Raven sin conversión); `confusion_matrix`,
`recall_per_class`, `iou_distribution`, `ablation_progression`, `qualitative_detections` y
`detection_timeline` quedaron de la evaluación del DETR anterior (`checkpoint` perdido) y se
regenerarán cuando haya una corrida DETR nueva en `runs/`.

## `boxes_per_split` se construye con `empty_ratio = 0`

El notebook llama a `build_manifest` sin ventanas de fondo, mientras que la partición que se
materializa en `data/processed/` lleva un 25 %. La figura describe el reparto de cajas por
clase, que no cambia; el conteo de ventanas, sí.

## `data/recording_durations.csv`

Caché de la duración de cada `.wav` de `data/raw/` (leerlas todas tarda minutos). Borrarlo
si cambia el material recibido; el notebook lo vuelve a construir.
