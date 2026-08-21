# Figuras de la tesis

Las figuras de `main.tex` **no se editan a mano**: se generan desde los datos y desde el
código, y se vuelven a generar cuando cualquiera de los dos cambia.

| Cuaderno | Qué produce |
|---|---|
| `notebooks/thesis_figures.ipynb` | Diagramas (árbol de problemas, arquitectura, pipelines), figuras de datos (galería de llamadas, anidamiento, PCEN) y figuras de resultados (recall por clase, matriz de confusión, ejemplos cualitativos, tabla de Raven) |
| `notebooks/dataset_report.ipynb` | Las figuras del conjunto: `annotations_per_pair`, `class_geometry`, `class_geometry_facets`, `boxes_per_split`, `unreachable_events`, `boxes_per_window`, `window_example` |
| `notebooks/re1_dataset_figures.ipynb` | Las figuras del documento de RE1 (`docs/RE1_dataset_curado.md`): `re1_curation_funnel`, `re1_split_composition`, `re1_boxes_per_class_split`, `re1_long_events`, `re1_box_geometry`. Además escribe `docs/re1_stats.json`, del que salen todas las cifras de ese documento |

`GENERATED.md` es el índice que escribe el primero: para cada figura, dónde va en el
documento y el bloque `\begin{figure}` con su pie. Los mismos bloques están insertados,
comentados, en el sitio que les corresponde dentro de `main.tex`.

## Dos figuras que ya no describen el conjunto vigente

`boxes_per_split.png` y `unreachable_events.png` salen de un manifiesto construido con
`empty_ratio = 0` y bajo la regla de solapamiento anterior al commit `3c76d20`, que medía
el solape contra la duración completa del evento. Con la regla vigente ningún evento se
pierde por ser largo, y la partición materializada incluye un 25 % de ventanas de fondo.
Las reemplazan `re1_boxes_per_class_split.png` y `re1_long_events.png`; véase
`docs/RE1_dataset_curado.md`, §3.2 y §7.

## Qué hace falta para regenerarlas

- Secciones 1–6 del cuaderno: sólo `data/cleaned/`.
- `notebooks/re1_dataset_figures.ipynb`: `data/cleaned/` para todo, y además
  `data/processed/` para la sección 7, que compara lo recalculado contra el disco.
- Sección 7 (resultados): además un `.pth` en `checkpoints/` y el caché de
  `data/processed/`. En CPU conviene dejar `MAX_EVAL_WINDOWS` en unos cientos; **las
  cifras que se reportan en la tesis tienen que salir de una corrida con
  `MAX_EVAL_WINDOWS = None`**.
- `training_curves`: necesita el `metrics.jsonl` que `train.py` deja en `logs/<fecha>/`.
- Sección 8: son plantillas con valores de ejemplo (tiempo de revisión, análisis de
  falsos positivos, cronograma). Cada una avisa por consola mientras siga sin datos
  reales.

## Imágenes que no genera ningún cuaderno

`image1.png` … `image14.png` vienen del informe de avance E3 y `pucp-logo.png` de la
portada. La captura de Raven abriendo una tabla de selección del modelo (sección 5.5)
hay que tomarla a mano: es la prueba de que el archivo exportado abre sin conversión.
