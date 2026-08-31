# Figuras de la tesis

Las figuras de `main.tex` **no se editan a mano**: se generan desde los datos y desde el
código, y se vuelven a generar cuando cualquiera de los dos cambia. Los cuadernos ya no
llevan títulos ni subtítulos dentro del PNG: el nombre de la figura lo pone el `\caption`.

| Cuaderno | Qué produce |
|---|---|
| `notebooks/thesis_figures.ipynb` | Diagramas (árbol de problemas, arquitectura, pipelines), figuras de datos (galería de llamadas, anidamiento, PCEN) y figuras de resultados (barrido de score, recall por clase, matriz de confusión, ejemplos cualitativos) |
| `notebooks/dataset_report.ipynb` | Las figuras del conjunto: `annotations_per_pair`, `class_geometry`, `class_geometry_facets`, `boxes_per_split`, `boxes_per_window`, `window_example` |

## `data/`: las series, aparte del PNG

Las figuras que son series de datos y no diagramas escriben además su JSON en
`figures/data/<nombre>.json`. Sirve para redibujarlas en pgfplots sin volver a correr el
cuaderno, y para citar una cifra sin leerla del gráfico. Hoy lo escriben doce:
`annotations_per_pair`, `class_geometry`, `boxes_per_split`, `boxes_per_window`,
`recorded_vs_annotated`, `mel_axis`, `multiscale_pyramid`, `score_sweep`,
`recall_per_class`, `confusion_matrix`, `iou_distribution` y `ablation_progression`.

## Qué hace falta para regenerarlas

- Diagramas y figuras del conjunto: sólo `data/cleaned/`.
- Figuras de resultados: además un `.pth` en `checkpoints/` y el caché de
  `data/processed/`. Sin ellos la sección se salta y avisa por consola. En CPU conviene
  dejar `MAX_EVAL_WINDOWS` en unos cientos; **las cifras que se reportan en la tesis
  tienen que salir de una corrida con `MAX_EVAL_WINDOWS = None`**.

## Imágenes que no genera ningún cuaderno

`image3.png` … `image10.png` vienen del informe de avance E3, `wbs_tree.png` y
`pucp-logo.png` de la portada, y `raven_table_preview.png` hay que capturarla a mano: es
la prueba de que el archivo exportado abre en Raven sin conversión.

## `boxes_per_split` se construye con `empty_ratio = 0`

`dataset_report.ipynb` llama a `build_manifest` sin ventanas de fondo, mientras que la
partición que se materializa en `data/processed/` lleva un 25 %. La figura describe el
reparto de cajas por clase, que no cambia; el conteo de ventanas, sí.
