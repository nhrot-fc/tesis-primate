# Presentaciones

| Archivo | Qué es |
|---|---|
| `resultados_evaluacion.tex` | Presentación corta (beamer + metropolis, 8 láminas) de los resultados de evaluación sobre val y test |
| `figures/` | Las figuras que usa la presentación |

Estructura: conjunto y tarea · desbalance por clase · tabla de las arquitecturas ·
resultados duros (val y test) · frontera de Pareto recall–precisión · limitaciones de cada
enfoque.

```bash
cd docs/presentations && latexmk -pdf resultados_evaluacion.tex
```

Las figuras de contexto (ventaneo, criterio de IoU) vienen de `research/figures/`; el
`\graphicspath` apunta a las dos carpetas.

> **Pendiente.** `figures/*.png` salió de un script (`make_figures.py`, borrado) que leía
> `checkpoints/*_metrics.json`, del protocolo anterior (mAP@0,5, F3). Las cifras vigentes son
> las de `runs/comparacion/*.json` (`src/compare_models.py`); la sección 4 de
> `notebooks/figuras_documento.ipynb` las dibuja (`comparison_map`, `comparison_paired`,
> `comparison_per_class`, `score_sweep`). Antes de recompilar la presentación hay que
> reemplazar `dataset_support.png` y `pr_plane.png` por esas y actualizar la tabla de
> resultados de la lámina 4.
