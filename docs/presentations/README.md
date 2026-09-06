# Presentaciones

| Archivo | Qué es |
|---|---|
| `resultados_evaluacion.tex` | Presentación corta (beamer + metropolis, 8 láminas) de los resultados de evaluación sobre val y test |
| `make_figures.py` | Genera `figures/*.png` desde `checkpoints/*_{val,test}_metrics.json` |

Estructura: conjunto y tarea · desbalance por clase · tabla de las cuatro
arquitecturas · resultados duros (val y test) · frontera de Pareto
recall--precisión · limitaciones de cada enfoque.

Las cifras **no se escriben a mano**: salen de los `.json` de evaluación. Si se reevalúa
un modelo, se vuelve a correr el script y se recompila.

> **Pendiente.** `make_figures.py` todavía lee `checkpoints/*_{val,test}_metrics.json`,
> que los dejaba `src/evaluate.py`. Ese script ya no existe: ahora hay un solo reporte,
> `comparacion_modelos.json` de `src/compare_models.py`, con las mismas métricas más el
> punto pareado, los IC y la tabla por clase. Hay que repuntar `make_figures.py` a ese
> `.json` antes de regenerar las figuras.

```bash
python docs/presentations/make_figures.py            # desde la raíz del repo
cd docs/presentations && latexmk -pdf resultados_evaluacion.tex
```

`make_figures.py` produce siete figuras; la presentación usa `dataset_support.png` y
`pr_plane.png`. Las otras cinco (`metrics_test`, `per_class_test`, `val_vs_test`,
`recall_vs_support`, `review_cost`) quedan disponibles para la versión larga o para
la tesis.

Las figuras de contexto (ventaneo, criterio de IoU) vienen de `research/figures/`,
que produce `notebooks/thesis_figures.ipynb`; el `\graphicspath` apunta a las dos
carpetas.

Nota: `research/figures/{score_sweep,recall_per_class,confusion_matrix}.png` provienen
de una evaluación parcial (`MAX_EVAL_WINDOWS` acotado) y **no** se usan acá.
