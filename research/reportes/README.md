# Reportes por resultado esperado

Un documento LaTeX por resultado esperado de OE1 y OE2, cada uno con un cuaderno homónimo que
genera sus figuras, tablas y cifras en `figures/RE_x-y/`. El `.tex` nunca lleva un número a
mano: los toma de `figures/RE_x-y/valores.tex` (macros) y de los fragmentos `*.tex` (filas de
tabla) que deja el cuaderno.

| Documento | Cuaderno | Medio de verificación |
|---|---|---|
| `RE_1-1_protocolo-curacion.tex` | `RE_1-1_protocolo-curacion.ipynb` | protocolo de curación y qué toca cada regla |
| `RE_1-2_conjunto-particionado.tex` | `RE_1-2_conjunto-particionado.ipynb` | `data/processed/`: clases, partición, ventaneo |
| `RE_1-3_ficha-conjunto.tex` | `RE_1-3_ficha-conjunto.ipynb` | ficha (datasheet) del conjunto derivado |
| `RE_2-1_diseno-arquitectura.tex` | `RE_2-1_diseno-arquitectura.ipynb` | diseño experimental, arquitectura y líneas base |
| `RE_2-2_pipeline.tex` | `RE_2-2_pipeline.ipynb` | manual técnico del pipeline |
| `RE_2-3_informe-comparativo.tex` | `RE_2-3_informe-comparativo.ipynb` | informe comparativo de resultados |
| `RE_2-4_modelo-final.tex` | `RE_2-4_modelo-final.ipynb` | modelo final cargable y demostración |
| `RE_2-5_repositorio.tex` | — | índice del repositorio y sus releases |

Las figuras de matplotlib son PNG a 200 ppp, una imagen por panel y del ancho con el que se
imprimen (`reporte.TEXT_WIDTH_IN`, o una fracción), de modo que el `.tex` las coloca por
separado como subfiguras; cuando la lista de paneles sale de los datos (galería de clases,
ventanas de ejemplo, tramos de la demostración), el cuaderno escribe también el bloque de
subfiguras (`galeria.tex`, `ventanas.tex`, `demo.tex`). Los esquemas y los gráficos de pocos
puntos van en TikZ/pgfplots dentro del `.tex`.

## Regenerar las cifras

Desde esta carpeta (el cuaderno resuelve la raíz del proyecto subiendo hasta `pyproject.toml`):

```bash
cd research/reportes
uv run jupyter nbconvert --to notebook --execute --inplace RE_1-1_protocolo-curacion.ipynb
```

Los cuadernos de OE1 leen `data/cleaned/` y `data/processed/`; los de OE2 leen además `runs/*/`
(checkpoints, volcados y `runs/comparacion/comparacion_modelos.json`, que produce
`./evaluation.sh`). `RE_2-3` tarda unos minutos por el costo de inferencia en CPU.

## Compilar

Cada `.tex` es autónomo: carga `preambulo.tex`, sus `figures/RE_x-y/` y `../references.bib`.

```bash
cd research/reportes
tectonic --outdir build RE_1-1_protocolo-curacion.tex          # o:
pdflatex -output-directory build RE_1-1_protocolo-curacion && bibtex build/RE_1-1_protocolo-curacion && pdflatex ... && pdflatex ...
```

Las marcas `[CITA: …]` (rojo) señalan dónde falta una referencia y para qué; `[TODO: …]`
(naranja), decisiones o pasos pendientes. Ambas se definen en `preambulo.tex` y se quitan
redefiniéndolas vacías.
