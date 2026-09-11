# tesis-primate

Detección de vocalizaciones de primates en espectrogramas: un detector de objetos (Faster
R-CNN, AST + Deformable DETR o YOLO) dibuja cajas tiempo–frecuencia sobre ventanas de 3 s y
las exporta como tablas de selección de Raven. Incluye el visor de escritorio con el que un
analista revisa y corrige esas detecciones.

Los documentos del proyecto (tesis, informes de resultados esperados, protocolo de
comparación) están indexados en [docs/README.md](docs/README.md).

## Instalación

Requiere Python 3.12 y [uv](https://docs.astral.sh/uv/). Las dependencias están en capas
([pyproject.toml](pyproject.toml)):

| Para | Comando |
|---|---|
| Visor + inferencia con Faster R-CNN (despliegue) | `uv sync --no-dev` |
| …y además checkpoints DETR / YOLO | `uv sync --no-dev --extra detr --extra yolo` |
| Preparar datos, entrenar, comparar | `uv sync --group train` |
| Todo (investigación, notebooks, linters) | `uv sync --all-extras --all-groups` |

El backbone AST se descarga de Hugging Face la primera vez y queda en `hf/`.

## Datos

```
data/
  raw/<nombre_comun>__<CODIGO>/*.wav + *.txt   # audio y tablas de Raven, por especie
  cleaned/                                     # anotaciones normalizadas (prepare_annotations.py)
  processed/{train,val,test}.pt + meta.json    # caché de ventanas mel (prepare_data.py)
  yolo/                                        # export PNG para Ultralytics (export_yolo.py)
runs/<corrida>/                                # checkpoints, métricas, volcados, punto de operación
```

Qué es cada especie y cada tipo de llamada: [src/data/species.py](src/data/species.py) y
[docs/call_type_notes.md](docs/call_type_notes.md). Qué hay en el caché actual:
[notebooks/dataset_report.ipynb](notebooks/dataset_report.ipynb).

## Pipeline

```bash
uv run python src/prepare_annotations.py          # raw/ -> cleaned/
uv run python src/prepare_data.py                 # cleaned/ -> processed/ (ventanas, splits, rango dB)
uv run python src/export_yolo.py                  # processed/ -> yolo/ (sólo para YOLO)

uv run python src/train.py --arch frcnn           # o detr | yolo; --hp clave=valor pisa hiperparámetros
uv run python src/train.py --arch detr --hp frontend=logmel --cfg epochs=20

./evaluation.sh                                   # vuelca val/test de cada runs/*/best.pt y compara
uv run python src/compare_models.py runs/*/*_predictions.pt --output runs/comparacion/comparacion_modelos
```

`compare_models.py` elige en val el umbral de cada modelo con la precisión mínima del
protocolo, lo mide una sola vez en test con IC por bootstrap de grabaciones, y deja
`operating_point.json` junto al checkpoint: es el umbral con el que arranca el visor.
`fuse_predictions.py` combina volcados de varios modelos (WBF) en uno que se compara igual.

## Visor

```bash
uv run python src/main.py
```

Abre un audio (o arrastrarlo), un checkpoint (`Ctrl+M`) y detecta (`Ctrl+R`). Muestra las
anotaciones de Raven que estén junto al `.wav`, las detecciones del modelo y, si se carga
una tabla de hallazgos, una cola de revisión con aceptar/rechazar. `F1` lista los
controles. Exporta imágenes del tramo y tablas en formato Raven.

Inferencia sin interfaz: `inference.predictor.predict(loaded, audio)` devuelve la tabla de
Raven de una grabación entera (ventanas solapadas fundidas).

## Código

```
src/
  core/        config.py: rutas, semilla, parámetros de señal y de detección compartidos
  data/        especies y etiquetas, anotaciones, manifest de ventanas, caché, datasets, aumento
  models/      Detector base, registry (arquitecturas, extras), FRCNN, AST-DETR (+PCEN), YOLO
  training/    Trainer (FRCNN/DETR), YOLO vía Ultralytics, checkpoints
  evaluation/  métricas, volcados, protocolo de comparación, reporte
  inference/   predictor sobre grabaciones completas
  viewer/      la aplicación PyQt6
  utils/       audio (mel, dB, ventanas) y cajas (NMS anidado, conversión)
```

Convenciones: cajas `cxcywh` normalizadas a la ventana, `x` es tiempo y `y` frecuencia en
escala mel; etiquetas `especie/llamada` en minúscula. Las constantes que comparten
evaluación, inferencia y visor (umbral por defecto, piso de score, NMS, tope de
detecciones) viven en `core/config.py`.

Calidad: `uv run ruff check src && uv run ruff format src && uv run ty check src`.

## Figuras y notebooks

| Notebook | Qué hace |
|---|---|
| [notebooks/figuras_documento.ipynb](notebooks/figuras_documento.ipynb) | Genera las figuras de la tesis en `research/figures/`, una sección por capítulo del documento |
| [notebooks/dataset_report.ipynb](notebooks/dataset_report.ipynb) | Reporte textual del caché: pares, splits, huella para comparar dos copias |
| [notebooks/TF_Experimental_Training.ipynb](notebooks/TF_Experimental_Training.ipynb) | Línea base histórica (clasificador binario en TensorFlow/Colab). No corre en este entorno; sus figuras están en el informe de agosto |

La tesis se compila con `make` (LaTeX en `research/`).
