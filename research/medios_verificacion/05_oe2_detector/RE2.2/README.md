# RE2.2. Pipeline reproducible de entrenamiento, evaluación e inferencia

**Medio de verificación.** Scripts de datos, entrenamiento, evaluación e inferencia; módulos de `src/pipelines/`; `pyproject.toml` y manual técnico.

**Indicador.** El pipeline genera los splits procesados, entrena desde un conjunto de configuración, evalúa un checkpoint y ejecuta inferencia sobre un WAV; las dependencias y comandos de ejecución están declarados.

**Dónde lo desarrolla el documento.** Sección 5.3; la sección 5.3.6 contrasta el indicador.

| Elemento del indicador | Dónde se comprueba |
|---|---|
| Genera los splits procesados | `prepare_annotations.py` y `prepare_data.py` (listado de abajo); su resultado, en la carpeta RE1.2 |
| Entrena desde un conjunto de configuración | `train.py --arch`; un registro por corrida en `registros/`, con sus métricas por época |
| Evalúa un checkpoint | `dump_predictions.py` y `compare_models.py`; su resultado, en la carpeta RE2.3 |
| Ejecuta inferencia sobre un WAV | `detect.py`; su resultado, en la carpeta RE2.4 |
| Dependencias declaradas | `pyproject.toml` (aquí) y `uv.lock` (raíz del repositorio) |
| Comandos de ejecución declarados | El listado de abajo, que es el Listado 5.1 del documento |

Interfaz del pipeline (scripts de `src/`):

```bash
# 1. datos (capitulo 4)
prepare_annotations.py [--force]     raw/ -> cleaned/
prepare_data.py        [--force]     cleaned/ -> processed/
export_yolo.py                       processed/ -> yolo/ (solo Ultralytics)

# 2. entrenamiento: --arch elige el preset; --hp y --cfg lo pisan
train.py --arch detr  --name detr_t10_logmel --hp frontend=logmel
train.py --arch frcnn --name frcnn_v3
train.py --arch yolo  --name yolo26s_v3 --hp model=yolo26s
train.py --arch detr  --limit 64 --cfg epochs=1      # prueba de humo
notebooks/pretrain_birds_train.py --name detr_birds_v2
notebooks/pretrain_birds_train.py --finetune <aves>/best.pt --name <corrida>

# 3. evaluación: volcar primero, medir después
dump_predictions.py --run <corrida> [--device cuda:1]
compare_models.py runs/*/*_predictions.pt --output <salida>
./evaluation.sh [corridas...]        las dos cosas, solo lo desactualizado

# 4. inferencia
detect.py <audio o carpeta> --model runs/<corrida>/best.pt
```
