# RE2.5. Código y modelo disponibles en un repositorio

**Medio de verificación.** Repositorio de control de versiones.

**Indicador.** La URL del repositorio permite localizar el código de curación, entrenamiento, evaluación e inferencia, junto con las instrucciones de ejecución y los artefactos del modelo final que se hayan decidido publicar.

**Dónde lo desarrolla el documento.** Sección 5.6.

| Elemento del indicador | Dónde se comprueba |
|---|---|
| URL del repositorio | https://github.com/nhrot-fc/tesis-primate |
| Código de curación, entrenamiento, evaluación e inferencia | Tabla de abajo |
| Instrucciones de ejecución | `README.md` del repositorio y el listado de la carpeta RE2.2 |
| Artefactos del modelo final | https://github.com/nhrot-fc/tesis-primate/releases/tag/v0.2.0 (carpeta RE2.4) |

*Tabla 5.19 del documento.*

| Qué | Dónde | Documentado en |
|---|---|---|
| Curación | `src/prepare_annotations.py`, `src/data/annotations.py`, `src/data/species.py` | RE1.1 |
| Conjunto y partición | `src/prepare_data.py`, `src/data/manifest.py`, `src/data/cache.py` | RE1.2, RE1.3 |
| Arquitecturas | `src/models/` | RE2.1 |
| Entrenamiento | `src/train.py`, `src/training/` | RE2.2 |
| Evaluación | `src/dump_predictions.py`, `src/compare_models.py`, `src/evaluation/`, `evaluation.sh` | RE2.1, RE2.3 |
| Inferencia | `src/detect.py`, `src/inference/` | RE2.4 |
| Instrucciones de ejecución | `README.md`, `pyproject.toml`, Listado 5.1 | RE2.2 |
| Modelo final | El *checkpoint* autocontenido de `yolo26s_v3` con su `operating_point.json`, en la versión v0.2.0 | Sección 5.5.1 |
