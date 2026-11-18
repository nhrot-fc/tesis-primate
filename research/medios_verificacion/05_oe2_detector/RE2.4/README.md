# RE2.4. Modelo final cargable y demostración funcional

**Medio de verificación.** Checkpoint `.pth`, `labels.json` y ejecución de `src/infer.py`.

**Indicador.** El checkpoint se carga sin claves incompatibles y la inferencia sobre un WAV produce un archivo tabulado con las columnas de Raven, incluidas coordenadas tiempo–frecuencia, especie, tipo de llamada y puntuación.

**Dónde lo desarrolla el documento.** Sección 5.5; la demostración está en la sección 5.5.3.

| Elemento del indicador | Dónde se comprueba |
|---|---|
| El checkpoint se carga sin claves incompatibles | `verificacion_carga.txt` |
| La inferencia sobre un WAV produce un archivo tabulado | `transcripcion_detect.txt` (la orden y su salida) y `demo.detections.txt` (la tabla) |
| … con las columnas de Raven, coordenadas tiempo–frecuencia, especie, tipo de llamada y puntuación | Cabecera de `demo.detections.txt` |
| Checkpoint y vocabulario | Versión v0.2.0 del repositorio, archivo `detector-2.0.0-model-yolo26s_v3.zip`; sus sumas, en `SHA256_checkpoint.txt` |
| Punto de operación | `operating_point.json` |

El checkpoint no se guarda aquí: está en https://github.com/nhrot-fc/tesis-primate/releases/tag/v0.2.0, dentro de `detector-2.0.0-model-yolo26s_v3.zip`, con las rutas de `SHA256_checkpoint.txt`. El indicador nombra un `.pth` y `src/infer.py`, nombres del prototipo: el checkpoint final es un `.pt` autocontenido, que lleva su vocabulario, y la inferencia la ejecuta `src/detect.py`.
