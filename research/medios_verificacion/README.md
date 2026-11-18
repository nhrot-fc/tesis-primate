# Medios de verificación

La evidencia de los once resultados esperados de la tesis, por capítulo y por resultado (Anexo B del documento). Cada carpeta tiene un `README.md` con el medio de verificación y el indicador copiados del documento, la sección que desarrolla el resultado y una tabla que lleva cada elemento del indicador a su archivo.

| RE | Carpeta | Archivos |
|---|---|---|
| RE1.1 | `04_oe1_curacion/RE1.1/` | `README.md`, `correcciones_pares.csv`, `reglas.csv`, `sinonimos.csv`, `vocabulario.csv` |
| RE1.2 | `04_oe1_curacion/RE1.2/` | `README.md`, `grabaciones_entrenamiento.txt`, `grabaciones_prueba.txt`, `grabaciones_validacion.txt`, `labels.json`, `meta.json`, `registro_de_carga.txt` |
| RE1.3 | `04_oe1_curacion/RE1.3/` | `README.md` |
| RE2.1 | `05_oe2_detector/RE2.1/` | `README.md` y subcarpetas |
| RE2.2 | `05_oe2_detector/RE2.2/` | `README.md`, `pyproject.toml` y subcarpetas |
| RE2.3 | `05_oe2_detector/RE2.3/` | `README.md`, `comparacion_modelos_v3.json`, `comparacion_modelos_v3.txt`, `matriz_confusion.png` y subcarpetas |
| RE2.4 | `05_oe2_detector/RE2.4/` | `README.md`, `SHA256_checkpoint.txt`, `demo.detections.txt`, `operating_point.json`, `transcripcion_detect.txt`, `verificacion_carga.txt` |
| RE2.5 | `05_oe2_detector/RE2.5/` | `README.md` |
| RE3.1 | `06_oe3_revision/RE3.1/` | `README.md`, `protocolo.md` |
| RE3.2 | `06_oe3_revision/RE3.2/` | `README.md`, `grabaciones_sesion.csv`, `test_cpu_times.csv`, `tiempos_sesion.csv` |
| RE3.3 | `06_oe3_revision/RE3.3/` | `README.md`, `respuestas_revision.csv`, `revision_ejemplos.png` |

No contiene las grabaciones, las tablas de Raven ni el caché de ventanas, que pertenecen al equipo de investigación, ni los nombres de quienes revisaron.

Se regenera con `make` y después `make medios`, desde la raíz del repositorio. `generar.py` lee las salidas de los cuadernos de los capítulos, las constantes del código de curación, las corridas de `runs/` y las planillas que devolvió el equipo. Las dos últimas no están en git.
