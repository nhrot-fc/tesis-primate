# Capítulo 5 — Detección y Evaluación del Modelo (OE2)

> **OE2.** Implementar y evaluar un detector que reciba un espectrograma y
> devuelva cajas tiempo–frecuencia con especie, tipo de llamada y confianza,
> comparándolo con una línea base establecida.

Sólo RE2.3 está redactado. Los otros cuatro resultados existen como reporte
aparte (`research/reportes/`) y como código, pero no en el cuerpo del documento.

## Qué va aquí

| Sección | Contenido | RE |
|---|---|---|
| Introducción | Enuncia OE2 y anuncia las cinco secciones de resultado | — |
| Diseño experimental, arquitectura propuesta y línea base | AST-Deformable-DETR (dimensión 128, 64 consultas, 3 niveles), hiperparámetros y la línea base declarada | **RE2.1** |
| Pipeline reproducible de entrenamiento, evaluación e inferencia | Preprocesamiento, pérdida, emparejamiento, selección de checkpoint, comandos | **RE2.2** |
| Protocolo de evaluación | Los tres ejes —detección, encuadre, clasificación— y por qué se puntúan por separado | (debería ser la primera subsección de RE2.3) |
| Informe comparativo de resultados | Configuraciones comparadas, punto de operación, comparación a carga igual, lectura por clase, qué establece la comparación | **RE2.3** |
| Modelo final cargable y demostración funcional | Checkpoint, `labels.json`, exportación a tabla de selección de Raven | **RE2.4** |
| Código y modelo disponibles en un repositorio | URL, contenido publicado, licencia, comandos de reproducción | **RE2.5** |
| Discusión | Las cinco preguntas de la pauta | — |

## Cómo se redacta

- Cada sección de resultado cierra con su verificación: RE2.2 se verifica
  ejecutando el pipeline, RE2.4 cargando el checkpoint y abriendo la salida en
  Raven, RE2.5 con la URL del repositorio. Describir el resultado sin decir cómo
  se comprobó no cumple la pauta.
- La comparación se reporta sobre **tres ejes separados** —detección, encuadre y
  clasificación—, porque el IOV de RE2.3 los exige por separado y porque el
  hallazgo del proyecto es justamente que la clasificación no es el problema.
- Las cifras no se escriben a mano: salen de `runs/comparacion/*.json` y de los
  volcados de predicciones.
- El punto de operación de este capítulo (umbral elegido en validación) **no es**
  el que compromete RE3.1 (confianza 0,5). Son dos preguntas distintas y el
  documento tiene que decir cuál usa en cada sitio.

## Imágenes

De `notebook/figuras_05_oe2_detector.ipynb`.

| Figura | Qué muestra | Estado |
|---|---|---|
| `architecture.png` | Arquitectura propuesta | vigente |
| `multiscale_pyramid.png`, `deformable_sampling.png`, `decoder_layer.png` | Pirámide multiescala, muestreo deformable, capa del decodificador | vigente |
| `training_pipeline.png`, `inference_pipeline.png`, `hungarian_matching.png` | Los dos pipelines y el emparejamiento húngaro | vigente |
| `score_sweep.png` | Barrido del umbral de confianza | vigente |
| `recall_per_class.png`, `confusion_matrix.png`, `iou_distribution.png`, `ablation_progression.png`, `qualitative_detections.png`, `detection_timeline.png` | Resultados de la evaluación | **quedaron de una evaluación DETR anterior cuyo checkpoint se perdió**: hay que regenerarlas con la comparación vigente |
| `raven_table_preview.png` | Tabla exportada abierta en Raven | captura manual; es la prueba visual de RE2.4 |
| `training_curves.png`, `comparison_map.png`, `comparison_paired.png`, `comparison_per_class.png` | Curvas de entrenamiento y comparación entre modelos | **generadas pero sin usar en el `.tex`**: son las que deberían sustituir a las figuras viejas |

Para regenerarlas hacen falta las corridas en `runs/` (`metrics.jsonl`,
`*_predictions.pt` y `runs/comparacion/*.json`, que produce `./evaluation.sh`).

## Extra

- `protocolo_comparacion.md` — cómo se ponen Faster R-CNN, AST-Deformable DETR y
  YOLO en la misma tabla: presupuesto igual de cajas, IoU de acierto, umbral
  elegido en validación, test medido una vez. Venía de `docs/`; el código es
  `src/evaluation/protocol.py`.
- `experimento_detr_coco.md` — el experimento que iguala la política de
  inicialización (Deformable DETR con pesos COCO sobre el AST). Alimenta la
  discusión de RE2.3 sobre arquitectura frente a inicialización.
- **Material por trasladar desde `research/reportes/`**:
  - `RE_2-1_diseno-arquitectura.tex` → formulación, arquitectura completa,
    líneas base y diseño experimental. Es lo que hoy falta entero.
  - `RE_2-2_pipeline.tex` → esquema, instalación, datos, entrenamiento,
    evaluación, inferencia, reproducibilidad y mapa del código.
  - `RE_2-3_informe-comparativo.tex` → contrastar con lo ya redactado aquí y
    quedarse con una sola versión.
  - `RE_2-4_modelo-final.tex` → modelo final, formato de salida, demostración y
    visor.
  - `RE_2-5_repositorio.tex` → repositorio, contenido publicado y no publicado,
    licencia, versionado.

## Pendientes

- [ ] Redactar RE2.1, RE2.2, RE2.4 y RE2.5 (hoy son `% TODO`).
- [ ] Regenerar las figuras de resultados con la comparación vigente y retirar
      las que vienen del DETR perdido.
- [ ] Decidir qué pasa con el prototipo YOLOv2 del informe E3: o entra como
      línea base explícita, o se menciona sólo en conclusiones como iteración
      descartada.
- [ ] Convertir «Protocolo de evaluación» en la primera subsección de RE2.3.
- [ ] Redactar la discusión.
