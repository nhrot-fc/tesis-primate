# Protocolo de comparación

Cómo se ponen Faster R-CNN, AST-Deformable DETR y YOLO en la misma tabla. El código es
[src/evaluation/protocol.py](../src/evaluation/protocol.py); esto son las decisiones y su porqué.

## Pasos

1. **Se vuelcan predicciones, no métricas.** `dump_predictions.py` guarda por ventana cajas,
   scores y etiquetas con `score >= 0.001` y sin tope, para val y test, junto al `best.pt`.
   Todas las cifras salen después de `compare_models.py` sobre esos volcados.
2. **Mismo presupuesto.** `equalize` deja las 100 mejores cajas por ventana (`MAX_DETECTIONS`)
   para todos: torchvision corta en 100 y ultralytics en 300, y sin techo común la cola de la
   curva PR es del framework, no del modelo. Ninguna ventana tiene más de 9 cajas anotadas.
3. **Acierto a IoU 0,3** (`MATCH_IOU`), por clase. mAP@0,3 es la métrica libre de umbral.
4. **Umbral por modelo, elegido en val.** De la rejilla 0,01–0,99, el de mayor recall entre los
   que cumplen la precisión mínima: 0,70 (objetivo declarado) y 0,50 (secundario). Un umbral
   fijo mide calibración, no detección: cada cabeza reparte sus scores a su manera.
5. **Test se mide una vez**, en ese umbral, con IC 95 % por bootstrap de 1 000 remuestreos
   **sobre grabaciones**: ventanas vecinas comparten la mitad del audio y la calidad (SNR,
   distancia, viento) es de la grabación entera. Remuestrear ventanas da IC ~50 % más angostos
   de lo que corresponde.
6. **Las clases de ventana van aparte.** Si la mediana del ancho de caja de una clase ocupa
   ≥ 95 % del clip (`as/hc`, `pt/dc`), detectarla es clasificar la ventana; se reportan como
   recall/precisión de ventana y no entran al macro.

El umbral del primer criterio queda en `runs/<corrida>/operating_point.json`: es con el que
arranca el visor y el que usa `detect.py`.

## Por qué IoU 0,3

Se mide si la llamada se **encontró**, no si la caja quedó ajustada: el eje de frecuencia lo
dibuja el anotador con criterio variable y las llamadas son cortas (mediana 0,4 s; p5 77 ms).
A IoU 0,5 el corrimiento temporal tolerado es un tercio del ancho de la caja, 26 ms en el
decil corto; a 0,3 sube a la mitad. No más abajo: a 0,25 una predicción de banda completa
acertaría un tercio de las cajas sin encuadrar nada (26 % a 0,3, 9 % a 0,5).

## Lo que no está igualado y se declara

- **Receta de entrenamiento**: cada framework con la suya (`train.py: PRESETS`). DETR 30
  épocas, lote 8, lr 2e-4; Faster R-CNN 12 épocas, lote 4, lr 1e-4; YOLO 30 épocas, lote 32
  con el optimizador y la aumentación de imagen de ultralytics.
- **Elección de `best.pt`**: DETR y Faster R-CNN por mAP@0,3 en val, la métrica que después
  se reporta. YOLO por el `fitness` de ultralytics (mAP@0,5:0,95 con su propio validador).
  Igualar el umbral no iguala esto: la diferencia está en qué época se guardó.
- **Preentrenamiento**: el DETR parte del AST (AudioSet) y entrena el transformer desde cero;
  Faster R-CNN y YOLO parten de COCO completo. `experimento_detr_coco.md` iguala esa política.

## Reproducir

```bash
uv run python src/dump_predictions.py --run <corrida>       # una vez por modelo, con GPU
uv run python src/compare_models.py runs/*/*_predictions.pt --output runs/comparacion/comparacion_modelos
```

`./evaluation.sh` hace las dos cosas para todos los `runs/*/best.pt`. Sale un `.txt` legible y
un `.json` con las mismas cifras: mAP@0,3, puntos pareados por criterio con IC, tabla por
clase y clases de ventana.
