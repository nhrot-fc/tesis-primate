# Protocolo de comparación entre los tres detectores

Qué está igualado entre el Deformable-DETR, Faster R-CNN y YOLO26, y qué no. Lo segundo
importa tanto como lo primero: son las diferencias que la tesis tiene que declarar al
presentar la tabla de resultados.

## 1. Lo que es idéntico para los tres

### Datos

Los tres consumen las mismas ventanas, con el mismo split por archivo de audio:

| Split | Ventanas | Cajas |
|---|---|---|
| train | 21 966 | 27 022 |
| val | 9 446 | 10 090 |
| test | 8 007 | 7 985 |

`data/processed/*.pt` es la fuente única. `data/yolo/` es una reexportación de ese mismo
caché (ultralytics no acepta tensores en memoria), y `train_yolo.py` aborta si el export
no corresponde a la versión actual del caché, para que no se entrene contra otras clases
sin darse cuenta.

### Punto de operación y métricas

Todo esto vive en `pipelines/metrics.py` y `pipelines/detection_pipeline.py`, y corre
igual para las tres arquitecturas:

| | Valor |
|---|---|
| Umbral de score | `>= 0.5` |
| IoU para contar acierto | `>= 0.5` |
| Supresión de cajas anidadas | `suppress_nested`, IoU 0.3 |
| Umbrales de AP agnóstico | 0.25 y 0.5 |
| Selección de checkpoint | `operating_score` con beta=3 (ver sección 3) |
| Score mínimo al detectar | 0.001 |
| Tope de detecciones por clip | 64 |

`evaluate.py` evalúa cualquier checkpoint desde `data/processed/*.pt`, incluido el
de YOLO: su adaptador rehace el PNG en memoria con la misma función que usó el
exportador. Verificado que las imágenes salen idénticas salvo 6 píxeles por millón que
difieren en un nivel de gris (redondeo float32 contra float64).

### Dos decisiones de igualación que parecen arbitrarias y no lo son

**`box_score_thresh = 0.001` en Faster R-CNN.** Torchvision descarta internamente todo lo
que baje de 0.05. Con el default, la cola de la curva de precisión-recall de Faster R-CNN
quedaba cortada mientras el DETR y YOLO reportaban hasta 0.001, y su AP salía
subestimado por un default heredado. No subir ese valor de vuelta.

**`MAX_DETECTIONS = 64`.** El DETR no puede emitir más cajas que sus queries (64);
torchvision corta en 100 y ultralytics en 300. Sin un techo común, los dos últimos
consiguen una cola de AP más larga por diseño y no por mérito. Ninguna ventana del
dataset tiene más de 9 cajas anotadas, así que el recorte no pierde nada real.

## 2. Diferencias arquitectónicas inherentes

No se pueden igualar sin desvirtuar el modelo; se declaran.

| | DETR | Faster R-CNN | YOLO26 |
|---|---|---|---|
| Parámetros | 89.6 M (3.1 M entrenables) | 43.4 M | 10.0 M |
| Preentrenamiento | AST sobre AudioSet, congelado | COCO | COCO |
| Cajas candidatas | 64 queries | RPN + 100 propuestas | 300 (end-to-end) |
| NMS propio | ninguno | sí, IoU 0.5 | ninguno (end-to-end) |
| Entrada al modelo | mel 1 x 128 x 331 | 3 x 416 x 1024 | 3 x 512 x 512 |

El DETR entrena 3.1 M de parámetros contra los 43.4 M de Faster R-CNN, porque su
backbone AST está congelado. Es una diferencia de capacidad efectiva, no solo de tamaño.

Los tres parten de pesos preentrenados, pero de corpus distintos: el DETR de audio
(AudioSet), los otros dos de fotos naturales (COCO).

## 3. Diferencias de receta de entrenamiento

**Decisión tomada: cada framework corre con su receta estándar.** El objetivo es comparar
pipelines completos, no arquitecturas aisladas. Estas son las diferencias que eso
implica, y hay que enunciarlas al reportar resultados.

| | DETR | Faster R-CNN | YOLO26 |
|---|---|---|---|
| Épocas | 40 | 30 | 100 (con `patience=30`) |
| Batch | 16 | 8 | 32 |
| Optimizador | AdamW 2e-4 + OneCycle | AdamW 1e-4 + OneCycle | `auto` de ultralytics + cos_lr |
| Jitter de cajas | sí | sí | no |
| Aumentación de imagen | ninguna | ninguna | translate 0.05, scale 0.2, brillo 0.2 |
| **Elección del `best`** | `operating_score` beta=3 | `operating_score` beta=3 | **mAP@0.5:0.95** |

### La que más pesa: cómo se elige el checkpoint

DETR y Faster R-CNN guardan la época con mejor `operating_score` beta=3 sobre validación
--una F-beta que pondera el recall nueve veces más que la precisión, medida al punto de
operación 0.5--. YOLO deja esa decisión a ultralytics, cuyo `fitness` es:

```python
w = [0.0, 0.0, 0.0, 1.0]  # [P, R, mAP@0.5, mAP@0.5:0.95]
```

es decir, **mAP@0.5:0.95 puro**: una métrica dominada por la precisión de localización de
la caja, indiferente al punto de operación y mucho menos sensible al recall.

Consecuencia concreta: si la tesis reporta recall al punto de operación, el `best.pt` de
YOLO fue elegido optimizando otra cosa. Puede quedar por debajo de su propio mejor
resultado en la métrica reportada. **No es que YOLO detecte peor: es que su checkpoint se
eligió con otro criterio.** Al comparar, decirlo.

Si en algún momento se quiere cerrar esa brecha, hay dos caminos:

1. Un callback `on_fit_epoch_end` en `train_yolo.py` que evalúe con
   `pipelines.detection_pipeline` y guarde por `operating_score`. Correcto, pero agrega
   una pasada de validación por época.
2. Entrenar con `save_period=1` y elegir después, offline, con `evaluate.py`. Cuesta
   unos 2 GB de checkpoints y una evaluación por época.

### La segunda: regularización asimétrica

La asimetría va en las dos direcciones, así que no favorece obviamente a nadie:

- YOLO es el único que ve aumentación de imagen (corrimiento, escala, brillo).
- DETR y Faster R-CNN son los únicos que ven jitter de cajas
  (`BoxJitter(scale=0.15, shift=0.10, min_size=0.02)`), un regularizador sobre las
  etiquetas que ultralytics no expone.

## 4. Reproducir la comparación

```bash
python src/prepare_data.py                         # data/processed/*.pt (fuente única)
python src/export_yolo.py                          # reexporta a data/yolo/

python src/train.py --arch detr                    # Deformable-DETR
python src/train.py --arch frcnn --device cuda:0   # Faster R-CNN
python src/train_yolo.py --device 0                # YOLO26

python src/evaluate.py --run <corrida> --split test   # los tres, mismo comando
```

Cada corrida vive en `runs/<corrida>/`: `config.json` con todo lo que la definió,
`metrics.jsonl` con una línea por época, `train.log`, y dos checkpoints --`best.pt`, el de
mejor F-beta, y `last.pt`, que además lleva optimizador y scheduler--. Relanzar el mismo
comando retoma la corrida donde se cortó; el barrido de la ablación se pide en un solo
comando, p. ej. `python src/train.py --time-stride 10 5 2`.

`evaluate.py` escribe un `.txt` legible y un `.json` con las mismas cifras, al lado del
checkpoint. Las métricas son las mismas para los tres: recall y precisión al punto de
operación, la F-beta (beta=3) que elige el checkpoint, y mAP@0.5 y mAP@0.5:0.95.
