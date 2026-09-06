# Protocolo de comparación entre los tres detectores

Qué está igualado entre el Deformable-DETR, Faster R-CNN y YOLO26, y qué no. Lo segundo
importa tanto como lo primero: son las diferencias que la tesis tiene que declarar al
presentar la tabla de resultados.

El EAT+DINO (`--arch dino`) entra por el mismo camino que el DETR --mismo `Trainer`, mismo
`TrainConfig`, mismas métricas-- y todo lo de la sección 1 le aplica igual; su
arquitectura y su relación con el paper de referencia están en `docs/eat_dino.md`.

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

Todo esto vive en `src/evaluation/metrics.py` y `src/evaluation/evaluator.py`, y corre
igual para las tres arquitecturas:

| | Valor | Constante |
|---|---|---|
| Umbral de score | `>= 0.5` | `TrainConfig.score_threshold` |
| IoU para contar acierto | `>= 0.3` | `MATCH_IOU` |
| Supresión de cajas anidadas | `suppress_nested`, IoU 0.3 | `TrainConfig.nms_iou` |
| mAP reportadas | 0.3, 0.5 y 0.5:0.95 | `MAP_THRESHOLDS` |
| Selección de checkpoint | F-beta con beta=2 | `BETA` |
| Ruido reportado | falsos positivos por hora | `fp_per_hour` |
| Score mínimo al detectar | 0.001 | `SCORE_FLOOR` |
| Tope de detecciones por clip | 64, sólo al comparar | `EQUALIZED_MAX_DET` |

`dump_predictions.py` evalúa cualquier checkpoint desde `data/processed/*.pt`, incluido
el de YOLO: su adaptador rehace el PNG en memoria con la misma función que usó el
exportador. Verificado que las imágenes salen idénticas salvo 6 píxeles por millón que
difieren en un nivel de gris (redondeo float32 contra float64).

### Por qué el IoU de acierto es 0.3 y no 0.5

Lo que la tesis quiere medir es si la llamada se **encontró**, no si la caja quedó
ajustada. El eje de frecuencia lo dibuja el anotador con criterio variable, y las llamadas
son cortas. Sobre las 7 985 cajas de test:

| | ancho (tiempo) | alto (frecuencia) |
|---|---|---|
| p5 | 0.026 (77 ms) | 0.050 |
| mediana | 0.134 (401 ms) | 0.199 |
| p95 | 1.000 | 0.622 |

Con IoU 0.5 y encuadre frecuencial perfecto, el corrimiento temporal tolerado es el 33%
del ancho de la propia caja: 134 ms en la mediana y **26 ms en el decil corto**. Eso es
resolución de encuadre, y castiga sistemáticamente a las clases de llamada breve, que son
las que la tesis necesita detectar. A IoU 0.3 la tolerancia sube al 54% del ancho.

**Por qué 0.3 y no 0.25.** Bajar más degrada la métrica hasta volverla ganable con cajas
gordas. Una predicción de banda completa --tiempo exacto, toda la frecuencia-- acertaría:

| IoU de acierto | cajas que una predicción de banda completa acertaría |
|---|---|
| 0.25 | 34% |
| 0.30 | 26% |
| 0.50 | 9% |

A 0.25 se regala un tercio de las cajas sin encuadrar nada. Además, `mAP50` y `mAP50-95`
se siguen reportando al lado de `mAP30` justamente para que engordar cajas no salga
gratis: para subir las tres hay que detectar **y** encuadrar.

### Por qué beta = 2

`beta**2` es la razón de costos entre un falso negativo y un falso positivo. **beta=2
declara que perderse una llamada cuesta como cuatro falsos positivos**: descartar un
recorte de 3 s son segundos de revisión, y una llamada perdida no se recupera sin volver
al campo. Con beta=3 la razón sería 9 y la F deja de distinguir un detector útil de uno
que dispara a todo.

Es una declaración de preferencia entre recall y precisión, **no** un ajuste de encuadre
(eso es `MATCH_IOU`) ni un control del ruido (eso es el umbral de score). Los tres ejes se
mueven por separado y conviene no confundirlos al justificar la elección.

### Cómo se audita el ruido: `fp_per_hour`

El beta dice cuánto se valora el recall; `fp_per_hour` dice qué costó, en la unidad en la
que alguien decide: cuántos recortes hay que descartar por hora de audio.

Se calcula sobre el audio **procesado** (`n_ventanas * clip_len_s`), no sobre audio de
campo: el split lleva un 25% de ventanas vacías (`EMPTY_RATIO`) mientras una grabación
real es casi toda vacía. **No es la tasa de campo y no hay que reportarla como tal.** Es
la misma vara para los tres detectores, que es para lo que sirve.

### Dos decisiones de igualación que parecen arbitrarias y no lo son

**`box_score_thresh = 0.001` en Faster R-CNN.** Torchvision descarta internamente todo lo
que baje de 0.05. Con el default, la cola de la curva de precisión-recall de Faster R-CNN
quedaba cortada mientras el DETR y YOLO reportaban hasta 0.001, y su AP salía
subestimado por un default heredado. No subir ese valor de vuelta.

**`EQUALIZED_MAX_DET = 64`.** El DETR no puede emitir más cajas que sus queries (64);
torchvision corta en 100 y ultralytics en 300. Sin un techo común, los dos últimos
consiguen una cola de AP más larga por diseño y no por mérito. Ninguna ventana del
dataset tiene más de 9 cajas anotadas, así que el recorte no pierde nada real. Lo aplica
`protocol.equalize` al comparar, no `dump_predictions.py`: el volcado guarda todo y el
recorte es una decisión de la comparación.

## 2. Diferencias arquitectónicas inherentes

No se pueden igualar sin desvirtuar el modelo; se declaran.

| | DETR | Faster R-CNN | YOLO26 |
|---|---|---|---|
| Parámetros | 89.6 M (3.1 M entrenables) | 43.4 M | 10.0 M |
| Preentrenamiento | AST sobre AudioSet, congelado | COCO | COCO |
| Cajas candidatas | 64 queries | RPN + 100 propuestas | 300 (end-to-end) |
| NMS propio | ninguno | sí, IoU 0.5 | ninguno (end-to-end) |
| Entrada al modelo | mel 1 x 128 x 331 | 3 x 396 x 1024 | 3 x 512 x 512 |

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
| Épocas | 30 | 30 | 50 (con `patience=30`) |
| Batch | 16 | 8 | 32 |
| Optimizador | AdamW 2e-4 + OneCycle | AdamW 1e-4 + OneCycle | `auto` de ultralytics + cos_lr |
| Jitter de cajas | sí | sí | no |
| Aumentación de imagen | ninguna | ninguna | translate 0.05, scale 0.2, brillo 0.2 |
| **Elección del `best`** | F-beta beta=2 | F-beta beta=2 | **mAP@0.5:0.95** |

### La que más pesa: cómo se elige el checkpoint

DETR y Faster R-CNN guardan la época con mejor F-beta sobre validación, medida al punto de
operación (score 0.5, IoU 0.3). YOLO deja esa decisión a ultralytics, cuyo `fitness` es
(verificado en 8.4.135, `ultralytics/utils/metrics.py:1009`):

```python
w = [0.0, 0.0, 0.0, 1.0]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
```

es decir, **mAP@0.5:0.95 puro**: una métrica dominada por la precisión de localización de
la caja, indiferente al punto de operación y mucho menos sensible al recall. Ni siquiera
es la mAP de acá: el validador interno de ultralytics corre con `conf=0.001`,
`max_det=300`, por clase y sin `suppress_nested`.

El sesgo se compone: DETR y Faster R-CNN se seleccionan por **la métrica exacta que
después se reporta, al umbral exacto que después se usa**, así que quedan optimizados para
ser permisivos en 0.5. YOLO no se selecciona por ninguna de las dos cosas.

**No es que YOLO detecte peor: es que su checkpoint se eligió con otro criterio.** Al
comparar, decirlo.

Si en algún momento se quiere cerrar esa brecha, hay dos caminos:

1. Un callback `on_fit_epoch_end` en `train_yolo.py` que evalúe con
   `src/evaluation/evaluator.py` y guarde por F-beta. Correcto, pero agrega una pasada de
   validación por época.
2. Entrenar con `save_period=1` y elegir después, offline, volcando cada época con
   `dump_predictions.py`. Cuesta unos 2 GB de checkpoints y una evaluación por época.

### El umbral fijo de 0.5 no es el mismo punto de operación para los tres

Este es el confundido que queda abierto, y hay que declararlo hasta que se cierre. Cortar
las tres curvas en score 0.5 no las evalúa en el mismo lugar de su curva
precisión-recall: mide **calibración**, no capacidad de detección. Las cabezas de
clasificación son distintas (BCE sobre asignación TAL en YOLO, softmax sobre propuestas en
Faster R-CNN, queries en el DETR) y no hay ninguna razón para que 0.5 signifique lo mismo
en las tres.

Medido sobre test con el protocolo **anterior** (IoU 0.5, checkpoints de
`checkpoints/*_test_metrics.json`), recalculando la F-beta a distintos beta:

| run | R | P | F1 | F1.5 | F2 | F3 | mAP50 | mAP50-95 | dets >= 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| yolo | 0.532 | **0.805** | **0.640** | 0.594 | 0.570 | 0.550 | **0.520** | **0.264** | 5 277 |
| detr ts10 | 0.682 | 0.499 | 0.576 | 0.613 | 0.635 | 0.658 | 0.347 | 0.138 | 10 914 |
| detr ts5 | 0.694 | 0.517 | 0.592 | 0.628 | 0.650 | 0.671 | 0.373 | 0.143 | 10 733 |
| frcnn | **0.777** | 0.499 | 0.608 | 0.663 | 0.699 | 0.736 | 0.487 | 0.211 | 12 447 |

**El ranking se invierte exactamente cuando beta cruza 1.** YOLO gana mAP50, mAP50-95,
precisión y F1, y pierde toda F con beta > 1. No detecta peor: está parado en otro punto
de la curva, y emite 5 277 cajas sobre 0.5 contra las 12 447 de Faster R-CNN sobre el
mismo split. Subir el beta agrava esto, porque el recall a umbral fijo *es* una medición
de calibración y cuanto más alto el beta más la domina.

**El cierre correcto es elegir el umbral por modelo sobre validación** --el que maximiza
F-beta, o el que la maximiza sujeto a un presupuesto de `fp_per_hour`-- y reportar test en
ese punto. Recién ahí el beta mide preferencia ecológica y no calibración. La fila a
umbral 0.5 fijo se puede conservar como secundaria. Eso es lo que hace la sección 4.

### La segunda: regularización asimétrica

La asimetría va en las dos direcciones, así que no favorece obviamente a nadie:

- YOLO es el único que ve aumentación de imagen (corrimiento, escala, brillo).
- DETR y Faster R-CNN son los únicos que ven jitter de cajas
  (`BoxJitter`: ±15% de escala y ±10% de corrimiento en los dos ejes, `min_size=0.02`), un
  regularizador sobre las etiquetas que ultralytics no expone.

## 4. El protocolo de comparación

Las secciones anteriores describen cómo se evalúa **un** modelo. Ponerlos a todos en la
misma tabla es otro problema, y evaluar cada corrida por separado no lo resuelve: cada una
escribe sus métricas con el tope de detecciones de su framework y su propio umbral, y después las
cifras se comparan como si fueran el mismo experimento. No lo son.

La medición que lo muestra, sobre las exportaciones que hay hoy en `checkpoints/`:

| modelo | det/ventana | cajas >= 0.5 | TP | cajas por TP | IoU de la evaluación |
|---|---|---|---|---|---|
| FRCNN | **22.0** | 12 447 | 6 211 | 2.00 | 0.5 |
| DETR ts5 | 8.0 | 10 733 | 5 549 | 1.93 | 0.5 |
| DETR ts10 | 6.0 | 10 914 | 5 446 | 2.00 | 0.5 |
| EAT+DINO t16 | 13.1 | 5 811 | 4 678 | 1.24 | **0.3** |
| EAT+DINO t2 | 3.9 | 5 808 | 4 757 | 1.22 | **0.3** |
| YOLO26 | **3.1** | 6 123 | 5 021 | 1.22 | **0.3** |

El presupuesto de detecciones va de 3.1 a 22.0 por ventana --un factor 7-- y el IoU de
evaluación cambia entre grupos. Con eso, el mAP de YOLO está subestimado y el de Faster
R-CNN inflado por construcción, y las seis filas no pertenecen a la misma tabla.

### Los seis pasos

**1. Se vuelcan predicciones crudas, no métricas.** `dump_predictions.py` guarda por
ventana `{boxes, scores, labels}` con `score >= 0.001` y **sin tope**, para val y para
test. Todas las cifras de la tesis salen después de un único script sobre esos volcados,
así que cualquier número se puede auditar hasta la caja que lo produjo.

**2. Mismo presupuesto para todos.** `equalize` recorta a `EQUALIZED_MAX_DET = 64`. Es el
tope más bajo de los tres frameworks: el DETR no puede emitir más de 64 cajas ni queriendo,
así que cualquier tope mayor le regala cola de curva PR a los otros dos sin que hayan
detectado nada más. Ninguna ventana del dataset tiene más de 9 cajas anotadas.

**3. Un solo IoU primario.** 0.3 como principal (la justificación está en la sección 1) y
0.5 y 0.5:0.95 al lado, en la misma fila y para todos los modelos.

**4. El umbral se elige en val, con cuatro criterios.** Barrerlo sobre test y reportar el
mejor es ajustar un hiperparámetro contra el conjunto de reporte. Los cuatro criterios
responden preguntas distintas y por eso se reportan los cuatro: `precision >= 0.70` (el
objetivo declarado), `precision >= 0.50` (donde opera hoy el Faster R-CNN), `FP/h <= 100`
(el presupuesto de quien revisa) y `max F-beta` (con el que se eligieron los checkpoints).
Entre los umbrales que cumplen se toma el de mayor recall, no el más bajo: la precisión no
es monótona en el umbral y "el más bajo que cumple" premia el ruido de un punto aislado.

**5. IC 95% por bootstrap sobre grabaciones, no sobre ventanas.** Con ventanas de 3 s y
salto de 1.5 s, dos vecinas comparten la mitad del audio y suelen contener la misma
vocalización; además la calidad de la grabación (SNR, distancia, viento) es de la grabación
entera. Remuestrear ventanas da intervalos falsamente angostos: sobre datos sintéticos con
esa correlación adentro, un 49% más angostos (`[0.420, 0.492]` contra `[0.385, 0.527]`).
El mapa ventana -> grabación lo escribe `prepare_data.py` junto con el caché
(`data/processed/<split>_sources.json`). Si falta, `python src/prepare_data.py
--sources-only` lo reescribe sin recalcular un solo mel: el manifiesto es determinista
--misma selección de clases, mismo `SEED`, mismo `split_manifest`-- y lo caro es el
espectrograma, no los encabezados de los audios.

**6. Lo que no es detección se reporta aparte.** `as/hc` y `pt/dc` tienen medianas de 11.0 s
y 6.7 s en ventanas de 3 s, así que su caja mediana ocupa el clip entero (ancho 1.000
contra 0.49 de la siguiente clase). Detectarlas es clasificar la ventana, no localizar un
evento. Van en su propio bloque y no entran al macro; la columna `mAP todas` conserva el
número viejo para poder comparar con las tablas anteriores.

### La tabla que sale

```bash
python src/dump_predictions.py --run <corrida>          # val y test, predicciones crudas
python src/compare_models.py checkpoints/*_predictions.pt
```

`compare_models.py` escribe un `.txt` legible y un `.json` con las mismas cifras, en cuatro
bloques que responden cuatro preguntas distintas:

- **libre de umbral** (mAP@0.3, mAP@0.5, mAP@0.5:0.95): cuál modelo detecta mejor.
- **puntos pareados**, uno por criterio: cuál conviene usar, y con qué IC.
- **descomposición** (`evaluation/decomposition.py`): AP agnóstico de clase, IoU por eje
  sobre lo emparejado, top-1 y acierto de género, y cuánto mAP recupera un oráculo de
  etiquetas y uno de cajas. Responde por qué falla, que las otras dos no.
- **clases de ventana**: recall y precisión de ventana para `as/hc` y `pt/dc`.

### Lo que este protocolo no arregla

Sigue en pie lo de la sección 3: el checkpoint de YOLO se eligió con `fitness` de
ultralytics (mAP@0.5:0.95 puro) y los otros con F-beta al punto de operación. Igualar el
umbral **no** iguala eso, porque la diferencia está en qué época se guardó. Y las recetas
de entrenamiento siguen siendo las de cada framework. Las dos cosas se declaran al
reportar; ninguna se puede leer como "detecta peor".

## 5. Reproducir la comparación

```bash
python src/prepare_data.py                         # data/processed/*.pt (fuente única)
python src/export_yolo.py                          # reexporta a data/yolo/

python src/train.py --arch detr                    # Deformable-DETR
python src/train.py --arch dino                    # EAT + DINO (docs/eat_dino.md)
python src/train.py --arch frcnn --device cuda:0   # Faster R-CNN
python src/train_yolo.py --device 0                # YOLO26

python src/dump_predictions.py --run <corrida>      # los cuatro, mismo comando
python src/compare_models.py checkpoints/*_predictions.pt
```

Cada corrida vive en `runs/<corrida>/`: `config.json` con todo lo que la definió,
`metrics.jsonl` con una línea por época, `train.log`, y dos checkpoints --`best.pt`, el de
mejor F-beta, y `last.pt`, que además lleva optimizador y scheduler--. Relanzar el mismo
comando retoma la corrida donde se cortó; el barrido de la ablación se pide en un solo
comando, p. ej. `python src/train.py --time-stride 10 5 2`.

Un solo script corre el modelo y uno solo reporta. `dump_predictions.py` guarda las cajas
crudas de val y de test al lado del checkpoint; `compare_models.py` escribe un `.txt`
legible y un `.json` con las mismas cifras, a partir de esos volcados y sin volver a mirar
un espectrograma. Todo lo que se reporta --el umbral por modelo, el presupuesto de
detecciones, los intervalos, la tabla por clase-- sale del protocolo de la sección 4, y
cualquier número se puede auditar hasta la caja que lo produjo.

Con un solo modelo el reporte sale igual: es el mismo protocolo, sin nadie con quien
compararlo.

```bash
python src/dump_predictions.py --run <corrida>            # una vez por modelo, con GPU
python src/compare_models.py checkpoints/*_predictions.pt # una vez, con todos, en CPU
```

> Los `checkpoints/*_metrics.txt` y `.json` que están en el repo son del protocolo
> anterior (beta=3, IoU de acierto 0.5, sin `fp_per_hour`) y los dejó un `evaluate.py` que
> ya no existe. Hay que regenerarlos con los dos comandos de arriba antes de usarlos en
> ninguna tabla.
