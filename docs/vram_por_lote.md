# Peso de los datos y VRAM por lote

Números medidos sobre este repo (no estimados de tabla) para poder calcular a mano
cuánta VRAM pide cada detector antes de lanzar una corrida.

Todo en MiB (`2^20` bytes) salvo donde diga MB. Todos los tensores en `float32`.

## 1. Una unidad de cada dataset

Los tres modelos ven las **mismas ventanas**: 3 s de audio, mel de 128 bandas x 331
tramas. Lo que cambia es en qué formato llega.

| | DETR | Faster R-CNN | YOLO |
|---|---|---|---|
| Fuente | `data/processed/*.pt` | `data/processed/*.pt` | `data/yolo/images/` |
| Unidad en disco | 165.5 KiB | 165.5 KiB | 119.0 KiB (PNG) |
| Tensor que entra al modelo | `1 x 128 x 331` | `3 x 416 x 1024` | `3 x 512 x 512` |
| **Bytes de esa entrada** | **0.162 MiB** | **4.88 MiB** | **3.00 MiB** |

Notas sobre cada uno:

- **DETR**: el `.pt` guarda el mel de potencia crudo, `1 x 128 x 331 x 4 B = 169 472 B`.
  El modelo lo consume tal cual.
- **Faster R-CNN**: lee el mismo `.pt`, pero `SpectrogramFasterRCNN.to_images` lo pasa a
  3 canales y el `GeneralizedRCNNTransform` de torchvision lo reescala con
  `min_size=512, max_size=1024` -> `396 x 1024`, y al agrupar el lote rellena a
  múltiplos de 32 -> **`416 x 1024`**. O sea: la entrada real es **30x más pesada que el
  mel original**, y es el modelo el que decide ese tamaño, no el dataset.
- **YOLO**: el PNG de 512x512 pesa 119 KiB en disco, pero se decodifica a
  `512 x 512 x 3` uint8 (0.75 MiB) y se normaliza a float32 -> **3.00 MiB**.

Cajas y etiquetas son ruido a esta escala: 1.07 cajas por ventana de promedio (máx. 9),
menos de 1 MiB para un split entero.

### Totales por split

| Split | Ventanas | `data/processed` | `data/yolo` (PNG) |
|---|---|---|---|
| train | 21 966 | 3 561 MiB | 2 553 MiB |
| val | 9 446 | 1 531 MiB | 1 096 MiB |
| test | 8 007 | 1 298 MiB | 931 MiB |

`data/processed` se carga **entero a RAM** (`CachedCallBoxDataset` hace `torch.load` de
todo el `.pt`): entrenar el DETR o Faster R-CNN necesita ~5 GiB de RAM de sistema solo
para train+val, antes de tocar la GPU. YOLO no: ultralytics lee los PNG de disco por
lote.

## 2. Dónde se va la VRAM

```
VRAM ≈ COSTO_FIJO + B x COSTO_POR_MUESTRA
```

El costo fijo no depende del batch; el otro sí, y es el que manda.

### Costo fijo: pesos + gradientes + estado del optimizador

Con AdamW (2 estados por parámetro entrenable) y todo en fp32:

```
fijo = params_totales x 4 B  +  params_entrenables x 4 B  +  params_entrenables x 8 B
       └── pesos ──────────┘    └── gradientes ────────┘    └── momentos de Adam ──┘
```

| Modelo | Params totales | Entrenables | Pesos | Grads | AdamW | **Fijo** |
|---|---|---|---|---|---|---|
| DETR (dim=128, 64 queries) | 89.58 M | 3.07 M | 341.7 | 11.7 | 23.4 | **377 MiB** |
| Faster R-CNN R50-FPN v2 | 43.38 M | 43.2 M | 165.5 | 164.8 | 329.6 | **660 MiB** |
| YOLO26n | 2.51 M | 2.51 M | 9.6 | 9.6 | 19.1 | **38 MiB** |
| YOLO26s | 9.97 M | 9.97 M | 38.0 | 38.0 | 76.1 | **152 MiB** |
| YOLO26m | 21.81 M | 21.81 M | 83.2 | 83.2 | 166.4 | **333 MiB** |

Al fijo de YOLO **sumale una copia más de los pesos**: ultralytics mantiene un modelo
EMA (`+9.6 / +38.0 / +83.2 MiB` según el tamaño).

El DETR tiene 89 M de parámetros pero solo 3 M entrenables (el AST está congelado), así
que su costo fijo es bajísimo comparado con su tamaño.

### Costo por muestra: entrada + activaciones

| Modelo | Entrada | Activaciones (cota superior) |
|---|---|---|
| DETR | 0.162 MiB | ~1 193 MiB |
| Faster R-CNN | 4.88 MiB | ~1 668 MiB |
| YOLO26s | 3.00 MiB | ~299 MiB |

> **Leé esto antes de usar la columna de activaciones.** Es la suma de *todas* las
> salidas intermedias de todos los módulos en un forward de entrenamiento con `B=1`,
> medida con hooks. Es una **cota superior**, no el pico real: PyTorch libera y reutiliza
> buffers, así que el máximo simultáneo suele ser **2-3x menor**. Sirve para comparar
> modelos entre sí y para saber que no te vas a pasar; para el número exacto, medilo
> (sección 4).

Dos cosas que explican esos números:

- El DETR gasta muchísimo para tener solo 3 M de parámetros entrenables porque el AST
  congelado **igual guarda sus activaciones**: `ASTBackbone.forward` no usa `no_grad` a
  propósito (ver el comentario en [`backbone.py`](../src/architectures/backbone.py)),
  porque la proyección de parches sí se entrena y un `no_grad` cortaría la cadena.
- Faster R-CNN es el más caro por el tamaño de entrada: `416 x 1024` contra `512 x 512`
  de YOLO es 1.6x más píxeles, y encima ResNet50-FPN es una red más pesada que YOLO26s.

## 3. Cuenta de servilleta

Faster R-CNN con `--batch 8`:

```
660 MiB  +  8 x (4.88 + 1668) MiB  ≈  660 + 13 384  ≈  13.7 GiB   (cota superior)
                                                       ~5-7 GiB   (pico real esperado)
```

YOLO26s con `--batch 32`:

```
152 + 38 (EMA)  +  32 x (3.00 + 299) MiB  ≈  190 + 9 664  ≈  9.6 GiB  (cota superior)
                                                             ~3-4 GiB (pico real esperado)
```

Reglas prácticas que salen de la tabla:

- **AMP (`amp=True`, que ya está activo en YOLO) baja las activaciones casi a la mitad**,
  porque pasan a fp16. No toca el costo fijo (los pesos maestros siguen en fp32).
- Duplicar el batch duplica las activaciones. Si no entra, bajá el batch antes que la
  resolución: `imgsz` entra al cuadrado.
- En Faster R-CNN, bajar `MIN_SIZE/MAX_SIZE` en
  [`faster_rcnn.py`](../src/architectures/faster_rcnn.py) es la palanca más fuerte: pasar
  de `512/1024` a `384/768` deja la entrada en `320 x 768` y recorta píxeles y
  activaciones a ~58 %.

## 4. Medirlo de verdad (30 segundos en la máquina con GPU)

La cota superior de arriba sirve para dimensionar; esto da el número exacto. Corré con
dos batches distintos y despejá la recta:

```python
import torch
from domain.dataset import CachedCallBoxDataset, collate_fn
from architectures.faster_rcnn import SpectrogramFasterRCNN, to_torchvision_targets

ds = CachedCallBoxDataset("data/processed/val.pt")
model = SpectrogramFasterRCNN(n_classes=25, db_low=-54.68, db_high=29.29).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for B in (1, 2, 4):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    images, targets = collate_fn([ds[i] for i in range(B)])
    images = images.cuda()
    targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
    losses = model(images, to_torchvision_targets(targets))
    torch.stack(list(losses.values())).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"B={B}: pico {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
```

Con dos puntos cualesquiera:

```
por_muestra = (pico(B2) - pico(B1)) / (B2 - B1)
fijo        = pico(B1) - por_muestra x B1
batch_maximo = (VRAM_disponible x 0.9 - fijo) / por_muestra
```

El `0.9` deja aire para la fragmentación del allocator; sin ese margen se cae con OOM
cerca del límite teórico.

Para YOLO no hace falta nada de esto: ultralytics acepta `batch=-1` (o un float entre
0.0 y 1.0) y calibra solo el lote más grande que entra en la GPU.

## Cómo se obtuvieron estos números

- Tamaños de dataset: recorriendo `data/processed/*.pt` y `data/yolo/images/*`.
- Forma de la entrada de Faster R-CNN: ejecutando su `model.transform` sobre un lote real.
- Parámetros: `sum(p.numel() for p in model.parameters())` sobre cada modelo con 25 clases.
- Activaciones: `register_forward_hook` sobre cada módulo hoja, sumando los bytes de las
  salidas en un forward de entrenamiento con `B=1`, en CPU.

Si cambiás `Parameters` en [`core/config.py`](../src/core/config.py) (`clip_len_s`,
`n_mels`, `hop_length`) o `IMAGE_SIZE` en
[`create_yolo_dataset.py`](../src/create_yolo_dataset.py), todos estos números se mueven.
