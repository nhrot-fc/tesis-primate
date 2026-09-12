# Requisitos de hardware

Para correr el detector de vocalizaciones en un equipo propio (visor de escritorio o
procesamiento por lotes). Se distribuye como carpeta autocontenida: no hay nada más que
instalar.

## Resumen

| | Mínimo (sólo CPU) | Recomendado (GPU) |
|---|---|---|
| Procesador | 4 núcleos x86-64 | 6+ núcleos x86-64 |
| Memoria RAM | 8 GB (6 GB libres al procesar) | 16 GB |
| Tarjeta gráfica | no hace falta | NVIDIA con **4 GB+ de VRAM** (GTX 1650 / RTX 3050 o superior) y driver actualizado¹ |
| Disco libre | 3 GB (paquete CPU) | 7 GB (paquete GPU) |

¹ Sólo el driver de NVIDIA; el paquete trae sus propias librerías CUDA.

## Tiempo de procesamiento por modelo

Medido en CPU con 8 hilos sobre una grabación de 108 s a 44,1 kHz. En un portátil
corriente esperá entre 1,5 y 2 veces más; con GPU los tres modelos procesan una hora de
audio en menos de un minuto.

| Modelo | Tamaño | Velocidad en CPU | RAM que usa | 1 h de audio en CPU |
|---|---|---|---|---|
| Faster R-CNN (el más preciso) | 166 MB | 1,4× tiempo real | 4,8 GB | ~45 min |
| AST-Deformable DETR | 340 MB (+ 380 MB de backbone) | 11× tiempo real | 1,5 GB | ~6 min |
| YOLO26s (el más rápido) | 39 MB | 42× tiempo real | 1,4 GB | ~1,5 min |

Referencia de tamaños: un WAV mono de 44,1 kHz y 16 bits ocupa **~5,3 MB por minuto**, así que
250 MB son unos 47 minutos de grabación (24 en estéreo).

## Disco por paquete

| Paquete | Descomprimido | Para quién |
|---|---|---|
| CPU | ~2 GB | cualquier PC de 64 bits |
| GPU (CUDA) | ~6 GB | PC con NVIDIA; también funciona sin GPU, usando la CPU |

Ambos incluyen el intérprete, las librerías, los modelos y el backbone. Extraer en una ruta
corta (p. ej. `C:\detector\`): las rutas largas de Windows dan problemas al descomprimir.
