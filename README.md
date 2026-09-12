# Detector de vocalizaciones de primates

Un detector de objetos (Faster R-CNN, AST-Deformable DETR o YOLO) dibuja cajas
tiempo–frecuencia sobre el espectrograma de una grabación y las exporta como **tablas de
selección de Raven**, para revisarlas en Raven Pro o en el visor incluido.

Se distribuye como una carpeta portable para Windows: se descomprime y se usa. **No hay que
instalar Python, CUDA ni ninguna librería.**

## Descarga

En [Releases](https://github.com/nhrot-fc/tesis-primate/releases), dos tipos de zip:

| Zip | Qué es | Tamaño |
|---|---|---|
| `detector-<versión>-win64-cpu.zip` | el programa, para cualquier PC de 64 bits | ~0,5 GB |
| `detector-<versión>-win64-cuda.zip` | el programa con soporte NVIDIA; sin GPU usa la CPU | ~3 GB |
| `detector-<versión>-model-<nombre>.zip` | un modelo; hacen falta uno o más | 25 MB – 0,7 GB |

1. Descomprime **un** zip `win64` en una ruta corta, p. ej. `C:\detector\`.
2. Arrastra cada zip `model` sobre `Agregar-modelo.bat` dentro de esa carpeta.

Si un zip viene en partes (`.zip.001`, `.zip.002`, …), descárgalas todas en la misma carpeta
con su `unir.bat` y ejecútalo; deja el zip entero.

## Requisitos de software

| | Necesario | Notas |
|---|---|---|
| Sistema operativo | **Windows 10 u 11, 64 bits** | Sin permisos de administrador: todo queda en la carpeta descomprimida |
| Python, CUDA, Visual C++ | **nada** | Van dentro del paquete |
| Controlador NVIDIA (sólo paquete `cuda`) | serie **570 o más nueva** | Es lo único que el paquete no trae. Se actualiza desde [nvidia.com/drivers](https://www.nvidia.com/drivers). Con controladores desde la serie 527 suele funcionar; si torch no ve la GPU, el paquete sigue en CPU |
| Raven Pro (opcional) | 1.5 o 1.6 | Para revisar las tablas junto al audio. Son texto tabulado: también las abren Excel o LibreOffice |
| Internet | **no** | Ni para instalar ni para detectar |
| 7-Zip (opcional) | — | Sólo para abrir zips en partes sin `unir.bat` |

Detalles de hardware (RAM, GPU, disco y cuánto tarda cada modelo):
[docs/system_requirements.md](docs/system_requirements.md).

Dos avisos de Windows:

- Descomprime en una **ruta corta** (`C:\detector\`): el paquete tiene rutas largas y el
  explorador de Windows a veces falla con más de 260 caracteres.
- El primer arranque es más lento: el antivirus revisa cientos de archivos nuevos. No hay
  ningún `.exe` propio, sólo el `python.exe` firmado por python.org y tres archivos `.bat`.

## Uso

| Archivo | Qué hace |
|---|---|
| `Visor.bat` | Visor de espectrogramas. `Ctrl+O` abre un audio (WAV, FLAC, MP3), `Ctrl+M` un modelo de `models\`, `Ctrl+R` detecta. `F1` lista los controles. Exporta tablas de Raven e imágenes del tramo |
| `Detectar.bat` | Procesa carpetas enteras: arrastra una carpeta (o varios audios) sobre el archivo y deja `<audio>.detections.txt` junto a cada grabación. Con más de un modelo en `models\` pregunta cuál usar |
| `Agregar-modelo.bat` | Arrastra encima el zip de un modelo: queda en `models\<nombre>\` listo para usar. Un modelo por subcarpeta, con su umbral de operación |
| `LEEME.txt` | Esto mismo, dentro del paquete |
| `visor.log` | Aparece si el visor falla; dice por qué |

Con Raven Pro: abre el audio y luego *File → Open Selection Table* con su
`.detections.txt`. La columna `Score` permite filtrar dentro de Raven; el umbral con el que
se escribió la tabla es el que se eligió para ese modelo en la comparación.

## Para desarrolladores

Python 3.12 y [uv](https://docs.astral.sh/uv/). `uv sync --no-dev --extra detr --extra yolo`
corre el visor e inferencia con cualquier checkpoint; `uv sync --all-extras --all-groups` todo
(datos, entrenamiento, notebooks, linters).

```bash
uv run python src/prepare_annotations.py                 # data/raw -> data/cleaned
uv run python src/prepare_data.py                        # -> data/processed (ventanas, splits)
uv run python src/train.py --arch frcnn                  # detr | detr_coco | yolo; --hp k=v, --cfg k=v
./evaluation.sh                                          # vuelca val/test de runs/*/best.pt y compara
uv run python src/main.py                                # visor
uv run python src/detect.py --model runs/frcnn carpeta/  # por lotes
```

Cómo se comparan los modelos: [docs/protocolo_comparacion.md](docs/protocolo_comparacion.md).
Especies y llamadas: [src/data/species.py](src/data/species.py) y
[docs/call_type_notes.md](docs/call_type_notes.md). Informes entregados (PDF): [docs/informes/](docs/informes/).

### Releases

`deploy/build_windows.py` arma en `dist/` (ignorado por git) dos clases de zip, desde Linux:
`runtime --variant cpu|cuda` (Python embebido, librerías, `src/`, lanzadores, `models/` vacía)
y `models runs/<corrida>…` (un zip por modelo con su `models/<corrida>/` y los `hf/` que
necesite; sin carpeta raíz, se vuelca dentro del runtime). Los zips van como *assets* del
release, no al repo; si uno pasa de 1,9 GB sale en partes con un `unir.bat`.

```bash
deploy/release.sh v0.1.0 runs/frcnn runs/yolo26s_coco          # runtime cpu + modelos + gh release
VARIANTS="cpu cuda" deploy/release.sh v0.1.0 runs/frcnn         # también el runtime cuda
REPO=usuario/repo-privado deploy/release.sh v0.1.0 runs/frcnn   # publicar en otro repo
```

Necesita `gh` autenticado. Antes de publicar, probar el zip en un Windows limpio (sin Python,
sin red): `Agregar-modelo.bat`, `Visor.bat`, `Detectar.bat`, el audio y cada modelo.
