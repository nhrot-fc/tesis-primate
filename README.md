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
2. Descomprime cada zip `model` en esa misma carpeta, junto a `Detector.exe` (no dentro de
   `models\`): el zip ya trae `models\<nombre>\`.

Si un zip viene en partes (`.zip.001`, `.zip.002`, …), descárgalas todas en la misma carpeta
con su `join.bat` y ejecútalo; deja el zip entero.

## Requisitos de software

| | Necesario | Notas |
|---|---|---|
| Sistema operativo | **Windows 10 u 11, 64 bits** | Sin permisos de administrador: todo queda en la carpeta descomprimida |
| Python, CUDA, Visual C++ | **nada** | Van dentro del paquete |
| Controlador NVIDIA (sólo paquete `cuda`) | serie **570 o más nueva** | Es lo único que el paquete no trae. Se actualiza desde [nvidia.com/drivers](https://www.nvidia.com/drivers). Con controladores desde la serie 527 suele funcionar; si torch no ve la GPU, el paquete sigue en CPU |
| Raven Pro (opcional) | 1.5 o 1.6 | Para revisar las tablas junto al audio. Son texto tabulado: también las abren Excel o LibreOffice |
| Internet | **no** | Ni para instalar ni para detectar |
| 7-Zip (opcional) | — | Sólo para abrir zips en partes sin `join.bat` |

Detalles de hardware (RAM, GPU, disco y cuánto tarda cada modelo):
[docs/system_requirements.md](docs/system_requirements.md).

Dos avisos de Windows:

- Descomprime en una **ruta corta** (`C:\detector\`): el paquete tiene rutas largas y el
  explorador de Windows a veces falla con más de 260 caracteres.
- El primer arranque es más lento: el antivirus revisa cientos de archivos nuevos.
- `Detector.exe` no está firmado: la primera vez SmartScreen avisa ("Windows protegió tu PC");
  *Más información → Ejecutar de todas formas*. Es un lanzador de 80 KB
  ([deploy/launcher/launcher.c](deploy/launcher/launcher.c)) que arranca el `python.exe`
  firmado por python.org que va dentro del paquete.

## Uso

La interfaz está en inglés. Una sola ventana, como un cliente de correo: la lista de grabaciones a la izquierda (*Recordings*), el espectrograma al centro y la tabla de cajas a la derecha (*Boxes*); los dos paneles se abren y cierran desde los extremos de la barra.

| | Qué hace |
|---|---|
| `Detector.exe` → una grabación | Arrastra un audio (WAV, FLAC, MP3) o `Ctrl+O`; el modelo ya viene elegido (el último usado) y hay **un solo botón de acento a la vez**, el siguiente paso: **Detect** hasta que hay detecciones, luego **Review**, y dentro de la revisión **Accept**. Encima se puede abrir una tabla de Raven con el mismo `Ctrl+O` o arrastrándola (con columna `Score` va a la capa *Detections*, sin ella a *Annotations*; el `.detections.txt` vecino se abre solo con el audio), filtrar por score, oír, borrar cajas y guardar la tabla o una imagen. Como en Raven, el ancho de ventana y la altura de banda se eligen de dos listas y hay un scroll para cada eje; la barra de tiempo lleva marcadas las cajas de toda la grabación y ⏮ ⏭ saltan entre detecciones. **Review** recorre las detecciones una por una: cada caja se encuadra en tiempo y banda, con asas para retocarla y el cabezal en su inicio (`Espacio` la hace sonar); se acepta con `A`/`Enter` (tras corregir la especie si hace falta), se rechaza con `R`/`Supr`, `N`/`P` la saltan y `Esc` sale. Las decisiones se apuntan al instante en un archivo temporal por grabación y al volver a revisarla se retoman solas; *Save → Annotations table…* guarda las aceptadas junto a las demás. `F1` lista los controles |
| `Detector.exe` → una carpeta | Arrástrala sobre el `.exe` o sobre la ventana, o `Ctrl+Shift+O`: sus grabaciones se listan a la izquierda con su duración y cuántas cajas tiene ya cada una. Un clic (o `↑`/`↓`) abre la grabación con su tabla; **Detect all** corre el modelo sobre toda la lista con el score elegido (arranca en el punto de operación del modelo) y deja `<audio>.detections.txt` junto a cada una al terminar; si la que está abierta recibe tabla, se carga sola. La carpeta se recuerda entre sesiones |
| `detect.exe` | Lo mismo que *Detect all* desde consola, para scripts: `detect.exe --model models\frcnn D:\grabaciones` |
| `README.txt` | Esto mismo, dentro del paquete |
| `viewer.log` | Aparece si el visor falla; dice por qué |

Cargar un audio descarga las capas anteriores: primero el audio, después sus tablas.

Con Raven Pro: abre el audio y luego *File → Open Selection Table* con su
`.detections.txt`. La columna `Score` permite filtrar dentro de Raven; el umbral con el que
se escribió la tabla es el que se eligió para ese modelo en la comparación.

## Para desarrolladores

Python 3.12 y [uv](https://docs.astral.sh/uv/). `uv sync --no-dev --extra detr --extra yolo`
corre el visor e inferencia con cualquier checkpoint; `uv sync --all-extras --all-groups` todo
(datos, entrenamiento, notebooks, linters).

```bash
uv run python src/prepare_annotations.py                 # data/raw -> data/cleaned (marca requires_review; notebooks/auditoria_anotaciones.ipynb las revisa)
uv run python src/prepare_data.py                        # -> data/processed (ventanas, splits)
uv run python src/train.py --arch frcnn                  # detr | detr_coco | yolo; --hp k=v, --cfg k=v
./evaluation.sh                                          # vuelca val/test de runs/*/best.pt y compara
uv run python src/main.py [audio-o-carpeta]              # visor; sin models/ lista runs/
uv run python src/detect.py --model runs/frcnn carpeta/  # por lotes
uv run python src/kfold.py --arch yolo --name yolo26s_v3_kfold5                # predicciones fuera de muestra sobre train
uv run python src/find_issues.py runs/yolo26s_v3_kfold5/yolo26s_v3_kfold5_train_predictions.pt  # CLOD -> hallazgos, contrastados con data/unified
uv run python src/find_issues.py runs/yolo26s_v3/yolo26s_v3_val_predictions.pt              # ídem val y test; notebooks/curacion_iter1.ipynb los decide a ojo -> curation/
```

Cómo se comparan los modelos: [docs/protocolo_comparacion.md](docs/protocolo_comparacion.md).
Especies y llamadas: [src/data/species.py](src/data/species.py) y
[docs/call_type_notes.md](docs/call_type_notes.md). Informes entregados (PDF): [docs/informes/](docs/informes/).

### Releases

`deploy/build_windows.py` arma en `dist/` (ignorado por git) dos clases de zip, desde Linux:
`runtime --variant cpu|cuda` (Python embebido, librerías, `src/`, `Detector.exe` y `detect.exe`,
`models/` vacía) y `models runs/<corrida>…` (un zip por modelo con su `models/<corrida>/` y
los `hf/` que necesite; sin carpeta raíz, se vuelca dentro del runtime). Los `.exe` son
[deploy/launcher/launcher.c](deploy/launcher/launcher.c) compilado con zig (uv lo baja como
paquete `ziglang`; no hace falta instalar nada). Los zips van como *assets* del release, no al
repo; si uno pasa de 1,9 GB sale en partes con un `join.bat`.

```bash
deploy/release.sh v0.1.0 runs/frcnn runs/yolo26s_coco          # runtime cpu + modelos + gh release
VARIANTS="cpu cuda" deploy/release.sh v0.1.0 runs/frcnn         # también el runtime cuda
REPO=usuario/repo-privado deploy/release.sh v0.1.0 runs/frcnn   # publicar en otro repo
```

Necesita `gh` autenticado. Antes de publicar, probar el zip en un Windows limpio (sin Python,
sin red): `Detector.exe` con un audio, una carpeta y el zip de cada modelo; `detect.exe`.
