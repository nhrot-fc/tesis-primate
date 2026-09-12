# Detector de vocalizaciones de primates

Un detector de objetos (Faster R-CNN, AST-Deformable DETR o YOLO) dibuja cajas
tiempo–frecuencia sobre el espectrograma de una grabación y las exporta como **tablas de
selección de Raven**, para revisarlas en Raven Pro o en el visor incluido.

Se distribuye como una carpeta portable para Windows: se descomprime y se usa. **No hay que
instalar Python, CUDA ni ninguna librería.**

## Descarga

En [Releases](https://github.com/nhrot-fc/tesis-primate/releases), el zip de la última versión:

| Paquete | Para quién | Tamaño |
|---|---|---|
| `detector-primates-<versión>-win64-cpu.zip` | cualquier PC de 64 bits | ~0,7 GB (2 GB descomprimido) |
| `detector-primates-<versión>-win64-cuda.zip` | PC con tarjeta NVIDIA; sin ella usa la CPU | ~4 GB (6 GB descomprimido) |

Si un paquete viene en partes (`.zip.001`, `.zip.002`, …), descárgalas todas en la misma
carpeta junto con su `unir.bat` y ejecútalo: deja el zip entero. Con 7-Zip se puede abrir
la parte `.001` directamente.

## Requisitos de software

| | Necesario | Notas |
|---|---|---|
| Sistema operativo | **Windows 10 u 11, 64 bits** | Sin permisos de administrador: todo queda en la carpeta descomprimida |
| Python, CUDA, Visual C++ | **nada** | Van dentro del paquete |
| Controlador NVIDIA (sólo paquete `cuda`) | serie **570 o más nueva** | Es lo único que el paquete no trae. Se actualiza desde [nvidia.com/drivers](https://www.nvidia.com/drivers). Con controladores desde la serie 527 suele funcionar; si torch no ve la GPU, el paquete sigue en CPU |
| Raven Pro (opcional) | 1.5 o 1.6 | Para revisar las tablas junto al audio. Son texto tabulado: también las abren Excel o LibreOffice |
| Internet | **no** | Ni para instalar ni para detectar |
| 7-Zip (opcional) | — | Sólo para abrir paquetes en partes sin `unir.bat` |

Detalles de hardware (RAM, GPU, disco y cuánto tarda cada modelo):
[docs/system_requirements.md](docs/system_requirements.md).

Dos avisos de Windows:

- Descomprime en una **ruta corta** (`C:\detector\`): el paquete tiene rutas largas y el
  explorador de Windows a veces falla con más de 260 caracteres.
- El primer arranque es más lento: el antivirus revisa cientos de archivos nuevos. No hay
  ningún `.exe` propio, sólo el `python.exe` firmado por python.org y dos archivos `.bat`.

## Uso

| Archivo | Qué hace |
|---|---|
| `Visor.bat` | Visor de espectrogramas. `Ctrl+O` abre un audio (WAV, FLAC, MP3), `Ctrl+M` un modelo de `models\`, `Ctrl+R` detecta. `F1` lista los controles. Exporta tablas de Raven e imágenes del tramo |
| `Detectar.bat` | Procesa carpetas enteras: arrastra una carpeta (o varios audios) sobre el archivo y deja `<audio>.detections.txt` junto a cada grabación. Con más de un modelo en `models\` pregunta cuál usar |
| `models\` | Un modelo por subcarpeta: el checkpoint y su umbral de operación. Para añadir otro, copiar su carpeta |
| `LEEME.txt` | Esto mismo, dentro del paquete |
| `visor.log` | Aparece si el visor falla; dice por qué |

Con Raven Pro: abre el audio y luego *File → Open Selection Table* con su
`.detections.txt`. La columna `Score` permite filtrar dentro de Raven; el umbral con el que
se escribió la tabla es el que se eligió para ese modelo en la comparación.

## Para desarrolladores

Entorno, datos, entrenamiento, comparación de modelos y cómo se arma el paquete de Windows:
[docs/desarrollo.md](docs/desarrollo.md). Documentos de la tesis e informes:
[docs/README.md](docs/README.md).
