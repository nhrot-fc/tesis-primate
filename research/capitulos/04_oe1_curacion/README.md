# Capítulo 4 — Curación del Conjunto de Datos (OE1)

> **OE1.** Definir un protocolo de curación y estandarización, y aplicarlo sobre
> las anotaciones entregadas por el equipo de investigación para construir un
> conjunto de datos consistente de cajas tiempo–frecuencia etiquetadas por
> especie y tipo de llamada.

Es el capítulo más avanzado del documento: está redactado casi entero.

## Qué va aquí

| Sección | Contenido | RE |
|---|---|---|
| Introducción | Enuncia OE1 y anuncia las tres secciones de resultado | — |
| Análisis del conjunto original | Material recibido, defectos detectados, composición, geometría por clase, sílabas y frases | contexto |
| Protocolo de curación y estandarización documentado | Carga y consolidación, normalización de etiquetas, saneamiento geométrico, vocabulario cerrado y selección | **RE1.1** |
| Conjunto acústico curado y particionado | Partición por grabación, ventaneo, densidad por ventana, representación de las cajas | **RE1.2** |
| Balance del conjunto de experimento | Reparto de cajas por clase y partición | contexto de RE1.2 |
| Ficha del conjunto derivado | Datasheet: propósito, composición, recolección, transformaciones, diccionario, usos, limitaciones | **RE1.3** |
| Discusión | Las cinco preguntas de la pauta | — |

## Cómo se redacta

- Una sección por resultado esperado, y cada una responde cuatro cosas en este
  orden: qué es el resultado, cómo se alcanzó (con los métodos del Capítulo 1),
  cómo se reproduce (comando concreto) y cómo se verifica con el medio de
  verificación declarado en la Tabla de IOV de OE1.
- Ninguna cifra se escribe a mano: sale del cuaderno o del propio
  `data/processed/meta.json`.
- La distinción entre **conjunto primario** (lo que entrega el equipo de
  investigación) y **conjunto derivado** (lo que produce esta tesis) tiene que
  mantenerse en todo el capítulo: la contribución es la segunda capa.
- «Análisis del conjunto original» y «Balance del conjunto de experimento» no
  son resultados esperados. Hoy son secciones sueltas; lo correcto es que la
  primera sea el diagnóstico dentro de la introducción o una subsección de RE1.1,
  y la segunda una subsección de RE1.2. Está sin hacer para no romper la prosa.

## Imágenes

De `notebook/figuras_04_oe1_curacion.ipynb`, salvo las `imageN.png`, que son
capturas heredadas del informe de avance E3 y no se regeneran.

| Figura | Qué muestra |
|---|---|
| `image4.png`, `image5.png` | Archivo de anotaciones de Raven; Raven con una grabación anotada |
| `image6.png`, `image7.png` | Etiquetas únicas encontradas en especie y en tipo de llamada |
| `image8.png`, `image9.png`, `image10.png` | Grabaciones con corrupción leve y severa |
| `call_gallery.png` | Galería de llamadas, una por clase |
| `class_geometry_facets.png` | Duración frente a ancho de banda, una faceta por clase |
| `nesting_phrase.png` | Anidamiento frase–sílaba sobre una grabación real |
| `curation_pipeline.png` | Etapas del protocolo |
| `annotations_per_pair.png` | Anotaciones por par especie/llamada y umbral del experimento |
| `split_by_recording.png` | Partición por archivo de grabación |
| `windowing.png`, `window_example.png`, `box_coordinates.png` | Ventaneo de 3 s, una ventana como la ve el modelo, coordenadas de la caja |
| `boxes_per_split.png`, `boxes_per_window.png` | Cajas por clase y partición; densidad de cajas por ventana |
| `class_geometry.png` | Geometría por clase **(en `figures/` pero sin usar en el `.tex`)** |

Las series de datos de cada figura quedan en `figures/data/*.json`, para citar
una cifra sin leerla del gráfico.

Para regenerarlas hacen falta `data/cleaned/` y, para `window_example` y
`boxes_per_split`, `data/processed/`.

## Extra

- `notas_tipos_llamada.md` — notas de campo por tipo de llamada (qué significa
  cada llamada y cómo se dibuja su caja en Raven). Venía de `docs/`. Es la
  fuente del vocabulario junto con `src/data/species.py`.
- **Material por trasladar desde `research/reportes/`**, que es donde hoy vive
  la versión más detallada:
  - `RE_1-1_protocolo-curacion.tex` → regla por regla, con el inventario de
    encabezados, el mapa de sinónimos y el conteo de lo que toca cada regla.
  - `RE_1-2_conjunto-particionado.tex` → selección de clases, partición,
    ventaneo, representación de entrada y entregables.
  - `RE_1-3_ficha-conjunto.tex` → la ficha completa (propósito, composición,
    recolección, transformaciones, diccionario de datos, usos, distribución,
    limitaciones). Es la sección más incompleta del capítulo frente a su reporte.

  Al traerlos: prefijar sus `\label` (colisionan `tab:especies`, `tab:columnas`,
  `sec:seleccion`, `sec:ventaneo`), bajar un nivel sus encabezados y cargar sus
  macros de cifras (`figures/RE_x-y/valores.tex`).

## Pendientes

- [ ] Redactar la discusión.
- [ ] Absorber los tres reportes RE y luego borrarlos.
- [ ] Decidir si `class_geometry.png` entra o se retira.
- [ ] `% TODO` abierto: decidir con el equipo si el conjunto se libera y en qué
      términos.
