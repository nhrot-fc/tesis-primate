# Capítulo 4 — Curación del Conjunto de Datos (OE1)

> **OE1.** Definir un protocolo de curación y estandarización, y aplicarlo sobre
> las anotaciones entregadas por el equipo de investigación para construir un
> conjunto de datos consistente de cajas tiempo–frecuencia etiquetadas por
> especie y tipo de llamada.

Capítulo redactado por completo, con los tres reportes `RE_1-*` de
`research/reportes/` ya absorbidos. Es el modelo de cómo se escriben los
capítulos de objetivo.


> **30-09-2026 (tarde).** Se quitaron del `.tex` el párrafo sobre cómo se generan las cifras y los bloques «Cómo reproducirlo» con comandos, y la subsección «Distribución y mantenimiento»: el documento no es un manual y la distribución está fuera de alcance (ver `capitulos/README.md`). Los cuatro `\todo` que pedían datos al equipo se convirtieron en limitaciones redactadas.

> **06-10-2026.** Revisión tras la observación del jurado («no se proporciona
> acceso claro a la ficha», figuras sin interpretación, referencias adelantadas):
>
> - Cada `\rotulo{Medio de verificación.}` dice dónde está la evidencia: el
>   código en el repositorio público y, como los datos no se publican, la
>   subsección de verificación que la reproduce en el documento.
> - Nueva **Tabla 4.19 (`tab:verificacion-re13`)**: los nueve elementos del IOV
>   de RE1.3 y el apartado, tabla o figura donde está cada uno. La introducción
>   de la ficha remite a ella por sección, no por tabla.
> - Las referencias que apuntaban a tablas a 11–23 páginas de distancia
>   (`tab:reglas`, `tab:archivos`, `lst:meta`, `tab:cleaned`,
>   `tab:verificacion-re13`) ahora apuntan a la subsección o se resolvieron en el
>   texto. La galería pasó de `[p]` a `[tbp]`.
> - Todas las figuras y tablas se presentan, se muestran y se interpretan.

> **06-10-2026 (noche), revisión estricta contra la rúbrica:**
>
> - Cada RE tiene un `\rotulo{Cómo reproducirlo.}`.
> - Cada «Cómo se alcanzó» nombra la fase de CRISP-DM del Capítulo 1 que
>   aplica.
> - RE1.3 ya no dice que la acompaña «el README del repositorio», porque ese
>   README es del programa y no del conjunto. La verificación declara la
>   desviación respecto del medio comprometido.

> **07-10-2026.** Medios de verificación (`research/medios_verificacion/`,
> Anexo B):
>
> - **RE1.3: la desviación del README ya no existe.** El README del conjunto
>   que nombra el medio de verificación es
>   `medios_verificacion/04_oe1_curacion/RE1.3/README.md`. Es la ficha entera,
>   en el orden de los nueve elementos del IOV, y se genera con las mismas
>   cifras que el capítulo (`make medios`). Cambiaron en consecuencia el
>   «Resultado», el «Medio de verificación», el cierre de la verificación y la
>   fila «Normalización log-mel» de la Tabla 4.19. Esa fila apuntaba a la
>   normalización de las cajas, que es otra cosa.
> - **RE1.1 y RE1.2** remiten a sus carpetas. La comprobación 1 de RE1.2
>   («cargan sin error») cita además los dos usos registrados del caché: la
>   exportación a YOLO, con 0 imágenes corruptas, y la evaluación.
> - **`prepare_data.log` se quitó** de la Tabla 4.17 y del listado del
>   diccionario. `prepare_data.py` llama a `setup_logging()` sin archivo, así
>   que ese registro no existe.
> - **Representación de entrada.** Decía que la compresión log y la
>   normalización eran la primera capa de cada modelo. Ahora precisa dónde
>   ocurren en cada uno: el modelo propuesto, Faster R-CNN dentro del modelo, y
>   YOLO y RT-DETR al exportar.
> - La discusión decía «diez limitaciones conocidas» y la lista tiene doce.
>   Corregido.
> - **Ojo: el `data/` de esta máquina es una copia del 20-08-2026** y no es el
>   caché del documento. Tiene 21 966/9 446/8 007 ventanas frente a
>   21 751/9 364/7 598, y `cleaned/` en formato viejo y sin LW. El caché
>   bueno está en el servidor (`/home/fcandia/tesis-primate`), donde se
>   entrenó todo. El código actual, sobre el `raw/` local, reproduce exacta la
>   curación (19 211, 643, 17 495, 25 clases), pero no la partición. Antes de
>   volver a ejecutar el cuaderno de este capítulo, traer `data/` del servidor
>   o ejecutarlo allí.

## Qué va aquí

| Sección | Contenido | RE |
|---|---|---|
| Introducción | Enuncia OE1, anuncia las tres secciones de resultado y explica de dónde sale cada cifra | — |
| El material recibido | Estructura y formato, inventario y los cinco defectos detectados | diagnóstico |
| Protocolo de curación y estandarización documentado | Esquema, las nueve reglas, verificación y mediciones, fuera de alcance | **RE1.1** |
| Conjunto acústico curado y particionado | Selección de clases, anidamiento frase–sílaba, partición por grabación, ventaneo, representación de entrada, entregables y verificación | **RE1.2** |
| Ficha del conjunto derivado | Propósito, composición (con el párrafo de desbalance), geometría, recolección, transformaciones, diccionario de datos, usos, versión, limitaciones y verificación (Tabla 4.19) | **RE1.3** |
| Discusión | Las cinco preguntas de la pauta | — |

Cada sección de resultado abre con tres rótulos fijos —**Resultado**, **Cómo se
alcanzó**, **Medio de verificación**— y cierra con una subsección de
**verificación y mediciones** que recorre el IOV punto por punto. Es la
estructura que piden las pautas y conviene mantenerla en los capítulos 5 y 6.

## Cómo se redacta

- **Ninguna cifra se escribe a mano.** Todas son macros de
  `figures/valores.tex` (`\nKept`, `\pctExperiment`, `\largestClass`…) que
  escribe el cuaderno. Si una cifra no tiene macro, es que falta generarla.
- Las filas de las tablas también se generan: el `.tex` pone la cabecera y hace
  `\input{\figOE/tablas/<nombre>} \\`, y el cuaderno escribe el cuerpo.
- `\figOE` es la ruta de las figuras, definida al inicio del capítulo.
- La distinción entre **conjunto primario** (lo que entrega el equipo) y
  **conjunto derivado** (lo que produce la tesis) se mantiene en todo el
  capítulo: la contribución es la segunda capa.

## Qué figura va en matplotlib y cuál en LaTeX

La regla que sigue el capítulo, y que conviene aplicar igual en los demás:

| Tipo de figura | Herramienta | Por qué |
|---|---|---|
| **Esquemas** sin datos: orden de las reglas, flujo del caché, reparto por grabación, asignación a ventanas, normalización de la caja, capas del conjunto | **TikZ**, dentro del `.tex` | Usan la tipografía y el color del documento, se editan como texto y no se pixelan |
| **Gráficos de pocas barras**: cajas por ventana, hora del día, mes | **pgfplots**, desde `figures/data/*.dat` | Mismo motivo; el cuaderno sólo aporta la serie |
| **Imágenes**: espectrogramas, galería de clases, ventanas del caché | **matplotlib** (PNG a 200 ppp) | Son píxeles; LaTeX no los puede dibujar |
| **Nubes y barras con muchas categorías**: anotaciones por par (57), facetas por clase (24), cajas por clase y partición | **matplotlib** | Cientos o miles de puntos; en pgfplots el `.dat` y el tiempo de compilación se disparan |

Las figuras de matplotlib usan la misma paleta que los esquemas TikZ
(`research/estilos.tex`), y se generan del ancho con el que se imprimen para que
no haya que escalarlas.

## Imágenes

Generadas por `notebook/figuras_04_oe1_curacion.ipynb`:

| Figura | Qué muestra |
|---|---|
| `anotaciones_por_par.png` | Cada par especie/llamada de `cleaned/` sobre el umbral, coloreado por destino |
| `anidamiento_frase.png` | Una frase `lw/cc` con sus sílabas `lw/cs` dentro, sobre el espectrograma |
| `cajas_por_clase_particion.png` | Cajas por clase y partición, con las anotaciones como referencia |
| `geometria_facetas.png` | Duración × ancho de banda, un panel por clase |
| `formas_ac_bc.png`, `formas_sm_fs.png` | Las dos etiquetas cuya nube se parte en dos |
| `galeria_<clase>.png` (9) | Una llamada de cada clase sobre el espectrograma; el bloque de subfiguras lo escribe el cuaderno en `tablas/galeria.tex` |
| `ventana_densa/larga/vacia.png` | Tres ventanas del caché tal como las recibe el modelo. Van al ancho del texto y como **tres figuras separadas**, no como subfiguras de un tercio: a ese tamaño no se veían las cajas |

Capturas que **no** genera el cuaderno (vienen del trabajo de campo y del informe
de avance E3; se conservan tal cual):

| Figura | Qué muestra |
|---|---|
| `tabla_raven_cruda.png` | Una tabla de selección de Raven como se recibe |
| `raven_anotando.png` | Raven Pro durante la anotación |
| `etiquetas_especie_crudas.png`, `etiquetas_llamada_crudas.png` | Los valores distintos hallados en cada columna manual |
| `aviso_audio_danado.png` | Advertencias de audio dañado al cargar |
| `audio_corrupcion_leve.png`, `audio_corrupcion_severa.png` | Los dos grados de corrupción de la señal |

## Qué produce el cuaderno

```
figures/
├── valores.tex          122 macros con cada cifra del capítulo
├── tablas/*.tex         las filas de 12 tablas (sin cabecera ni \bottomrule)
├── data/*.dat           series para pgfplots
├── meta.json            copia del meta del caché, que el capítulo lista entero
└── *.png                las figuras de matplotlib
```

Para regenerarlo hacen falta `data/raw/` (la auditoría del protocolo relee las
tablas crudas), `data/cleaned/` y `data/processed/`:

```bash
cd research/capitulos/04_oe1_curacion/notebook
uv run jupyter nbconvert --to notebook --execute --inplace figuras_04_oe1_curacion.ipynb
```

## Extra

- `notas_tipos_llamada.md` — notas de campo por tipo de llamada: qué significa
  cada una y cómo se dibuja su caja en Raven. Es la fuente del vocabulario junto
  con `src/data/species.py`.

## Pendientes

- [x] ~~El caché parecía ir una pasada por detrás.~~ **Resuelto el 30-09.** No lo
      estaba: `data/cleaned/` arrastraba 633 marcas de revisión de una corrida hecha
      con un `species.py` al que le faltaban las cuatro entradas `unofficial_*`
      (`pt/sqc`, `lw/tj`, `pt/pp`, `pt/bp`), y el commit `7b88820` había quitado
      `lw/tj → lw/trino` de `JOINED_LABELS`. Se restauró la fusión y se regeneró
      `cleaned/` con `--force`: las reglas vuelven a dar **las 25 clases exactas de
      `labels.json`**, así que el caché es válido y no hay que reentrenar nada.
- [ ] Resolver con el equipo los pares marcados para revisión, en especial si
      `as/cp`, `as/pp` e `as/ip` son tipos legítimos de *Alouatta sara*.
- [ ] Completar los datos de campo de la ficha: número y ubicación de los
      sensores, configuración de grabación y criterio de selección de lo que se
      anotó. *El sitio ya está (06-10-2026): Tambopata Research Center, Reserva
      Nacional Tambopata. La limitación pasó a llamarse «Procedencia de campo
      incompleta».*
- [ ] Decidir si el conjunto derivado se libera y en qué términos.
- [ ] El medio de verificación congelado de RE1.3 (Capítulo 1) dice «Capítulo de
      curación, README y metadatos del conjunto», pero **no existe un README
      del conjunto** en `data/`. O se escribe (puede resumir la ficha y remitir
      a ella), o se declara en la verificación que la ficha del capítulo cumple
      esa función.
- [ ] **Faltan diez citas.** Las marcas `[CITA: …]` se quitaron del `.tex` el
      02-10-2026 por pedido del usuario, así que las afirmaciones están en el
      documento **sin respaldo bibliográfico**. Lo que hay que buscar, y dónde:

  | Afirmación | Referencia que falta |
  |---|---|
  | El protocolo es código y no una edición manual | protocolos de monitoreo acústico estandarizables y reproducibles; `gibb_2018` sirve de partida |
  | Recorte a Nyquist | teorema de muestreo de Nyquist–Shannon |
  | Fuga entre particiones por duplicados | efecto de las instancias duplicadas sobre la evaluación |
  | Hace falta medir el acuerdo entre anotadores | `leroy_reliability_2018` es candidata |
  | Fuga por ventanas solapadas de una misma grabación | un precedente de SED que reporte la inflación al partir por clip |
  | Ventanas vacías como negativos | muestreo de fondo en SED y su efecto sobre los falsos positivos |
  | Escala mel y banco HTK | Stevens y Volkmann, o Young et al. (HTK) |
  | AudioMoth y Raven Pro | Hill et al. (2018); K. Lisa Yang Center for Conservation Bioacoustics |
  | Variabilidad entre anotadores | `leroy_reliability_2018` o equivalente sobre etiquetas fuertes |
  | Variabilidad entre sensores de bajo costo | efecto sobre los detectores |
- [x] ~~Comentarios del jurado sobre E1 (material, grafías, partición, referencias adelantadas)~~ **Resuelto el 07-10-2026.** §4.2.1 «Origen y formato» abre con el contexto (proyecto, sitio, fechas, horario, quién anotó, qué se entregó). Las grafías van en la Tabla «Grafías de una misma clase». La figura del reparto tenía flechas que no seguían el algoritmo (con 7, 5, 4, 2 y 1 cajas, la segunda va a train): corregidas y recorridas en el texto. Se quitaron cinco referencias adelantadas (regla de especie, anidamiento, ventanas vacías ×2, ventaneo).
- [ ] Para E3: las Figuras de valores crudos son capturas del cuaderno; convendría pasarlas a tabla. Quedan referencias adelantadas de hoja de ruta y de verificación (aceptables) y algunas de contenido (geometría, anidamiento, limitaciones).
