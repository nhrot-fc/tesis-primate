# Capítulo 4 — Curación del Conjunto de Datos (OE1)

> **OE1.** Definir un protocolo de curación y estandarización, y aplicarlo sobre
> las anotaciones entregadas por el equipo de investigación para construir un
> conjunto de datos consistente de cajas tiempo–frecuencia etiquetadas por
> especie y tipo de llamada.

Capítulo redactado por completo. Es el modelo de cómo se escriben los
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

## Pendientes

En la lista única de `research/capitulos/README.md`.

> **17-11-2026. Revisión externa de cifras.** Cuaderno: el destino de un par sin elegibles es
> «marcada para revisión» antes que «frase» (`sb/pcs`), `nBelowThreshold` cuenta solo los `<100`,
> nuevas `nLeftOutReview`/`nLeftOutVocab`; el desbalance es en cajas (`as/hc` 6 933 / `sb/sc` 126 =
> 55×); `largestPairN` cuenta las elegibles (2 453). No se corrió el cuaderno (en local
> `data/cleaned` es copia de `raw`): las salidas se corrigieron a mano con los valores que daría.
> Texto: 3 238+81+19+17+1 ilegible; 2 823 emparejadas (4 ilegibles dentro) + 12; 643 = 336+180+128−1;
> nota de la Tabla 4.8 (19 213 → 19 211); 41 pares → 37 tras la fusión; 49 % de `sb/spc` sobre 16 kHz;
> catorce limitaciones (nueva: partición por grabación, no por seguimiento).
