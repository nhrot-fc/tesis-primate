# Capítulos de la tesis

Un capítulo, una carpeta. `main.tex` solo tiene el preámbulo, la portada, los
índices y un `\include` por capítulo; todo el contenido vive aquí.

```
capitulos/<nn>_<nombre>/
├── <nn>_<nombre>.tex   el capítulo: se incluye desde main.tex, no se compila solo
├── README.md           qué va en el capítulo, cómo se redacta, qué falta
├── figures/            sus imágenes (+ figures/data/ con las series)
└── notebook/           el cuaderno que genera esas imágenes
```

| Carpeta | Capítulo | Objetivo |
|---|---|---|
| `01_generalidades` | Generalidades | — |
| `02_marco_referencial` | Marco Referencial | — |
| `03_estado_del_arte` | Estado del Arte | — |
| `04_oe1_curacion` | Curación del Conjunto de Datos | **OE1** |
| `05_oe2_detector` | Detección y Evaluación del Modelo | **OE2** |
| `06_oe3_utilidad` | Tiempo de Revisión y Análisis de Errores | **OE3** |
| `07_conclusiones` | Conclusiones | — |
| `08_anexo_plan` | Plan del Proyecto (anexo) | — |

## Las reglas que fijan esta estructura

De `objetivos.md` y `pautas.md`, y no se negocian:

1. **Los objetivos y los resultados esperados están congelados.** Una vez
   presentados no cambian ni ellos ni sus entregables. El Capítulo 1 los copia
   literalmente.
2. **Un capítulo por objetivo específico**, y dentro, **una sección por
   resultado esperado**. De ahí que OE3 tenga capítulo propio en lugar de ir al
   final del de OE2.
3. Cada capítulo de objetivo lleva **Introducción** (enuncia el OE), las
   secciones de resultado y **Discusión**.
4. Cada resultado esperado se redacta **completo en su capítulo**: qué es, cómo
   se alcanzó con los métodos del Capítulo 1, cómo se verifica con el medio de
   verificación declarado y qué mediciones sostienen su IOV. No se delega a un
   anexo ni a un documento aparte; si el medio de verificación es extenso, va un
   resumen y el anexo se referencia.
5. La discusión resume los resultados sin añadir ninguno nuevo, los interpreta,
   los contrasta con la literatura del Capítulo 3, analiza su generalización y
   declara sus limitaciones.

## Tono: es documentación de investigación, no un manual

Tres decisiones editoriales que se tomaron el 30-09-2026 y conviene no deshacer:

1. **No se explica al lector de dónde salen las cifras.** Nada de «este capítulo
   obtiene sus cifras del cuaderno X»: se describe lo que se hizo y se muestra el
   resultado. Que las cifras se generen con código es una propiedad del trabajo,
   no un párrafo del documento, y ningún jurado va a ejecutar nada. El cuaderno y
   su trazabilidad se documentan **en estos README**, no en el `.tex`.
2. **No se escriben instructivos.** Fuera los bloques «Cómo reproducirlo» con
   comandos de consola. La única excepción es RE2.2, cuyo IOV exige literalmente
   que «las dependencias y comandos de ejecución estén declarados»: ahí el
   listado se presenta como *la interfaz del pipeline*, que es el entregable, no
   como una guía de uso.
3. **La distribución y el mantenimiento están fuera de alcance**, y así lo
   declara la sección de exclusiones del anexo. No se discuten licencias,
   empaquetado, *releases* ni publicación del conjunto: son decisiones del equipo
   de investigación. Lo que RE2.5 compromete es que el repositorio exista y
   permita localizar el código y el *checkpoint* final, nada más.

**Nada de marcas visibles en el PDF.** El 02-10-2026 se eliminaron las 22
`\todocita` y las 2 `\todo` que quedaban: las macros siguen definidas en
`estilos.tex` para uso temporal, pero no se dejan en el documento. Las citas que
faltaban se anotaron en el README del capítulo correspondiente, con la afirmación
y la referencia que hay que buscar.

También: **lo que no se puede averiguar se declara como limitación, no como
`\todo`.** Si falta un dato de procedencia o no se puede distinguir un error de
anotación de una categoría legítima, eso es una limitación del conjunto y va en
la sección de limitaciones del capítulo, redactada.

## Estructura frente a las pautas

`pautas.md` trae dos bloques y los dos se aplican:

1. **Un capítulo por objetivo específico** (4, 5 y 6), cada uno con Introducción,
   una sección por resultado esperado (qué es, cómo se alcanzó, cómo se verificó
   y qué se midió contra el IOV) y una **Discusión** con los cinco puntos que
   pide la pauta: qué se obtuvo, qué significa, consistencia con la literatura,
   generalización y limitaciones. Los capítulos 4 y 5 ya cumplen los cinco.
2. **Una conclusión de proyecto** (capítulo 7). El usuario dudó de si debía
   existir (02-10-2026) y se quedó porque el segundo bloque de la pauta pide
   decir explícitamente si se cumplieron los RE, los OE, **el objetivo general**
   y si la propuesta resuelve el problema central: las dos últimas no caben en
   ningún capítulo de objetivo. Se mantiene **corto a propósito**, sin repetir
   las discusiones y sin cifras nuevas.

El capítulo 7 está **bloqueado por el capítulo 6**: el veredicto sobre el
objetivo general sale de RE3.2 (minutos por hora de audio, manual frente a
asistido) y hasta tenerlo no se enuncia. Lo que falta está marcado con
comentarios dentro del `.tex`, no con `\todo` visibles.

## Figuras y tablas: cómo se colocan y cómo se titulan

Decisiones del 30-09-2026, sobre una lectura del PDF impreso:

1. **El pie es corto y es un nombre, no un párrafo.** De tres a siete palabras,
   sin punto final, sin rutas ni nombres de función. Era trampa poner un
   `\caption[corto]{párrafo largo}`: el índice quedaba limpio y bajo la figura
   había tres renglones de explicación. Si algo hay que explicar, se explica en
   el cuerpo, que es donde el lector ya está mirando. El `[corto]` opcional sólo
   tiene sentido cuando el pie de verdad no cabe en el índice, y hoy no hace
   falta en ningún flotante.
2. **Ningún flotante sin `\ref` y sin texto alrededor.** Una página de puras
   figuras se lee como un volcado de imágenes. Cada figura y cada tabla se
   introduce desde el párrafo anterior y, si van varias seguidas, se parte el
   párrafo para que cada una tenga el suyo. Los parámetros de colocación
   (`\topfraction` y compañía) están subidos en `main.tex` justamente para que
   LaTeX no fabrique páginas de solo flotantes.
3. **Nada de raya (`---`) en el cuerpo.** El inciso va entre paréntesis o entre
   comas. La raya media de los compuestos (`tiempo--frecuencia`) sí se queda.
4. **Un panel por figura cuando el panel tiene detalle.** Tres espectrogramas en
   subfiguras de un tercio de ancho no se ven; valen más como tres figuras al
   ancho del texto, apiladas, con una frase entre medias.
5. **Si una figura y una tabla dicen lo mismo, sobra una.** Así se eliminó la
   tabla de fases de CRISP-DM, que repetía la figura del ciclo.

## Compilar

Desde la raíz del repositorio:

```bash
make            # compila research/build/main.pdf y resume los avisos
make warnings   # vuelve a mostrar los avisos del último log
```

Las rutas de `\includegraphics` son relativas a `main.tex`, así que la
compilación se lanza siempre desde `research/` (lo hace el Makefile). Para
trabajar en un solo capítulo, descomenta el `\includeonly` de `main.tex`.

## Qué está pendiente

Cada `README.md` de capítulo termina con su propia lista. En grande:

- OE1 (cap. 4) está **redactado entero** (RE1.1–RE1.3 + discusión), con sus tres
  reportes absorbidos.
- OE2 (cap. 5) está **redactado entero** (RE2.1–RE2.5 + discusión), sobre la
  comparación vigente `runs/comparacion/comparacion_modelos_v3`. Sus cinco
  reportes están absorbidos.
- OE3 (cap. 6) está sin redactar, y **RE3.2 necesita una medición de tiempo que
  todavía no se ha hecho**. Es lo único que responde directamente al objetivo
  general.
- Las conclusiones (cap. 7) se escriben al final.

`research/reportes/` ya solo conserva `RE_3-3` sin trasladar; los ocho de OE1 y
OE2 están absorbidos y se pueden borrar. **Ojo:** `RE_2-1` y `RE_2-3` describen
una comparación anterior y sus cifras ya no son válidas.
