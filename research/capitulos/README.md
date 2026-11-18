# Capítulos de la tesis

Un capítulo, una carpeta. `main.tex` solo tiene el preámbulo, la portada, los
índices y un `\include` por capítulo; todo el contenido vive aquí.

```
capitulos/<nn>_<nombre>/
├── <nn>_<nombre>.tex   el capítulo: se incluye desde main.tex, no se compila solo
├── README.md           qué va en el capítulo, cómo se redacta, qué falta
├── figures/            valores.tex (macros), tablas/, data/ (series de pgfplots) e imágenes
└── notebook/           el cuaderno que los genera
```

**Cuadernos (15-11-2026).** Son guiones cortos, de unas 25 celdas como máximo y sin
explicaciones: el capítulo explica. El estilo y las salidas comunes están en
`src/utils/figuras.py` (paleta de `estilos.tex`, caja anotada verde y continua, caja del
modelo celeste y discontinua, espectrograma magma; `Salidas` escribe macros, filas, series y
figuras). `make figuras` los ejecuta todos; `CAP="04 05"` elige capítulos. Las celdas marcadas
`SERVIDOR` necesitan el caché de ventanas y la GPU del documento, que sólo están en el servidor:
allí se corre `TESIS_SERVIDOR=1 make figuras CAP="04 05"`. Fuera del servidor esas celdas no
corren y sus cifras y figuras se conservan. Las anotaciones se limpian en memoria desde
`data/raw/` (`utils.figuras.anotaciones`), así que ningún cuaderno depende de `data/cleaned/`.

| Carpeta | Capítulo | Objetivo |
|---|---|---|
| `01_generalidades` | Generalidades | — |
| `02_marco_referencial` | Marco Referencial | — |
| `03_estado_del_arte` | Estado del Arte | — |
| `04_oe1_curacion` | Curación del Conjunto de Datos | **OE1** |
| `05_oe2_detector` | Detección y Evaluación del Modelo | **OE2** |
| `06_oe3_revision` | Tiempo de Revisión y Análisis de Errores | **OE3** |
| `07_conclusiones` | Conclusiones | — |
| `08_anexo_plan` | Plan del Proyecto (anexo) | — |
| `09_anexo_medios` | Medios de Verificación (anexo) | — |

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

   > **Enmienda del 06-10-2026.** `pautas.md` («indicar cómo se ha alcanzado
   > dicho resultado y cómo reproducirlo») y el rubro de 12 puntos de la rúbrica
   > lo piden de forma literal. Por eso cada RE vuelve a tener un párrafo
   > `\rotulo{Cómo reproducirlo.}` de dos o tres oraciones: qué se ejecuta,
   > sobre qué entrada y por qué da el mismo resultado. **Sigue sin haber bloques
   > de comandos.** El de RE1.3 menciona el cuaderno del capítulo, porque la
   > ficha se reproduce precisamente regenerándolo: es la única excepción a la
   > regla 1.
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

El veredicto sobre el objetivo general sale de RE3.2 (minutos por hora de audio,
manual frente a asistido). El 17-11-2026 el autor lo dejó en «Cumplido», con el
corpus focal declarado como limitación.

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
make                # compila research/build/main.pdf y resume los avisos (necesita biber)
make warnings       # vuelve a mostrar los avisos del último log
```

Las rutas de `\includegraphics` son relativas a `main.tex`, así que la
compilación se lanza siempre desde `research/` (lo hace el Makefile). Para
trabajar en un solo capítulo, descomenta el `\includeonly` de `main.tex`.

> **06-10-2026 (noche).** biber quedó instalado y APA 7 está activo y
> revisado:
>
> - `\DeclareCaseLangs*{spanish}` pone los títulos en mayúscula de oración.
>   Las siglas y los nombres propios se protegen con llaves en
>   `references.bib`.
> - Los rangos de páginas usan raya corta.
> - Los preprints son `@online` con `titleaddon = {Preprint}` y
>   `organization = {arXiv}`.
> - Los conjuntos de datos son `@dataset` con
>   `titleaddon = {Conjunto de datos}`.
>
> **14-11-2026, formato del curso (`research/Formato(2025).docx`).** Times New
> Roman de 12 pt, doble espacio, márgenes de 2,54 cm y párrafos sin sangría.
> **Los rótulos siguen en APA 7**, como antes: «**Figura 1**» en negrita y el
> título en cursiva en la línea siguiente, a la izquierda y **encima** de la
> figura o de la tabla (el `\caption` va al principio del entorno). La plantilla
> pone el título de la figura debajo, pero la rúbrica exige APA 7 «en todo el
> documento», y el autor eligió APA 7. La fuente de una figura ajena va en la
> «*Nota.*», debajo. `icomma` hace que «2,0» no salga «2, 0» en las fórmulas;
> por eso los intervalos se escriben `[0, 1]`, con espacio.

> **06-10-2026.** Cambios en la compilación:
>
> - **APA 7** con biblatex-apa y biber: los DOI como `https://doi.org/…`. Desde el 17-11-2026,
>   «&» entre los dos últimos autores en las citas entre paréntesis y en la lista (con la coma
>   de APA), como el ejemplo de `Formato(2025).docx`; «y» en la cita narrativa del texto y «&»
>   en la que va dentro de tablas y figuras (APA 8.17). Es lo que hace biblatex-apa por defecto.
>   El 17-11-2026 se quitó la rama natbib + apalike, que no era APA 7: sin
>   `biber` el documento no compila.
> - **Auxiliares en `research/build/aux/`**, ignorada por git. El PDF se copia
>   a `research/build/main.pdf` sólo si la compilación termina bien. Antes,
>   pdflatex lo borraba al fallar, y las subcarpetas que exige `\include` con
>   `-output-directory` no existían: `make` fallaba desde cero.
> - `make clean` ya no borra los PDF versionados.
> - «Referencias» va antes de los anexos y entra en el índice (APA 7).

> **17-11-2026. Redacción y limpieza.** Se partieron las oraciones encadenadas con «:» y
> «;» (de 139 marcadas a 10, todas listas o ecuaciones). Las rutas, los *scripts* y las
> funciones salieron de la prosa: solo quedan donde el IOV o el medio los nombra (RE1.2,
> RE1.3, RE2.2, RE2.4, RE2.5) y en el protocolo de RE3.1. El resto del texto dice «el
> programa de curación», «el de preparación», etc., y remite a `tab:donde` (cap. 5). Se
> borraron la presentación al equipo, los cambios desde el E2, los reportes `RE_*`,
> `unused.bib` y las notas que el PDF no usa; el Makefile ya no tiene `slides` ni
> `cambios`, y sin biber el documento no compila.

> **17-11-2026. Medios de verificación.** `research/medios_verificacion/generar.py` arma las once
> carpetas del Anexo B, cada una con su `README.md` (medio e IOV copiados del cap. 1, sección del
> documento, tabla elemento → archivo). Corre en local y en CPU; no necesita el servidor. Lee los
> `valores.tex` y las `tablas/` de los caps. 4–6, los `.aux` (números de sección), las constantes de
> curación, `runs/` (configuraciones, registros, volcados, puntos de operación, `best.pt`) y
> `06_oe3_revision/team/`. RE1.2 sale de los volcados: sus 1 248/690/601 grabaciones y sus ventanas
> coinciden con el cap. 4, y el registro de carga es el escaneo de Ultralytics (0 corruptas). El
> `best.pt` local tiene el mismo SHA-256 que el de la versión v0.2.0. Sin nombres: la planilla de tiempos
> pierde `notes` y `Observer`, y las personas van como A y B.

> **17-11-2026. Cuadernos comprobados en local.** No hizo falta el servidor (que además no es
> accesible desde aquí: el salto por `enterprise` pide contraseña). Los cuadernos de los caps. 1, 4,
> 5 y 6 se corrieron hacia una carpeta aparte (`TESIS_SALIDAS`), con las cifras de servidor
> conservadas, y todas las salidas (`valores.tex`, `tablas/`, `data/`, PNG) salen idénticas a las del
> documento. Eso comprueba lo que se había corregido a mano: grabaciones únicas (cap. 1), destino de
> los pares excluidos y par mayor elegible (cap. 4) y los controles (cap. 6). El desbalance en cajas
> (cap. 4, celda de servidor) se comprobó con las cajas del volcado de entrenamiento: `as/hc` 6 933,
> `sb/sc` 126, 55 veces, 11 clases bajo 300. El cap. 1 leía rutas del servidor en
> `recording_durations.csv`; ahora las resuelve bajo el `data/raw/` local. `make medios` no cambió
> nada.

## Qué está pendiente

Esta es la **única lista de pendientes** del proyecto (17-11-2026). Los README de cada
capítulo guardan la historia de lo resuelto, pero ya no llevan pendientes propios, y el
`.tex` no lleva comentarios `PENDIENTE`. Lo que se resuelva se borra de aquí.

### Datos que solo tiene el equipo

- [ ] **Datos de campo y de esfuerzo del equipo** (opcional: el texto ya los declara como
      faltantes). Son las horas grabadas y anotadas, los minutos de especialista por hora y
      el tiempo disponible, que el problema central menciona y no cuantifica. También la
      ganancia, la distancia al animal y el criterio con que se eligió lo que se anotó. El
      equipo preguntó si se refieren a la Call Library o a la campaña reciente.

### Antes de entregar el E3 (viernes 20-11)

- [ ] **Hacer commit y push** de todo lo del 17-11 (lo hace el autor). Nada tiene commit todavía.
      Incluir `05_oe2_detector/figures/tablas/pares.tex`, que lo necesita el PDF, la carpeta
      `research/medios_verificacion/` (la URL del Anexo B existe solo después del push) y los
      borrados (`research/presentacion/`, `research/reportes/`, `unused.bib`, `docs/informes/` y
      las notas sueltas de los caps. 4 y 5).
