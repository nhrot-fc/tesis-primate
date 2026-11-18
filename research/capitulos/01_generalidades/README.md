# Capítulo 1 — Generalidades

Fija el problema y el contrato del proyecto: qué se quiere resolver, qué se
compromete a entregar y con qué se va a verificar. Todo lo que los capítulos de
objetivos dan por sentado se declara aquí.

## Qué va aquí

> **16-11-2026.** Observaciones: transición brusca del problema general al
> seleccionado, falta una causa sobre los datos de validación, y «validar no es
> el objetivo del proyecto». La entrada de 1.1 describe ahora el proceso del
> monitoreo en seis etapas (`fig:proceso`: instalación, grabación, búsqueda,
> delimitación y etiquetado, revisión, uso), con las tres manuales agrupadas como
> anotación, y de ahí pasa al enunciado. C3 pasa a «Faltan datos para validar los
> detectores»: una referencia de calidad conocida y el experimento de tiempo, los
> dos productos de OE3. OE3 ya no se justifica como «el evaluar del OG». OG, OE e
> IOV no cambiaron.

> **14-11-2026.** Árbol reestructurado por observación del jurado: cada OE ataca
> una causa, las causas son independientes y los efectos no se encadenan. La C1
> anterior se dividió en dos: lo que resuelve una regla quedó en C1 y el
> desacuerdo de criterio y las omisiones pasaron a C3. Se agregó el fin (el
> inverso de los efectos) en la sección de objetivos, y una columna «Se mide con»
> en la trazabilidad. El texto de OG, OE e IOV no cambió (opción A).

> **07-10-2026.** Calificación estricta, Problemática 2,25/3:
>
> - **C1 → problema central.** Tiene ahora su párrafo: la vía directa es que lo
>   ya anotado vuelve al especialista (`\nReview` anotaciones), y la indirecta
>   pasa por C2. Las viñetas del árbol coinciden con las tres clases de defecto
>   de C1; «repositorios casi vacíos» salió, porque era escasez y no ruido.
> - **Tabla 1.1.** El texto aclara que el material es la selección que el equipo
>   organizó para anotar. Sin eso, el 81 % anotado parecía contradecir el
>   problema.
> - Fase 3 ya no presenta PCEN como variante.
> - Faltaba un dato de magnitud del propio equipo. Desde el 17-11-2026 el
>   capítulo da los minutos por hora de audio de la sesión del cap. 6; el resto
>   está en la lista única de pendientes.

> **06-10-2026.** Secciones 1.1 a 1.6 reestructuradas tras la observación del
> jurado (Problemática 0/4, Objetivos 1,75/4). Diagnóstico, cambios, propuestas
> sobre las tablas de IOV: estaban en `REESTRUCTURACION.md`, que ya no existe.
>
> **06-10-2026 (segunda pasada).** Tras la revisión completa (Redacción 1/2:
> repeticiones, tablas «al aire», referencias adelantadas), se recortaron las
> ideas repetidas entre la problemática, el árbol y la brecha; E3 remite a la
> relevancia en vez de repetirla; las dos tablas de herramientas se presentan una
> a una y la primera se interpreta (sólo Raven viene impuesta por el problema).

Desde el 14-11-2026 las secciones siguen el formato del curso
(`research/Formato(2025).docx`): 1.1 Problemática, 1.2 Objetivos y 1.3 Métodos y
procedimientos, con las subsecciones de la plantilla.

| Sección | Contenido |
|---|---|
| 1.1 Problemática | Entrada narrativa: primates difíciles de seguir → PAM → el 8 Primates Project del TRC y sus ocho especies → problema general (el análisis no sigue a la grabación) → proceso en seis etapas (Figura 1.1) → la anotación como paso más lento → enunciado del problema |
| 1.1.1 Árbol de problemas | Figura y su lectura. Un problema central y **tres causas independientes**: C1, faltan anotaciones utilizables (→ OE1); C2, encontrar las vocalizaciones exige recorrer la grabación entera (→ OE2); C3, faltan datos para validar los detectores (referencia de calidad conocida y experimento de tiempo), y por eso todo se revisa (→ OE3). Tres efectos sobre los estudios, sin flechas entre ellos |
| 1.1.2 Descripción | El material y su anotación (Raven, Tabla 1.1), magnitud, problema central, causas (cada una con su medida) y efectos |
| 1.1.3 Problema seleccionado | Por qué se elige: relevancia (aquí queda la conservación), soluciones existentes por forma de salida, vacío en dos ejes (taxón y tarea) y alcance |
| 1.2 Objetivos | Fin (inverso de los efectos), 1.2.1 OG y 1.2.2 OE, **texto congelado** (`objetivos.md`), con un párrafo «OE*n* frente a C*n*» |
| 1.2.3 Resultados esperados | Términos y «Aporte de RE*x.y*» por objetivo |
| 1.2.4 Mapeo | Las tres tablas de IOV, **texto congelado**, con la fila del objetivo como en el formato, y la trazabilidad causa → objetivo → resultados → medida |
| 1.3 Métodos y procedimientos | Tabla por resultado (método, herramientas, validación), CRISP-DM, herramientas y reproducibilidad: es la sección a la que apuntan los «cómo se alcanzó» de los capítulos 4, 5 y 6 |

## Cómo se redacta

- **Los objetivos y los resultados esperados no se tocan.** El enunciado del OG,
  el de cada OE y las tres tablas de IOV se copian literalmente de
  `objetivos.md`. Si un capítulo posterior no puede cumplir un IOV, se declara
  como limitación ahí; no se reescribe el compromiso aquí.
- La problemática carga el peso de las citas y es narrativa; el árbol de
  problemas es la síntesis estructurada, con una cita por elemento. No repetir
  el mismo texto en las dos.
- La sección de métodos define el *medio de verificación* de cada RE. Cada
  resultado de los capítulos 4–6 debe poder decir «se verificó con el medio
  definido en el Capítulo 1».

## Imágenes
> **30-09-2026.** Los diagramas de globos y flechas (árbol de problemas, CRISP-DM, cadena de la señal, EDT) se pasaron de matplotlib a **TikZ dentro del `.tex`**: usan la tipografía del documento, no se pixelan y el flotante se coloca donde corresponde. Los PNG se borraron; no hay que regenerarlos.


> **30-09-2026 / 02-10-2026.** La figura de barras `recorded_vs_annotated.png`
> pasó a ser tabla: apretaba ocho especies en dos paneles y los números no se
> leían. Y las **dos** tablas por especie que había (material recibido y material
> anotado) se fundieron en una sola, la 1.1, porque repetían columnas y sus
> totales no cuadraban entre sí ni con el capítulo 4.
>
> El cuaderno de este capítulo ya no dibuja nada: su única celda de salida escribe
> `figures/tabla_material.tex` (las filas) y `figures/valores.tex` (las macros con
> prefijo `g`, cargadas desde `main.tex`). Las celdas que generaban `problem_tree`
> y `crisp_dm` se eliminaron.
>
> **Ojo con `recording_durations.csv`**: es un caché y no se invalida solo. Si
> cambia `raw/`, hay que borrarlo antes de reejecutar o las cifras del capítulo 1
> se quedan en el estado anterior. Así se coló la discrepancia de 2026.

Las tres figuras que quedan las genera **este** cuaderno. Hasta el 02-10-2026 las
generaba el del capítulo 2, que no las usa, y existían copias idénticas en las dos
carpetas: regenerar un capítulo dejaba las del otro viejas sin avisar.

| Figura | Qué muestra | Origen |
|---|---|---|
| `annotation_example.png` | Anotación tiempo–frecuencia en Raven | cuaderno |
| `output_forms.png` | Tres formas de salida de un evento (segmento, límites, caja) | cuaderno |
| `iou_criterion.png` | Criterio de IoU | cuaderno |

El logo de la portada vive en `research/figures/pucp-logo.png`, fuera de los
capítulos, porque lo usa la carátula y no este capítulo.

## Pendientes

En la lista única de `research/capitulos/README.md`.

