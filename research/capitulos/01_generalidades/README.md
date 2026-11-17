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
> - **Pendiente, y es lo que más pesa:** un dato de magnitud del propio
>   equipo, como horas grabadas frente a anotadas o minutos de especialista por
>   hora de audio. Hoy toda la magnitud es de literatura.

> **06-10-2026.** Secciones 1.1 a 1.6 reestructuradas tras la observación del
> jurado (Problemática 0/4, Objetivos 1,75/4). Diagnóstico, cambios, propuestas
> sobre las tablas de IOV y decisiones pendientes: `REESTRUCTURACION.md`.
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

- [ ] La prosa que acompaña al OG y a RE3.1–RE3.3 se reescribió para el texto
      congelado (tiempo de revisión, revisión experta). Releerla entera: quedan
      giros del encuadre anterior («carga de revisión» como indicador principal).
- [x] 06-10-2026 (noche), árbol con dos causas, a pedido del autor. La C3
      anterior («no se ha medido cuánto ahorra la asistencia») era falsa: la
      propia §1.1.3 cita mediciones de Heinicke, HOWLish y Hong. Su contenido se
      repartió así:
      - «se midió para detectar una especie, no para la anotación completa»
        pasó a la brecha (eje de la tarea), como vacío;
      - la referencia incompleta pasó a C1, como ruido.

      La C2 anterior era la solución en negativo («no hay herramientas»). Ahora
      describe el material: llamadas escasas y breves que obligan a recorrer
      todo el audio (Heinicke: 179 h → 360 h).

      En los resultados esperados, cada OE abre con su vínculo al árbol y sus
      productos, las reglas de RE1.1 se presentan por el defecto que corrigen,
      y se quitaron dos justificaciones que no correspondían: la partición
      justificada por la variación de etiquetas (Koga) y la ficha justificada
      por transparencia (Gebru).
- [x] 06-10-2026 (noche), revisión estricta contra la rúbrica:
      - las fases de CRISP-DM y «Procedimientos» tienen etiqueta
        (`sec:fase2`…`sec:fase6`, `sec:procedimientos`), a la que remiten los
        «Cómo se alcanzó» de los capítulos 4 y 5;
      - la Fase 3 describe las aumentaciones que de verdad se usaron, no
        desplazamiento de tono ni estiramiento temporal, y presenta PCEN como
        variante implementada, no como comparación hecha;
      - §1.3.3 nombra la propuesta de solución;
      - se definen DETR, UICN e IC en su primer uso.
- [x] ~~`% TODO VERIFICAR` sobre Heinicke et al. (2015).~~ Resuelto el
      06-10-2026: Taï, Costa de Marfil (lo confirma Kalan et al. 2015).
- [ ] Decidir si los medios de verificación y los IOV pueden cambiar
      (`REESTRUCTURACION.md`, sección 4), y alinear después el Anexo A.
- [x] ~~Nombrar el sitio de grabación en §1.1.2.~~ Resuelto el 06-10-2026:
      Tambopata Research Center, Reserva Nacional Tambopata (Madre de Dios). La
      brecha (§1.3) aprovecha que el conjunto de aves de Inkaterra es de la
      misma región.
- [x] ~~Nombrar el equipo de investigación.~~ Resuelto el 07-10-2026: el
      8 Primates Project de Rainforest Expeditions (Wired Amazon), dirigido por
      Mark Bowler, se cita como `wiredamazon_8primates`. La misma página da el
      horario de campo (5–11 y 16–18 h), que explica el sesgo horario del
      capítulo 4.
- [ ] La tabla CRISP-DM se eliminó: decía exactamente lo mismo que la figura del
      ciclo, que sí indica objetivo y capítulo por fase. Comprobar que la figura
      sigue apuntando a los capítulos correctos tras la separación de OE2 y OE3
      (fase 5 → capítulos 5 y 6).
- [x] ~~El total de grabaciones del Capítulo 1 y el del Capítulo 4 difieren en
      una.~~ Resuelto el 02-10-2026: el cuaderno descartaba en silencio el único
      WAV ilegible (`night_monkey__AA/20240527_175248.wav`) y el capítulo 4 sí lo
      cuenta. Ahora se cuenta con duración 0 y los dos capítulos dicen 3 356. Las
      cifras del cuerpo salen de `valores.tex`, no escritas a mano.
- [x] ~~Repeticiones~~ **Resuelto el 07-10-2026.** El vínculo C1→C2 se explicaba cuatro veces (nodo del árbol, lectura de la figura, C1 y C2): C2 ya no lo repite. Se quitaron la segunda «decisión final sobre cada caja», la segunda «tercera forma de salida de la Figura 1.3» y las definiciones de mAP y FP/TP que el párrafo posterior copiaba de la Tabla de métricas.
- [x] ~~Árbol y texto desalineados~~ **Resuelto el 07-10-2026.** El nodo central reproduce el enunciado literal; el título de C2 lleva la misma cláusula que el árbol («y encontrarlas exige recorrer la grabación entera»); el sub-punto «ningún detector cubre estas especies» pasó a «detectores limitados a una o pocas especies» (lo que la evidencia sostiene, y sin contradecir E2, que cita detectores de aulladores); E1 ya no repite el sesgo de muestreo de C1 (era un ciclo efecto → causa); la vía indirecta de C1 nombra el sesgo; E2 se ata al tiempo disponible.
- [x] ~~Decisión del autor pendiente~~ **Resuelto el 07-10-2026 por el autor:** el problema central pasa a «Anotar las vocalizaciones de primates neotropicales en grabaciones de monitoreo acústico pasivo exige más tiempo de especialista del que está disponible» (la cláusula del ritmo frente a la recolección se movió a E1). C1 pasa a «La anotación manual es inconsistente y exige una segunda pasada del especialista»: su vía directa al problema es esa segunda pasada (643 anotaciones), y el rótulo «Sesgo» salió de C1 porque no explicaba el tiempo (el desbalance sigue en §1.1.2 y en la ficha de RE1.3). C2 dice «encontrarlas a mano»: las dos causas parten de que la anotación es manual, que se dice en la lectura del árbol y no como nodo (sería la negación de la solución). Antes: el enunciado incluía su propio efecto («y por ello avanza mucho más lento que la recolección» = E1) y dos magnitudes, aunque §Problema central dice «una sola». Recomendado: cortar la cláusula en el enunciado, el árbol y el Capítulo 7.
- [x] ~~Revisión de coherencia problema–árbol–objetivos~~ **Resuelto el 07-10-2026.** La vía directa de C1 es ahora la revisión que hace un especialista de mayor experiencia sobre lo que anotaron otros (dato del autor sobre el flujo del equipo), con la evidencia del Capítulo 4: 129 especies que contradicen la carpeta, 180 cajas sin tipo de llamada, pares imposibles y 643 anotaciones que ninguna regla decide. Las 866 filas de ruido no se citan como error porque pueden ser marcas intencionales. E2 ya no usa la cita de detectores de aullador (era la misma evidencia que la causa C2: circular); se apoya en el reparto del propio material, con la salvedad de la duración de cada llamada. El sub-punto de C2 pasó a «el costo crece con las horas grabadas». OE3 se justifica como el «evaluar» del OG.
- [x] ~~Citar el flujo de revisión del equipo~~ **17-11-2026:** citado como comunicación personal de B. Gacheva (`\cpEquipo`): la coordinación dirigió, instruyó y verificó la anotación, que hicieron voluntarios. La etapa 5 de `fig:proceso` y C1 ya no dicen «un especialista de mayor experiencia». Además, §material declara que el corpus es una biblioteca focal grabada con micrófono parabólico, y que el fin del sistema es apoyar el PAM del equipo (AudioMoth); el alcance y la explicación del OG lo dicen.
- [x] **17-11-2026, redacción de C1 y C3 (pedido del autor: el vínculo con el problema no se veía).** C1 pasa de «Faltan anotaciones utilizables para estas especies» a «Lo anotado sale con errores de registro que hay que corregir a mano». La falta de un estándar y de conjuntos públicos pasa a ser una viñeta. C3 pasa de «Faltan datos para validar los detectores, y por eso todo se revisa» a «No se sabe cuánto se equivocan las anotaciones ni las propuestas, y por eso todo se revisa». «Datos de validación» sigue en el párrafo, que es lo que pidió la observación del 16/11. El cambio se aplicó en el árbol, los títulos de párrafo, la trazabilidad, «OE frente a C», el problema seleccionado, el cap. 7 y la presentación `research/cambios/`. El problema central no cambia (el autor preguntó si agregar «lenta»: no se agregó, por redundante).
