# Capítulo 1 — Generalidades

Fija el problema y el contrato del proyecto: qué se quiere resolver, qué se
compromete a entregar y con qué se va a verificar. Todo lo que los capítulos de
objetivos dan por sentado se declara aquí.

## Qué va aquí

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

| Sección | Contenido |
|---|---|
| Problemática | Definición del problema: narrativa y sin desglosar el enunciado. Primates difíciles de seguir → PAM → el 8 Primates Project del TRC y sus ocho especies → la anotación como paso más lento → enunciado. Luego «El material y su anotación» (Raven, Tabla 1.1), magnitud y relevancia |
| Árbol de problemas | Un solo problema central y **dos causas** (06-10-2026, noche): C1, las únicas anotaciones disponibles tienen ruido, sesgo e inconsistencias; C2, las vocalizaciones son escasas y breves dentro de horas de audio. C1 → C2 punteado. Efectos E1–E3 por actor. OE3 no cuelga de una causa: mide el problema central y la calidad de la referencia (C1) |
| Brecha de investigación | Soluciones existentes por forma de salida, vacío en dos ejes (taxón y tarea) y alcance de la tesis |
| Objetivos | OG y OE1–OE3, **texto congelado** (`objetivos.md`), más un párrafo «OE*n* frente a C*n*» por objetivo |
| Resultados esperados | Las tres tablas de IOV, **texto congelado**; términos definidos antes de cada tabla y «Aporte de RE*x.y*» después |
| Trazabilidad | Causa → objetivo → resultados → aporte, y problema central → OG |
| Métodos y procedimientos | CRISP-DM, herramientas y reproducibilidad: es el capítulo al que apuntan los «cómo se alcanzó» de los capítulos 4, 5 y 6 |

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
- [ ] **Citar el flujo de revisión del equipo** como comunicación personal (APA 7, solo en el texto) cuando M. Bowler lo confirme. Hay un comentario PENDIENTE en el párrafo de las dos vías de C1.
