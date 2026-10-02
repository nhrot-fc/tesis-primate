# Capítulo 1 — Generalidades

Fija el problema y el contrato del proyecto: qué se quiere resolver, qué se
compromete a entregar y con qué se va a verificar. Todo lo que los capítulos de
objetivos dan por sentado se declara aquí.

## Qué va aquí

| Sección | Contenido |
|---|---|
| Problemática | PAM, el cuello de botella del análisis, trabajos previos y el vacío que ocupa la tesis |
| Árbol de problemas | Condiciones subyacentes, problema central, causas C1–C3 y efectos E1–E3 |
| Objetivos | OG y OE1–OE3, **texto congelado** (`objetivos.md`) |
| Resultados esperados | Las tres tablas de IOV, **texto congelado**, más la justificación de cada RE |
| Trazabilidad | Causa → efecto → objetivo → resultados esperados |
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
- [ ] `% TODO VERIFICAR` sobre Heinicke et al. (2015): confirmar que son primates
      africanos y que la redacción no insinúa que sean neotropicales.
- [ ] La tabla CRISP-DM se eliminó: decía exactamente lo mismo que la figura del
      ciclo, que sí indica objetivo y capítulo por fase. Comprobar que la figura
      sigue apuntando a los capítulos correctos tras la separación de OE2 y OE3
      (fase 5 → capítulos 5 y 6).
- [x] ~~El total de grabaciones del Capítulo 1 y el del Capítulo 4 difieren en
      una.~~ Resuelto el 02-10-2026: el cuaderno descartaba en silencio el único
      WAV ilegible (`night_monkey__AA/20240527_175248.wav`) y el capítulo 4 sí lo
      cuenta. Ahora se cuenta con duración 0 y los dos capítulos dicen 3 356. Las
      cifras del cuerpo salen de `valores.tex`, no escritas a mano.
