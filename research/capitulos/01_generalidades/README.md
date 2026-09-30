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

Todas salen de `notebook/figuras_01_generalidades.ipynb` salvo donde se indique.

| Figura | Qué muestra | Origen |
|---|---|---|
| `annotation_example.png` | Anotación tiempo–frecuencia en Raven | cuaderno |
| `output_forms.png` | Tres formas de salida de un evento (segmento, límites, caja) | cuaderno |
| `problem_tree.png` | Árbol de problemas | cuaderno |
| `recorded_vs_annotated.png` | Material recibido por especie | cuaderno (cachea duraciones en `figures/data/recording_durations.csv`) |
| `crisp_dm.png` | Las seis fases de CRISP-DM | cuaderno |
| `iou_criterion.png` | Criterio de IoU | cuaderno |

El logo de la portada vive en `research/figures/pucp-logo.png`, fuera de los
capítulos, porque lo usa la carátula y no este capítulo.

## Pendientes

- [ ] La prosa que acompaña al OG y a RE3.1–RE3.3 se reescribió para el texto
      congelado (tiempo de revisión, revisión experta). Releerla entera: quedan
      giros del encuadre anterior («carga de revisión» como indicador principal).
- [ ] `% TODO VERIFICAR` sobre Heinicke et al. (2015): confirmar que son primates
      africanos y que la redacción no insinúa que sean neotropicales.
- [ ] Comprobar que la tabla CRISP-DM sigue apuntando a los capítulos correctos
      tras la separación de OE2 y OE3 (fase 5 → capítulos 5 y 6).
