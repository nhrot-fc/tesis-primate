# Pendientes de RE3.1, RE3.2 y RE3.3

Anotado el 06-10-2026, a pedido del autor: «los RE3.x aún no te preocupes, pero
anótalos». Nada de esto se resolvió todavía. Según el cronograma del Anexo A,
OE3 corresponde a **E3** (entregable final, 23/11/26).

Los textos del OG, de los OE y de las tablas de IOV están congelados
(`objetivos.md`). Lo que sigue son los puntos en que esos compromisos chocan
con lo que ya existe, o en que todavía falta trabajo.

## 1. Punto de operación (RE3.1)

> **Resuelto el 07-10-2026.** El autor reescribió el IOV de RE3.1: «Especifica el
> subconjunto de prueba, el punto de operación elegido en validación y el umbral de
> IoU para verdadero positivo, fijados antes de medir, …». Se actualizaron
> `objetivos.md`, la Tabla 1.6, §1.5.3, §1.7.1 y la Tabla A.1. El capítulo 6 debe
> reportar en el umbral del modelo seleccionado (0,11 para `yolo26s_v3`) y
> emparejar a IoU 0,3, como el capítulo 5. Lo que sigue es el análisis anterior.

- **El IOV congelado** de RE3.1 compromete confianza 0,5, NMS IoU 0,3 y
  verdadero positivo a IoU 0,5. Lo repiten la Tabla A.1 (criterios de
  aceptación) y los comentarios de `06_oe3_utilidad.tex`.
- **El modelo final** (YOLO26s, `yolo26s_v3`) opera a **0,11**, umbral elegido
  en validación (Capítulo 5, RE2.4). El protocolo de comparación empareja a
  IoU 0,3.
- **Decisión pendiente:**
  - o se reporta en el punto comprometido (0,5) y el de validación va como
    lectura secundaria;
  - o se pide cambiar el IOV a «el punto de operación fijado en validación
    (RE2.4)». Las propuestas están en
    `01_generalidades/REESTRUCTURACION.md`, sección 4.
- **Después de decidir:** alinear el IOV, la Tabla A.1 del anexo y la sección
  RE3.1 del capítulo 6.

> **09-10-2026.** Los puntos 2 y 3 tienen ya protocolo (`protocolo_oe3.md`), paquete para el
> equipo y cuaderno de análisis; ver el README del capítulo. Falta fechar el protocolo, enviar
> el paquete y recibir las respuestas. La muestra de RE3.3 sale de prueba con el modelo final, no
> del k-fold: el k-fold y CLOD siguen sirviendo para la descripción del corpus entero.

## 2. Medición del tiempo de revisión (RE3.2)

- Es la única cifra que responde al objetivo general: minutos por hora de audio
  en condición manual y asistida, sobre material comparable, con el orden de
  revisión declarado, más el tiempo de procesamiento del modelo por hora de
  audio.
- **Requiere una sesión cronometrada con el equipo de investigación, que no se
  ha hecho.** Es el riesgo R3 del anexo.
- El costo de inferencia por modelo ya está medido (Capítulo 5, sección de
  costo de inferencia); falta el del revisor humano.

## 3. Análisis de errores (RE3.3)

- **Categorías.** El IOV compromete tres (evento no anotado, ruido, error de
  encuadre). El análisis existente usa cuatro (A, B, C, D). La correspondencia
  ya está en la tabla de taxonomía del Capítulo 1:
  - A y B: error de encuadre;
  - C: error de clase, que se reporta aparte;
  - D: evento no anotado o ruido, según lo resuelva el revisor.
- **Material por trasladar:** `research/reportes/RE_3-3_taxonomia-discrepancias`
  (tabla de hallazgos, caracterización de D, ejemplos). Necesita el volcado
  fuera de muestra del modelo seleccionado (k-fold).
- **Revisión experta.** La muestra de predicciones de alta confianza sin
  anotación se resuelve en la misma sesión con el equipo que alimenta RE3.2.

## 4. Cronograma (Bloque 3 del Anexo A)

El cronograma original que entregó el autor tenía en el Bloque 3 tareas del OE3
anterior, que ya no coinciden con los RE3.x congelados. En la Tabla
`tab:cronograma` se reetiquetaron con los paquetes de la EDT (6.3 a 6.6) y **se
conservaron las fechas** del original:

| Original | En el anexo |
|---|---|
| 3.1 RE3.1 «Metodología de validación frente a anotaciones de expertos», 22/10–25/10 | 3.1 RE3.1 «Metodología de validación», mismas fechas |
| (no había 3.2; hueco entre el 25/10 y el 02/11) | — |
| 3.3 RE3.2 «Acuerdo inter-anotador sobre subconjunto de prueba», 02/11–09/11 | 3.2 RE3.2 «Medición del tiempo de revisión manual y asistida», mismas fechas |
| 3.4 RE3.2 «Informe de análisis comparativo modelo vs experto», 09/11–14/11 | 3.3 RE3.3 «Análisis de errores y revisión experta de la muestra», mismas fechas |
| 4.1 «Redacción del informe final», asignada a RE3.3 | 4.1 «Cierre», ver el README del anexo |

**Hay que confirmar** las tareas y las fechas. El acuerdo entre anotadores ya
no es un RE: figura como trabajo futuro en las conclusiones.

## 5. Lo que depende de RE3.x en otros capítulos

- `07_conclusiones.tex`: filas de RE3.1–RE3.3 en la tabla de cumplimiento, el
  párrafo de OE3, el veredicto del OG y el párrafo de OE3 frente a C3 en «La
  propuesta frente al problema central». Todos marcados con `PENDIENTE`.
- `01_generalidades`: releer la prosa del OG y de RE3.1–RE3.3 (pendiente del
  README).
- `06_oe3_utilidad.tex`: el capítulo entero es un esqueleto.
