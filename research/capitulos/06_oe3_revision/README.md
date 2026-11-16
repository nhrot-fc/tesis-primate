# Capítulo 6 — Tiempo de Revisión y Análisis de Errores (OE3)

> **OE3.** Cuantificar la reducción del tiempo de revisión que aporta el sistema y caracterizar
> sus errores mediante revisión experta de una muestra de predicciones.

El `.tex` está redactado (15-11-2026) con las respuestas de las tareas 1 y 2. Todas sus cifras
son macros `\u…` de `figures/valores.tex`, y sus dos tablas generadas son
`figures/tablas/sesion.tex` y `figures/tablas/estratos.tex`; las escribe el cuaderno de figuras.

## Archivos

| Archivo | Qué es |
|---|---|
| `06_oe3_revision.tex` | el capítulo |
| `protocolo.md` | las reglas de la evaluación (RE3.1); quedan fijadas con el commit de esta versión, antes de enviar el paquete |
| `instrucciones.md` | plantilla del `INSTRUCTIONS.md` que recibe el revisor |
| `notebook/paquete_06_oe3_revision.ipynb` | arma `team/package/` y `team/data/` |
| `notebook/figuras_06_oe3_revision.ipynb` | lee `team/answers/` y escribe `figures/` (macros `\u…`) |
| `team/` | **fuera de git**: claves de la revisión ciega y ~1 GB de audio |

Contenido de `team/`. Al equipo solo se envía `package/`: `data/` tiene las claves de la revisión ciega.

| Carpeta o archivo | Qué es |
|---|---|
| `message.md` | el mensaje de Discord a la persona de contacto del equipo, en inglés |
| `package/` | lo que se envía: `INSTRUCTIONS.md` y una carpeta por tarea |
| `answers/` | lo que devuelve el equipo, tal cual |
| `data/` | lo que **no** se envía: `test_detections.csv` (tablas de la herramienta en prueba), `test_population.csv`, `check_boxes_key.csv` y `dataset_quality_key.csv` (claves de las tareas 2 y 3), `timed_session_folders.csv` (carpetas de la tarea 1), `test_summary.json` y `test_cpu_times.csv` |

- El cuaderno del paquete reproduce la misma muestra mientras no cambien la semilla ni
  `runs/yolo26s_v3/hallazgos_val.csv` y `runs/yolo26s_v3_kfold5/hallazgos_train.csv`.
- `data/test_cpu_times.csv` salió de la corrida del 09-10-2026 que generó
  `test_detections.csv`. El cuaderno actual no lo regenera.

## Cómo se redacta

- Secciones: RE3.1 (desde `protocolo.md`), RE3.2, RE3.3 y la discusión con las cinco preguntas
  de la pauta.
- RE3.2 es la única cifra que responde al objetivo general.
- No se anticipa ningún ahorro, ni en la introducción ni en el resumen.
- Una predicción sin anotación que nadie revisó es indicio de omisión, no prueba.
- La unidad es la fila de la tabla de Raven, no la detección por ventana del capítulo 5, así que
  las cifras no se comparan una a una con las de ese capítulo. Ya está medido:

  | Medida | Por fila | Por ventana |
  |---|---|---|
  | Precisión | 0,45 (2 150 de 4 792) | 0,665 |
  | Cobertura | 0,78 | 0,81 |

## Criterio del autor (14-11-2026)

**OE3 y sus RE no se reajustan.** Cada sección reporta lo que se obtuvo frente a su IOV,
tal como está definido, y lo que no salió como se planeó se explica como limitación. No se
cambia el énfasis del capítulo hacia la tarea 2.

## Estado (15-11-2026)

- **Tarea 1 hecha** (sesión del 11 al 13-11; `times.xlsx` solo trae horas).
- **Tarea 2 hecha** (15-11): las 1 000 filas respondidas, en `answers/boxes.xlsx`. Las 310
  grabaciones vuelven repartidas en `answers/checked/` (225) y en una carpeta por persona dentro
  de `answers/audio/` (42 y 43), intercaladas a lo largo de la planilla. Las tablas
  `.boxes.txt` vuelven idénticas a las enviadas. Falta la nota que pide `INSTRUCTIONS.md`:
  quién respondió qué filas, sus años de experiencia y si anotaron antes alguna de estas
  grabaciones.
- **Tarea 3**, el mes siguiente: queda fuera de E3.
- El protocolo quedó **fijado en git con el commit 63f84a8 (10-10-2026)**, antes de la
  sesión. Esa versión (`protocolo_oe3.md`) ya trae todas las reglas, también las de la sesión
  cronometrada. La versión del 27-10 (`protocolo.md`) reordena el texto y no tiene commit.

Desviaciones que el capítulo debe declarar:

- la sesión la hicieron dos personas del equipo, no una. Las notas de `times.xlsx` dicen
  quién hizo qué: se repartieron las carpetas 1 y 4, y la 2 y la 3 las hizo sobre todo una
  de ellas. En el capítulo van como «persona A» y «persona B», sin nombres;
- dentro de los intervalos de las carpetas 1, 2 y 3 hay 15 minutos de revisión de la otra
  persona;
- la tarea 2 la respondió más de una persona y cada fila una sola vez, así que no hay kappa
  (protocolo, §5, «Acuerdo»). Dentro de una misma especie, las carpetas difieren sobre todo en
  cuánto usan `box` frente a `correct`;
- la planilla volvió como `.xlsx`, no como `.csv`, con una fila final de totales y tres erratas
  en `answer`: C046 `corect`, C445 `other ` (con espacio) y C671 `too short`, que se lee `box`;
- el comentario «missing selection of vocalizations», que está en todas las filas de 239 de las
  310 grabaciones, no es parte del formulario y no se usa. La tabla mostraba solo las cajas
  sorteadas, así que siempre faltaban llamadas por seleccionar: no mide lo que omite el modelo;
- de las 633 cajas «nada anotado», 30 tienen al menos la mitad dentro de una anotación de otra
  clase (22 respondidas `other`, casi siempre nombrando esa clase). La lectura del protocolo
  las cuenta como evento no anotado; sin ellas, la proporción estratificada baja de 75,9 % a
  71,5 %. Se reporta como sensibilidad, sin cambiar la regla.

## Cuaderno de figuras (15-11-2026)

- Lee `.xlsx` (sin openpyxl: el archivo es un zip de XML) o `.csv`, corrige a la vista las tres
  erratas (`TYPOS`) y no lee la columna de notas de `times.xlsx`, que trae nombres.
- Las personas van como A y B por el orden alfabético de sus carpetas en `answers/audio/`:
  A es la que solo había trabajado con aulladores; B, la que tenía experiencia con las ocho
  especies (columna «Experience» de `times.xlsx`).
- La cobertura de la sesión se mide sobre la referencia de las 56 grabaciones: una tabla
  devuelta vacía no saca su grabación de la cuenta (protocolo, §7). Una grabación asistida
  devuelta vacía cuenta todas sus propuestas como borradas.
- Lecturas adicionales, marcadas como tales en el capítulo: la sensibilidad de las cajas dentro
  de una anotación de otra clase, el remuestreo por grabación (semilla 42) y el reparto del
  error aparente con la precisión según el revisor.
- La única figura es `revision_ejemplos.png`; el reparto por estrato va en la tabla, con sus
  intervalos.

## Pendientes

- [x] ~~Cuaderno de figuras con las respuestas reales.~~ Hecho el 15-11-2026.
- [x] ~~Redactar RE3.1, RE3.2, RE3.3 y la discusión.~~ Hecho el 15-11-2026, sin reajustar OE3
      ni sus RE: cada resultado se reporta frente a su IOV y lo que no salió como se planeó va a
      las limitaciones.
- [x] ~~Decidir qué hacer con `protocolo.md`~~ **16-11-2026:** ya tiene commit (c39dda6, d7f8adb). El capítulo cita el commit
      63f84a8 (10-10-2026) como la fecha de las reglas. Si se hace commit ahora, el mensaje
      debe decir que es la versión reordenada del 27-10, sin cambios de reglas; su encabezado
      («La fecha es la del commit de esta versión») y «Cambios después del commit (ninguno)»
      ya no son exactos.
- [ ] Llenar las carpetas de OE3 de `research/medios_verificacion/`, que el Anexo B ya describe
      (ver `09_anexo_medios/README.md`). Los tiempos van sin la columna de notas.
- [ ] `research/reportes/RE_3-3_taxonomia-discrepancias` describe el corpus entero con el
      k-fold en train. No entró al capítulo: es el material de la tarea 3, que queda fuera de E3 (el equipo la hará en 2027).
- [x] ~~En `07_conclusiones.tex`, las filas de RE3.1–RE3.3, el párrafo de OE3 y el veredicto
      del objetivo general.~~ Hecho el 15-11-2026. El veredicto es «parcialmente cumplido»: el
      capítulo 1 fija como criterio de éxito la reducción del tiempo, y la sesión no la muestra.
- El capítulo 1 no se tocó, por decisión del autor (14-11-2026).
