# Protocolo de evaluación de OE3 (RE3.1)

Reglas fijadas antes de enviar el paquete al equipo. La fecha es la del commit de esta versión.
No cambian después; un cambio forzoso se anota al final, con fecha y motivo.

Implementación: `notebook/paquete_06_oe3_revision.ipynb` (paquete),
`notebook/figuras_06_oe3_revision.ipynb` (análisis), `src/analysis/review.py` (reglas) e
`instrucciones.md` (lo que lee el revisor).

## 1. Qué se evalúa

| | |
|---|---|
| Modelo | `yolo26s_v3` (capítulo 5) |
| Umbral | score ≥ 0,11, elegido en validación |
| Salida | tabla de Raven de `detect.py` por grabación: ventanas de 3 s, salto de 1,5 s, NMS del modelo, duplicados entre ventanas fundidos |
| Unidad | una fila de esa tabla, no una detección por ventana como en el capítulo 5 |

## 2. Datos

- Prueba: 601 grabaciones, 4,93 h. El modelo no las vio al entrenar ni al elegir el umbral.
- Referencia: anotaciones del equipo limpiadas con RE1.1, de las 25 clases del modelo, sin las
  filas en revisión.
- Se excluyen las detecciones de las clases de ventana `as/hc` y `pt/dc`.
- Las tareas 1 y 2 usan grabaciones distintas.

## 3. Emparejamiento y categorías

Acierto: misma clase e IoU ≥ 0,3, en segundos × escala mel. En orden de score, cada detección
toma la anotación libre de mayor IoU; cada anotación se usa una vez.

Una detección sin acierto toma la primera categoría que cumpla:

| | Categoría | Regla |
|---|---|---|
| 1 | Caja distinta | IoU ≥ 0,3 con una anotación de su clase ya tomada |
| 2 | Otra llamada | IoU ≥ 0,3 con una anotación de otra clase |
| 3 | Caja distinta | IoU > 0 con una anotación de su clase |
| 4 | Nada anotado | ninguna de las anteriores |

Una «nada anotado» que toca una anotación fuera de la referencia (en revisión, o de una clase que
el modelo no tiene) no se revisa.

## 4. Tarea 1: tiempo de revisión (RE3.2)

**Material.** 4 carpetas de 14 grabaciones de prueba: 2 de cada una de las 7 especies con algún
tipo entrenado (sin *Cebus*). Cada grabación dura ≤ 2 min y tiene ≥ 1 anotación. Es el mejor de
20 000 sorteos para igualar minutos y anotaciones: 5,9–6,1 min y 47–51 anotaciones por carpeta,
23,8 min en total.

**Condiciones.** Manual: anotar desde cero. Asistida: partir de la tabla del modelo, con su
score, y borrar, corregir y agregar. Mismo revisor, estándar y ajustes de Raven.

**Orden.** La tarea 1 va antes que la 2, cuando el revisor aún no conoce las cajas del modelo.
Carpetas: manual, asistida, asistida, manual.

**Medidas.**

- Tiempo de una carpeta: término − inicio − pausas.
- Por condición: minutos de trabajo por hora de audio, y el cociente asistida / manual. No se fija
  un ahorro esperado.
- Calidad de cada tabla entregada frente a la referencia (§3): cobertura y precisión.
- Cada caja del modelo, en la asistida: igual (la entregada más parecida tiene IoU ≥ 0,9 y su
  clase), corregida (IoU ≥ 0,3) o borrada. Agregadas: las entregadas con IoU < 0,3 frente a todas
  las del modelo. Una tabla del modelo devuelta sin cambios cuenta como aceptada.
- Costo del modelo: minutos de cómputo por hora de audio en GPU (capítulo 5) y en CPU.

## 5. Tarea 2: revisión de cajas (RE3.3)

**Muestra.** 1 000 cajas en 310 grabaciones, al azar dentro de cada estrato (semilla 42). IC 95 %
en el peor caso: ±6 puntos con 300 cajas, ±4 con las 633 de «nada anotado».

| Estrato | Población | Muestra |
|---|---|---|
| Nada anotado, score > 0,5 | 583 | 300 |
| Nada anotado, 0,3–0,5 | 638 | 200 |
| Nada anotado, 0,11–0,3 | 1 016 | 133 |
| Caja distinta | 220 | 220 |
| Otra llamada | 87 | 87 |
| Acierto (control) | 2 150 | 60 |

**Planilla.** Grabaciones en orden aleatorio y, dentro de cada una, cajas por tiempo. Se responde
de arriba abajo sin saltar filas.

**Ciego.** El revisor ve la caja y la clase propuesta. No ve el score, el estrato ni la
referencia, y no sabe que hay controles.

**Respuesta.** Una palabra por caja, más un comentario opcional:

| Respuesta | Significa |
|---|---|
| `correct` | es la llamada propuesta y la caja la encuadra |
| `other` | es una llamada de primate de otra especie o tipo, aunque la caja también esté mal |
| `box` | es la llamada propuesta, con la caja mal puesta |
| `no` | no hay llamada de primate |
| `?` | no se puede decir |

**Lectura.**

| Estrato | `correct` | `other` | `box` | `no` |
|---|---|---|---|---|
| Nada anotado | evento no anotado | evento no anotado | evento no anotado | ruido |
| Caja distinta | anotación por corregir | error de clase | error de encuadre | ruido |
| Otra llamada | anotación por corregir | error de clase | error de encuadre | ruido |
| Acierto | acierto confirmado | desacuerdo con la referencia | ídem | ídem |

`?` es incierto. El error de clase se reporta aparte: no es una de las tres categorías del IOV.

**Estimación.**

- Por estrato: proporción de cada respuesta, con IC de Wilson al 95 %.
- «Nada anotado» en conjunto: estimador estratificado, con cada tramo pesado por su población e
  IC normal. De ahí salen los eventos no anotados en prueba y por hora de audio.

**Si queda incompleta.** Se analizan las filas respondidas, que son las primeras: una muestra por
grabaciones de cada estrato. Se reporta el número revisado por estrato y que el IC de Wilson no
corrige por grabación.

**Acuerdo.** Si una segunda persona responde por separado, kappa de Cohen sobre la respuesta y
sobre si hay llamada. Si no, se declara como limitación.

## 6. Tarea 3, opcional: calidad del corpus

Fuera del IOV. Son los hallazgos de CLOD (`src/find_issues.py`) sobre predicciones fuera de
muestra: en train, del k-fold `yolo26s_v3_kfold5`; en validación, del modelo final. Se responden
con las mismas cinco palabras.

| Hallazgo | Qué indica | Caja mostrada | Train | Val. | Se confirma con |
|---|---|---|---|---|---|
| `missing` | llamada predicha con confianza, sin anotar | la del modelo | 1 961 | 734 | `correct`, `other`, `box` |
| `spurious` | anotación que ninguna predicción toca | la anotación | 2 232 | 785 | `no` |
| `location` | anotación mal ubicada | la anotación | 1 310 | 487 | `box` |
| `label` | anotación con otra etiqueta | la anotación | 164 | 55 | `other` |

- 7 728 hallazgos en 1 538 grabaciones. Un `missing` entra solo si ninguna fila cruda está bajo
  la caja.
- El revisor no sabe si la caja es del modelo o una anotación.
- Orden: al azar dentro de cada tipo, con los tipos intercalados. Se responde de arriba abajo
  hasta donde alcance.
- Estimación por tipo: proporción confirmada (IC de Wilson, sobre lo respondido, incluidos los
  `?`) × hallazgos del tipo = errores de anotación estimados. La proporción de `no` entre los
  `missing` de train es la tasa de falsos positivos del k-fold.

## 7. Reglas

- Ninguna regla cambia después de ver respuestas.
- Ninguna caja ni grabación se descarta por su resultado. Las filas sin respuesta se cuentan y se
  declaran.
- La sesión no se repite.

## 8. Limitaciones previstas

- El audio de prueba es más denso en llamadas que una grabación continua: el tiempo por hora no
  se extrapola a una temporada.
- Un solo revisor y 23,8 min de audio.
- El revisor puede haber anotado alguna de estas grabaciones: se le pregunta y se declara.

## Cambios después del commit

(ninguno)
