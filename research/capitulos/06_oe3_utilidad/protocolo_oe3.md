# Protocolo de evaluación de OE3 (RE3.1)

> **Fijado el: ____ /____ /2026.** Se fecha *antes* de enviar el paquete al equipo y antes de
> leer cualquier respuesta. Después de esa fecha, nada de lo que sigue se cambia a la vista de
> los resultados: ni el umbral, ni los estratos, ni las reglas de lectura. Lo que haya que
> cambiar por fuerza mayor se anota al final, con su fecha y su motivo.

Este documento es la base de la sección de metodología del capítulo 6. Lo implementan
`notebook/preparar_revision.ipynb` (arma la muestra y la sesión) y `notebook/analisis_oe3.ipynb`
(lee lo que devuelve el equipo); el código común está en `src/analysis/review.py`.

## 1. Qué se evalúa

El sistema tal como lo usaría el equipo:

- **Modelo:** el seleccionado en el capítulo 5 (`yolo26s_v3`, RE2.4).
- **Punto de operación:** score ≥ 0,11, el umbral elegido en validación. No se vuelve a elegir.
- **Salida:** la tabla de Raven que `detect.py` escribe para cada grabación. La herramienta
  recorre la grabación en ventanas de 3 s con salto de 1,5 s, aplica el NMS propio del modelo y
  el tope de detecciones por ventana, y funde los duplicados entre ventanas solapadas.

La unidad de análisis es **una fila de esa tabla**, no una detección por ventana como en el
capítulo 5. Por eso las cifras de este capítulo no se comparan una a una con las del 5.

## 2. Subconjunto de prueba

Las 601 grabaciones de la partición de prueba (4,93 h de audio), que el modelo no vio ni al
entrenar ni al elegir el umbral. La revisión experta y la sesión cronometrada usan grabaciones
**distintas** de este mismo subconjunto: quien revisó una grabación la conocería en la sesión.

## 3. Referencia y emparejamiento

- **Referencia:** las anotaciones del equipo, limpiadas con el protocolo de RE1.1, de las 25
  clases del modelo y sin las filas que quedaron en revisión.
- **Verdadero positivo:** una detección y una anotación de la misma clase con IoU ≥ 0,3. El
  emparejamiento es voraz por score y cada anotación se toma una sola vez. El IoU se mide en
  segundos × escala mel, la misma escala de frecuencia que ve el modelo.
- **Fuera del análisis:** las dos clases de ventana (`as/hc`, `pt/dc`), cuya caja ocupa la
  ventana entera, como en RE3.3.

## 4. Categorías automáticas

Cada detección que no acierta cae en una sola categoría, en este orden (la regla de RE3.3):

| Categoría | Regla |
|---|---|
| Caja distinta | se superpone con una anotación de su clase sin acertarla (IoU < 0,3, o la anotación ya la tomó otra detección, o cubre varias) |
| Otra llamada | encuadra (IoU ≥ 0,3) una anotación de otra clase |
| Nada anotado | no toca ninguna anotación de su clase ni encuadra una de otra |

Una detección «nada anotado» que se superpone con una fila en revisión, una frase o un tipo de
llamada sin clase se cuenta aparte y no se revisa: ahí ya hay algo anotado.

## 5. Revisión de una muestra de cajas (RE3.3)

**Población y estratos.** Las detecciones de §4, más los aciertos como controles:

| Estrato | Población | Muestra | Para qué |
|---|---|---|---|
| Nada anotado, score ≥ 0,5 | 583 | 300 | el de alta confianza que pide el IOV |
| Nada anotado, 0,3–0,5 | 638 | 200 | |
| Nada anotado, 0,11–0,3 | 1 016 | 133 | cubrir todo el rango de operación |
| Caja distinta | 220 | 220 | todas |
| Otra llamada | 87 | 87 | todas |
| Acierto (control) | 2 150 | 60 | medir qué tan exigente es el revisor |

Las 98 detecciones «nada anotado» que caen sobre otra fila (§4) quedan fuera. En total, 1 000
cajas en 310 grabaciones, unas 16 a 18 horas de trabajo.

Muestreo aleatorio simple dentro de cada estrato, semilla 42; «caja distinta» y «otra llamada» van
completas. Con 300 cajas, una proporción queda con un intervalo del 95 % de ±6 puntos en el peor
caso; con las 633 de «nada anotado» juntas, ±4.

**Ciego.** El revisor ve la caja, la especie y la llamada propuestas. No ve el score, el estrato
ni la anotación de referencia, y no sabe que hay controles. Trabaja con una copia del audio, sin
las tablas originales a mano.

**Formulario.** Una fila por caja y una sola palabra por fila:

| Respuesta | Cuándo | Veredicto |
|---|---|---|
| `correct` | es una llamada de primate del tipo propuesto y la caja la encuadra | correcta |
| `other` | es una llamada de primate, pero de otra especie o tipo | error de clase |
| `box` | es la llamada propuesta, pero la caja está mal puesta | error de encuadre |
| `no` | no hay ninguna llamada de primate en la caja | ruido |
| `?` | no se puede decir | incierto |

Si la especie o el tipo y la caja están mal a la vez, el revisor escribe `other`. Un comentario
libre opcional dice cuál es la llamada correcta o qué es el sonido.

**Lectura.** Según el estrato, el veredicto se lee como la categoría del IOV:

| Estrato | `correct` | Otras respuestas |
|---|---|---|
| Nada anotado | **evento no anotado**; también con `other` o `box`: hay una llamada que la referencia no tiene | `no`: ruido · `?`: incierto |
| Caja distinta | la caja del modelo es la buena: anotación por corregir | `box`: error de encuadre · `other`: error de clase · `no`: ruido |
| Otra llamada | la etiqueta del modelo es la buena: anotación por corregir | `other`: error de clase · `box`: error de encuadre · `no`: ruido |
| Acierto | acierto confirmado | desacuerdo del revisor con la referencia |

Las tres categorías que compromete el IOV son evento no anotado, ruido y error de encuadre. El
error de clase no cabe en ninguna de las tres y se reporta aparte, igual que lo incierto, que no
se reparte entre las demás.

**Estimación.** Por estrato, la proporción de cada veredicto con su intervalo de Wilson al 95 %.
Para «nada anotado» en conjunto, cada tramo de score pesa lo que pesa en la población (estimador
estratificado, intervalo normal), y de ahí el número estimado de eventos no anotados en prueba y
por hora de audio.

**Acuerdo.** Basta una persona. Si una segunda responde también la planilla, o parte de ella, sin
hablarlo con la primera, se reporta el kappa de Cohen sobre la respuesta y sobre si hay una
llamada. Con una sola persona, se declara como limitación.

## 6. Medición del tiempo de revisión (RE3.2)

**Material.** Cuatro carpetas de 14 grabaciones de prueba, dos de cada especie con algún tipo
entrenado (5,9 a 6,1 min de audio cada carpeta, 24 min en total), de hasta 2 min y con al menos
una anotación. De muchas elecciones al azar se elige la que deja las cuatro carpetas más parecidas
en minutos y en anotaciones: quedaron con 47 a 51 anotaciones cada una (7,8 a 8,4 por minuto). Quedan
fuera las grabaciones de la muestra de §5 y *Cebus*, sin ningún tipo entrenado (la condición
asistida no propondría nada).

**Condiciones.**

- *Manual:* el revisor anota la grabación desde cero, con su protocolo habitual.
- *Asistida:* el revisor parte de la tabla de la herramienta y borra, corrige y agrega hasta
  dejarla como la entregaría.

En las dos, el mismo estándar de calidad y los mismos ajustes de Raven.

**Orden.** Una persona hace las cuatro carpetas en el orden manual, asistida, asistida, manual,
para que el aprendizaje a lo largo de la sesión caiga parejo en las dos condiciones.

**Medida.** Por carpeta, tiempo de trabajo = hora de término − hora de inicio − pausas anotadas.
Se suma por condición y se divide por las horas de audio: **minutos de trabajo por hora de
audio**, en cada condición, y su cociente asistida / manual. No se fija ningún ahorro esperado:
se reporta lo que salga.

**Calidad de lo entregado.** Para que un ahorro de tiempo no oculte un trabajo peor, las tablas
entregadas en cada condición se comparan con la referencia con la regla de §3: cobertura y
precisión. En la condición asistida, además, qué pasó con cada propuesta (se dejó igual, se
corrigió o se borró) y cuántas cajas se agregaron.

**Costo de la máquina.** Minutos de cómputo por hora de audio en GPU (capítulo 5), en CPU (la
corrida que arma el paquete) y en el computador del equipo, si lo reporta.

## 7. Calidad del corpus (complemento de RE3.3)

Fuera de lo que compromete el IOV, y para contrastar las anotaciones del corpus entero, el equipo
resuelve los hallazgos de CLOD (`src/find_issues.py`) de train y validación. Prueba queda fuera:
ya está en §5.

- **Hallazgos:** sobre predicciones fuera de muestra (en train, del k-fold `yolo26s_v3_kfold5`;
  en validación, del modelo final), CLOD marca anotaciones que parecen faltar (`missing`: el
  modelo ve con confianza una llamada sin anotar), sobrar (`spurious`: ninguna predicción la toca),
  tener otra etiqueta (`label`) o estar mal ubicadas (`location`). Un `missing` sólo entra si no
  hay ninguna fila en la tabla cruda debajo de la caja: si la hay, la anotación existe (en el
  conjunto pero sin ligar con la caja, en revisión, con una clase fuera del modelo o descartada
  por la limpieza) y no es una omisión.
- **Tamaño:** 7 728 hallazgos en 1 538 grabaciones:

  | Tipo | Train (k-fold) | Validación | Total |
  |---|---|---|---|
  | `missing` | 1 961 | 734 | 2 695 |
  | `spurious` | 2 232 | 785 | 3 017 |
  | `location` | 1 310 | 487 | 1 797 |
  | `label` | 164 | 55 | 219 |

- **Formulario:** el mismo de §5. Cada hallazgo es una caja: la del modelo en `missing`, la
  anotación original en los demás. El revisor no sabe cuál es cuál.
- **Orden:** al azar dentro de cada tipo e intercalado entre tipos. El equipo revisa desde arriba
  todo lo que alcance; lo revisado es una muestra al azar de cada tipo.
- **Lectura:** un hallazgo se confirma con `correct`, `other` o `box` si es `missing` (una
  omisión de la anotación; `no` es un falso positivo del modelo), con `no` si es `spurious`, con
  `other` si es `label` y con `box` si es `location`. La proporción confirmada por tipo, con su
  intervalo de Wilson, multiplicada por los hallazgos de ese tipo, estima cuántos errores de
  anotación tiene el corpus. La proporción de `no` entre los `missing` de train es la tasa de
  falsos positivos del k-fold entre sus predicciones confiadas sin anotación.

## 8. Qué no se hace

- No se cambia el umbral ni ninguna regla después de ver respuestas.
- No se descarta ningún ítem ni ninguna grabación a la vista de su resultado. Las filas sin
  respuesta o ilegibles se cuentan y se declaran.
- No se repite la sesión para mejorar una cifra.

## 9. Limitaciones previstas

- El audio de prueba es la selección que el equipo anotó, más densa en llamadas que una
  grabación continua: el tiempo por hora de audio no se extrapola sin más a una temporada
  entera.
- Una sola persona y 24 min de audio en la sesión: la variación entre personas y entre
  grabaciones queda poco medida.
- Si el revisor es uno de los anotadores originales, puede reconocer grabaciones; se le pregunta
  y se declara.

## Cambios después de la fecha

(ninguno)
