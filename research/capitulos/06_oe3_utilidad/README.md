# Capítulo 6 — Tiempo de Revisión y Análisis de Errores (OE3)

> **OE3.** Cuantificar la reducción del tiempo de revisión que aporta el sistema
> y caracterizar sus errores mediante revisión experta de una muestra de
> predicciones.

Capítulo nuevo: antes OE3 vivía como cuatro secciones sueltas al final del
capítulo de OE2. Está sin redactar; hoy es la estructura y el encargo.

> **06-10-2026.** Los pendientes de RE3.1–RE3.3, incluido el reetiquetado del
> Bloque 3 del cronograma, están reunidos en `PENDIENTES_RE3.md`.

> **09-10-2026.** Preparado todo lo de la revisión con el equipo:
>
> - `protocolo_oe3.md`: el protocolo de RE3.1, por fechar **antes** de enviar el paquete.
> - `notebook/preparar_revision.ipynb`: corre la herramienta sobre las 601 grabaciones de prueba,
>   clasifica cada detección con la regla de RE3.3 y arma `revision/paquete_equipo/`: 1 000 cajas
>   que se marcan con una palabra; 4 carpetas de ~6 min de audio (manual, asistida, asistida,
>   manual); y la tarea de calidad del corpus, con los hallazgos de CLOD de train (k-fold) y
>   validación en orden aleatorio, para revisar todos los que alcancen. Instrucciones de una
>   página en inglés. No pide datos nuevos.
> - Tarea de calidad: 7 728 hallazgos en 1 538 grabaciones (train del k-fold y validación). Los
>   `missing` con una fila cruda debajo no entran: la anotación existe.
>   `runs/yolo26s_v3/hallazgos_val.csv` se pasó al formato nuevo de `find_issues.py` con su
>   `export()`, sobre el contraste con `unified/` que hizo el servidor; el original quedó en
>   `hallazgos_val_formato_viejo.csv`.
> - `revision/` está en `.gitignore`: las claves (`clave_*.csv`) no están en git. Volver a correr
>   `preparar_revision.ipynb` las reproduce mientras no cambien los `hallazgos_*.csv` de `runs/`.
> - `notebook/analisis_oe3.ipynb`: lee lo que devuelva el equipo en `revision/respuestas/` y
>   escribe `figures/valores.tex` (macros `\u…`) y las figuras. Probado con respuestas simuladas.
> - `revision/mensaje_equipo.md`: el pedido a Mark, en inglés.
>
> Hallazgo: en la tabla de Raven que ve el revisor, con los duplicados entre ventanas fundidos,
> la precisión es ~0,45 (2 150 aciertos de 4 792 cajas), no 0,665 como por ventana. La cobertura
> casi no cambia (0,78 frente a 0,81).

## Qué va aquí

| Sección | Contenido | RE |
|---|---|---|
| Introducción | Enuncia OE3 y anuncia las tres secciones de resultado | — |
| Metodología de validación comparativa | Subconjunto de prueba, punto de operación, emparejamiento, métricas y protocolo de revisión experta, fechados antes de medir | **RE3.1** |
| Medición comparativa del tiempo de revisión | Minutos por hora de audio en condición manual y asistida, más el costo de inferencia del modelo | **RE3.2** |
| Análisis cualitativo de errores y hallazgos | Muestra de predicciones de alta confianza clasificada en evento no anotado, ruido y error de encuadre, con ejemplos sobre el espectrograma | **RE3.3** |
| Discusión | Las cinco preguntas de la pauta | — |

## Cómo se redacta

- **RE3.2 es la única cifra que responde al objetivo general.** Sin los minutos
  por hora de audio en las dos condiciones, el OG no queda demostrado. Todo lo
  demás del capítulo la acompaña.
- La metodología se fecha **antes** de correr la evaluación: es lo que impide
  elegir a posteriori el umbral que mejor queda.
- El material de las dos condiciones tiene que ser comparable y el orden de
  revisión declarado, para que el aprendizaje del revisor no se confunda con el
  efecto de la asistencia.
- El IOV prohíbe fijar un porcentaje de ahorro antes de medirlo: no se anticipa
  ninguna cifra, ni en la introducción ni en el resumen.
- Mientras una predicción sin anotación no la resuelva el revisor, su parecido
  con los eventos anotados es **indicio** de una omisión en la referencia, no
  prueba. Con revisión experta pasa a ser medición.

## Conflictos a resolver antes de redactar

1. ~~**Punto de operación.**~~ **Resuelto el 07-10-2026:** el IOV de RE3.1 ahora dice
   «el punto de operación elegido en validación y el umbral de IoU para verdadero
   positivo, fijados antes de medir». Lo que sigue es el texto anterior.
   El IOV de RE3.1 estaba comprometido con confianza 0,5,
   NMS IoU 0,3 y verdadero positivo a IoU 0,5. El protocolo implementado
   (`src/evaluation/protocol.py`, descrito en
   `../05_oe2_detector/protocolo_comparacion.md`) empareja a IoU 0,3 y elige el
   umbral por modelo en validación. Esto último es correcto para comparar
   arquitecturas (RE2.3), pero este capítulo reporta en el punto comprometido; el
   umbral de validación, si se incluye, va rotulado como lectura secundaria.
2. **Categorías del análisis de errores.** El IOV compromete tres —evento no
   anotado, ruido, error de encuadre— y el trabajo hecho usa cuatro (A encuadre,
   B unidad, C clase, D sin anotación). Hay que declarar explícitamente la
   correspondencia: A (y B) → error de encuadre; D resuelto como evento válido →
   evento no anotado; D resuelto como falso positivo → ruido. C (clase
   incorrecta) no entra en ninguna de las tres y necesita su propio párrafo.
3. **La revisión experta está comprometida.** Ya no es un resultado condicional:
   RE3.1 pide su protocolo y RE3.3 pide sus resultados. Lo que se declara como
   limitación es el tamaño de la muestra revisada, no su ausencia.

## Imágenes

Ninguna todavía. Las que pide el capítulo:

| Figura | Qué mostraría | Para |
|---|---|---|
| Curva de cobertura frente a propuestas por hora | Por qué baja el tiempo: cuántas propuestas hay que inspeccionar y cuánto audio queda sin ninguna | RE3.2 |
| Panel de espectrogramas | Aciertos, errores comunes y detecciones correctas ausentes de la referencia — las tres las pide el IOV | RE3.3 |
| Distribución de banda, duración y puntuación | Predicciones sin anotación frente a los eventos anotados de cada clase | RE3.3 |

## Extra

- **Material por trasladar desde `research/reportes/`**:
  `RE_3-3_taxonomia-discrepancias.tex` y su cuaderno reparten las discrepancias
  A/B/C/D sobre todo el corpus (k-fold en train) y dejan la tabla de hallazgos,
  la caracterización de la categoría D y los ejemplos. Es casi todo RE3.3.
  Necesita el volcado fuera de muestra del modelo seleccionado
  (`python src/kfold.py --arch yolo --name <run>_kfold5`).
- Las mediciones de tiempo de inferencia por modelo están en
  `docs/system_requirements.md`; aquí van las del modelo seleccionado.
- No existe reporte de RE3.1 ni de RE3.2: hay que escribirlos desde cero, y
  **RE3.2 requiere una sesión de cronometraje que todavía no se ha hecho**.

## Pendientes

- [ ] Revisar `protocolo_oe3.md` y **fecharlo antes** de enviar el paquete; después,
      pasarlo a la sección de metodología.
- [x] Copiar `hallazgos_train.csv` del servidor y volver a correr `preparar_revision.ipynb`.
- [ ] Enviar `revision/mensaje_equipo.md` y el paquete (~1 GB, por Drive).
- [ ] Copiar las respuestas a `revision/respuestas/` y correr `notebook/analisis_oe3.ipynb`.
- [ ] Cronometrar la revisión manual y la asistida sobre material comparable.
- [ ] Absorber `RE_3-3_taxonomia-discrepancias` y sus figuras.
- [ ] Redactar la discusión.
