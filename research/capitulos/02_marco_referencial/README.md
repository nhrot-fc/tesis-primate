# Capítulo 2 — Marco Referencial

Los conceptos que el resto del documento usa sin volver a definir: qué es un
evento acústico, cómo se representa el audio en tiempo–frecuencia y qué
significa detectar una caja sobre un espectrograma.

## Qué va aquí

| Sección | Contenido |
|---|---|
| Marco teórico | Formulación del problema de detección, asimetría de los ejes del espectrograma, detección de objetos sobre audio |
| Marco conceptual | Paisaje sonoro, eventos acústicos, representaciones tiempo–frecuencia, SED, clasificación jerárquica, anotaciones inconsistentes |

## Cómo se redacta

- Es un capítulo de definiciones, no de resultados: nada de cifras propias.
  Cuando una decisión de diseño se apoya en un concepto de aquí, la decisión va
  en el capítulo del objetivo y aquí queda solo el concepto.
- Cada subsección cierra apuntando al capítulo que usa ese concepto
  (`\ref{cap:curacion}`, `\ref{cap:deteccion}`), para que el lector sepa dónde
  se cobra la definición.
- La subsección de clasificación jerárquica es la que justifica excluir las
  clases de frase en el Capítulo 4: mantener ese hilo explícito.

## Imágenes
> **30-09-2026.** Los diagramas de globos y flechas (árbol de problemas, CRISP-DM, cadena de la señal, EDT) se pasaron de matplotlib a **TikZ dentro del `.tex`**: usan la tipografía del documento, no se pixelan y el flotante se coloca donde corresponde. Los PNG se borraron; no hay que regenerarlos.


De `notebook/figuras_02_marco_referencial.ipynb`, salvo `image3.png`. La celda de
`signal_chain` se eliminó porque esa cadena hoy se dibuja en TikZ dentro del
`.tex`, y las de `annotation_example`, `output_forms` e `iou_criterion` se mudaron
al cuaderno del capítulo 1 el 02-10-2026: las usa ese capítulo, no este, y había
copias idénticas en las dos carpetas.

| Figura | Qué muestra | Origen |
|---|---|---|
| `stft_tradeoff.png` | Compromiso tiempo–frecuencia de la STFT | cuaderno |
| `mel_axis.png` | Escala mel frente a la lineal | cuaderno |
| `frontends.png` | Los tres *front-ends* intercambiables (none / logmel / PCEN) | cuaderno |
| `image3.png` | Segmentación tiempo–frecuencia | heredada del informe de avance E3, **no se regenera** |

## Pendientes

- [ ] `image3.png` es una captura antigua y desentona con el resto de figuras.
      Decidir si se rehace en el cuaderno o se retira.
