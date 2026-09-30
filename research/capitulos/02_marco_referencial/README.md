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

De `notebook/figuras_02_marco_referencial.ipynb`, salvo `image3.png`.

| Figura | Qué muestra | Origen |
|---|---|---|
| `signal_chain.png` | Cadena de la señal, de la onda al tensor | cuaderno |
| `stft_tradeoff.png` | Compromiso tiempo–frecuencia de la STFT | cuaderno |
| `mel_axis.png` | Escala mel frente a la lineal | cuaderno |
| `frontends.png` | Los tres *front-ends* intercambiables (none / logmel / PCEN) | cuaderno |
| `image3.png` | Segmentación tiempo–frecuencia | heredada del informe de avance E3, **no se regenera** |

## Pendientes

- [ ] `image3.png` es una captura antigua y desentona con el resto de figuras.
      Decidir si se rehace en el cuaderno o se retira.
