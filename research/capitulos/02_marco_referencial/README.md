# Capítulo 2 — Marco Referencial

Los conceptos que el resto del documento usa sin volver a definir: qué es un
evento acústico, cómo se representa el audio en tiempo–frecuencia y qué
significa detectar una caja sobre un espectrograma.

> **06-10-2026.** Pasada contra repeticiones tras la revisión del jurado: se
> retiró la figura de segmentación (`image3.png`), se quitó la tabla de especies
> (repetía la Tabla 1.1), la tabla de familias pasó a ser conceptual, cada figura
> se interpreta después de mostrarse y lo que ya define el Capítulo 1 (causas,
> soluciones, métricas, problema) se cita en vez de repetirse.

> **07-10-2026.** PCEN se descartó: el autor lo probó y lo dejó por su bajo
> rendimiento, pero no hay cifras publicables de esa prueba, así que el texto
> no la menciona. §2.2.3 ya no promete comparar log-mel con PCEN: presenta PCEN
> como alternativa de la literatura y justifica el log-mel porque es la entrada
> con que se preentrenó el AST (`gong_ast_2021`, nueva). `frontends.png` se
> recortó a dos paneles (mel de potencia y log-mel), porque el cuaderno no corre
> con el `data/` de esta máquina (ver el README del capítulo 4). La celda del
> cuaderno ya dibuja solo esos dos, así que regenerarla en el servidor da la
> misma figura. La cadena de la señal dice ahora 22,05 kHz, como el capítulo 1.

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
| `frontends.png` | Mel de potencia y log-mel sobre la misma ventana (PCEN se descartó el 07-10-2026) | cuaderno |
| ~~`image3.png`~~ | ~~Segmentación tiempo–frecuencia~~ | **retirada del capítulo el 06-10-2026**; el archivo sigue en la carpeta |

## Pendientes

- [x] ~~`image3.png` es una captura antigua y desentona con el resto de
      figuras.~~ Retirada el 06-10-2026: mostraba la misma caja que la Figura 1.1
      (anotación en Raven), a la que ahora remite el texto. Borrar el archivo si
      no vuelve.
