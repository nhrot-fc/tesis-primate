# RE3.3. Análisis cualitativo de errores y hallazgos

**Medio de verificación.** Informe de análisis y espectrogramas anotados.

**Indicador.** Clasifica una muestra de predicciones de alta confianza en evento no anotado, ruido y error de encuadre, e incluye ejemplos visuales de aciertos, errores comunes y detecciones correctas ausentes de la anotación de referencia.

**Dónde lo desarrolla el documento.** Sección 6.4; la sección 6.4.6 contrasta el indicador.

| Elemento del indicador | Dónde se comprueba |
|---|---|
| Predicciones de alta confianza revisadas | `respuestas_revision.csv`, estrato `none_alta` (confianza mayor que 0,5) |
| Clasificadas en evento no anotado, ruido y error de encuadre | `respuestas_revision.csv`, columna `categoria`, que aplica la lectura del protocolo a `estrato` y `respuesta` |
| Espectrogramas anotados | `revision_ejemplos.png` |

Una fila por caja revisada. `estrato` es de dónde salió la caja; `respuesta`, lo que contestó el revisor a ciegas; `persona`, A o B cuando la grabación volvió separada por persona. La caja es la del modelo, no una anotación del equipo.
