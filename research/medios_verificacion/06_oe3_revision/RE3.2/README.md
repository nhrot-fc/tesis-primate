# RE3.2. Medición comparativa del tiempo de revisión

**Medio de verificación.** Informe de validación, registros de tiempo y medición de inferencia.

**Indicador.** Reporta minutos por hora de audio en condición manual y asistida, describe el material comparable y el orden de revisión, y presenta el tiempo de procesamiento del modelo por hora de audio. No fija un porcentaje de ahorro antes de medirlo.

**Dónde lo desarrolla el documento.** Sección 6.3; la sección 6.3.5 contrasta el indicador.

| Elemento del indicador | Dónde se comprueba |
|---|---|
| Minutos por hora de audio, manual y asistida | `tiempos_sesion.csv`, columna `min_por_hora_de_audio` |
| Material comparable | `grabaciones_sesion.csv`: grabaciones, duración, anotaciones y propuestas de cada carpeta |
| Orden de revisión declarado | `tiempos_sesion.csv`: carpetas 1 a 4 en el orden en que se hicieron |
| Tiempo de procesamiento del modelo | `test_cpu_times.csv`: segundos de CPU y duración de cada grabación de prueba |

Las horas vienen de la planilla que devolvió el equipo, sin sus notas ni la columna de quién anotó, que traen nombres. El trabajo de cada carpeta es término menos inicio, menos las pausas.
