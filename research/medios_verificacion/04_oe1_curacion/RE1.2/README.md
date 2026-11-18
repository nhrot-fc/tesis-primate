# RE1.2. Conjunto acústico curado y particionado para el experimento

**Medio de verificación.** Directorio `data/processed/`: etiquetas, metadatos y tres archivos `.pt`.

**Indicador.** Los archivos procesados cargan sin error; `meta.json` registra la semilla, el criterio de selección, las exclusiones, la etiqueta usada y la normalización; el código divide por archivo de grabación con proporciones 60/22,5/17,5 y no por ventanas aisladas.

**Dónde lo desarrolla el documento.** Sección 4.4; la sección 4.4.5 aplica las tres comprobaciones del indicador.

| Elemento del indicador | Dónde se comprueba |
|---|---|
| Los archivos procesados cargan sin error | `registro_de_carga.txt`: 0 imágenes corruptas al exportar y todas las ventanas leídas al evaluar |
| `meta.json` registra la semilla | `meta.json`, clave `seed` (42) |
| … el criterio de selección | `meta.json`, clave `min_pair_count` (100) |
| … las exclusiones | `meta.json`, claves `excluded_labels` y `joined_labels` |
| … la etiqueta usada | `labels.json`, índice → clase `especie/llamada` |
| … la normalización | `meta.json`, clave `db_range` y bloque `params` |
| El código divide por archivo de grabación | `grabaciones_*.txt`; `registro_de_carga.txt`, punto 3: ninguna grabación compartida |
| Proporciones 60/22,5/17,5 | `meta.json`, clave `split_ratios` ([0.6, 0.225, 0.175]) |

El caché (`data/processed/`) no se publica, porque son espectrogramas de grabaciones del equipo. `meta.json` es el del caché con que se entrenaron los modelos; las listas de grabaciones y los conteos salen de los volcados de predicciones de esas mismas corridas.
