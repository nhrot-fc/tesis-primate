# RE1.1. Protocolo de curación y estandarización documentado

**Medio de verificación.** Documento del protocolo.

**Indicador.** Define la normalización de columnas y etiquetas, el mapa de sinónimos, el descarte de ruido y cajas degeneradas, la corrección de pares imposibles, el recorte al límite de Nyquist y el tratamiento de registros fuera de vocabulario.

**Dónde lo desarrolla el documento.** Sección 4.3, que es el documento del protocolo. La sección 4.3.3 contrasta cada regla con el código.

| Elemento del indicador | Dónde se comprueba |
|---|---|
| Normalización de columnas y etiquetas | `reglas.csv`, filas de tipo de llamada y de especie |
| Mapa de sinónimos | `sinonimos.csv` (los de todas las especies y los nombres legibles por especie) |
| Descarte de ruido y cajas degeneradas | `reglas.csv`, filas `noise` y caja degenerada |
| Corrección de pares imposibles | `correcciones_pares.csv`; sus filas, en `reglas.csv` |
| Recorte al límite de Nyquist | `reglas.csv`, fila de frecuencia superior |
| Tratamiento de registros fuera de vocabulario | `vocabulario.csv`; las filas marcadas, en `reglas.csv` |

Las tablas salen de las constantes del código de curación (`src/data/annotations.py` y `src/data/species.py`). Los conteos de `reglas.csv` son los de la auditoría del capítulo 4 (Tabla 4.8), sobre las filas crudas.
