# RE3.1. Metodología de validación comparativa

**Medio de verificación.** Documento de metodología y protocolo de revisión experta.

**Indicador.** Especifica el subconjunto de prueba, el punto de operación elegido en validación y el umbral de IoU para verdadero positivo, fijados antes de medir, las métricas, el emparejamiento y el procedimiento para revisar discrepancias y eventos potencialmente omitidos.

**Dónde lo desarrolla el documento.** Sección 6.2; la sección 6.2.7 contrasta el indicador.

| Elemento del indicador | Dónde se comprueba |
|---|---|
| Subconjunto de prueba | `protocolo.md`, §2 |
| Punto de operación elegido en validación | `protocolo.md`, §1 |
| Umbral de IoU para verdadero positivo | `protocolo.md`, §3 |
| Fijados antes de medir | Commit `63f84a8` del 2026-10-10, anterior a la sesión (11–13/11/2026) y a la revisión de cajas; `protocolo.md`, «Cambios después del commit» |
| Métricas | `protocolo.md`, §4 y §5 |
| Emparejamiento | `protocolo.md`, §3 |
| Procedimiento para revisar discrepancias y eventos omitidos | `protocolo.md`, §5 |

Commit completo: `63f84a830ca5fad81b5c3a202c5d9318e0c979b6`. `protocolo.md` es la versión actual; la del commit se ve con `git show 63f84a8`.
