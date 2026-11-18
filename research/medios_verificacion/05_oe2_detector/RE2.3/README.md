# RE2.3. Informe comparativo de resultados

**Medio de verificación.** Informe de experimentación y archivos de métricas.

**Indicador.** Reporta, como mínimo, recall, precisión, F1, IoU medio, AP a IoU 0,25 y 0,5, recall por clase y matriz de confusión; separa detección, encuadre y clasificación y justifica la selección del modelo.

**Dónde lo desarrolla el documento.** Sección 5.4, que es el informe de experimentación; la sección 5.4.9 contrasta el indicador magnitud por magnitud.

| Elemento del indicador | Dónde se comprueba |
|---|---|
| Recall, precisión, F1 | `tablas/resultados_punto_de_operacion.csv` (Tabla 5.11); recall y precisión también en `comparacion_modelos_v3.txt`, punto 1 |
| IoU medio | `tablas/iou_medio.csv`, sobre los pares emparejados (Figura 5.12); el `.txt` da el mediano, punto 2 |
| AP a IoU 0,25 y 0,5 | `tablas/resultados_punto_de_operacion.csv`, columnas mAP |
| Recall por clase | `tablas/recall_por_clase.csv` (Tabla 5.15); `comparacion_modelos_v3.txt`, punto 3 |
| Matriz de confusión | `matriz_confusion.png` |
| Separa detección, encuadre y clasificación | `tablas/ejes.csv` (Tabla 5.13); `comparacion_modelos_v3.txt`, punto 2 |
| Justifica la selección del modelo | `tablas/diferencias_pareadas.csv` (Tabla 5.12) y sección 5.2.4 |
| Umbral de cada modelo | `punto_de_operacion/*.json` |

Las tablas de `tablas/` son las del documento, que el cuaderno del capítulo 5 calcula a partir de los mismos volcados que `comparacion_modelos_v3.txt`.
