# Anexo A — Plan del Proyecto

Va como anexo (`\appendix` en `main.tex`), después de las conclusiones. Es el
plan de gestión: justificación, alcance, entregables, EDT, cronograma, costos,
recursos y riesgos.

## Qué va aquí

| Sección | Contenido |
|---|---|
| Justificación y viabilidad | Beneficios esperados; viabilidad técnica, temporal y económica |
| Definición y alcance | Alcance, entregables con criterio de aceptación, exclusiones, limitaciones |
| Planificación | EDT, cronograma, estimación de costos, recursos y gestión de riesgos |

## Cómo se redacta

- Los entregables **son** los resultados esperados del Capítulo 1: la tabla de
  criterios de aceptación tiene que tener las mismas once filas, ni una más.
  Cada vez que se toque un IOV hay que revisar esta tabla, la EDT y los riesgos.
- El cronograma y los costos se escriben en tiempo futuro respecto del plan
  original; no se reescriben con lo que finalmente pasó, para eso está la
  discusión de cada capítulo.

## Imágenes
> **30-09-2026.** Los diagramas de globos y flechas (árbol de problemas, CRISP-DM, cadena de la señal, EDT) se pasaron de matplotlib a **TikZ dentro del `.tex`**: usan la tipografía del documento, no se pixelan y el flotante se coloca donde corresponde. Los PNG se borraron; no hay que regenerarlos.


| Figura | Qué muestra | Origen |
|---|---|---|

## Extra

Se alineó con los objetivos congelados: se eliminó el entregable RE3.4
(condicional), la medición cronometrada volvió al alcance, el paquete 6.6 de la
EDT pasó a alimentar RE3.2 y RE3.3, y el riesgo R3 dejó de decir que ningún
resultado depende de un revisor externo.

## Pendientes

- [ ] El cronograma no tiene diagrama de Gantt; si el jurado lo pide, sale de
      la EDT.
- [ ] Revisar que los costos sigan reflejando el hardware usado (las corridas
      pasaron por varias GPU).
