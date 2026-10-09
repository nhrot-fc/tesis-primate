# Anexo A — Plan del Proyecto

Va como anexo (`\appendix` en `main.tex`), después de las conclusiones. Es el
plan de gestión: justificación, alcance, entregables, EDT, cronograma, costos,
recursos y riesgos.

> **06-10-2026.** La EDT, la tabla de criterios de aceptación y cada una de las
> cinco tablas de costos tienen ahora una frase que las presenta y otra que las
> interpreta; antes iban seguidas sin texto entre ellas.

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
  discusión de cada capítulo. **Excepción:** la columna *Estado* del cronograma
  se actualiza en cada entrega, porque el rubro «Desarrollo de resultados de
  acuerdo con el cronograma» compara el avance con el plan.

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

- [x] ~~**El jurado lo pidió (06-10-2026):** «El cronograma debe permitir
      identificar de manera inmediata qué RE, productos e hitos debían estar
      completados para E1 y cuáles corresponden a etapas posteriores».~~
      Resuelto el 06-10-2026. La Tabla A.3 (`tab:cronograma`) agrupa las
      tareas del cronograma del autor por la entrega del calendario del curso:
      - E1: 24/08/26, semana 2, documento del curso anterior (Bloque 0).
      - E2: 07/10/26, semana 8, entregable parcial.
      - E3: 23/11/26, semana 15, entregable final.
      - Exposición final: semanas 16 y 17.

      Regla: cada tarea va en la primera entrega posterior a su fecha de fin.
      Cambios respecto del original:
      - estados actualizados (2.1 y 2.4–2.6 ya están completos);
      - cifras quitadas de los nombres de tarea (19 029 anotaciones, 70/10/20),
        porque no coinciden con las macros de los capítulos;
      - Bloque 3 reetiquetado con los RE3.x congelados (ver
        `06_oe3_utilidad/PENDIENTES_RE3.md`);
      - **4.1 «Redacción del informe final» termina el 20/11, no el 26/11,**
        para que llegue a E3 (el 20/11 se envía al asesor). Confirmar con el
        autor.
- [x] ~~**Error aritmético en «Capital intelectual»:** 600 h × 5,00 USD/h son
      3 000,00, no 2 500,00.~~ Resuelto el 06-10-2026: se conservaron las horas
      y la tarifa (los supuestos) y se corrigió lo derivado. Tesista 3 000,00,
      subtotal 4 000,00, total general 7 349,88.
- [x] ~~El criterio de aceptación de RE3.1 repite el IOV congelado (confianza 0,5).~~
      **Resuelto el 07-10-2026:** el IOV de RE3.1 se reescribió («punto de operación
      elegido en validación») y la Tabla de criterios de aceptación lo sigue.
- [ ] Revisar que los costos sigan reflejando el hardware usado (las corridas
      pasaron por varias GPU).
