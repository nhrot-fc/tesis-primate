# Capítulo 5 — Detección y Evaluación del Modelo (OE2)

> **OE2.** Implementar y evaluar un detector que reciba un espectrograma y
> devuelva cajas tiempo–frecuencia con especie, tipo de llamada y confianza,
> comparándolo con una línea base establecida.

Redactado completo (30-09-2026) sobre la comparación vigente
`runs/comparacion/comparacion_modelos_v3` (21-09). **La versión anterior de RE2.3
era de otra comparación** —cuatro configuraciones con el AST congelado, partición
de 8 007 ventanas— y se reescribió entera.

> **06-10-2026.** Revisión tras la observación del jurado («no se encontró
> referencia clara al medio de verificación» de RE2):
>
> - Los medios ahora son accesibles: RE2.1 y RE2.2 apuntan al repositorio
>   público (`\rRepoUrl`) y al commit `\rCommit`; RE2.4 y RE2.5, a la versión
>   v0.2.0, con el nombre del zip y su SHA-256; RE2.3 nombra los archivos de
>   métricas. Se corrigieron rutas que no coincidían con el código
>   (`src/pipelines/`, `.pth`, `src/infer.py`).
> - Se quitaron todas las remisiones al Capítulo 6, que no está desarrollado.
> - Las 23 figuras y tablas se interpretan después de mostrarse.
> - **Corrección de hecho:** el capítulo decía en tres lugares que el desbalance
>   entre clases era de «dos órdenes de magnitud». Es de `\classImbalance` = 23
>   veces (Capítulo 4), algo más de un orden. Corregido aquí y en las
>   conclusiones.

> **06-10-2026 (noche), revisión estricta contra la rúbrica:**
>
> - Cada RE tiene un `\rotulo{Cómo reproducirlo.}`.
> - Cada «Cómo se alcanzó» nombra la fase de CRISP-DM o el procedimiento del
>   Capítulo 1 que aplica.
> - RE2.5 distingue los dos caminos: desde el código (`uv.lock` y las etapas de
>   RE2.2) y desde el `README`, que es la guía del programa para Windows y no
>   explica la instalación desde el código fuente.
> - Se definen en su primer uso ViT, MLP, FPN y GIoU.

## Qué va aquí

| Sección | Contenido | RE |
|---|---|---|
| Introducción | Enuncia OE2, anuncia las cinco secciones y declara cómo se generan las cifras | — |
| Diseño experimental, arquitectura propuesta y líneas base | Formulación, AST-Deformable-DETR completo, tres líneas base, recetas, protocolo y regla de decisión | **RE2.1** |
| Pipeline reproducible | Esquema, instalación, los cuatro grupos de comandos, ciclo de entrenamiento, artefactos, costo y reproducibilidad | **RE2.2** |
| Informe comparativo | Punto de operación, tres ejes, cobertura frente a carga, lectura por clase, ejemplos cualitativos, selección y costo de inferencia | **RE2.3** |
| Modelo final y demostración | Checkpoint, formato de Raven, ejecución de `detect.py` sobre una grabación de prueba | **RE2.4** |
| Código y modelo en un repositorio | Árbol, qué se publica y qué no, comandos de reproducción | **RE2.5** |
| Discusión | Las cinco preguntas de la pauta | — |

Cada sección de resultado abre con `\rotulo{Resultado.}` / `\rotulo{Cómo se
alcanzó.}` / `\rotulo{Medio de verificación.}` y cierra con un apartado de
verificación que contrasta el IOV elemento por elemento.

## Cifras

Ninguna se escribe a mano —y **el `.tex` no lo menciona**, por la decisión
editorial de `capitulos/README.md`: eso se documenta aquí, no en el documento.

Salen de `notebook/figuras_05_oe2_detector.ipynb`, que deja en `figures/` las
macros (`valores.tex`, 221 con prefijo `m`/`b`/`p`/`r`/`c` para no chocar con las
del capítulo 4 —el cuaderno lo comprueba y falla si hay colisión), las filas de
cada tabla (`tablas/`), las series de pgfplots (`data/`) y las tres figuras que
son imagen.

El cuaderno **importa el mismo `src/evaluation/protocol.py` que usa
`compare_models.py`** y comprueba que lo que calcula coincide con
`runs/comparacion/comparacion_modelos_v3.json`.

## Qué queda fuera a propósito

- **El paquete portable y la licencia.** La distribución está fuera del alcance
  (exclusiones del anexo). RE2.5 compromete que el repositorio exista y permita
  localizar el código y el *checkpoint* final; nada más. La versión v0.2.0 se
  cita sólo como el lugar donde está publicado el *checkpoint*, no se documenta
  cómo se arma.
- **El tono de manual.** El Listado 5.1 declara la interfaz del *pipeline* porque
  el IOV de RE2.2 lo exige; no hay bloques «cómo reproducirlo» en el resto.

## Qué va en TikZ y qué en matplotlib

Todos los **esquemas** (arquitectura, pirámide, capa del decodificador, muestreo
deformable, emparejamiento, líneas base, protocolo, pipeline, ciclo de
entrenamiento, flujo de inferencia) van en **TikZ**, y todas las **curvas**
(duración por clase, curvas de entrenamiento, distribución del IoU, cobertura
frente a carga) en **pgfplots**. Sólo son PNG las tres que son imagen:

| Figura | Qué muestra |
|---|---|
| `matriz_confusion.png` | 25×25 celdas: demasiado densa para TikZ |
| `detecciones_cualitativas.png` | Seis espectrogramas con cajas |
| `raven_compatibility.png` | Raven Lite 2.0.5 con `20240117_162715.detections.txt` abierto como *Selection Table* sobre su WAV: la prueba visual de RE2.4. La entregó el usuario (02-10-2026), no la genera el cuaderno |
| `demo_grabacion.png` | El tramo de 12 s de la grabación de la demostración que concentra más anotaciones. A 35 s por el ancho del texto las cajas eran ilegibles; las cifras que reportan las macros siguen siendo las de la grabación entera |

El cuaderno también deja `demo.detections.txt`, la tabla que produjo la
demostración de RE2.4, como evidencia; el `.tex` reproduce sus primeras filas en
una tabla LaTeX y **no incluye la transcripción de consola**, que se quitó por la
decisión editorial de `capitulos/README.md`.

## Resultado de la comparación

Cuatro configuraciones, mismo caché, misma partición, mismo protocolo:

| | umbral | recall (test) | IC 95 % | mAP@0,3 | min GPU / h audio |
|---|---|---|---|---|---|
| AST-Def-DETR (propuesta) | 0,21 | 0,797 | [0,771; 0,825] | 0,577 | 0,30 |
| Faster R-CNN R50-FPN v2 | 0,69 | 0,781 | [0,753; 0,808] | 0,574 | 0,88 |
| **YOLO26s** | 0,11 | **0,811** | [0,787; 0,835] | **0,620** | **0,12** |
| RT-DETR-L | 0,39 | 0,808 | [0,786; 0,829] | 0,601 | 0,30 |

- **La métrica primaria no separa a ninguna pareja**: los cuatro intervalos se
  solapan. Decide la secundaria (mAP@0,3) y gana YOLO26s, que además es el más
  barato.
- **La clasificación no es el problema**: ≥ 0,981 de los pares emparejados llevan
  la clase correcta y ≥ 0,991 la especie. El cuello de botella es detectar y
  encuadrar.
- **A carga igual (3 000 propuestas/hora) las cuatro cubren 0,793–0,807.** La
  elección del punto de operación pesa más que la de arquitectura.
- **La hipótesis de diseño no se confirma** (la propuesta debía superar a las
  líneas base y quedó tercera de cuatro). Desde el 07-10-2026 el capítulo lo dice
  una sola vez y sin atenuarlo: el veredicto, en «Selección del modelo y qué establece
  la comparación»; el contraste con RT-DETR-L (misma familia de cabeza, extractor de
  imágenes), que apunta al preentrenamiento en audio como lo que no aporta; y la
  consecuencia favorable para el sistema (el modelo más pequeño basta). La lista
  de «tres factores» que leía como excusas se redujo a un párrafo de alcance, y la
  hipótesis se enuncia como «supera», igual que en la discusión; antes decía
  «iguala o supera» en un lugar y «superara» en otro.

## Pendientes

En la lista única de `research/capitulos/README.md`.

> **17-11-2026. Revisión externa (errores, contradicciones y observaciones críticas).**
>
> - Nuevas celdas del cuaderno, que corren con los volcados de `runs/` (no necesitan el servidor):
>   **diferencias pareadas** (`tab:pares`, bootstrap pareado por grabación de Δrecall y ΔmAP@0,3,
>   en prueba y en validación, y el piso alternativo de 0,50) y **fuga por sesión** (`rLeak*`).
>   Las cifras se calcularon en local y se copiaron a `valores.tex`/`tablas/pares.tex`.
> - **Cambia la lectura**: con la diferencia pareada, Faster R-CNN queda por debajo de las otras
>   tres; la propuesta lo supera (+0,017 [0,004; 0,029]) y empata con YOLO26s y RT-DETR-L.
>   YOLO26s empata con RT-DETR-L también en mAP y gana por el desempate de costo. El criterio
>   «intervalos que se solapan» se reemplazó por el pareado *después* de ver los resultados: está
>   declarado en el texto. En validación la regla elige el mismo modelo.
> - **Fuga por sesión**: 575 de 601 grabaciones de prueba comparten especie y día con alguna de
>   entrenamiento; en las 26 restantes el recall baja a 0,48–0,59. Declarado como limitación
>   (cap. 4, 5, 6) y trabajo futuro (cap. 7). Corregirlo exige repartir por día y reentrenar.
> - IoU medio (lo pide el IOV de RE2.3) añadido junto al mediano; tasa de la propuesta 2×10⁻⁴
>   (el cuaderno redondeaba el exponente); sesgo inicial −log((1−π)/π); Faster R-CNN congela tallo
>   y primera etapa; versiones v0.2.0 / 2.0.0 / 0.2.5 explicadas; cifras por ventana declaradas
>   como no extrapolables (las del producto están en el cap. 6).
> - El capítulo ya no cita un *commit* para los resultados (decisión del autor, 17-11-2026):
>   cita el repositorio y dice que las cifras salen de ejecutar los *scripts* de RE2.2. Se
>   quitaron `rCommit`, `rNCommits` y `rVersion` del cuaderno y de `valores.tex`. El *commit*
>   del cap. 6 (`63f84a8`) se queda: es la prueba de que el protocolo se fijó antes de medir.
