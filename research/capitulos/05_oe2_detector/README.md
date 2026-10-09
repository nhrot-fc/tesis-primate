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

## Extra

- `protocolo_comparacion.md` — notas previas sobre cómo se ponen los modelos en
  la misma tabla; el código es `src/evaluation/protocol.py` y la sección 5.2.6
  del capítulo lo documenta.
- `experimento_detr_coco.md` — el experimento que iguala la política de
  inicialización. Alimenta la discusión sobre arquitectura frente a
  inicialización.

## Pendientes

- [x] ~~RT-DETR-L no tiene preset en `src/train.py`.~~ **Resuelto el 02-10-2026.**
      La receta se recuperó del `config.json` de `runs/rtdetr_l_v3` y entró como
      `--arch rtdetr`: AdamW con `lr0=1e-4`, `weight_decay=1e-4` y lote 16.
      `training/yolo.fit` elige la clase de Ultralytics por el prefijo `rtdetr` y
      sólo escribe optimizador y tasa en `config.json` cuando se fijan a mano, que
      es como están en la corrida publicada y como **no** están en la de YOLO.
      Probado de punta a punta con una corrida de humo de una época.
- [ ] **Las dos ablaciones que separarían extractor de cabeza** están
      implementadas y sin correr (el capítulo las deja como trabajo futuro; si se
      corren antes de E3, `detr_resnet` es la que responde la hipótesis con la
      misma receta): AST congelado (`--hp frozen_backbone=true`) y el
      Deformable DETR de COCO sobre el mismo mel (`--arch detr_resnet`). PCEN
      (`--hp frontend=pcen`) ya no cuenta como ablación: el autor lo descartó por
      bajo rendimiento (07-10-2026), y el capítulo describe solo el log-mel. El interruptor del extractor congelado **no existía**
      hasta el 02-10-2026, aunque el capítulo decía que sí: se añadió a
      `ASTBackbone` (congela los parámetros y lo deja en modo `eval`, para que no
      aporte dropout ni estadísticas de lote nuevas).
- [x] ~~**Publicar las métricas de RE2.3.**~~ **Resuelto el 07-10-2026.** Están
      en `research/medios_verificacion/05_oe2_detector/RE2.3/`, junto al punto de
      operación de cada corrida y la matriz de confusión. Los cinco «Medio de
      verificación» del capítulo remiten a su carpeta del Anexo B, y la de RE2.4
      añade la comprobación de carga estricta del checkpoint (708 de 708
      tensores) y su SHA-256. Falta hacer push para que la URL del anexo exista.
- [ ] **No hay techo de evaluación**: falta medir el acuerdo entre anotadores
      sobre una muestra. Es la limitación más seria del capítulo.
- [ ] **Faltan once citas.** Las marcas `[CITA: …]` se quitaron del `.tex` el
      02-10-2026, así que las afirmaciones están **sin respaldo bibliográfico**:
      AST (Gong et al. 2021, ya en `references.bib` como `gong_ast_2021`) y AudioSet (Gemmeke et al. 2017); ViTDet (Li et al.
      2022); DETR (Carion et al. 2020); PCEN (Wang et al. 2017); pérdida focal
      (Lin et al. 2017), GIoU (Rezatofighi et al. 2019) y algoritmo húngaro
      (Kuhn 1955); Powdermill (Chronister et al. 2021) y PteroSet; Faster R-CNN
      (Ren et al. 2015) y FPN (Lin et al. 2017); YOLO26 / Ultralytics; AdamW
      (Loshchilov y Hutter 2019) y OneCycle (Smith 2018); *cluster bootstrap*
      (Efron y Tibshirani 1993); y el software (PyTorch, Transformers,
      Ultralytics). La única que sí se cableó es la definición de AP, que ahora
      cita `lin_2014` (COCO), ya presente en `references.bib`.
- [x] ~~Repeticiones entre resultados y discusión~~ **Resuelto el 07-10-2026.** «Selección del modelo» queda en resultado (regla, veredicto, contraste con RT-DETR-L); la lista de «tres cosas útiles», el párrafo de alcance y el de Zhu salieron de allí porque la discusión y las limitaciones ya los cubren. La discusión remite a las cifras de la sección de ejes en vez de repetirlas, y la limitación de las clases de ventana ya no copia la regla de la sección de métricas.
