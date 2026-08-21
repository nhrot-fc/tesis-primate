# Figuras generadas

Producidas por `notebooks/thesis_figures.ipynb` (diagramas, figuras de datos y de
resultados) y por `notebooks/dataset_report.ipynb` (figuras del conjunto).
No edites los PNG a mano: se regeneran corriendo el cuaderno.

## `annotation_example.png`

- **Dónde va**: Cap. 1, sección 1.1 (Figura 1.1) - antes de la Tabla 1.1
- **Etiqueta**: `fig:annotation-example`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.85\textwidth]{figures/annotation_example.png}
    \caption{Ejemplo de anotación tiempo--frecuencia tal como la produce el analista en Raven: cada evento se delimita en tiempo y en banda de frecuencia, y se etiqueta con especie y tipo de llamada. Las cuatro coordenadas rotuladas son las cuatro primeras componentes de la ecuación~\eqref{eq:tupla}.}
    \label{fig:annotation-example}
\end{figure}
```

## `output_forms.png`

- **Dónde va**: Cap. 1, sección 1.1.3 (Figura 1.2) - cierre de 'Investigaciones anteriores'
- **Etiqueta**: `fig:output-forms`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/output_forms.png}
    \caption{Las tres formas de salida que la literatura llama detección de eventos sonoros, sobre el mismo fragmento. Sólo la tercera devuelve la tupla completa de la ecuación~\eqref{eq:tupla} y puede sustituir el insumo del analista; las dos primeras lo complementan. Es la causa C3 hecha visible.}
    \label{fig:output-forms}
\end{figure}
```

## `problem_tree.png`

- **Dónde va**: Cap. 1, sección 1.2 (Figura 1.3) - tras el párrafo que cita a Veselý
- **Etiqueta**: `fig:problem-tree`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/problem_tree.png}
    \caption{Árbol de problemas. Las tres causas sostienen el problema central y este produce dos efectos directos, de los que se deriva un tercero. Las condiciones subyacentes S1 y S2 se dibujan fuera del eje causal: enmarcan el problema pero esta tesis no actúa sobre ellas.}
    \label{fig:problem-tree}
\end{figure}
```

## `crisp_dm.png`

- **Dónde va**: Cap. 1, sección 1.6.1 - junto a la Tabla 1.6
- **Etiqueta**: `fig:crisp-dm`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/crisp_dm.png}
    \caption{Las seis fases de CRISP-DM aplicadas al proyecto, con el objetivo específico que atiende cada una y el capítulo que la documenta. El proceso es cíclico: la evaluación devuelve preguntas a la comprensión de los datos.}
    \label{fig:crisp-dm}
\end{figure}
```

## `recorded_vs_annotated.png`

- **Dónde va**: Cap. 1, sección 1.1.1 (evidencia local) y Cap. 4, sección 4.1.3
- **Etiqueta**: `fig:recorded-vs-annotated`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/recorded_vs_annotated.png}
    \caption{Composición del material recibido por especie: horas de audio entregadas, horas que llegaron acompañadas de un archivo de anotación de Raven y minutos de evento efectivamente delimitado. Sobre 33 horas de grabación hay 8.6 horas de evento anotado, el 26.3% del material.}
    \label{fig:recorded-vs-annotated}
\end{figure}
```

## `signal_chain.png`

- **Dónde va**: Cap. 2, apertura del Marco Conceptual (sección 2.2)
- **Etiqueta**: `fig:signal-chain`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/signal_chain.png}
    \caption{Recorrido de la señal desde la vocalización hasta la etiqueta revisable, con la pérdida que introduce cada eslabón. Los valores son los parámetros efectivos de preprocesamiento del proyecto.}
    \label{fig:signal-chain}
\end{figure}
```

## `stft_tradeoff.png`

- **Dónde va**: Cap. 2, sección 2.1.2 - o junto a la tabla de preprocesamiento del Cap. 4
- **Etiqueta**: `fig:stft-tradeoff`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/stft_tradeoff.png}
    \caption{Efecto de la ventana de análisis sobre el mismo evento. Una ventana corta resuelve el tiempo y difumina la frecuencia; una larga hace lo contrario. El proyecto usa 1024 muestras (23\,ms), por debajo de la duración del evento más corto del corpus.}
    \label{fig:stft-tradeoff}
\end{figure}
```

## `mel_axis.png`

- **Dónde va**: Cap. 2, sección 2.2.3 (De la señal a la representación)
- **Etiqueta**: `fig:mel-axis`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/mel_axis.png}
    \caption{La escala mel y su precio. El mapa \texttt{hz\_to\_y} concentra la mitad del alto de la imagen por debajo de unos pocos kilohercios, donde caen las especies graves; a cambio, cada banda cubre cada vez más hercios conforme sube la frecuencia, y con ella crece el error de la coordenada vertical de la caja.}
    \label{fig:mel-axis}
\end{figure}
```

## `pcen_vs_logmel.png`

- **Dónde va**: Cap. 2, sección 2.2.3 - donde se declara el criterio de comparación
- **Etiqueta**: `fig:pcen-vs-logmel`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/pcen_vs_logmel.png}
    \caption{Compresión logarítmica frente a normalización de energía por canal sobre el mismo clip. La comparación se resuelve en el Capítulo~5 bajo el criterio declarado: el \textit{recall} de eventos de baja relación señal-ruido al punto de operación.}
    \label{fig:pcen-vs-logmel}
\end{figure}
```

## `box_coordinates.png`

- **Dónde va**: Cap. 4, sección 4.3.4 (Representación y normalización de las cajas)
- **Etiqueta**: `fig:box-coordinates`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/box_coordinates.png}
    \caption{De la anotación al marco normalizado. El tiempo se divide entre la longitud del clip y la frecuencia pasa por la escala mel de \texttt{hz\_to\_y}, de modo que una caja significa lo mismo en píxeles que en pantalla y eventos de escalas muy distintas son comparables por IoU.}
    \label{fig:box-coordinates}
\end{figure}
```

## `iou_criterion.png`

- **Dónde va**: Cap. 1 sección 1.6.1 (Tabla de métricas) o Cap. 5 (protocolo de evaluación)
- **Etiqueta**: `fig:iou-criterion`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/iou_criterion.png}
    \caption{IoU e IoMin sobre los tres casos que decide el criterio de evaluación. Como la frontera del evento es borrosa incluso entre anotadores, el barrido de umbrales incluye valores bajos; IoMin, implementado en \texttt{architectures/iou.py}, es el que reconoce el anidamiento.}
    \label{fig:iou-criterion}
\end{figure}
```

## `prisma_flow.png`

- **Dónde va**: Cap. 3, sección 3.1.3 (Figura 3.1) - tras la tabla de criterios
- **Etiqueta**: `fig:prisma-flow`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/prisma_flow.png}
    \caption{Flujo de la revisión de literatura. De los 39 documentos identificados en Scopus y Web of Science, los criterios de inclusión y exclusión de la Tabla~\ref{tab:criterios} dejan el conjunto que sostiene las cinco preguntas de revisión.}
    \label{fig:prisma-flow}
\end{figure}
```

## `curation_pipeline.png`

- **Dónde va**: Cap. 4, apertura de la sección 4.2 (Protocolo de curación)
- **Etiqueta**: `fig:curation-pipeline`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/curation_pipeline.png}
    \caption{El protocolo de curación con el conteo que sobrevive a cada paso. Las dos primeras filas corresponden a RE1.1 (limpieza y estandarización) y la tercera a RE1.2 (particionado y ventaneo).}
    \label{fig:curation-pipeline}
\end{figure}
```

## `windowing.png`

- **Dónde va**: Cap. 4, sección 4.3.2 (El ventaneo y su defecto)
- **Etiqueta**: `fig:windowing`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/windowing.png}
    \caption{Los cuatro destinos de una anotación bajo el ventaneo de 3\,s con salto de 1.5\,s. El cuarto caso es el defecto silencioso: los eventos que superan el doble de la longitud del clip no pueden asignarse a ninguna ventana y no llegan nunca al modelo.}
    \label{fig:windowing}
\end{figure}
```

## `split_by_recording.png`

- **Dónde va**: Cap. 4, sección 4.3.1 (Partición por archivo de grabación)
- **Etiqueta**: `fig:split-by-recording`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/split_by_recording.png}
    \caption{Partición por ventana frente a partición por archivo. Con un salto igual a la mitad del clip, dos ventanas consecutivas comparten la mitad de su audio; asignarlas por separado filtraría material de entrenamiento a la partición de prueba.}
    \label{fig:split-by-recording}
\end{figure}
```

## `nesting_phrase.png`

- **Dónde va**: Cap. 4, sección 4.1.6 (Sílabas y frases) - referenciada desde el Cap. 1 (Figura 1.4)
- **Etiqueta**: `fig:nesting-phrase`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.92\textwidth]{figures/nesting_phrase.png}
    \caption{Anidamiento entre frase y sílaba sobre una grabación real. La caja discontinua es la anotación de frase; las llenas son las sílabas anotadas por separado dentro de ella: el 100% de las frases LW/CC contiene al menos una. Las clases de frase se excluyen del vocabulario porque un detector de conjunto plano no puede representar esa jerarquía.}
    \label{fig:nesting-phrase}
\end{figure}
```

## `call_gallery.png`

- **Dónde va**: Cap. 4, sección 4.1.3 (Composición del conjunto curado)
- **Etiqueta**: `fig:call-gallery`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/call_gallery.png}
    \caption{Un ejemplo real de cada una de las clases del experimento, ordenadas por frecuencia mínima. La caja marca la anotación; el resto del panel es contexto. La comparación entre paneles muestra a la vez la separación por banda entre especies y la semejanza entre tipos de llamada de una misma especie.}
    \label{fig:call-gallery}
\end{figure}
```

## `architecture.png`

- **Dónde va**: Cap. 5, sección 5.1 (Figura 5.1)
- **Etiqueta**: `fig:architecture`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/architecture.png}
    \caption{Arquitectura propuesta. El espectrograma log-mel pasa por una capa PCEN entrenable, un Audio Spectrogram Transformer congelado y una pirámide multiescala; el decodificador deformable emite un conjunto de 64 candidatos, cada uno con su clase y su caja. Sólo los bloques marcados como entrenables reciben gradiente.}
    \label{fig:architecture}
\end{figure}
```

## `multiscale_pyramid.png`

- **Dónde va**: Cap. 5, sección 5.1 - junto a la tabla de desviaciones de la receta estándar
- **Etiqueta**: `fig:multiscale-pyramid`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/multiscale_pyramid.png}
    \caption{La pirámide multiescala y las duraciones reales. Cada nivel reconstruye el mapa del AST a una resolución distinta; la columna de un nivel cubre entre 9 y 38\,ms de audio, y las clases del experimento se reparten a ambos lados de esos pasos.}
    \label{fig:multiscale-pyramid}
\end{figure}
```

## `deformable_sampling.png`

- **Dónde va**: Cap. 5, sección 5.1 - al explicar la atención deformable
- **Etiqueta**: `fig:deformable-sampling`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/deformable_sampling.png}
    \caption{Puntos de muestreo de una \textit{query} en la inicialización radial de Zhu et al. Los desplazamientos se expresan como fracción de la caja de referencia, de modo que la vecindad que la \textit{query} inspecciona crece con el evento que sigue: es lo que permite atender con el mismo mecanismo a llamadas de decenas de milisegundos y de segundos.}
    \label{fig:deformable-sampling}
\end{figure}
```

## `decoder_layer.png`

- **Dónde va**: Cap. 5, sección 5.1 - junto a la tabla de desviaciones
- **Etiqueta**: `fig:decoder-layer`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/decoder_layer.png}
    \caption{Una capa del decodificador y el refinamiento iterativo de la caja. Las seis capas comparten estructura y cada una emite su propia predicción; la caja predicha por una capa es la referencia de muestreo de la siguiente.}
    \label{fig:decoder-layer}
\end{figure}
```

## `hungarian_matching.png`

- **Dónde va**: Cap. 5, sección 5.2 (Pipeline de entrenamiento)
- **Etiqueta**: `fig:hungarian-matching`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/hungarian_matching.png}
    \caption{Emparejamiento húngaro entre \textit{queries} y anotaciones. El costo combina clase, distancia $L_1$ entre cajas y GIoU con los pesos de \texttt{HungarianMatcher}; la asignación resultante es uno a uno, y las \textit{queries} sin pareja se entrenan contra la clase de no-objeto. Las cajas del ejemplo son ilustrativas; el cálculo es el del código.}
    \label{fig:hungarian-matching}
\end{figure}
```

## `training_pipeline.png`

- **Dónde va**: Cap. 5, sección 5.2 (Pipeline de entrenamiento e inferencia)
- **Etiqueta**: `fig:training-pipeline`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/training_pipeline.png}
    \caption{Pipeline de entrenamiento. El \textit{checkpoint} se elige por una F-$\beta$ con $\beta = 3$ sobre la partición de validación, que pondera el \textit{recall} nueve veces más que la precisión; la partición de prueba se reserva para el reporte final.}
    \label{fig:training-pipeline}
\end{figure}
```

## `inference_pipeline.png`

- **Dónde va**: Cap. 5, sección 5.2 y sección 5.5 (Exportación a Raven)
- **Etiqueta**: `fig:inference-pipeline`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/inference_pipeline.png}
    \caption{Pipeline de inferencia. La grabación se recorre con el mismo ventaneo y el mismo preprocesamiento del entrenamiento; el posprocesamiento funde las detecciones repetidas entre ventanas solapadas y escribe una tabla de selección con las columnas que espera Raven.}
    \label{fig:inference-pipeline}
\end{figure}
```

## `score_sweep.png`

- **Dónde va**: Cap. 5, sección 5.3 (Protocolo de evaluación) o 5.4 (Resultados)
- **Etiqueta**: `fig:score-sweep`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/score_sweep.png}
    \caption{Efecto del umbral de confianza sobre el \textit{recall}, la precisión y la carga de revisión. La línea vertical marca el punto de operación con el que se seleccionó el \textit{checkpoint} y con el que se reportan todas las cifras.}
    \label{fig:score-sweep}
\end{figure}
```

## `recall_per_class.png`

- **Dónde va**: Cap. 5, sección 5.4 (Figura 5.2)
- **Etiqueta**: `fig:recall-per-class`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/recall_per_class.png}
    \caption{\textit{Recall} por clase al punto de operación, con la causa atribuida a cada caída. Las causas se calculan sobre el propio conjunto: fracción de anotaciones que el ventaneo no alcanza, anidamiento en una clase de frase excluida, heterogeneidad geométrica y duración mediana.}
    \label{fig:recall-per-class}
\end{figure}
```

## `confusion_matrix.png`

- **Dónde va**: Cap. 5, sección 5.4 (Resultados) y 5.8 (Análisis de errores)
- **Etiqueta**: `fig:confusion-matrix`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/confusion_matrix.png}
    \caption{Matriz de confusión sobre las \textit{queries} que el emparejamiento húngaro asignó a una anotación, normalizada por fila. Mide la etiqueta, no la detección: la última columna recoge las \textit{queries} emparejadas que aun así decidieron no-objeto.}
    \label{fig:confusion-matrix}
\end{figure}
```

## `iou_distribution.png`

- **Dónde va**: Cap. 5, sección 5.4 - junto a la tabla de resultados por eje
- **Etiqueta**: `fig:iou-distribution`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/iou_distribution.png}
    \caption{Distribución del IoU sobre los pares que el emparejamiento húngaro seleccionó. La masa se concentra por encima de 0,5 y cae antes de 0,75: el cuello de botella del sistema es el encuadre, no la detección.}
    \label{fig:iou-distribution}
\end{figure}
```

## `qualitative_detections.png`

- **Dónde va**: Cap. 5, sección 5.4 (Figura 5.3)
- **Etiqueta**: `fig:qualitative-detections`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/qualitative_detections.png}
    \caption{Cuatro ventanas de la partición de prueba con la anotación y la predicción superpuestas. Los casos se eligen por criterio e incluyen deliberadamente un encuadre flojo y una propuesta de alta confianza sin anotación detrás, que es la categoría que el análisis de errores tiene que resolver.}
    \label{fig:qualitative-detections}
\end{figure}
```

## `ablation_progression.png`

- **Dónde va**: Cap. 5, sección 5.4 - junto a la tabla de ablación
- **Etiqueta**: `fig:ablation-progression`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.92\textwidth]{figures/ablation_progression.png}
    \caption{Efecto acumulado de las tres decisiones de diseño sobre la precisión media agnóstica a la clase. Cada configuración incluye a la anterior; la última mejora los tres umbrales y además reduce el entrenamiento a una cuarta parte de las épocas.}
    \label{fig:ablation-progression}
\end{figure}
```

## `detection_timeline.png`

- **Dónde va**: Cap. 5, sección 5.5 (Exportación a Raven)
- **Etiqueta**: `fig:detection-timeline`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/detection_timeline.png}
    \caption{Salida del modelo sobre una grabación completa que no participó del entrenamiento, con la anotación del experto superpuesta. Es la forma en que el analista ve la pre-anotación antes de aceptarla o corregirla.}
    \label{fig:detection-timeline}
\end{figure}
```

## `review_time.png`

- **Dónde va**: Cap. 5, sección 5.7 (Tiempo de revisión)
- **Etiqueta**: `fig:review-time`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.85\textwidth]{figures/review_time.png}
    \caption{Minutos de analista por hora de audio, anotación manual frente a revisión asistida sobre el mismo material. El diseño de la prueba controla el efecto de aprendizaje alternando el orden de las condiciones entre analistas.}
    \label{fig:review-time}
\end{figure}
```

## `false_positive_analysis.png`

- **Dónde va**: Cap. 5, sección 5.8 (Análisis de errores)
- **Etiqueta**: `fig:false-positive-analysis`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/false_positive_analysis.png}
    \caption{Reparto de una muestra de 90 falsos positivos de alta confianza entre las tres categorías del protocolo. La primera categoría es la que cambia la interpretación de toda cifra de precisión: son eventos que el modelo encontró y la referencia no registra.}
    \label{fig:false-positive-analysis}
\end{figure}
```

## `wbs_tree.png`

- **Dónde va**: Anexo A, sección A.3.1 - junto a la Tabla A.5
- **Etiqueta**: `fig:wbs-tree`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/wbs_tree.png}
    \caption{Estructura de desglose del trabajo. Los paquetes de nivel superior siguen las fases de CRISP-DM; los de segundo nivel llevan el resultado esperado que entregan.}
    \label{fig:wbs-tree}
\end{figure}
```

## `gantt.png`

- **Dónde va**: Anexo A, sección A.3.2 (Cronograma) - donde está el TODO CRONOGRAMA
- **Etiqueta**: `fig:gantt`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/gantt.png}
    \caption{Cronograma por paquete de trabajo. Las dependencias se respetan por construcción: la preparación de datos precede al modelado, el modelado a la evaluación, y la medición del tiempo de revisión al análisis de errores.}
    \label{fig:gantt}
\end{figure}
```

## `risk_matrix.png`

- **Dónde va**: Anexo A, sección A.3.5 - junto a la Tabla A.9
- **Etiqueta**: `fig:risk-matrix`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.75\textwidth]{figures/risk_matrix.png}
    \caption{Riesgos del proyecto sobre el plano de probabilidad y severidad. La celda de probabilidad y severidad altas la ocupa un solo riesgo, la disponibilidad del tesista, que es el que gobierna la holgura del cronograma.}
    \label{fig:risk-matrix}
\end{figure}
```
