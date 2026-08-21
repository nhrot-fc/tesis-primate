### Bloque 1: Aceleración y ventajas de la recolección con PAM (Monitoreo Acústico Pasivo)

* **1. Sensores no invasivos y omnidireccionales:**
> "These are noninvasive, autonomous, usually omni-directional (sampling a three-dimensional sphere around the sensor), and offer the advantage of a larger detection area and fewer taxonomic restrictions than camera traps..."
> 
> 


* **Fuente:** Gibb 2018
* **Página:** 170-171
* **Sección:** 2. Passive Acoustics Applications in Ecology


* **2. Monitoreo continuo y especies raras:**
> "Their benefits over traditional surveys include continuous surveying for long periods with low manual effort, and the associated higher likelihood of detecting rarer or less vocally active species..."
> 
> 


* **Fuente:** Gibb 2018
* **Página:** 171
* **Sección:** 2. Passive Acoustics Applications in Ecology


* **3. Bajo costo y redes a gran escala:**
> "The arrival of low-cost, open-source sensors is now rapidly expanding access to PAM technologies..."
> "Such initiatives now enable deployment of multisensor networks at scale, involving both experts and volunteers..."
> 
> 


* **Fuente:** Gibb 2018
* **Páginas:** 169 (Abstract) y 170 (1. Introduction)


* **4. El "diluvio de datos" y la accesibilidad:**
> "This is both enabled and demanded by the twenty-first century data deluge: digital recording devices, data storage and sharing have become dramatically more widely available, and affordable for large-scale bioacoustic monitoring, including continuous audio capture..."
> 
> 


* **Fuente:** Stowell 2022
* **Página:** 2
* **Sección:** Introduction


* **5. Estandarización y escalabilidad para estudios ecológicos:**
> "Passive acoustic monitoring (PAM) is thus increasingly suited to objectives-driven survey and monitoring programmes, whose protocols must be standardisable, scalable, and financially sustainable..."
> 
> 


* **Fuente:** Gibb 2018
* **Página:** 170
* **Sección:** 1. Introduction



---

### Bloque 2: El cuello de botella y la despersonalización del sesgo manual

* **6. El límite del tiempo humano (cuello de botella):**
> "The resulting deluge of audio data means that a common bottleneck is the lack of person-time for trained analysts, heightening the importance of methods that can automate large parts of the workflow, such as machine learning."
> 
> 


* **Fuente:** Stowell 2022
* **Página:** 2
* **Sección:** Introduction


* **7. El análisis manual es subjetivo y difícil de cuantificar estadísticamente:**
> "Conducted manually, this process is time-consuming and subjective, and it is difficult to quantify biases related to analyst knowledge level, which may be particularly problematic in resource-limited conservation settings..."
> 
> 


* **Fuente:** Gibb 2018
* **Página:** 176
* **Sección:** 4.1 Developing a pipeline for automated sound identification


* **8. La automatización evita el sesgo inherente a los anotadores:**
> "Standardised post hoc analysis also avoids the skill level biases in species identification that often impact citizen science data..."
> 
> 


* **Fuente:** Gibb 2018
* **Página:** 171
* **Sección:** 2. Passive Acoustics Applications in Ecology


* **9. La escasez de expertos para tareas de categorización fina:**
> "...the sound events may require a subject expert to annotate them with the correct labels (for training), and this expert time is often in short supply. Such constraints are felt for fine categorical distinctions [...] and also for large-scale monitoring in which the data volume far exceeds the person hours available."
> 
> 


* **Fuente:** Stowell 2022
* **Página:** 15
* **Sección:** Small data: data augmentation, pre-training, embeddings


* **10. La IA como herramienta de estandarización, no como reemplazo humano:**
> "Firstly, AI does not replace expertise... As the use of these systems becomes even more standardised, they take on the role of expert peers, with whom we consult and debate... Hence, DL does not displace the role of experts, nor even of crowdsourcing; future work in the field will integrate the benefits of all three..."
> 
> 


* **Fuente:** Stowell 2022
* **Páginas:** 21-22
* **Sección:** A Roadmap for Bioacoustic Deep Learning


### Hecho 1: El volumen supera la capacidad de revisión (El análisis manual no escala)

* **1. El principal cuello de botella:**
> "The process of collecting annotations is thus the main bottleneck in building such methods."
> 
> 


* **Fuente:** Nguyen Hong Duc et al. 2021
* **Página:** 1
* **Sección:** 1. Introduction


* **2. Intensivo y laborioso:**
> "...annotation in marine bioacoustics, besides being resource intensive, laborious and time consuming, is compounded by the intrinsic difficulty in discriminating underwater acoustic sources."
> 
> 


* **Fuente:** Nguyen Hong Duc et al. 2021
* **Página:** 1
* **Sección:** 1. Introduction


* **3. Dificultad de recolección a gran escala:**
> "In SED [Sound Event Detection], it is very time-consuming to collect large-scale strong labels, and in most cases, multiple workers divide up the annotations to create a single dataset."
> 
> 


* **Fuente:** Koga et al. 2024
* **Página:** 1
* **Sección:** Abstract



### Hecho 2: La etiqueta producida depende de quién la produce (Variación entre anotadores)

* **4. La subjetividad inherente a la tarea:**
> "However, besides being laborious and resource intensive, recent studies have suggested that such a task could also be highly subjective with the generation of annotator specific errors."
> 
> 


* **Fuente:** Nguyen Hong Duc et al. 2021
* **Página:** 1
* **Sección:** Abstract


* **5. Variación en el tipo de sonido y el tiempo:**
> "In general, strong labels created by multiple annotators have large variations in the type of sound events and temporal onset/offset."
> 
> 


* **Fuente:** Koga et al. 2024
* **Página:** 1
* **Sección:** Abstract


* **6. Imposibilidad de determinar una etiqueta única:**
> "Through the annotations of multiple workers, uniquely determining the strong label is quite difficult because the dataset contains sounds that can be mistaken for similar classes and sounds whose temporal onset/offset is difficult to distinguish."
> 
> 


* **Fuente:** Koga et al. 2024
* **Página:** 1
* **Sección:** Abstract


* **7. Impacto de Leroy et al. (2018) sobre la falta de acuerdo:**
> "They revealed both a strong inter-annotator variability between two annotators (with less than 50% agreement between annotators), but also a poor agreement obtained with an annotator annotating the same audio segment twice."
> 
> 


* **Fuente:** Nguyen Hong Duc et al. 2021 (citando a Leroy et al. 2018)
* **Página:** 2
* **Sección:** 1. Introduction


* **8. La inevitabilidad de la variación temporal:**
> "The temporal variation in strong labels is inevitable... This not only affects the model training in SED, but also has a significant impact on the performance evaluation of the trained model."
> 
> 


* **Fuente:** Koga et al. 2024
* **Página:** 3
* **Sección:** IV.B Temporal Variation of Strong Labels


* **9. La subjetividad como un rol clave (Conclusión de Nguyen):**
> "The large differences among annotator behaviors show that subjectivity plays a key role in annotating underwater sounds, which should be included into automatic classification systems..."
> 
> 


* **Fuente:** Nguyen Hong Duc et al. 2021
* **Página:** 10
* **Sección:** 5. Conclusion


* **10. La dificultad para cuantificar los sesgos manuales:**
> "Conducted manually, this process is time-consuming and subjective, and it is difficult to quantify biases related to analyst knowledge level..."
> 
> 


* **Fuente:** Gibb et al. 2018
* **Página:** 176
* **Sección:** 4.1 Developing a pipeline for automated sound identification


### Paso 1: Clasificadores a nivel de segmento y priorización del *recall*

* **1. Reducción de tiempo con baja precisión bajo desbalance (HOWLish):**
> "HOWLish’s base precision was low (0.006, Table 2), but ultimately connected to the extremely imbalanced (7124:1) distribution of the test dataset... This quality control step took 0.74 h of operator time-approximately 15 times faster than manually annotating the entire holdout dataset."
> 
> 


* **Fuente:** Campos et al. (HOWLish)
* **Páginas:** 8 (Sección: Discussion) y 7 (Sección: Case study)



### Paso 2: Detección temporal (*onset* y *offset*)

* **2. Cajas delimitadoras puramente temporales (SEBB):**
> "In mathematical terms, we define SEBBs as quadruples $\hat{b}_j = (\hat{c}_j, \hat{t}_{on,j}, \hat{t}_{off,j}, \bar{y}_j)$ which intuitively represent sound event candidates defined by sound class $\hat{c}_j$, a fixed extent given by onset time $\hat{t}_{on,j}$ and offset time $\hat{t}_{off,j}$, plus an overall presence confidence score $\bar{y}_j$."
> 
> 


* **Fuente:** Ebbers et al. 2024 (SEBB)
* **Página:** 563
* **Sección:** 2.3 Sound Event Bounding Boxes



### Paso 3: Detección sobre el espectrograma como imagen 2D

* **3. El concepto de *Eventness* (Detección de objetos en espectrogramas):**
> "The key observation behind the eventness concept is that audio events reveal themselves as 2-dimensional time-frequency patterns with specific textures and geometric structures in spectrograms. These time-frequency patterns can then be viewed analogously to objects occurring in natural images..."
> 
> 


* **Fuente:** Pham et al. 2018 (Eventness)
* **Página:** 2491
* **Sección:** 1. Introduction


* **4. El análogo actual (Anotaciones 2D y modelo de dos etapas):**
> "The dataset includes precise time and frequency annotations gathered by dozens of bird enthusiasts across France... This comprises the development of a novel two-stage deep object detection model optimized for audio data, achieving competitive accuracy on the 45 most represented species..."
> 
> 


* **Fuente:** Airale et al. 2026 (NBM)
* **Página:** 1
* **Sección:** Abstract


### El Vacío y la Contribución (Por qué es necesaria esta tesis)

**Idea (a): Ni para primates neotropicales existe el dataset anotado ni el detector.**
*(Nota: Dado que los documentos proporcionados se enfocan en turbinas eólicas, pájaros, lobos y mamíferos marinos, el hecho de que ninguno aborde a los primates neotropicales amazónicos confirma tu premisa de que no existe. Esta ausencia en el estado del arte provisto es tu justificación).*

**Idea (b): La detección de cajas tiempo-frecuencia es un área poco explorada incluso fuera de la bioacústica.**
Este es el punto fuerte de Zhu & Sato (DCASE 2025). Ellos afirman claramente que detectar cajas 2D (tiempo y frecuencia) en lugar de solo límites temporales es un campo casi virgen:

* **1. Ignorar la frecuencia es un error limitante:**
> "Sound event detection (SED) aims to identify the time boundaries (i.e., onset and offset times) of specific sound events in audio recordings. However, existing research has largely overlooked an important aspect: the frequency ranges of these events."
> 
> 


* **Fuente:** Zhu & Sato 2025
* **Página:** 150
* **Sección:** 1. Introduction


* **2. La detección tiempo-frecuencia sigue sin explorarse:**
> "In some cases, temporal SED is insufficient as it discards critical information encoded in the frequency domain. Therefore, accurate detection of time-frequency bounding boxes can assist human investigators and enable in-depth analysis using machine learning. Despite its significance, the detection of time-frequency bounding boxes remains largely unexplored."
> 
> 


* **Fuente:** Zhu & Sato 2025
* **Página:** 150
* **Sección:** 1. Introduction
