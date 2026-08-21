### Stowell 2022

**C1 - Velocidad: la anotación no escala con el volumen grabado**

* "[...] a widespread and persistent issue in bioacoustic projects is the lack of large labelled datasets: [...] the sound events may require a subject expert to annotate them with the correct labels (for training), and this expert time is often in short supply."


* "Such constraints are felt for fine categorical distinctions [...] and also for large-scale monitoring in which the data volume far exceeds the person hours available."



### Koga et al. 2024

**C1 - Velocidad / C2 - Consistencia: doble función**

* "In SED, it is very time-consuming to collect large-scale strong labels, and in most cases, multiple workers divide up the annotations to create a single dataset."


* "In general, strong labels created by multiple annotators have large variations in the type of sound events and temporal onset/offset."



**C2 - Consistencia: La magnitud está medida**

* "To investigate the variation in strong labels, we release the LEAD dataset, which provides distinct strong labels for each clip annotated by 20 different annotators."



**C2 - Consistencia: Es inevitable, no corregible**

* "The temporal variation in strong labels is inevitable, but sufficient solutions have not been found in previous studies [14], [13]."


* "This not only affects the model training in SED, but also has a significant impact on the performance evaluation of the trained model."



### Nguyen Hong Duc et al. 2021

**C1 - Velocidad / C2 - Consistencia: La tarea es subjetiva**

* "However, besides being laborious and resource intensive, recent studies have suggested that such a task could also be highly subjective with the generation of annotator specific errors."



**C2 - Consistencia: La magnitud está medida (vía Leroy 2018)**

* "They revealed both a strong inter-annotator variability between two annotators (with less than 50% agreement between annotators), but also a poor agreement obtained with an annotator annotating the same audio segment twice."



**C2 - Consistencia: Es inevitable, no corregible**

* "The large differences among annotator behaviors show that subjectivity plays a key role in annotating underwater sounds, which should be included into automatic classification systems of underwater sounds."



### Gibb 2018

**C2 - Consistencia: La tarea es subjetiva**

* "Conducted manually, this process is time-consuming and subjective, and it is difficult to quantify biases related to analyst knowledge level, which may be particularly problematic in resource-limited conservation settings (Heinicke et al., 2015; Kalan et al., 2015)."



### Leroy et al. 2018

**C2 - Consistencia: La magnitud está medida**

* "Manual annotations exhibit strong inter- and intra-analyst variability, with less than 50% agreement between analysts."


* "The manual annotations showed considerable variability between different analysts (inter-analyst variability), but also between different rounds of data analysis conducted by the same analyst (intra-analyst variability)."

### Kershenbaum et al. 2016

**C2 - Consistencia: Es inevitable, no corregible (Granularidad sílaba/frase)**

* "Finally, there may be a complex hierarchical structure to the sequence, in which combinations of sounds, which might otherwise be considered fundamental units, always appear together, giving the impression of a coherent, larger unit of communication..."


* "...potential acoustic units for sequence analyses is that they can be hierarchically nested, such that a sequence of units can itself be considered as a unit and replaced with a label."


* "Identifying units is made all the more challenging because acoustic units produced by animals often exhibit graded variation in their features... but most analytical methods for unit classification assume that units can be divided into discrete, distinct categories..."


* "Definitions of units, how they are identified, and the semantic labels we assign them vary widely across researchers working with different taxonomic groups... or even within taxonomic groups..."


* "These units can be organised into longer sequences, of "notes," "trills," "syllables," "phrases," "motifs," and "songs"..."


### C3 - Forma de la salida: lo automático no devuelve lo que el analista necesita

**Kahl et al. 2021 [BirdNET]**

* *Probabilidad por segmento fijo:* "Test recordings were split into 3-s chunks, and species probabilities were retrieved and stored individually for each segment."



**Ebbers 2024 [Sound Event Bounding Boxes]**

* *Solo límites temporales:* "In mathematical terms, we define SEBBs as quadruples $\hat{b}_{j}=(\hat{c}_{j},\hat{t}_{on,j},\hat{t}_{off,j},\overline{y}_{j})$ which intuitively represent sound event candidates defined by sound class $\hat{c}_{j}$, a fixed extent given by onset time $\hat{t}_{on,j}$ and offset time $\hat{t}_{off,j}$, plus an overall presence confidence score $\overline{y}$."



**Venkatesh et al. 2022 [YOHO]**

* *Solo límites temporales:* "We convert the detection of acoustic boundaries into a regression problem instead of frame-based classification. This is done by having separate output neurons to detect the presence of an audio class and predict its start and end points."



**Escobar-Amado et al. 2024 [YOLOV5 / Bearded Seals]**

* *Por qué la frecuencia sí importa:* "Another advantage of using YOLOV5 over other typical DCS is that the predicted bounding boxes have embedded statistical information about the vocalization, such as the duration, bandwidth, and center frequency of the signals."



**Zhu & Sato 2025 [AST DETR]**

* *Por qué la frecuencia sí importa:* "Sound event detection (SED) aims to identify the time boundaries (i.e., onset and offset times) of specific sound events in audio recordings. However, existing research has largely overlooked an important aspect: the frequency ranges of these events."



**Airale et al. 2026 [NBM Dataset]** *(Nota: En los archivos provistos aparece con fecha de publicación 2026)*

* *Por qué la frecuencia sí importa:* "This comprises the development of a novel two-stage deep object detection model optimized for audio data, achieving competitive accuracy on the 45 most represented species, comparable to state-of-the-art systems trained on substantially larger datasets."



---

### Efectos (E1, E2, E3)

**Stowell 2022**

* *E1 - La mayor parte del audio permanece sin anotar:* "...large-scale monitoring in which the data volume far exceeds the person hours available."



**Koga et al. 2024 [LEAD Dataset]**

* *E2 - El ground truth es incompleto y heterogéneo:* "This not only affects the model training in SED, but also has a significant impact on the performance evaluation of the trained model."



**Zhu & Sato 2025 [AST DETR]**

* *E2 - El ground truth es incompleto y heterogéneo (justificación para IoU bajo):* "We observed relatively large misplacements compared with typical object detection in images. This result supports the observation that the boundaries of the sound events are blurred and difficult to determine. Hence, setting the IoU threshold to a small value is feasible."



**Gibb 2018**

* *E2 - El ground truth es incompleto y heterogéneo (enmascaramiento):* "Even when detection precision is high (few false positives), state-of-the-art methods regularly fail to distinguish faint, transient or partially masked calls, leading to high false-negative rates (low recall)..."


* *E3 - Respuestas ecológicas tardías o sobre muestras sesgadas (Estandarización):* "PAM is thus increasingly suited to objectives-driven survey and monitoring programmes, whose protocols must be standardisable, scalable, and financially sustainable..."


* *E3 - Respuestas ecológicas tardías o sobre muestras sesgadas (Detección):* "Their benefits over traditional surveys include continuous surveying for long periods with low manual effort, and the associated higher likelihood of detecting rarer or less vocally active species..."



**Heinicke et al. 2015**

* *E3 - Respuestas ecológicas tardías o sobre muestras sesgadas (Ahorro de tiempo en primates):* "In this study, we evaluated the performance of a passive acoustic monitoring system for four primate species in the highly noisy rain forest environment of the Taï National Park, Côte d’Ivoire."


* *E3 (Ahorro de tiempo):* "Despite the seemingly low precision, time investment for the manual removal of false positives in the system’s output was only 3–5% compared to a human collecting and processing the primate vocalization data."


### C3 - Forma de la salida: lo automático no devuelve lo que el analista necesita

**Campos et al. 2025**

* *Probabilidad por segmento fijo (Detection pipeline):* "[...] (1) input WAV files get segmented into 0.96 s long audio examples; (2) HOWLish predicts whether each example is not-wolf or wolf (continuous prediction value between 0 and 1, respectively); (3) prediction values get averaged by a moving window of W examples..."



---

### Efectos (E1, E2, E3)

**Campos et al. 2025**

* *E1 - La mayor parte del audio permanece sin anotar (El cuello de botella):* "As the number of acoustic recorders scales in the field, the task of detecting acoustic signals in recorded soundscapes quickly becomes a logistical bottleneck, hindering the scalability of PAM studies (Tuia et al., 2022)."


* *E3 - Ahorro de tiempo y esfuerzo manual (Abstract):* "HOWLish retrieved 81.3% of the observed howling events while offering a 15-fold reduction in operator time when compared to fully manual detection."


* *E3 - Ahorro de tiempo y esfuerzo manual (Case Study):* "The detection pipeline [...] required 8 h and 20 min of GPU time (no operator input) to process the entire holdout dataset (3789 h) and output 169 h of data as potentially containing wolf howls-a 22-fold reduction in the volume of data that needed to be manually processed by an operator."


## Raiz

**Stowell 2022**

* *El cuello de botella es la falta de tiempo-persona (Cita ancla):* "The resulting deluge of audio data means that a common bottleneck is the lack of person-time for trained analysts, heightening the importance of methods that can automate large parts of the workflow, such as machine learning."



**Gibb et al. 2018**

* *Sensores de bajo costo permiten desplegar redes a escala (El lado de la oferta):* "The arrival of low-cost, open-source sensors is now rapidly expanding access to PAM technologies, making it vital to evaluate where these tools can contribute to broader efforts in ecology and biodiversity research."


* *Despliegue a escala:* "For example, the recently released AudioMoth low-cost sensor has seen broad uptake... Such initiatives now enable deployment of multisensor networks at scale, involving both experts and volunteers..."



**Nguyen Hong Duc et al. 2021**

* *Recolectar anotaciones es el cuello de botella principal:* "The process of collecting annotations is thus the main bottleneck in building such methods."


