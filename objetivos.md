OG. Desarrollar y evaluar un sistema de pre-anotación asistida que proponga
cajas tiempo–frecuencia con especie y tipo de llamada sobre grabaciones de
PAM de primates amazónicos, con el fin de reducir el tiempo de revisión
experta.

OE1. Definir un protocolo de curación y estandarización, y aplicarlo s...obre las anotacio-
nes entregadas por el equipo de investigación para construir un conjunto de datos

consistente de cajas tiempo–frecuencia etiquetadas por especie y tipo de llamada.
OE2. Implementar y evaluar un detector que reciba un espectrograma y devuelva cajas
tiempo–frecuencia con especie, tipo de llamada y confianza, comparándolo con una
línea base establecida.
OE3. Cuantificar la reducción del tiempo de revisión que aporta el sistema y caracterizar
sus errores mediante revisión experta de una muestra de predicciones.



Tabla 1.4: Resultados esperados, medios de verificación e indicadores objetivamente verifica-
bles de OE1

RE Resultado Medio de
verificación

Indicador objetivamente
verificable (IOV)

RE1.1 Protocolo de curación
y estandarización
documentado.

Documento del
protocolo.

Define la normalización de columnas
y etiquetas, el mapa de sinónimos, el
descarte de ruido y cajas
degeneradas, la corrección de pares
imposibles, el recorte al límite de
Nyquist y el tratamiento de registros
fuera de vocabulario.

RE1.2 Conjunto acústico
curado y particionado
para el experimento.

Directorio
data/processed/:
etiquetas,
metadatos y tres
archivos .pt.

Los archivos procesados cargan sin
error; meta.json registra la semilla,
el criterio de selección, las
exclusiones, la etiqueta usada y la
normalización; el código divide por
archivo de grabación con
proporciones 60/22,5/17,5 y no por
ventanas aisladas.

RE1.3 Ficha del conjunto
derivado.

Capítulo de
curación,
README y
metadatos del
conjunto.

Describe las ocho especies, el
vocabulario, la composición, el
desbalance, el diccionario de campos,
el ventaneo de 3 s con salto de 1,5 s,
la normalización log-mel, las
exclusiones y las limitaciones
conocidas.




Tabla 1.5: Resultados esperados, medios de verificación e indicadores objetivamente verifica-
bles de OE2
...
RE Resultado Medio de
verificación

Indicador objetivamente
verificable (IOV)

RE2.1 Diseño experimental,
arquitectura
propuesta y línea
base.

Documento de
diseño y módulo de
arquitecturas.

Especifica la arquitectura
AST-Deformable-DETR
implementada, con dimensión 128,
64 consultas y 3 niveles, sus
hiperparámetros, y define la
comparación con la línea base
declarada.

RE2.2 Pipeline reproducible
de entrenamiento,
evaluación e
inferencia.

Scripts de datos,
entrenamiento,
evaluación e
inferencia; módulos
de
src/pipelines/;
pyproject.toml y
manual técnico.

El pipeline genera los splits
procesados, entrena desde un
conjunto de configuración, evalúa un
checkpoint y ejecuta inferencia sobre
un WAV; las dependencias y
comandos de ejecución están
declarados.

RE2.3 Informe comparativo
de resultados.

Informe de
experimentación y
archivos de
métricas.

Reporta, como mínimo, recall,
precisión, F1, IoU medio, AP a IoU
0,25 y 0,5, recall por clase y matriz
de confusión; separa detección,
encuadre y clasificación y justifica la
selección del modelo.

RE2.4 Modelo final cargable
y demostración
funcional.

Checkpoint .pth,
labels.json y
ejecución de
src/infer.py.

El checkpoint se carga sin claves
incompatibles y la inferencia sobre
un WAV produce un archivo
tabulado con las columnas de Raven,
incluidas coordenadas
tiempo–frecuencia, especie, tipo de
llamada y puntuación.

RE2.5 Código y modelo
disponibles en un
repositorio.

Repositorio de
control de
versiones.

La URL del repositorio permite
localizar el código de curación,
entrenamiento, evaluación e
inferencia, junto con las
instrucciones de ejecución y los
artefactos del modelo final que se
hayan decidido publicar.


Tabla 1.6: Resultados esperados, medios de verificación e indicadores objetivamente verifica-
bles de OE3

RE Resultado Medio de
verificación

Indicador objetivamente
verificable (IOV)

RE3.1 Metodología de
validación
comparativa.

Documento de
metodología y
protocolo de
revisión experta.

Especifica el subconjunto de prueba,
el punto de operación (confianza 0,5
y NMS IoU 0,3), el umbral IoU 0,5
para verdadero positivo, las
métricas, el emparejamiento y el
procedimiento para revisar
discrepancias y eventos
potencialmente omitidos.

RE3.2 Medición comparativa
del tiempo de
revisión.

Informe de
validación,
registros de tiempo
y medición de
inferencia.

Reporta minutos por hora de audio
en condición manual y asistida,
describe el material comparable y el
orden de revisión, y presenta el
tiempo de procesamiento del modelo
por hora de audio. No fija un
porcentaje de ahorro antes de
medirlo.

RE3.3 Análisis cualitativo de
errores y hallazgos.

Informe de análisis
y espectrogramas
anotados.

Clasifica una muestra de
predicciones de alta confianza en
evento no anotado, ruido y error de
encuadre, e incluye ejemplos visuales
de aciertos, errores comunes y
detecciones correctas ausentes de la
anotación de referencia.

