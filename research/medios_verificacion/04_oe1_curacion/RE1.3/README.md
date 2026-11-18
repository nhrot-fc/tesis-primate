# RE1.3. Ficha del conjunto derivado

**Medio de verificación.** Capítulo de curación, README y metadatos del conjunto.

**Indicador.** Describe las ocho especies, el vocabulario, la composición, el desbalance, el diccionario de campos, el ventaneo de 3 s con salto de 1,5 s, la normalización log-mel, las exclusiones y las limitaciones conocidas.

**Dónde lo desarrolla el documento.** Sección 4.5; la sección 4.5.10 ubica cada elemento.

Esta es la ficha en un solo archivo, en el orden de los nueve elementos del indicador. Las tablas y los textos se toman del capítulo 4 y de sus salidas, sin transcribir ninguna cifra; las tablas, figuras y secciones que citan son las del documento. Los metadatos del conjunto (`meta.json` y `labels.json`) están en la carpeta RE1.2.

## 1. Las ocho especies

*Tabla 4.16. Composición por especie*

| Especie | Cód. | Grab. | Horas | Anot. | Pares | Clases | Anot. en experimento |
|---|---|---|---|---|---|---|---|
| *Leontocebus weddelli* | LW | 410 | 3,7 | 4 159 | 8 | 5 | 3 450 |
| *Sapajus macrocephalus* | SM | 582 | 5,3 | 3 668 | 8 | 5 | 3 381 |
| *Alouatta sara* | AS | 778 | 7,2 | 3 619 | 2 | 2 | 3 269 |
| *Ateles chamek* | AC | 384 | 3,7 | 3 106 | 6 | 4 | 2 980 |
| *Saimiri boliviensis* | SB | 256 | 2,8 | 2 348 | 4 | 4 | 2 313 |
| *Plecturocebus toppini* | PT | 320 | 2,8 | 1 185 | 5 | 2 | 1 019 |
| *Aotus azarae* | AA | 82 | 0,8 | 1 095 | 3 | 3 | 1 083 |
| *Cebus cuscinus* | CC | 7 | 0,0 | 31 | 1 | 0 | 0 |
| **Total** |  | 2 691 | 25,0 | 19 211 | 37 | 25 | 17 495 |

## 2. Vocabulario

*Tabla 4.6. Vocabulario cerrado*

| Esp. | Tipos | Códigos de tipo de llamada |
|---|---|---|
| AA | 3 | `gc` (gulp call), `hm` (hoot call), `sc` (squeak call) |
| AC | 6 | `chc` (chitter call), `gc` (growl call), `cc` (contact call), `whc` (whinny call), `sc` (squeak call), `bc` (bark call) |
| AS | 2 | `hc` (howl call), `bc` (bark call) |
| CC | 1 | `cc` (contact call) |
| LW | 11 | `cc` (contact call), `cs` (contact syllable), `aa` (aerial alarm call), `ta` (terrestrial alarm call), `sqc` (squeal call), `phc` (phee call), `tr` (trino call), `tf` (trino fast call), `tj` (unofficial tj call), `tt` (trino transition call), `vc` (visual contact call) |
| PT | 5 | `dc` (duet call), `ac` (alarm call), `bp` (unofficial bellow phrase), `pp` (unofficial pant phrase), `sqc` (unofficial squeal call) |
| SB | 5 | `pcc` (peep contact call), `ppc` (play peep call), `lpc` (long peep call), `spc` (spit call), `sc` (shriek call) |
| SM | 8 | `cc` (contact call), `acc` (aggressive contact call), `sc` (squeal call), `pc` (purr call), `whc` (whistle call), `hic` (hip call), `fc` (food call), `fs` (food syllable) |

*Tabla 4.7. Pares marcados para revisión*

| Par | Filas | Par | Filas |
|---|---|---|---|
| `as/cp` | 91 | `sb/phc` | 3 |
| `as/pp` | 85 | `as/pc` | 3 |
| `as/ip` | 73 | `sm/chc` | 3 |
| `as/` | 53 | `lw/c` | 3 |
| `sm/` | 47 | `cc/` | 2 |
| `pt/` | 30 | `sb/cc` | 1 |
| `ac/` | 21 | `sb/hic` | 1 |
| `lw/a` | 20 | `as/chc` | 1 |
| `lw/b` | 15 | `sm/spc` | 1 |
| `lw/` | 15 | `aa/hf` | 1 |
| `sb/` | 11 | `aa/` | 1 |
| `sb/pcs` | 8 | `pt/d` | 1 |
| `sm/hc` | 8 | `pt/chc` | 1 |
| `aa/sqc` | 6 | `lw/t` | 1 |
| `as/cc` | 4 | `lw/chc` | 1 |
| `sm/sic` | 4 | `lw/lw` | 1 |

## 3. Composición

*Tabla 4.11. Composición de las particiones*

| Partición | Grabaciones | Ventanas | Vacías | % vacías | Cajas | GB |
|---|---|---|---|---|---|---|
| Entrenamiento | 1 248 | 21 751 | 4 608 | 21,2 % | 26 938 | 3,44 |
| Validación | 690 | 9 364 | 2 713 | 29,0 % | 10 114 | 1,48 |
| Prueba | 601 | 7 598 | 2 357 | 31,0 % | 7 931 | 1,20 |
| **Total** | 2 539 | 38 713 | 9 678 | 25,0 % | 44 983 | 6,13 |

*Tabla 4.12. Las 25 clases del experimento*

| Clase | Tipo de llamada | Anot. | Grab. | Train | Val | Test | Train % |
|---|---|---|---|---|---|---|---|
| `ac/bc` | bark call | 2 453 | 137 | 2 838 | 1 065 | 828 | 60 % |
| `lw/cs` | contact syllable | 2 279 | 248 | 2 667 | 1 000 | 779 | 60 % |
| `sm/cc` | contact call | 2 031 | 466 | 2 353 | 883 | 686 | 60 % |
| `as/bc` | bark call | 1 894 | 236 | 2 216 | 831 | 647 | 60 % |
| `as/hc` | howl call | 1 375 | 697 | 6 933 | 2 599 | 2 021 | 60 % |
| `sb/ppc` | play peep call (ppc, lpc) | 1 189 | 191 | 1 340 | 529 | 464 | 57 % |
| `aa/gc` | gulp call | 797 | 58 | 922 | 345 | 269 | 60 % |
| `sb/spc` | spit call | 729 | 166 | 859 | 323 | 250 | 60 % |
| `sm/fs` | food syllable | 691 | 89 | 799 | 301 | 232 | 60 % |
| `lw/trino` | trino (tr, tj, tt, tf) | 633 | 122 | 739 | 278 | 216 | 60 % |
| `pt/dc` | duet call | 632 | 292 | 1 944 | 729 | 567 | 60 % |
| `pt/sqc` | unofficial squeal call | 387 | 13 | 465 | 161 | 138 | 61 % |
| `sb/pcc` | peep contact call | 290 | 95 | 341 | 128 | 100 | 60 % |
| `lw/ta` | terrestrial alarm call | 265 | 80 | 311 | 116 | 91 | 60 % |
| `sm/hic` | hip call | 254 | 61 | 296 | 111 | 86 | 60 % |
| `sm/pc` | purr call | 242 | 83 | 283 | 106 | 83 | 60 % |
| `ac/chc` | chitter call | 229 | 142 | 281 | 105 | 82 | 60 % |
| `sm/sc` | squeal call | 163 | 33 | 195 | 73 | 56 | 60 % |
| `aa/sc` | squeak call | 159 | 33 | 183 | 68 | 53 | 60 % |
| `ac/sc` | squeak call | 158 | 31 | 188 | 71 | 54 | 60 % |
| `lw/sqc` | squeal call | 154 | 43 | 182 | 69 | 52 | 60 % |
| `ac/gc` | growl call | 140 | 65 | 191 | 71 | 56 | 60 % |
| `aa/hm` | hoot call | 127 | 24 | 148 | 54 | 44 | 60 % |
| `lw/vc` | visual contact call | 119 | 44 | 138 | 52 | 40 | 60 % |
| `sb/sc` | shriek call | 105 | 51 | 126 | 46 | 37 | 60 % |
| **Total** |  | 17 495 | 2 527 | 26 938 | 10 114 | 7 931 | 60 % |

## 4. Desbalance

La distribución es de cola larga en dos niveles. Entre especies, *Leontocebus weddelli* aporta 8 tipos de llamada y *Cebus cuscinus* uno solo, con tan pocas anotaciones que no llega a ser clase. Dentro de una especie la distribución es tan sesgada como entre especies: los diez pares mayores concentran el 76 % de las anotaciones en vocabulario y la cola llega hasta pares con una sola (Figura 4.8). Entre `ac/bc`, con 2 453 anotaciones, y `sb/sc`, con 105, median más de veinte veces.

## 5. Diccionario de campos

*Tabla 4.15. Contenido del caché*

| Archivo | Contenido |
|---|---|
| `train.pt`, `val.pt`, `test.pt` | Ventanas de cada partición: mel, cajas y etiquetas |
| `labels.json` | Índice de clase → nombre `especie/llamada`, las 25 clases en orden alfabético |
| `meta.json` | Semilla, criterio de selección, exclusiones y fusiones, proporciones de partición, rango en dB y parámetros del espectrograma |
| `train_sources.json`, … | Grabación y segundo de inicio de cada ventana, para volver al audio |

*Tabla 4.19. Columnas de las tablas curadas*

| Columna | Tipo | Significado |
|---|---|---|
| `species` | texto | Código de especie en minúscula (`lw`); el de la carpeta |
| `written_species` | texto | Lo que traía la columna `Species`, en minúscula |
| `call_type` | texto | Código del tipo de llamada normalizado (`cs`); puede estar fuera del vocabulario |
| `begin_time_s` | real | Inicio de la caja, en segundos desde el inicio de la grabación |
| `end_time_s` | real | Fin de la caja, en segundos; mayor que `begin_time_s` + 0,01 |
| `low_freq_hz` | real | Frecuencia inferior de la caja, en Hz |
| `high_freq_hz` | real | Frecuencia superior, en Hz; no mayor que 22 050 |
| `requires_review` | lógico | La fila no pudo asignarse con certeza a un par del vocabulario |

*Tabla 4.20. Estructuras del caché*

| Clave o archivo | Tipo | Significado |
|---|---|---|
| `images` | `float32` (N, 1, 128, 331) | Mel de potencia de cada ventana, sin comprimir |
| `boxes` | lista de (n_i, 4) | Cajas (c_x, c_y, w, h) en [0, 1]: x es tiempo, y frecuencia en escala mel |
| `labels` | lista de (n_i,) `int64` | Índice de clase de cada caja, según `labels.json` |
| `labels.json` | JSON | Índice → nombre de clase |
| `meta.json` | JSON | Listado 4.1 |
| `*_sources.json` | JSON | `recordings`, `recording_of_window` y `clip_start_s`: qué grabación y qué segundo origina cada ventana |

## 6. Ventaneo

Ventanas de 3 s con salto de 1,5 s. Las grabaciones tienen duración variable y el detector consume entradas de tamaño fijo. Por eso cada grabación se segmenta en ventanas de 3 s que avanzan 1,5 s. Una anotación entra en una ventana si el solapamiento entre ambas cubre al menos la mitad de la *parte visible* del evento, que es el mínimo entre su duración y la longitud de la ventana (Figura 4.12). Comparar contra la duración completa dejaría inalcanzable a todo evento de más de 6 s, porque no hay posición de la ventana en la que quepa la mitad de un aullido de 30 s. Con la parte visible, ese aullido entra en toda ventana que caiga dentro de él, y ninguna anotación del experimento queda sin ventana. Al entrar, la caja se recorta a los límites de la ventana.

*Tabla 4.13. Efecto del ventaneo en las clases largas*

| Clase | Anot. | Cajas | Cajas por anotación | Anot. >3 s | Cajas en el borde | Ventana entera |
|---|---|---|---|---|---|---|
| `as/hc` | 1 375 | 11 553 | 8,4 | 94,7 % | 99,6 % | 77,5 % |
| `pt/dc` | 632 | 3 240 | 5,1 | 91,0 % | 99,2 % | 63,5 % |
| Resto (23 clases) | 15 488 | 30 190 | 1,9 | 0,3 % | 12,1 % | 0,2 % |
| **Total** | 17 495 | 44 983 | 2,6 | 11,0 % | 40,8 % | 24,6 % |

## 7. Normalización log-mel

*Tabla 4.14. Parámetros del espectrograma*

| Parámetro | Valor | Justificación |
|---|---|---|
| Frecuencia de muestreo | 44 100 Hz | La de casi todas las grabaciones; las pocas que llegan a 48 kHz se remuestrean al leerlas. Su mitad, 22 050 Hz, es el techo de Nyquist que fija el límite superior del rango de frecuencia |
| Ventana / salto | 3,0 s / 1,5 s | Contiene entera a casi todas las clases; el costo es la fragmentación de las dos largas (Tabla 4.13) |
| Solapamiento mínimo | 0,5 | Un evento entra si la mitad de su parte visible cae en la ventana; medirlo contra la parte visible es lo que mantiene alcanzables a los eventos de más de 6 s |
| Longitud de ventana STFT | 1 024 (23 ms) | Más corta que los eventos más breves: 34 ms el mínimo, 81 ms la mediana de `sm/cc` |
| Función de ventana | Hann | Valor por defecto de `torchaudio`; no se comparó con otras |
| Puntos de la transformada rápida de Fourier | 4 096 | Relleno con ceros: densifica la rejilla de frecuencia sin tocar la resolución temporal |
| Salto STFT | 400 (9 ms) | Produce 331 tramas por ventana |
| Bandas mel | 128 | Lo que espera el extractor preentrenado del modelo propuesto |
| Escala mel | HTK (q=2 595, f_quiebre=700 Hz) | Convención por defecto de `torchaudio` (Yang et al., 2022); es la misma fórmula con la que se mapean las cajas |
| Rango de frecuencia | 25 Hz – 22 050 Hz | Un techo de 16 kHz cortaba por arriba el 49 % de las cajas de `sb/spc` |
| Rango en dB | -54,3 a 27,9 dB | Percentiles 1 y 99,9 de 10log₁₀(mel+10⁻⁶) sobre 2 000 ventanas de entrenamiento; los modelos que pintan el mel como imagen normalizan con él |

Rango en dB del caché (`meta.json`, `db_range`): -54,27 a 27,87 dB.

## 8. Exclusiones

*Tabla 4.10. Pares excluidos del experimento*

| Par | Tipo de llamada | Anot. | Motivo |
|---|---|---|---|
| `lw/cc` | contact call | 536 | excluida (frase) |
| `sm/fc` | food call | 126 | excluida (frase) |
| `as/cp` |  | 91 | marcada para revisión |
| `as/pp` |  | 85 | marcada para revisión |
| `lw/aa` | aerial alarm call | 80 | <100 anotaciones |
| `sm/whc` | whistle call | 76 | <100 anotaciones |
| `as/ip` |  | 73 | marcada para revisión |
| `as/` |  | 53 | marcada para revisión |
| `sm/` |  | 47 | marcada para revisión |
| `pt/ac` | alarm call | 44 | <100 anotaciones |
| `pt/pp` | unofficial pant phrase | 43 | <100 anotaciones |
| `pt/bp` | unofficial bellow phrase | 39 | <100 anotaciones |
| `ac/whc` | whinny call | 35 | <100 anotaciones |
| `lw/phc` | phee call | 34 | <100 anotaciones |
| `pt/` |  | 30 | marcada para revisión |
| `ac/cc` | contact call | 29 | <100 anotaciones |
| `cc/cc` | contact call | 29 | <100 anotaciones |
| `ac/` |  | 21 | marcada para revisión |
| `lw/a` |  | 20 | marcada para revisión |
| `lw/b` |  | 15 | marcada para revisión |
| `lw/` |  | 15 | marcada para revisión |
| `sb/` |  | 11 | marcada para revisión |
| `sb/pcs` |  | 8 | marcada para revisión |
| `sm/acc` | aggressive contact call | 8 | <100 anotaciones |
| `sm/hc` |  | 8 | marcada para revisión |
| `aa/sqc` |  | 6 | marcada para revisión |
| `as/cc` |  | 4 | marcada para revisión |
| `sm/sic` |  | 4 | marcada para revisión |
| `sb/phc` |  | 3 | marcada para revisión |
| `as/pc` |  | 3 | marcada para revisión |
| `sm/chc` |  | 3 | marcada para revisión |
| `lw/c` |  | 3 | marcada para revisión |
| `cc/` |  | 2 | marcada para revisión |
| `sb/cc` |  | 1 | marcada para revisión |
| `sb/hic` |  | 1 | marcada para revisión |
| `as/chc` |  | 1 | marcada para revisión |
| `sm/spc` |  | 1 | marcada para revisión |
| `aa/hf` |  | 1 | marcada para revisión |
| `aa/` |  | 1 | marcada para revisión |
| `pt/d` |  | 1 | marcada para revisión |
| `pt/chc` |  | 1 | marcada para revisión |
| `lw/t` |  | 1 | marcada para revisión |
| `lw/chc` |  | 1 | marcada para revisión |
| `lw/lw` |  | 1 | marcada para revisión |

## 9. Limitaciones conocidas

- **Una sola anotación por llamada.** Anotaron varias personas a lo largo del tiempo, pero cada llamada la anotó una sola, y no hay medición de acuerdo entre anotadores (Nguyen Hong Duc et al., 2021). El error de etiquetado residual no puede separarse del error del modelo.
- **La columna de tipo de llamada llega sucia y no se puede sanear sin el equipo.** 32 pares quedan fuera del vocabulario cerrado y no figuran en el vocabulario que el equipo documenta (8 Primates Project, 2025). No hay forma de separar el error tipográfico de la subdivisión legítima aún no documentada (sección 4.3.2). El protocolo los marca y los conserva en lugar de decidir, pero el conjunto de experimento se construye asumiendo que lo que quedó fuera es ruido. Si parte de ello fuera repertorio real, el vocabulario está incompleto. Es la vía de mejora más directa para una versión futura del detector, y lo que requiere es ampliar la referencia.
- **Procedencia de campo incompleta.** De cada grabación se conoce el sitio y, por sus metadatos, el equipo. El material no viene acompañado del protocolo de campo (ganancia, distancia al animal y criterio con que se eligieron los archivos que se anotaron). La ficha reconstruye lo que se puede leer del propio corpus y nada más. El horario de campo del proyecto explica el sesgo horario que se describe arriba, pero no permite saber cuánto de él se debe también al ritmo de actividad de las especies.
- **Cajas dibujadas con criterio variable en frecuencia.** El eje de frecuencia lo fija el analista a ojo, lo que obliga a emparejar las predicciones con un umbral de IoU tolerante.
- **Fragmentación de las llamadas largas.** `as/hc` y `pt/dc` sobreviven al ventaneo como series de cajas recortadas al borde de la ventana, sin ningún mecanismo que las reagrupe en el evento original (Tabla 4.13). Corregirlo exigiría una segunda longitud de ventana para esas clases, o una agregación que reconstruya el evento completo a partir de sus fragmentos antes de puntuarlo. Ninguna de las dos se implementó.
- **Frases excluidas.** Las clases de frase quedan fuera por anidamiento y dejan sin cubrir el nivel superior de la jerarquía vocal.
- **Cobertura parcial del repertorio.** Los tipos de llamada bajo el umbral de 100 anotaciones y los 32 pares fuera de vocabulario no entran al experimento; *Cebus cuscinus* no tiene ninguna clase.
- **Escasez en 11 clases**, que aportan menos de 300 cajas de entrenamiento cada una. Cuando una de ellas obtiene mal *recall*, la causa puede ser el modelo o puede ser que nunca vio material suficiente.
- **Etiquetas que esconden dos formas.** `ac/bc` y `sm/fs` agrupan cada una dos formas de caja distintas (Figura 4.21), porque la guía del equipo deja a criterio del anotador incluir los armónicos.
- **Partición por grabación, no por seguimiento.** Ninguna grabación está en dos particiones, pero casi todas las de prueba comparten especie y día con alguna de entrenamiento. Son del mismo seguimiento, con el mismo individuo, fondo y equipo. El Capítulo 5 mide cuánto puede inflar eso las cifras (sección 5.7).
- **Copias de grabaciones en la capa primaria.** El mismo audio aparece en varias carpetas de especie, anotado solo para la especie de cada carpeta; la capa derivada las une, pero la llamada de una tercera especie no anotada en ninguna copia sigue sin caja.
- **Sesgo horario y estacional.** La mayoría de las grabaciones son matinales y se concentran en pocos meses (Figura 4.22); no hay garantía de que el fondo acústico de otras horas o estaciones esté representado.
- **Solo la especie focal.** En cada grabación se anotaron solo las llamadas del primate al que se apuntaba (equipo de investigación del 8 Primates Project, comunicación personal, 17 de noviembre de 2026); las de otras especies quedan sin caja.
- **Grabaciones focales, no de PAM.** Todo el material se grabó a mano, con micrófono parabólico o con teléfono, dirigido al primate que vocaliza. El PAM del equipo, al que el sistema quiere apoyar, usa grabadoras AudioMoth fijas, en las que las llamadas llegan más lejanas y con más fondo. La transferencia a ese material no está medida.
