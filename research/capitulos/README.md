# Capítulos de la tesis

Un capítulo, una carpeta. `main.tex` solo tiene el preámbulo, la portada, los
índices y un `\include` por capítulo; todo el contenido vive aquí.

```
capitulos/<nn>_<nombre>/
├── <nn>_<nombre>.tex   el capítulo: se incluye desde main.tex, no se compila solo
├── README.md           qué va en el capítulo, cómo se redacta, qué falta
├── figures/            sus imágenes (+ figures/data/ con las series)
└── notebook/           el cuaderno que genera esas imágenes
```

| Carpeta | Capítulo | Objetivo |
|---|---|---|
| `01_generalidades` | Generalidades | — |
| `02_marco_referencial` | Marco Referencial | — |
| `03_estado_del_arte` | Estado del Arte | — |
| `04_oe1_curacion` | Curación del Conjunto de Datos | **OE1** |
| `05_oe2_detector` | Detección y Evaluación del Modelo | **OE2** |
| `06_oe3_utilidad` | Tiempo de Revisión y Análisis de Errores | **OE3** |
| `07_conclusiones` | Conclusiones | — |
| `08_anexo_plan` | Plan del Proyecto (anexo) | — |

## Las reglas que fijan esta estructura

De `objetivos.md` y `pautas.md`, y no se negocian:

1. **Los objetivos y los resultados esperados están congelados.** Una vez
   presentados no cambian ni ellos ni sus entregables. El Capítulo 1 los copia
   literalmente.
2. **Un capítulo por objetivo específico**, y dentro, **una sección por
   resultado esperado**. De ahí que OE3 tenga capítulo propio en lugar de ir al
   final del de OE2.
3. Cada capítulo de objetivo lleva **Introducción** (enuncia el OE), las
   secciones de resultado y **Discusión**.
4. Cada resultado esperado se redacta **completo en su capítulo**: qué es, cómo
   se alcanzó con los métodos del Capítulo 1, cómo se reproduce, cómo se
   verifica con el medio de verificación declarado y qué mediciones sostienen su
   IOV. No se delega a un anexo ni a un documento aparte; si el medio de
   verificación es extenso, va un resumen y el anexo se referencia.
5. La discusión resume los resultados sin añadir ninguno nuevo, los interpreta,
   los contrasta con la literatura del Capítulo 3, analiza su generalización y
   declara sus limitaciones.

## Compilar

Desde la raíz del repositorio:

```bash
make            # compila research/build/main.pdf y resume los avisos
make warnings   # vuelve a mostrar los avisos del último log
```

Las rutas de `\includegraphics` son relativas a `main.tex`, así que la
compilación se lanza siempre desde `research/` (lo hace el Makefile). Para
trabajar en un solo capítulo, descomenta el `\includeonly` de `main.tex`.

## Qué está pendiente

Cada `README.md` de capítulo termina con su propia lista. En grande:

- OE1 (cap. 4) está **redactado entero**, con sus tres reportes ya absorbidos y todas
  sus cifras generadas por su cuaderno. Sirve de modelo para los capítulos 5 y 6.
- OE2 (cap. 5) solo tiene redactado RE2.3.
- OE3 (cap. 6) está sin redactar, y **RE3.2 necesita una medición de tiempo que
  todavía no se ha hecho**.
- Las conclusiones (cap. 7) se escriben al final.

`research/reportes/` conserva un documento por resultado esperado, más detallado
que el cuerpo en casi todo. Es la fuente de la que hay que trasladar; se borra
cuando no quede nada útil en él.
