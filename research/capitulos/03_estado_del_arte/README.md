# Capítulo 3 — Estado del Arte

La revisión de literatura, con su metodología declarada y sus resultados
organizados por pregunta de revisión. Es el capítulo que se cita en las
discusiones de los capítulos 4, 5 y 6 cuando toca responder «¿son mis
resultados consistentes con trabajos previos?».

> **07-10-2026.** Calificación estricta: la revisión no decía con cuántos trabajos
> se quedó y citaba como resultados dos anteriores a su propio criterio de 2020.
>
> - **Nueva Tabla 3.3 (`tab:estudios`)**: los nueve trabajos en que se apoya la
>   síntesis y la pregunta a la que responde cada uno. He et al. (2016) y
>   Lostanlen et al. (2019) se marcan como antecedentes, fuera de la búsqueda.
> - **La síntesis es ahora la sección 3.3** (`sec:sintesis-revision`); antes
>   colgaba de P5.
> - Citas para BirdVox, Macaulay, AnuraSet y Watkins en P4, y la frase de BirdNET
>   en P2 recuperó su puntuación.
> - **Pendiente (solo el autor lo sabe):** los conteos del flujo de selección. El
>   PRISMA del commit `9d6c92d` tiene «??» en todas las etapas, y hay un
>   comentario `PENDIENTE` al inicio de §3.2.

> **06-10-2026.** Las dos tablas de la metodología quedaban «al aire»: la de
> cadenas de búsqueda no tenía lectura posterior y la de criterios se comentaba
> antes de mostrarse. Ahora las dos siguen el orden presentar → mostrar →
> interpretar.

## Qué va aquí

| Sección | Contenido |
|---|---|
| Metodología de la revisión | Preguntas P1–P5, estrategia de búsqueda, criterios de inclusión y exclusión |
| Resultados de la revisión | Una subsección por pregunta: detección tiempo–frecuencia, clasificación jerárquica, ruido y variabilidad entre sensores, conjuntos y aumento de datos, transferencia |

## Cómo se redacta

- La revisión se lee como el respaldo de las decisiones, no como un catálogo:
  cada subsección termina diciendo qué toma esta tesis de ese cuerpo de trabajo.
- Los trabajos que sirven de comparación cuantitativa (BirdNET, HOWLish,
  Heinicke et al., los detectores 2D en bioacústica) deben quedar aquí con su
  cifra, porque las discusiones de los capítulos 5 y 6 se comparan contra ellas.

## Imágenes

Ninguna. El diagrama PRISMA del flujo de selección se eliminó en una limpieza
anterior (`figures/prisma_flow.png`, recuperable del historial de git).

## Pendientes

- [ ] Decidir si vuelve el diagrama de flujo de la selección de estudios: es la
      figura que normalmente se le pide a una revisión con metodología declarada.
- [ ] Dejar explícita la línea base establecida contra la que compara OE2
      (`\citep{gonzalez_yolo-based_2025}`), porque el IOV de RE2.1 exige que la
      comparación esté definida.
- [x] ~~PCEN repetido del Capítulo 2~~ **Resuelto el 07-10-2026.** La sección de P1 remite a la sección de representaciones del Capítulo 2 y solo añade el dato de Nolasco et al. (2023).
