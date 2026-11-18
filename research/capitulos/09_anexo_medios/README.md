# Anexo B. Medios de verificación

Describe la carpeta `research/medios_verificacion/`, que reúne la evidencia de cada IOV
que no cabe en el documento. Se creó el 07-10-2026, para responder a dos observaciones
de la revisión de E1:

- «no se encontró referencia clara al medio de verificación» de RE2;
- «no se proporciona acceso claro a la ficha del conjunto».

## Cómo se mantiene

- `make medios` regenera las carpetas de OE1 y OE2 y el zip
  `research/build/medios_verificacion.zip`, que se entrega junto al PDF.
- Las cifras salen de las macros y los fragmentos de los cuadernos de los capítulos 4
  y 5, y de `runs/`. Después de volver a ejecutar esos cuadernos, hay que correr
  `make medios`.
- `06_oe3_revision/README.md` y el `README.md` raíz se escriben a mano.

## Pendientes

En la lista única de `research/capitulos/README.md`.

