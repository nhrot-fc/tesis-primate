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

## Pendiente

- **La carpeta no existe (15-11-2026).** Ni `research/medios_verificacion/` ni `generar.py`
  están en el disco ni en el historial de git (se notó el 27-10), así que `make medios` falla y
  el anexo describe una carpeta que no está. Hay que reconstruir `generar.py` antes de E3. Las
  carpetas de OE3 que el anexo ya describe (15-11-2026):
  - `06_oe3_revision/RE3.1`: `protocolo.md`;
  - `06_oe3_revision/RE3.2`: `tiempos_sesion.csv` (carpeta, inicio, término y pausas de
    `times.xlsx`, **sin** la columna de notas, que trae nombres) y `test_cpu_times.csv`;
  - `06_oe3_revision/RE3.3`: `respuestas_revision.csv` (ID, estrato, clase, respuesta y
    categoría de cada caja) y `revision_ejemplos.png`.
  La clave de la tarea 3 (`dataset_quality_key.csv`) no se publica mientras esa tarea siga
  abierta.
- **Lista de grabaciones de entrenamiento.** `grabaciones_train.txt` de RE1.2 sólo puede
  generarse donde está el caché con que se entrenó: el servidor `fcandia`. El
  `data/processed/` de esta máquina es una copia anterior, del 20-08-2026, con otras
  particiones: 21 966/9 446/8 007 ventanas frente a las 21 751/9 364/7 598 del
  documento. Hay que correr `make medios` en el servidor, o copiar de ahí
  `data/processed/train_sources.json`.
- **URL de la carpeta.** El anexo cita `.../tree/main/research/medios_verificacion`, que
  existe sólo después de hacer push de la carpeta.
