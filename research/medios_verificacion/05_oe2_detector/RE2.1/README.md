# RE2.1. Diseño experimental, arquitectura propuesta y línea base

**Medio de verificación.** Documento de diseño y módulo de arquitecturas.

**Indicador.** Especifica la arquitectura AST-Deformable-DETR implementada con todos sus hiperparámetros, y define antes de entrenar la comparación con las líneas base declaradas: configuraciones, partición, métricas y regla de selección.

**Dónde lo desarrolla el documento.** Sección 5.2, que es el documento de diseño; la sección 5.2.5 contrasta cada elemento del indicador.

| Elemento del indicador | Dónde se comprueba |
|---|---|
| Arquitectura AST-Deformable-DETR implementada | Módulo de arquitecturas del repositorio (`src/models/`); `configuraciones/detr_t10_logmel_birds_v3.json`, clave `architecture` |
| Todos sus hiperparámetros | `configuraciones/detr_t10_logmel_birds_v3.json`, bloque `hparams` |
| Configuraciones comparadas | Los cuatro `configuraciones/*.json`; las dos de Ultralytics, también con su `*_ultralytics.yaml` |
| Partición | Bloque `dataset` de cada configuración: el mismo `meta.json` del caché en las cuatro |
| Métricas y regla de selección | Sección 5.2.4 del documento |
