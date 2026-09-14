# Presentaciones

Una carpeta por exposición. Cada una toma sus cifras y figuras de `research/reportes/figures/`
(lo que dejan los cuadernos de los reportes) y arma lo demás con un script propio, de modo que
ninguna cifra se escribe a mano.

| Carpeta | Contenido |
|---|---|
| `avance-2026-09/` | Exposición de avance: RE1.1 a RE2.3, con el prototipo de 2024 como contraste |

## Compilar

```bash
cd research/presentaciones/avance-2026-09
uv run python armar_figuras.py      # esquemas TikZ de los reportes, figuras del prototipo, estado de la corrida
tectonic --outdir build avance.tex  # o pdflatex
```

`armar_figuras.py` se vuelve a ejecutar cuando cambian los reportes o avanza la corrida en curso.
