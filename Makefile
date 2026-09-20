MAIN         := main
RESEARCH_DIR := research
BUILD_DIR    := $(RESEARCH_DIR)/build
PDF          := $(BUILD_DIR)/$(MAIN).pdf
LOG          := $(BUILD_DIR)/$(MAIN).log
FIGURES      := $(wildcard $(RESEARCH_DIR)/figures/*)
LATEX_FLAGS  := -interaction=nonstopmode -halt-on-error -file-line-error \
                -output-directory=build

MANUAL_DIR   := docs/manual
MANUAL_PDF   := $(MANUAL_DIR)/build/manual.pdf
DEMO         := resources/demo.mp4

.PHONY: all clean warnings help manual screenshots demo

all: $(PDF)

$(PDF): $(RESEARCH_DIR)/$(MAIN).tex $(RESEARCH_DIR)/references.bib $(FIGURES)
	@mkdir -p $(BUILD_DIR)
	cd $(RESEARCH_DIR) && pdflatex $(LATEX_FLAGS) $(MAIN).tex
	cd $(RESEARCH_DIR) && bibtex build/$(MAIN)
	cd $(RESEARCH_DIR) && pdflatex $(LATEX_FLAGS) $(MAIN).tex
	cd $(RESEARCH_DIR) && pdflatex $(LATEX_FLAGS) $(MAIN).tex
	@$(MAKE) --no-print-directory warnings

# pdflatex escribe el log en ISO-8859, y grep lo trata como binario: sin
# -a no muestra ni una línea y el documento parece estar limpio.
warnings:
	@test -f $(LOG) || { echo 'No hay log: corre "make" primero.'; exit 1; }
	@echo '--- Cajas desbordadas (texto fuera del margen) ---'
	@grep -a -A2 'Overfull\|Underfull' $(LOG) || echo '  ninguna'
	@echo '--- Citas o referencias sin resolver ---'
	@grep -ao "Citation \`[^']*'" $(LOG) | sort -u || echo '  ninguna'
	@grep -a 'Reference .* undefined' $(LOG) || true

# La guía de usuario del paquete de Windows (viewer.exe y detect.exe). Sin bibliografía:
# dos pasadas bastan para el índice y las referencias. build_windows.py la mete en el zip.
manual: $(MANUAL_PDF)

$(MANUAL_PDF): $(MANUAL_DIR)/manual.tex $(wildcard $(MANUAL_DIR)/fig/*.png)
	@mkdir -p $(MANUAL_DIR)/build
	cd $(MANUAL_DIR) && pdflatex $(LATEX_FLAGS) manual.tex
	cd $(MANUAL_DIR) && pdflatex $(LATEX_FLAGS) manual.tex

# Las capturas del manual, desde el propio visor (plataforma offscreen de Qt).
screenshots:
	uv run python $(MANUAL_DIR)/screenshots.py

# El vídeo de demostración: el visor manejado por un guion fijo, YOLO en CPU, a ffmpeg.
demo:
	uv run python $(MANUAL_DIR)/demo.py --out $(DEMO)

clean:
	@rm -rf $(BUILD_DIR) $(MANUAL_DIR)/build

help:
	@printf '%s\n' \
	  'make          Compila el documento y resume los avisos' \
	  'make warnings Vuelve a mostrar los avisos del último log' \
	  'make manual   Compila la guía de usuario (docs/manual/build/manual.pdf)' \
	  'make screenshots  Regenera las capturas del manual desde el visor' \
	  'make demo     Graba el vídeo de demostración (resources/demo.mp4)' \
	  'make clean    Borra los archivos generados'
