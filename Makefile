MAIN         := main
RESEARCH_DIR := research
BUILD_DIR    ?= $(RESEARCH_DIR)/build
PDF          := $(BUILD_DIR)/$(MAIN).pdf
# Los auxiliares van a su propia carpeta y el PDF se copia a $(PDF) sólo si la
# compilación termina bien: pdflatex borra el PDF de salida cuando falla, y
# research/build/main.pdf está versionado.
AUX_DIR      := $(BUILD_DIR)/aux
AUX_ABS      := $(abspath $(AUX_DIR))
LOG          := $(AUX_DIR)/$(MAIN).log
CHAPTER_DIRS := $(wildcard $(RESEARCH_DIR)/capitulos/*/)
CHAPTERS     := $(wildcard $(RESEARCH_DIR)/capitulos/*/*.tex)
FIGURES      := $(wildcard $(RESEARCH_DIR)/capitulos/*/figures/*) \
                $(wildcard $(RESEARCH_DIR)/figures/*)
LATEX_FLAGS  := -interaction=nonstopmode -halt-on-error -file-line-error

# Bibliografía. Con biber instalado se usa biblatex-apa (APA 7); sin él,
# natbib con apalike, que no es APA 7. main.tex elige según \usarbiber.
# Se puede forzar con: make BIB=apa  o  make BIB=apalike
BIB          ?= $(if $(shell command -v biber 2>/dev/null),apa,apalike)
ifeq ($(BIB),apa)
TEX_INPUT    := -jobname=$(MAIN) "\def\usarbiber{}\input{$(MAIN).tex}"
BIB_RUN      := cd $(RESEARCH_DIR) && biber $(AUX_ABS)/$(MAIN)
else
TEX_INPUT    := $(MAIN).tex
BIB_RUN      := cd $(AUX_DIR) && BIBINPUTS=$(abspath $(RESEARCH_DIR)): bibtex $(MAIN)
endif
LATEX_RUN    := cd $(RESEARCH_DIR) && pdflatex $(LATEX_FLAGS) -output-directory=$(AUX_ABS) $(TEX_INPUT)

MANUAL_DIR   := docs/manual
MANUAL_PDF   := $(MANUAL_DIR)/build/manual.pdf
DEMO         := resources/demo.mp4

.PHONY: all clean warnings help manual screenshots demo medios

all: $(PDF)

# Con -output-directory, cada \include escribe su .aux en una subcarpeta del
# mismo nombre que la del capítulo, y pdflatex no la crea: hay que crearla antes.
# Los auxiliares de biblatex y los de natbib no se mezclan: al cambiar de BIB se
# vacía la carpeta de auxiliares.
$(PDF): $(RESEARCH_DIR)/$(MAIN).tex $(CHAPTERS) $(RESEARCH_DIR)/references.bib $(FIGURES)
	@test -f $(AUX_DIR)/.bib-$(BIB) || rm -rf $(AUX_DIR)
	@for d in $(CHAPTER_DIRS); do mkdir -p "$(AUX_DIR)/capitulos/$$(basename "$$d")"; done
	@touch $(AUX_DIR)/.bib-$(BIB)
	@echo '--- Bibliografía: $(BIB)'
	$(LATEX_RUN)
	$(BIB_RUN)
	$(LATEX_RUN)
	$(LATEX_RUN)
	@cp $(AUX_DIR)/$(MAIN).pdf $(PDF)
	@$(MAKE) --no-print-directory warnings

# pdflatex escribe el log en ISO-8859, y grep lo trata como binario: sin
# -a no muestra ni una línea y el documento parece estar limpio.
warnings:
	@test -f $(LOG) || { echo 'No hay log: corre "make" primero.'; exit 1; }
	@echo '--- Cajas desbordadas (texto fuera del margen) ---'
	@grep -a -A2 'Overfull\|Underfull' $(LOG) || echo '  ninguna'
	@echo '--- Flotantes más altos que la página ---'
	@grep -a 'Float too large' $(LOG) || echo '  ninguno'
	@echo '--- Citas o referencias sin resolver ---'
	@grep -aoE "Citation [\`'][^']*'" $(LOG) | sort -u | grep . || echo '  ninguna'
	@grep -a 'Reference .* undefined' $(LOG) || true

# Los medios de verificación de OE1 y OE2 (research/medios_verificacion/), regenerados desde
# los cuadernos y las corridas, y el zip que se adjunta al documento.
medios:
	uv run python $(RESEARCH_DIR)/medios_verificacion/generar.py
	@mkdir -p $(BUILD_DIR)
	@rm -f $(BUILD_DIR)/medios_verificacion.zip
	cd $(RESEARCH_DIR) && zip -qr $(abspath $(BUILD_DIR))/medios_verificacion.zip medios_verificacion -x '*/__pycache__/*'

# La guía de usuario del paquete de Windows (viewer.exe y detect.exe). Sin bibliografía:
# dos pasadas bastan para el índice y las referencias. build_windows.py la mete en el zip.
manual: $(MANUAL_PDF)

$(MANUAL_PDF): $(MANUAL_DIR)/manual.tex $(wildcard $(MANUAL_DIR)/fig/*.png)
	@mkdir -p $(MANUAL_DIR)/build
	cd $(MANUAL_DIR) && pdflatex $(LATEX_FLAGS) -output-directory=build manual.tex
	cd $(MANUAL_DIR) && pdflatex $(LATEX_FLAGS) -output-directory=build manual.tex

# Las capturas del manual, desde el propio visor (plataforma offscreen de Qt).
screenshots:
	uv run python $(MANUAL_DIR)/screenshots.py

# El vídeo de demostración: el visor manejado por un guion fijo, YOLO en CPU, a ffmpeg.
demo:
	uv run python $(MANUAL_DIR)/demo.py --out $(DEMO)

# Borra los auxiliares, no los PDF versionados (research/build/main.pdf y
# docs/manual/build/manual.pdf).
clean:
	@rm -rf $(AUX_DIR)
	@find $(BUILD_DIR) -maxdepth 1 -type f ! -name '$(MAIN).pdf' -delete 2>/dev/null || true
	@find $(MANUAL_DIR)/build -maxdepth 1 -type f ! -name 'manual.pdf' -delete 2>/dev/null || true

help:
	@printf '%s\n' \
	  'make          Compila el documento y resume los avisos' \
	  '              (APA 7 con biber si está instalado; si no, natbib + apalike)' \
	  'make BIB=apalike  Fuerza natbib + apalike aunque haya biber' \
	  'make warnings Vuelve a mostrar los avisos del último log' \
	  'make medios   Regenera los medios de verificación y su zip' \
	  'make manual   Compila la guía de usuario (docs/manual/build/manual.pdf)' \
	  'make screenshots  Regenera las capturas del manual desde el visor' \
	  'make demo     Graba el vídeo de demostración (resources/demo.mp4)' \
	  'make clean    Borra los archivos auxiliares (conserva los PDF)'
