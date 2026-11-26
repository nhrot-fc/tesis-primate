MAIN         := main
RESEARCH_DIR := research
BUILD_DIR    := $(RESEARCH_DIR)/build
PDF          := $(BUILD_DIR)/$(MAIN).pdf
# pdflatex borra el PDF cuando falla, y research/build/main.pdf está versionado:
# se compila en aux/ y el PDF se copia sólo si la compilación termina bien.
AUX_DIR      := $(BUILD_DIR)/aux
AUX_ABS      := $(abspath $(AUX_DIR))
LOG          := $(AUX_DIR)/$(MAIN).log
CHAPTER_DIRS := $(wildcard $(RESEARCH_DIR)/capitulos/*/)
SOURCES      := $(RESEARCH_DIR)/$(MAIN).tex $(RESEARCH_DIR)/estilos.tex $(RESEARCH_DIR)/references.bib \
                $(wildcard $(RESEARCH_DIR)/capitulos/*/*.tex) \
                $(wildcard $(RESEARCH_DIR)/capitulos/*/figures/*) \
                $(wildcard $(RESEARCH_DIR)/capitulos/*/figures/*/*) \
                $(wildcard $(RESEARCH_DIR)/figures/*)
LATEX_FLAGS  := -interaction=nonstopmode -halt-on-error -file-line-error
LATEX_RUN    := cd $(RESEARCH_DIR) && pdflatex $(LATEX_FLAGS) -output-directory=$(AUX_ABS) $(MAIN).tex

MANUAL_DIR   := docs/manual
MANUAL_PDF   := $(MANUAL_DIR)/build/manual.pdf

.PHONY: all warnings figuras medios presentacion manual screenshots demo clean help

all: $(PDF)

# Cada \include escribe su .aux en una subcarpeta con el nombre del capítulo, y
# pdflatex no la crea.
$(PDF): $(SOURCES)
	@for d in $(CHAPTER_DIRS); do mkdir -p "$(AUX_DIR)/capitulos/$$(basename "$$d")"; done
	$(LATEX_RUN)
	cd $(RESEARCH_DIR) && biber $(AUX_ABS)/$(MAIN)
	$(LATEX_RUN)
	$(LATEX_RUN)
	@cp $(AUX_DIR)/$(MAIN).pdf $(PDF)
	@$(MAKE) --no-print-directory warnings

# El log está en ISO-8859: sin -a, grep lo trata como binario y no muestra nada.
warnings:
	@test -f $(LOG) || { echo 'No hay log: corre "make" primero.'; exit 1; }
	@echo '--- Cajas desbordadas ---'
	@grep -a -A2 'Overfull\|Underfull' $(LOG) || echo '  ninguna'
	@echo '--- Flotantes más altos que la página ---'
	@grep -a 'Float too large' $(LOG) || echo '  ninguno'
	@echo '--- Citas o referencias sin resolver ---'
	@grep -aoE "Citation [\`'][^']*'" $(LOG) | sort -u | grep . || echo '  ninguna'
	@grep -a 'Reference .* undefined' $(LOG) || true

# Las celdas que necesitan el caché de ventanas o la GPU sólo corren en el servidor:
#   TESIS_SERVIDOR=1 make figuras CAP="04 05"
CAP          ?= 01 02 04 05 06
CUADERNOS    := $(foreach c,$(CAP),$(wildcard $(RESEARCH_DIR)/capitulos/$(c)_*/notebook/figuras_*.ipynb))
figuras:
	@for nb in $(CUADERNOS); do \
	  echo "--- $$nb"; \
	  (cd "$$(dirname "$$nb")" && uv run jupyter nbconvert --to notebook --execute --inplace "$$(basename "$$nb")") || exit 1; \
	done

medios:
	uv run python $(RESEARCH_DIR)/medios_verificacion/generar.py
	@rm -f $(BUILD_DIR)/medios_verificacion.zip
	cd $(RESEARCH_DIR) && zip -qr $(abspath $(BUILD_DIR))/medios_verificacion.zip medios_verificacion -x '*/__pycache__/*'

# La exposición final: toma las cifras de los capítulos, así que va después de `make`.
PRES_DIR     := $(RESEARCH_DIR)/presentacion
PRES_AUX     := $(abspath $(BUILD_DIR)/presentacion)
presentacion:
	uv run python $(PRES_DIR)/datos.py
	cd $(PRES_DIR) && pdflatex $(LATEX_FLAGS) -output-directory=$(PRES_AUX) presentacion.tex
	cd $(PRES_DIR) && pdflatex $(LATEX_FLAGS) -output-directory=$(PRES_AUX) presentacion.tex
	@cp $(PRES_AUX)/presentacion.pdf $(BUILD_DIR)/presentacion.pdf
	@grep -a -c 'Overfull' $(PRES_AUX)/presentacion.log | xargs -I{} echo 'Cajas desbordadas: {}'

# La guía de usuario del paquete de Windows; el visor la abre desde Help.
manual: $(MANUAL_PDF)

$(MANUAL_PDF): $(MANUAL_DIR)/manual.tex $(wildcard $(MANUAL_DIR)/fig/*.png)
	@mkdir -p $(MANUAL_DIR)/build
	cd $(MANUAL_DIR) && pdflatex $(LATEX_FLAGS) -output-directory=build manual.tex
	cd $(MANUAL_DIR) && pdflatex $(LATEX_FLAGS) -output-directory=build manual.tex

screenshots:
	uv run python $(MANUAL_DIR)/screenshots.py

demo:
	uv run python $(MANUAL_DIR)/demo.py --out resources/demo.mp4

clean:
	@rm -rf $(AUX_DIR)
	@find $(MANUAL_DIR)/build -maxdepth 1 -type f ! -name 'manual.pdf' -delete 2>/dev/null || true

help:
	@printf '%s\n' \
	  'make              Compila el documento (research/build/main.pdf) y resume los avisos' \
	  'make warnings     Vuelve a mostrar los avisos del último log' \
	  'make figuras      Ejecuta los cuadernos de figuras (CAP="04 05" elige capítulos)' \
	  'make medios       Regenera los medios de verificación y su zip' \
	  'make presentacion Compila la exposición final (research/build/presentacion.pdf)' \
	  'make manual       Compila la guía de usuario (docs/manual/build/manual.pdf)' \
	  'make screenshots  Regenera las capturas de la guía desde el visor' \
	  'make demo         Graba el vídeo de demostración (resources/demo.mp4)' \
	  'make clean        Borra los auxiliares (conserva los PDF)'
