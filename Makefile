MAIN         := main
RESEARCH_DIR := research
BUILD_DIR    := $(RESEARCH_DIR)/build
PDF          := $(BUILD_DIR)/$(MAIN).pdf
LOG          := $(BUILD_DIR)/$(MAIN).log
FIGURES      := $(wildcard $(RESEARCH_DIR)/figures/*)
LATEX_FLAGS  := -interaction=nonstopmode -halt-on-error -file-line-error \
                -output-directory=build

.PHONY: all clean warnings help

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

clean:
	@rm -rf $(BUILD_DIR)

help:
	@printf '%s\n' \
	  'make          Compila el documento y resume los avisos' \
	  'make warnings Vuelve a mostrar los avisos del último log' \
	  'make clean    Borra los archivos generados'
