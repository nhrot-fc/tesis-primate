# Makefile for LaTeX compilation
MAIN = main
RESEARCH_DIR = research
BUILD_DIR = $(RESEARCH_DIR)/build

LATEX = pdflatex
BIBTEX = biber
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error -output-directory=build

.PHONY: all clean help

all: $(BUILD_DIR)/$(MAIN).pdf

$(BUILD_DIR)/$(MAIN).pdf: $(RESEARCH_DIR)/$(MAIN).tex $(RESEARCH_DIR)/references.bib $(RESEARCH_DIR)/LTJournalArticle.cls
	@mkdir -p $(BUILD_DIR)
	@echo "Compiling LaTeX document..."
	cd $(RESEARCH_DIR) && $(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "Running BibTeX..."
	cd $(RESEARCH_DIR) && $(BIBTEX) build/$(MAIN)
	@echo "Recompiling to resolve references..."
	cd $(RESEARCH_DIR) && $(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	cd $(RESEARCH_DIR) && $(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "Compilation complete: $(BUILD_DIR)/$(MAIN).pdf"

clean:
	@echo "Cleaning auxiliary files..."
	@rm -rf $(BUILD_DIR)

help:
	@echo "LaTeX Makefile Usage:"
	@echo "  make          - Compile the document with bibliography"
	@echo "  make clean    - Remove auxiliary files"
	@echo "  make help     - Display this help message"
