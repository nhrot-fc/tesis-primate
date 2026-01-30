# Makefile for LaTeX document compilation

# Main document name (without .tex extension)
MAIN = main

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Compilation flags
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

.PHONY: all clean cleanall view help quick

# Default target: compile the document
all: $(MAIN).pdf

# Compile the PDF (with bibliography)
$(MAIN).pdf: $(MAIN).tex references.bib
	@echo "Compiling LaTeX document..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "Running BibTeX..."
	$(BIBTEX) $(MAIN)
	@echo "Recompiling to resolve references..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "Compilation complete: $(MAIN).pdf"

# Quick compile (without bibliography)
quick: $(MAIN).tex
	@echo "Quick compilation (no bibliography)..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	@rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.synctex.gz *.fls *.fdb_latexmk

# Clean all generated files including PDF
cleanall: clean
	@echo "Cleaning all generated files..."
	@rm -f $(MAIN).pdf

# View the PDF (requires a PDF viewer)
view: $(MAIN).pdf
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(MAIN).pdf; \
	elif command -v open > /dev/null; then \
		open $(MAIN).pdf; \
	else \
		echo "No PDF viewer found. Please open $(MAIN).pdf manually."; \
	fi

# Display help
help:
	@echo "LaTeX Makefile Usage:"
	@echo "  make          - Compile the document with bibliography"
	@echo "  make quick    - Quick compile without bibliography"
	@echo "  make clean    - Remove auxiliary files"
	@echo "  make cleanall - Remove all generated files including PDF"
	@echo "  make view     - Open the compiled PDF"
	@echo "  make help     - Display this help message"
