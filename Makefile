# Makefile for LaTeX document compilation

# Main document name (without .tex extension)
MAIN = main
BUILD_DIR = build

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Compilation flags
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD_DIR)

.PHONY: all clean help

# Default target: compile the document
all: $(BUILD_DIR)/$(MAIN).pdf

# Compile the PDF (with bibliography)
$(BUILD_DIR)/$(MAIN).pdf: $(MAIN).tex references.bib
	@mkdir -p $(BUILD_DIR)
	@echo "Compiling LaTeX document..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "Running BibTeX..."
	$(BIBTEX) $(BUILD_DIR)/$(MAIN)
	@echo "Recompiling to resolve references..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "Compilation complete: $(BUILD_DIR)/$(MAIN).pdf"

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	@rm -rf $(BUILD_DIR)

# Display help
help:
	@echo "LaTeX Makefile Usage:"
	@echo "  make          - Compile the document with bibliography"
	@echo "  make clean    - Remove auxiliary files"
	@echo "  make help     - Display this help message"
