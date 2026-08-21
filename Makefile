MAIN := main
RESEARCH_DIR := research
BUILD_DIR := $(RESEARCH_DIR)/build
LATEX_FLAGS := -interaction=nonstopmode -halt-on-error -output-directory=build

.PHONY: all clean help

all: $(BUILD_DIR)/$(MAIN).pdf

$(BUILD_DIR)/$(MAIN).pdf: $(RESEARCH_DIR)/$(MAIN).tex $(RESEARCH_DIR)/references.bib
	@mkdir -p $(BUILD_DIR)
	cd $(RESEARCH_DIR) && pdflatex $(LATEX_FLAGS) $(MAIN).tex
	cd $(RESEARCH_DIR) && bibtex build/$(MAIN)
	cd $(RESEARCH_DIR) && pdflatex $(LATEX_FLAGS) $(MAIN).tex
	cd $(RESEARCH_DIR) && pdflatex $(LATEX_FLAGS) $(MAIN).tex

clean:
	@rm -rf $(BUILD_DIR)

help:
	@printf '%s\n' 'make       Compile the document' 'make clean Remove generated files'
