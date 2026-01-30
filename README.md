# tesis-primate

A generic LaTeX project setup with TeXLive for writing academic papers.

## Prerequisites

You need to have TeXLive installed on your system:

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install texlive-full
```

### macOS
```bash
brew install --cask mactex
```

### Windows
Download and install TeXLive from: https://www.tug.org/texlive/

## Project Structure

```
.
├── main.tex           # Main LaTeX document
├── references.bib     # Bibliography file (BibTeX format)
├── Makefile          # Build automation
└── README.md         # This file
```

## Building the Document

### Using Make (Recommended)

The project includes a Makefile for easy compilation:

```bash
# Compile the document with bibliography
make

# Quick compile without bibliography
make quick

# Clean auxiliary files
make clean

# Clean all files including PDF
make cleanall

# View the compiled PDF
make view

# Display help
make help
```

### Manual Compilation

If you prefer to compile manually or don't have Make:

```bash
# Complete compilation with bibliography
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Quick compilation without bibliography
pdflatex main.tex
```

## Editing the Document

1. **Main Content**: Edit `main.tex` to add your content
2. **References**: Add your bibliography entries to `references.bib`
3. **Citations**: Use `\cite{key}` in your text to cite references

## Document Structure

The template includes the following sections:
- Abstract
- Introduction
- Related Work
- Methodology
- Results
- Discussion
- Conclusion
- Acknowledgments (optional)
- References (automatically generated from references.bib)

## Customization

### Changing Document Class
Edit the first line of `main.tex`:
```latex
\documentclass[12pt,a4paper]{article}
```

Common options:
- Paper size: `a4paper`, `letterpaper`
- Font size: `10pt`, `11pt`, `12pt`
- Document class: `article`, `report`, `book`

### Adding Packages
Add packages in the preamble of `main.tex`:
```latex
\usepackage{package-name}
```

## Continuous Integration

The project includes a GitHub Actions workflow that automatically:
- Compiles the PDF on every push
- Uploads the PDF as an artifact
- Makes it available for download

## Common Issues

### Missing Packages
If you get errors about missing packages, install them using your TeX distribution's package manager.

### Bibliography Not Showing
Make sure to:
1. Run `pdflatex` first
2. Run `bibtex`
3. Run `pdflatex` twice more

Or simply use `make` which handles this automatically.

## License

This template is provided as-is for academic and educational purposes.