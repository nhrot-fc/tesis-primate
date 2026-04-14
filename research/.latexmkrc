# latexmk configuration file
# This file configures latexmk for automatic compilation

# Use pdflatex by default
$pdf_mode = 1;

# Use bibtex for bibliography
$bibtex_use = 2;

# Disable continuous preview mode
$pvc = 0;

# Configure pdflatex with flags
$pdflatex = 'pdflatex -halt-on-error -interaction=nonstopmode %O %S';

# Clean up auxiliary files
$clean_ext = "aux bbl blg log out toc synctex.gz fls fdb_latexmk";

# Output directory
$out_dir = 'build';
