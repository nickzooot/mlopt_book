#!/bin/bash

# Build script for the ML Optimization Book

echo "Building ML Optimization Book..."

# Create necessary directories if they don't exist
mkdir -p build

# Create empty placeholder chapters if they don't exist
for chapter_dir in chapters/*/; do
    if [ ! -f "${chapter_dir}chapter.tex" ]; then
        echo "Creating empty placeholder for ${chapter_dir}chapter.tex"
        mkdir -p "${chapter_dir}"
        echo "% Placeholder for future content\n\\chapter{$(basename ${chapter_dir})}" > "${chapter_dir}chapter.tex"
    fi
done

# Run pdflatex and bibtex directly in the main directory for simplicity
echo "Running first pdflatex pass..."
pdflatex -interaction=nonstopmode main.tex || true
echo "Running bibtex..."
bibtex main || true
echo "Running second pdflatex pass..."
pdflatex -interaction=nonstopmode main.tex || true
echo "Running final pdflatex pass..."
pdflatex -interaction=nonstopmode main.tex || true

# Check if PDF was generated
if [ -f "main.pdf" ]; then
    # Copy final PDF to a nicely named file
    cp main.pdf "ML_Optimization_Book.pdf"
    echo "Build complete. Output is in ML_Optimization_Book.pdf"
else
    echo "Build failed. No PDF was generated."
fi 