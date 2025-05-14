# Machine Learning Optimization Book

This repository contains the source files for the "Machine Learning Optimization: Theory and Practice" book. The book covers various optimization methods used in machine learning, from fundamental algorithms to advanced techniques.

## Repository Structure

```
mlopt_book/
├── main.tex                   # Main LaTeX file that compiles the entire book
├── references.bib             # Bibliography file
├── chapter_template.tex       # Template for new chapters
├── chapters/                  # Individual chapter directories
│   ├── 01_introduction/
│   │   ├── chapter.tex        # Chapter content
│   │   └── figures/           # Chapter-specific figures
│   ├── 02_gradient_descent/
│   │   ├── chapter.tex
│   │   └── figures/
│   └── ...                    # Other chapter directories
├── figures/                   # Global figures directory
└── scripts/                   # Utility scripts (build, format, etc.)
```

## Getting Started

### Prerequisites

- LaTeX distribution (e.g., TeX Live, MiKTeX)
- Editor of your choice
- Git

### Building the Book

To compile the book, run:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Or use the provided build script:

```bash
./scripts/build.sh
```

## Contributing

### How to Contribute a Chapter

1. Identify which chapter you're working on from the list below
2. Create a branch named after your chapter: `git checkout -b chapter-XX-name`
3. Use the `chapter_template.tex` as a starting point
4. Place your chapter content in the appropriate directory (`chapters/XX_name/chapter.tex`)
5. Add any chapter-specific figures to the chapter's `figures/` directory
6. Submit a pull request when your chapter is ready for review

### Chapter Structure Guidelines

Each chapter should follow this general structure:
- Abstract/Overview
- Introduction (background, motivation)
- Main content sections (theory, algorithms, examples)
- References (cite using the standard `\cite{key}` command)

### Style Guidelines

- Use the provided LaTeX commands for theorems, proofs, etc.
- Place figures in the appropriate directory and reference them consistently
- Use consistent notation across chapters
- Follow mathematical writing best practices

## Chapter List

1. Introduction to Optimization
2. Gradient Descent Methods
3. Newton Method
4. Conjugate Gradient Method
5. Hessian-Free Newton and Convex Sets
6. Quasi-Newton Methods
7. Constrained Optimization and KKT Conditions
8. Dual Optimization Problems
9. Linear Programming and Log-Barrier Method
10. Non-Smooth Convex Optimization
11. Proximal Gradient Method
12. Projections and Proximal Operators
13. Stochastic Optimization

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

## Acknowledgments

This book is based on course materials from the Machine Learning Optimization course made by Dmitry Kropotov in Constructor University Bremen. 