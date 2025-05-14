# Contributor Guide for ML Optimization Book

This guide provides detailed instructions for contributing to the "Machine Learning Optimization: Theory and Practice" book.

## Chapter Assignments

Each chapter will be written by a different contributor. Please coordinate with the book editors to claim a chapter.

## Setting Up Your Environment

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd mlopt_book
   ```

2. Create a branch for your chapter:
   ```bash
   git checkout -b chapter-XX-name
   ```


## Chapter Development

### Chapter Structure

Each chapter should follow this structure:

1. **Abstract/Overview**: Brief summary of the chapter content (1 paragraph)
2. **Introduction**:
   - Background and motivation
   - Connection to other chapters
   - Outline of the chapter
3. **Main Content**:
   - Theoretical foundations
   - Algorithms and methods
   - Mathematical proofs (where appropriate)
   - Complexity analysis
   - Convergence properties
   - Illustrative examples
   - Code snippets (if relevant)
   - Visualizations
4. **Conclusion**:
   - Summary of key points

### Content Guidelines

- **Mathematical Notation**: Use consistent notation across chapters. See the notation guide below.
- **Algorithms**: Present algorithms in pseudocode using the algorithm environment.
- **Theorems and Proofs**: Use the theorem, lemma, and proof environments.
- **Figures**: Include clear, high-quality figures that illustrate concepts.
- **References**: Cite primary sources and influential papers.

### Figure Guidelines

For figures in your chapter:
- Place all figures in your chapter's figures directory: `chapters/XX_chapter_name/figures/`
- Reference these figures in your LaTeX code using relative paths: `\includegraphics{figures/figure_name.pdf}`
- Use vector formats (PDF, EPS) whenever possible
- Provide source files (TikZ code, Python scripts) for generated figures
- Use consistent styling across all figures in your chapter

### Notation Guide

To maintain consistency across chapters, please follow these notation conventions:

- Vectors: Bold lowercase (e.g., $\mathbf{x}$)
- Matrices: Bold uppercase (e.g., $\mathbf{A}$)
- Sets: Calligraphic font (e.g., $\mathcal{X}$)
- Scalars: Regular lowercase (e.g., $\alpha$)
- Functions: Regular lowercase (e.g., $f(x)$)
- Gradients: $\nabla f(\mathbf{x})$
- Hessians: $\nabla^2 f(\mathbf{x})$

## Technical Details

## Submission Process

1. Complete your chapter draft.
2. Build the document locally to ensure it compiles correctly.
3. Push your branch and create a pull request.
4. Address reviewer feedback.
5. Once approved, your chapter will be merged into the main branch.

## Resources

- [Numerical Optimization by Nocedal and Wright](https://link-to-book)
- [Convex Optimization by Boyd and Vandenberghe](https://link-to-book)
- [LaTeX Mathematical Symbols Cheat Sheet](https://link-to-resource)
- [TikZ Examples for Creating Figures](https://link-to-resource)

## Questions and Support

If you have any questions or need assistance, please contact the book editors at [email] or open an issue in the repository. 