# Paper Materials

LaTeX manuscript for "A Calibrated Transcriptomic Atlas of Porcine Development Reveals Conserved Molecular Programs with Humans".

## Structure

```
paper/
├── paper.tex              # Main manuscript (Nature format)
├── supplementary.tex      # Supplementary materials
├── paper.bbl              # Compiled bibliography
│
├── figures/               # Figure files
│   ├── output/pdf/       # Generated PDFs (main figures)
│   ├── R/                # R scripts for figure generation
│   └── README.md         # Figure documentation
│
└── pig-age-human/        # Bibliography and reference files
    └── pig-age-human.bib # BibTeX references
```

## Compilation

### Main Paper

```bash
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

Or using latexmk:
```bash
latexmk -pdf paper.tex
```

### Supplementary Materials

```bash
pdflatex supplementary.tex
```

### Clean Build Files

```bash
latexmk -c
# or manually:
rm -f *.aux *.log *.out *.fls *.fdb_latexmk *.synctex.gz *.bbl *.blg
```

## Main Manuscript Contents

1. **Abstract** - 150 words summarizing the calibrated transcriptomic atlas
2. **Introduction** - Motivation for molecular staging in pig research
3. **Results**
   - Comprehensive transcriptomic atlas (1,924 samples, 5 tissues)
   - ML framework for stage inference (LightGBM ordinal classification)
   - Functional landscape of development
   - Cross-species conservation with human muscle
4. **Discussion** - Implications and limitations
5. **Methods** - Technical details
6. **Figures** - 4 main figures

## Supplementary Contents

- **Supplementary Figure S1**: Cross-species sensitivity analysis
- **Supplementary Figure S2**: Tissue specificity and feature stability
- **Supplementary Figure S3**: Extended classification performance
- **Supplementary Table S1**: Train vs test performance
- **Supplementary Table S2**: Per-class performance metrics
- **Supplementary Table S3**: Functional annotations of key markers

## Requirements

- LaTeX distribution (TeX Live or MiKTeX)
- Packages: geometry, helvet, graphicx, booktabs, natbib, hyperref, etc.

## Formatting Notes

- **Target journal**: Nature (adaptable to other journals)
- **Font**: Helvetica (sans-serif)
- **Line numbering**: Enabled for peer review
- **Citation style**: Superscript numerical (Nature style)