# Pig Developmental Stage - Presentation Materials

Cardiff University LaTeX Beamer slides with auto-generated figures for the pig developmental stage classification project.

## Directory Structure

```
slides/
├── R/                          # Figure Generation Scripts
│   ├── slides_theme.R         # Cardiff-styled ggplot2 theme
│   └── generate_figures.R     # Master script for all panels
│
├── Figures/                    # Generated Figure Panels (PNG)
│   ├── fig1_panel_a_sample_distribution.png
│   ├── fig1_panel_b_classification_schemes.png
│   ├── fig2_panel_a_performance_metrics.png
│   ├── fig2_panel_b_confusion_matrices.png
│   ├── fig2_panel_c_age_correlation.png
│   ├── fig3_panel_a_umap.png
│   ├── fig3_panel_b_feature_overlap.png
│   ├── fig4_panel_a_go_enrichment.png
│   └── fig4_panel_b_marker_validation.png
│
├── templates/                  # LaTeX Templates
│   └── master_presentation.tex
├── theme/                      # Beamer Theme
│   └── cardiff_theme.sty
├── assets/                     # Images & Logos
├── build/                      # Compilation Files
└── Makefile                    # Build Automation
```

## Regenerating Figures

To regenerate all figure panels:

```bash
cd R
Rscript generate_figures.R
```

This creates 9 individual PNG panels in `Figures/`, optimized for:
- 16:9 slide aspect ratio
- Large fonts (14pt base) for readability
- Cardiff University color palette
- White backgrounds

## Available Figure Panels

| Panel | Description | Dimensions |
|-------|-------------|------------|
| `fig1_panel_a` | Sample distribution heatmap | 10x7 in |
| `fig1_panel_b` | Classification schemes | 10x6 in |
| `fig2_panel_a` | Performance metrics (BA with CI) | 10x6 in |
| `fig2_panel_b` | Confusion matrices (4 tissues) | 10x8 in |
| `fig2_panel_c` | Age vs stage correlation | 10x8 in |
| `fig3_panel_a` | UMAP clustering | 10x7 in |
| `fig3_panel_b` | Feature overlap analysis | 9x6 in |
| `fig4_panel_a` | GO enrichment bubble plot | 10x7 in |
| `fig4_panel_b` | Marker gene validation | 10x8 in |

## Using Figures in LaTeX

```latex
\begin{frame}{Sample Distribution}
    \includegraphics[width=\textwidth]{Figures/fig1_panel_a_sample_distribution}
\end{frame}
```

## Building Slides

Using Makefile (recommended):
```bash
make
```

Manual compilation:
```bash
export TEXINPUTS=.:./theme/:$TEXINPUTS
pdflatex templates/master_presentation.tex
```

## Customization

- **Colors/Fonts**: Edit `theme/cardiff_theme.sty`
- **Figure styling**: Edit `R/slides_theme.R`
- **Add images**: Place in `assets/`

## Cardiff Color Palette

| Color | Hex | Usage |
|-------|-----|-------|
| CardiffRed | `#D3374A` | Accents, highlights |
| CardiffBlack | `#22211F` | Text, titles |
| CardiffGrey | `#D3D3D2` | Backgrounds |
| CardiffDarkGrey | `#545351` | Secondary text |
