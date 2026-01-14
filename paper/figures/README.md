# Paper Figures

This directory contains all code and outputs for manuscript figures following **Nature publication standards**.

## Structure

```
paper/figures/
├── config/
│   └── nature_style.yaml      # Style configuration (colors, fonts, dimensions)
├── R/
│   ├── _theme.R               # Shared Nature theme & color palettes
│   ├── _utils.R               # Shared data loading utilities
│   ├── fig1_study_design.R    # Figure 1: Study framework
│   ├── fig2_model_performance.R # Figure 2: Classification performance
│   ├── fig3_functional_enrichment.R # Figure 3: Biological validation
│   └── fig4_cross_species.R   # Figure 4: Cross-species comparison
├── python/
│   └── fig4_data_prep.py      # Data preparation for Figure 4
├── output/
│   ├── pdf/                   # Publication-ready PDFs (vector)
│   ├── png/                   # Preview PNGs (300 DPI)
│   └── stats/                 # Summary statistics
├── Makefile                   # Build automation
└── README.md                  # This file
```

## Figures

| Figure | Description | Panels |
|--------|-------------|--------|
| **Fig 1** | Study Design & Methodology | A: Timeline, B: Sample heatmap, C: Scheme bar, D: ML workflow |
| **Fig 2** | Model Performance | A: Metrics heatmap, B: Confusion matrices, C: Age correlation |
| **Fig 3** | Functional Enrichment | A: GO term matrix, B: Pathway network |
| **Fig 4** | Cross-Species Comparison | A: Correlation scatter, B: Importance bars, C: Fold change pairs |

## Quick Start

### Generate All Figures

```bash
cd paper/figures
make all
```

### Generate Individual Figures

```bash
make fig1    # Study design
make fig2    # Performance
make fig3    # Enrichment
make fig4    # Cross-species (runs Python data prep first)
```

### Clean Outputs

```bash
make clean      # Remove PDFs/PNGs
make clean-all  # Remove all outputs including data
```

## Requirements

### R Packages

```r
install.packages(c(
  "tidyverse",
  "patchwork",
  "viridis",
  "ggrepel",
  "ggraph",
  "igraph",
  "gprofiler2",
  "readxl",
  "jsonlite",
  "yaml",
  "reshape2",
  "data.table",
  "scales"
))
```

### Python Packages

```bash
pip install pandas numpy scipy requests
```

## Nature Style Guidelines

All figures follow Nature publication standards:

| Element | Specification |
|---------|---------------|
| **Font** | Arial, sans-serif |
| **Font sizes** | Axis: 6-7pt, Title: 8pt, Panel label: 8pt bold |
| **Figure width** | 183mm (double column) |
| **Resolution** | 300 DPI |
| **Colors** | Colorblind-safe palette |
| **Background** | White (#FFFFFF) |
| **Gridlines** | None |
| **Panel labels** | Bold uppercase (A, B, C) |

## Customization

### Modify Colors

Edit `config/nature_style.yaml`:

```yaml
colors:
  tissues:
    Muscle: "#E64B35"
    Brain: "#4DBBD5"
    # ...
```

### Modify Theme

Edit `R/_theme.R`:

```r
nature_theme <- function(base_size = 7, base_family = "Arial") {
  # ...
}
```

### Add New Figure

1. Create `R/figN_description.R`
2. Source shared modules at top:
   ```r
   source(file.path(script_dir, "_theme.R"))
   source(file.path(script_dir, "_utils.R"))
   ```
3. Use `nature_theme()` for all plots
4. Use `save_figure()` to export
5. Add target to `Makefile`

## Output Files

After running `make all`:

```
output/
├── pdf/
│   ├── fig1_study_design.pdf
│   ├── fig2_model_performance.pdf
│   ├── fig3_functional_enrichment.pdf
│   └── fig4_cross_species.pdf
├── png/
│   ├── fig1_study_design.png
│   ├── fig2_model_performance.png
│   ├── fig3_functional_enrichment.png
│   └── fig4_cross_species.png
└── stats/
    ├── fig1_summary.txt
    ├── fig2_summary.txt
    ├── fig3_summary.txt
    ├── fig4_summary.txt
    ├── fig4_orthology_mapping.csv
    └── fig4_expression_stats.csv
```

## Troubleshooting

### Missing fonts

On macOS:
```bash
brew install --cask font-arial
```

On Linux:
```bash
sudo apt-get install ttf-mscorefonts-installer
```

### gprofiler2 API errors

The enrichment analysis requires internet connection. If g:Profiler is unavailable, Panel B of Figure 4 will show a placeholder.

### Cairo PDF issues

If `cairo_pdf` fails, the scripts will fall back to standard `pdf()` device.

---

*Generated for: Pig Developmental Stage Classification manuscript*
