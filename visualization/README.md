# Visualization Module

This module contains all R-based visualization components for creating publication-quality figures following Nature journal standards.

## Structure

```
visualization/
├── theme_configs/           # Theme and style configurations
│   └── nature_theme.R      # Nature journal formatting
├── nature_figures/          # Nature-formatted figures
│   ├── figure3_cross_tissue.R
│   └── test_outputs/       # Test figure outputs
├── plot_generators/        # Reusable plot functions
└── test_viz_pipeline.R    # Test visualization pipeline
```

## Quick Start

### Test the visualization pipeline
```bash
cd visualization
Rscript test_viz_pipeline.R
```

### Generate Figure 3
```bash
cd visualization/nature_figures
Rscript figure3_cross_tissue.R
```

## Nature Journal Standards

All figures follow Nature's formatting requirements:
- **Font**: Sans-serif (Arial/Helvetica), 8pt minimum
- **Dimensions**: 89mm (single column), 183mm (double column)
- **Format**: PDF or EPS, 300 DPI minimum
- **Colors**: RGB color mode, avoid red-green combinations
- **Style**: No titles, subplot labels (a, b, c), minimal design

## Components

### Theme Configuration
`nature_theme.R` provides:
- `nature_theme()`: Base Nature-compliant ggplot2 theme
- `add_subplot_label()`: Add subplot labels (a, b, c)
- `bar_to_dot_plot()`: Convert bar charts to dot plots
- `nature_colors()`: Accessible color palette
- `save_nature_figure()`: Export with correct specifications

### Plot Types
- **Dot plots**: Preferred over bar charts
- **Box plots**: With individual points overlay
- **Line plots**: For trends and comparisons
- **Heatmaps**: For matrix visualizations

## Usage Examples

### Apply Nature theme
```r
source("theme_configs/nature_theme.R")

p <- ggplot(data, aes(x, y)) +
  geom_point() +
  nature_theme()
```

### Add subplot label
```r
p <- add_subplot_label(p, "a")
```

### Save figure
```r
save_nature_figure(p, "figure_name", width = 89, height = 89)
```

## Dependencies
- ggplot2
- jsonlite
- ggpubr (optional, for multi-panel layouts)

## Output
All figures are saved to `nature_figures/` in PDF format at 300 DPI.