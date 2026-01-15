# ==============================================================================
# Shared Nature Publication Theme
# Paper: Pig Developmental Stage Classification
# ==============================================================================
# This module provides consistent styling across all figures following
# Nature publication guidelines.
# ==============================================================================

# Required packages
suppressPackageStartupMessages({
  library(ggplot2)
  library(yaml)
  library(scales)
  library(patchwork)
})

# ==============================================================================
# CONFIGURATION LOADING
# ==============================================================================

#' Get the project root directory
#' @return Character string of project root path
get_project_root <- function() {
  # Primary: use environment variable if set
  env_root <- Sys.getenv("PIG_PROJECT_ROOT")
  if (env_root != "" && dir.exists(env_root)) {
    return(env_root)
  }
  
  # Fallback: hardcoded path
  "/Users/tianyuan/Desktop/github_dev/pig-dev-stage"
}

#' Load style configuration from YAML
#' @return List of style parameters
load_style_config <- function() {
  config_path <- file.path(get_project_root(), "paper/figures/config/nature_style.yaml")
  
  if (!file.exists(config_path)) {
    warning("Config file not found, using defaults: ", config_path)
    return(get_default_config())
  }
  
  yaml::read_yaml(config_path)
}

#' Get default configuration (fallback)
get_default_config <- function() {
  list(
    fonts = list(
      family = "Arial",
      sizes = list(
        axis_text = 6, axis_title = 7, plot_title = 8,
        panel_label = 8, legend_text = 6, legend_title = 7
      )
    ),
    dimensions = list(
      double_column = 183, max_height = 247, dpi = 300
    ),
    elements = list(
      axis_line_width = 0.5, border_width = 0.5
    )
  )
}

# Load config on source
CONFIG <- load_style_config()

# ==============================================================================
# COLOR PALETTES
# ==============================================================================

#' Tissue color palette
TISSUE_COLORS <- c(

"Muscle" = "#E64B35",
  "Brain" = "#4DBBD5",
  "Liver" = "#00A087",
  "Blood" = "#3C5488",
  "Lung" = "#F39B7F",
  "Adipose" = "#FFD700",
  "Small intestine" = "#FFA500",
  "Testis" = "#9370DB"
)

#' Developmental stage colors (light to dark)
STAGE_COLORS <- c(
  "Infant_0_20d" = "#f5fbff",
  "Early childhood_21_59d" = "#b3d9f7",
  "Pre_pubertal_60_149d" = "#6bb3ef",
  "Post_pubertal_150_365d" = "#2d8be0",
  "Adult_>365d" = "#1565c0"
)

#' Stage display labels
STAGE_LABELS <- c(
  "Infant_0_20d" = "Infant",
  "Early childhood_21_59d" = "Early",
  "Pre_pubertal_60_149d" = "Pre-pub",
  "Post_pubertal_150_365d" = "Post-pub",
  "Adult_>365d" = "Adult"
)

#' Stage order for factors
STAGE_ORDER <- c(
  "Infant_0_20d",
  "Early childhood_21_59d",
  "Pre_pubertal_60_149d",
  "Post_pubertal_150_365d",
  "Adult_>365d"
)

#' Classification scheme colors
SCHEME_COLORS <- c(
  "4-class" = "#2166AC",
  "3-class" = "#67A9CF",
  "2-class" = "#D1E5F0"
)

#' Colorblind-safe primary palette (Wong)
PRIMARY_COLORS <- c(
  "#0072B2", "#D55E00", "#009E73", "#CC79A7",
  "#F0E442", "#56B4E9", "#E69F00", "#000000"
)

# ==============================================================================
# NATURE THEME FUNCTION
# ==============================================================================

#' Nature-compliant ggplot2 theme
#'
#' Creates a clean, publication-ready theme following Nature guidelines:
#' - Sans-serif font (Arial/Helvetica)
#' - Minimal gridlines (none by default)
#' - White background
#' - Font sizes: 5-8pt range
#' - Line weights: minimum 0.5pt
#'
#' @param base_size Base font size (default: 7)
#' @param base_family Font family (default: "Arial")
#' @param show_grid Show light grid lines (default: FALSE)
#' @return A ggplot2 theme object
#' @export
nature_theme <- function(base_size = 7, base_family = "Arial", show_grid = FALSE) {
  
  # Get sizes from config or use defaults
  sizes <- CONFIG$fonts$sizes
  if (is.null(sizes)) {
    sizes <- list(axis_text = 6, axis_title = 7, plot_title = 8,
                  legend_text = 6, legend_title = 7, strip_text = 7)
  }
  
  # Grid settings
  grid_major <- if (show_grid) {
    element_line(color = "gray92", linewidth = 0.3)
  } else {
    element_blank()
  }
  
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      # Text elements - NO title as panel labels are handled by patchwork
      plot.title = element_blank(),
      plot.subtitle = element_text(
        size = sizes$plot_subtitle %||% 6,
        color = "gray30",
        hjust = 0,
        margin = margin(b = 4)
      ),
      
      # Axis text (5-7pt per Nature)
      axis.text = element_text(
        size = sizes$axis_text %||% 6,
        color = "black"
      ),
      axis.text.x = element_text(
        margin = margin(t = 2)
      ),
      axis.text.y = element_text(
        margin = margin(r = 2)
      ),
      
      # Axis titles (7-8pt per Nature)
      axis.title = element_text(
        size = sizes$axis_title %||% 7,
        color = "black",
        face = "plain"
      ),
      axis.title.x = element_text(
        margin = margin(t = 6)
      ),
      axis.title.y = element_text(
        margin = margin(r = 6)
      ),
      
      # Axis lines and ticks (minimum 0.5pt per Nature)
      axis.line = element_line(
        color = "black",
        linewidth = 0.5
      ),
      axis.ticks = element_line(
        color = "black",
        linewidth = 0.5
      ),
      axis.ticks.length = unit(1.5, "mm"),
      
      # Legend (compact, inside figure)
      legend.text = element_text(
        size = sizes$legend_text %||% 6
      ),
      legend.title = element_text(
        size = sizes$legend_title %||% 7,
        face = "bold"
      ),
      legend.key.size = unit(3.5, "mm"),
      legend.key.height = unit(3.5, "mm"),
      legend.key.width = unit(3.5, "mm"),
      legend.background = element_rect(fill = "white", color = NA),
      legend.key = element_blank(),
      legend.box.background = element_blank(),
      legend.margin = margin(2, 2, 2, 2),
      legend.spacing = unit(1, "mm"),
      
      # Panel
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      panel.grid.major = grid_major,
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      
      # Facets (consistent styling)
      strip.background = element_rect(fill = "gray95", color = NA),
      strip.text = element_text(
        size = sizes$strip_text %||% 7,
        face = "bold",
        color = "black",
        margin = margin(3, 3, 3, 3)
      ),
      
      # Margins (consistent spacing)
      plot.margin = margin(5, 5, 5, 5, "mm")
    )
}

#' Nature theme variant for subpanels
#'
#' Slightly more compact margins for multi-panel figures
#' @param ... Arguments passed to nature_theme
#' @return A ggplot2 theme object
#' @export
nature_theme_panel <- function(...) {
  nature_theme(...) +
    theme(
      plot.margin = margin(3, 3, 3, 3, "mm"),
      legend.position = "none"
    )
}

# ==============================================================================
# SCALE FUNCTIONS
# ==============================================================================

#' Tissue color scale for ggplot2
#' @param ... Additional arguments passed to scale_fill_manual
#' @export
scale_fill_tissue <- function(...) {
  scale_fill_manual(values = TISSUE_COLORS, name = "Tissue", ...)
}

#' Tissue color scale for points/lines
#' @param ... Additional arguments passed to scale_color_manual
#' @export
scale_color_tissue <- function(...) {
  scale_color_manual(values = TISSUE_COLORS, name = "Tissue", ...)
}

#' Stage color scale for ggplot2
#' @param ... Additional arguments passed to scale_fill_manual
#' @export
scale_fill_stage <- function(...) {
  scale_fill_manual(values = STAGE_COLORS, name = "Stage", ...)
}

#' Performance gradient scale (red-yellow-green)
#' @param ... Additional arguments
#' @export
scale_fill_performance <- function(...) {
  scale_fill_gradient2(
    low = "#D73027",
    mid = "#FEE090",
    high = "#1A9850",
    midpoint = 0.5,
    limits = c(0, 1),
    name = "Score",
    ...
  )
}

#' Heatmap gradient scale
#' @param ... Additional arguments
#' @export
scale_fill_heatmap <- function(...) {
  scale_fill_viridis_c(option = "C", name = "Value", ...)
}

# ==============================================================================
# FIGURE DIMENSION HELPERS
# ==============================================================================

#' Get figure dimensions from config
#' @param fig_name Name of figure (e.g., "fig1", "fig2")
#' @return Named list with width and height in mm
get_fig_dimensions <- function(fig_name) {
  dims <- CONFIG$dimensions[[fig_name]]
  if (is.null(dims)) {
    # Default to double column width
    dims <- c(183, 160)
  }
  list(width = dims[1], height = dims[2])
}

#' Nature figure widths (mm)
FIG_WIDTH <- list(
  single = 89,
  onehalf = 120,
  double = 183
)

# ==============================================================================
# SAVE FUNCTION
# ==============================================================================
#' Save figure in Nature-compliant formats
#'
#' Saves figure as both PDF (vector) and PNG (raster) with appropriate
#' settings for Nature publication.
#'
#' @param plot A ggplot2 object
#' @param filename Base filename without extension
#' @param width Width in mm (default: 183 for double column)
#' @param height Height in mm
#' @param dpi Resolution for PNG (default: 300)
#' @param output_dir Output directory (default: paper/figures/output)
#' @export
save_figure <- function(plot, filename, width = 183, height = 160, dpi = 300,
                        output_dir = NULL) {
  
  if (is.null(output_dir)) {
    output_dir <- file.path(get_project_root(), "paper/figures/output")
  }
  
  # Ensure directories exist
  dir.create(file.path(output_dir, "pdf"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(output_dir, "png"), showWarnings = FALSE, recursive = TRUE)
  
  # Save PDF (vector, preferred for publication)
  pdf_path <- file.path(output_dir, "pdf", paste0(filename, ".pdf"))
  ggsave(
    pdf_path, plot,
    width = width, height = height, units = "mm",
    device = cairo_pdf
  )
  message("Saved: ", pdf_path)
  
  # Save PNG (raster, for preview/web)
  png_path <- file.path(output_dir, "png", paste0(filename, ".png"))
  ggsave(
    png_path, plot,
    width = width, height = height, units = "mm",
    dpi = dpi, type = "cairo"
  )
  message("Saved: ", png_path)
  
  invisible(list(pdf = pdf_path, png = png_path))
}

# ==============================================================================
# PANEL LABEL HELPER (Nature Standard)
# ==============================================================================

#' Nature-compliant panel annotation theme
#' 
#' Nature requires lowercase bold panel labels (a, b, c) positioned
#' consistently at top-left, outside the plot area.
#'
#' @return A patchwork plot_annotation object
#' @export
nature_panel_annotation <- function() {
  patchwork::plot_annotation(
    tag_levels = "a",
    theme = theme(
      plot.tag = element_text(
        size = 10,
        face = "bold",
        family = "Arial",
        color = "black",
        hjust = 0,
        vjust = 1
      ),
      plot.tag.position = c(0, 1)
    )
  )
}

#' Add Nature-style panel labels to combined plots
#' 
#' Wraps patchwork plot with proper Nature panel labels
#' @param combined_plot A patchwork plot object
#' @return Plot with panel annotations
#' @export
add_nature_labels <- function(combined_plot) {
  combined_plot + nature_panel_annotation()
}

#' Create individual panel label element
#' 
#' For manually adding labels when patchwork auto-labeling doesn't work
#' @param label Character label (e.g., "a", "b")
#' @return ggplot annotation layer
#' @export
panel_label <- function(label) {
  annotate(
    "text",
    x = -Inf, y = Inf,
    label = label,
    fontface = "bold",
    family = "Arial",
    size = 10 / .pt,  # Convert to ggplot units
    hjust = -0.5,
    vjust = 1.5
  )
}

# ==============================================================================
# NULL COALESCING OPERATOR
# ==============================================================================
`%||%` <- function(x, y) if (is.null(x)) y else x

# ==============================================================================
# PRINT CONFIGURATION INFO
# ==============================================================================
message("Nature theme loaded successfully")
message("  - Font: ", CONFIG$fonts$family %||% "Arial")
message("  - Figure width: ", CONFIG$dimensions$double_column %||% 183, "mm")
message("  - DPI: ", CONFIG$dimensions$dpi %||% 300)
