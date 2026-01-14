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
#' - Sans-serif font (Arial)
#' - Minimal gridlines
#' - White background
#' - Appropriate font sizes
#'
#' @param base_size Base font size (default: 7)
#' @param base_family Font family (default: "Arial")
#' @return A ggplot2 theme object
#' @export
nature_theme <- function(base_size = 7, base_family = "Arial") {
  
  # Get sizes from config or use defaults
  sizes <- CONFIG$fonts$sizes
  if (is.null(sizes)) {
    sizes <- list(axis_text = 6, axis_title = 7, plot_title = 8,
                  legend_text = 6, legend_title = 7, strip_text = 7)
  }
  
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      # Text elements
      plot.title = element_text(
        size = sizes$plot_title %||% 8,
        face = "bold",
        color = "black",
        hjust = 0,
        margin = margin(b = 4)
      ),
      plot.subtitle = element_text(
        size = sizes$plot_subtitle %||% 6,
        color = "gray30",
        hjust = 0,
        margin = margin(b = 4)
      ),
      
      # Axis text
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
      
      # Axis titles
      axis.title = element_text(
        size = sizes$axis_title %||% 7,
        color = "black",
        face = "plain"
      ),
      axis.title.x = element_text(
        margin = margin(t = 4)
      ),
      axis.title.y = element_text(
        margin = margin(r = 4)
      ),
      
      # Axis lines and ticks
      axis.line = element_line(
        color = "black",
        linewidth = CONFIG$elements$axis_line_width %||% 0.5
      ),
      axis.ticks = element_line(
        color = "black",
        linewidth = CONFIG$elements$axis_line_width %||% 0.5
      ),
      axis.ticks.length = unit(1.5, "mm"),
      
      # Legend
      legend.text = element_text(
        size = sizes$legend_text %||% 6
      ),
      legend.title = element_text(
        size = sizes$legend_title %||% 7,
        face = "bold"
      ),
      legend.key.size = unit(3, "mm"),
      legend.background = element_blank(),
      legend.key = element_blank(),
      legend.box.background = element_blank(),
      legend.margin = margin(0, 0, 0, 0),
      
      # Panel
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      
      # Facets
      strip.background = element_blank(),
      strip.text = element_text(
        size = sizes$strip_text %||% 7,
        face = "bold",
        color = "black"
      ),
      
      # Margins
      plot.margin = margin(4, 4, 4, 4, "mm")
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
# PANEL LABEL HELPER
# ==============================================================================

#' Add panel labels (A, B, C) to plots
#' Uses patchwork annotation
#' @param ... Plots to combine
#' @export
add_panel_labels <- function(...) {
  # This is handled by patchwork::plot_annotation(tag_levels = "A")
  # Just a reminder function
  message("Use: plot_annotation(tag_levels = 'A') with patchwork")
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
