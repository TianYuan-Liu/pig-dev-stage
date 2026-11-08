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

#' Locate the directory containing this theme module
#'
#' Resolves the directory of `_theme.R` whether the figure scripts are run via
#' `Rscript --file=...`, `source()`-d, or sourced from a sibling utility. Returns
#' NA_character_ if the location cannot be determined.
#' @return Character string of the directory, or NA_character_
get_theme_dir <- function() {
  # Case 1: a parent frame is an active source() call. source() evaluates with
  # an `ofile` variable bound in its frame holding the path being sourced. This
  # works regardless of getOption("keep.source").
  for (fr in rev(sys.frames())) {
    of <- tryCatch(get("ofile", envir = fr, inherits = FALSE),
                   error = function(e) NULL)
    if (!is.null(of) && is.character(of) && length(of) == 1L && nzchar(of)) {
      d <- dirname(normalizePath(of, mustWork = FALSE))
      if (dir.exists(file.path(d)) && file.exists(file.path(d, "_theme.R"))) {
        return(d)
      }
    }
  }
  # Case 2: keep.source on -> this function carries a srcref pointing at _theme.R
  this <- tryCatch(sys.function(0L), error = function(e) NULL)
  sf <- tryCatch(attr(attr(this, "srcref"), "srcfile"), error = function(e) NULL)
  if (!is.null(sf) && !is.null(sf$filename) && nzchar(sf$filename)) {
    d <- dirname(normalizePath(sf$filename, mustWork = FALSE))
    if (dir.exists(d)) return(d)
  }
  # Case 3: Rscript --file=... (the entry figure script lives alongside this
  # module in paper/figures/R/)
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    d <- dirname(normalizePath(sub("--file=", "", file_arg[1]), mustWork = FALSE))
    if (dir.exists(d)) return(d)
  }
  NA_character_
}

# Capture the module directory at source time. While `_theme.R` is being
# source()-d, the source() frame (with its `ofile`) is still live, so this
# resolves reliably even under getOption("keep.source") == FALSE (the Rscript
# default) and even when get_project_root() is later called after source()
# returns. Stored in a package-private environment to avoid clobbering globals.
.fig_env <- new.env(parent = emptyenv())
.fig_env$theme_dir <- get_theme_dir()

#' Get the project root directory
#'
#' Resolution order:
#'   1. PIG_PROJECT_ROOT environment variable (if set and exists)
#'   2. Auto-detect by walking up from this module's own location until a
#'      directory containing `paper/figures/R/` is found. This makes the figure
#'      scripts read data from the repository they actually live in (e.g. a git
#'      worktree) rather than a stale sibling clone.
#'   3. Hardcoded fallback path (last resort, for interactive sessions where the
#'      script location cannot be determined).
#' @return Character string of project root path
get_project_root <- function() {
  # Primary: use environment variable if set
  env_root <- Sys.getenv("PIG_PROJECT_ROOT")
  if (env_root != "" && dir.exists(env_root)) {
    return(env_root)
  }

  # Secondary: auto-detect from this module's location (worktree-safe). Prefer
  # the directory captured at source time, then fall back to live detection.
  theme_dir <- if (!is.null(.fig_env$theme_dir) && !is.na(.fig_env$theme_dir)) {
    .fig_env$theme_dir
  } else {
    get_theme_dir()
  }
  if (!is.na(theme_dir)) {
    d <- theme_dir
    # Walk up at most 6 levels looking for the project layout marker
    for (i in seq_len(6)) {
      if (dir.exists(file.path(d, "paper", "figures", "R"))) {
        return(d)
      }
      parent <- dirname(d)
      if (identical(parent, d)) break
      d <- parent
    }
    # Common case: this module lives at <root>/paper/figures/R/_theme.R
    candidate <- dirname(dirname(dirname(theme_dir)))
    if (dir.exists(file.path(candidate, "paper", "figures", "R"))) {
      return(candidate)
    }
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

#' Developmental stage colors (light to dark), 6-stage scheme.
#' Boundaries [3,28,90,165,540] d: Neonatal/Suckling/Juvenile/Pre-pubertal/
#' Post-pubertal/Adult. House blue family (matches SEQ_RAMP endpoints).
STAGE_COLORS <- c(
  "Neonatal" = "#f5fbff",
  "Suckling" = "#cfe3f5",
  "Juvenile" = "#9ecae1",
  "Pre-pubertal" = "#6bb3ef",
  "Post-pubertal" = "#3182bd",
  "Adult" = "#08519c"
)

#' Stage display labels (short forms for compact axes)
STAGE_LABELS <- c(
  "Neonatal" = "Neonatal",
  "Suckling" = "Suckling",
  "Juvenile" = "Juvenile",
  "Pre-pubertal" = "Pre-pub",
  "Post-pubertal" = "Post-pub",
  "Adult" = "Adult"
)

#' Stage order for factors (developmental order)
STAGE_ORDER <- c(
  "Neonatal",
  "Suckling",
  "Juvenile",
  "Pre-pubertal",
  "Post-pubertal",
  "Adult"
)

#' Classification scheme colors (sequential blue; shares the SEQ_RAMP family so
#' the sample heatmap and the scheme bars read as one hue across Fig 1).
SCHEME_COLORS <- c(
  "6-class" = "#08306b",
  "5-class" = "#2166AC",
  "4-class" = "#4292C6",
  "3-class" = "#67A9CF",
  "2-class" = "#D1E5F0"
)

#' Colorblind-safe primary palette (Wong)
PRIMARY_COLORS <- c(
  "#0072B2", "#D55E00", "#009E73", "#CC79A7",
  "#F0E442", "#56B4E9", "#E69F00", "#000000"
)

# ==============================================================================
# COLOUR CONTRACT  (single source of truth for ALL figures)
# ------------------------------------------------------------------------------
# Every figure script sources _theme.R and uses ONLY the constants / scale_*()
# helpers below. No figure script may hardcode hex codes or call viridis /
# matplotlib palettes. Design rule (Nature): one figure = one neutral family +
# one signal family + one accent family; saturated red/green is reserved for
# DIRECTIONAL cues (up vs down, gain vs loss), never for arbitrary categories.
#
#   categorical identity ........ TISSUE_COLORS        scale_*_tissue()
#   ordered stage ............... STAGE_COLORS         scale_fill_stage()
#   sequential magnitude/counts . SEQ_RAMP             scale_fill_house_seq()
#   signed / z-scored values .... DIVERGING_RAMP       scale_fill_house_div()
#   developmental direction ..... DIRECTION_COLORS     scale_*_direction()
#   transfer substrate (Fig 4) .. SUBSTRATE_COLORS
#   reference / status colours .. CHANCE_COLOR, SIG_COLOR, NULL_COLOR,
#                                 NA_FILL, NEUTRAL_GREY
#   many-level batch ............ categorical_palette(n)   (legend suppressed)
# ==============================================================================

#' Sequential single-hue ramp (light -> house blue). For counts / magnitudes.
#' Endpoints match the STAGE_COLORS blue family so heatmaps stay in one hue.
SEQ_RAMP <- c(low = "#f5fbff", high = "#08306b")

#' Diverging ramp (house blue - white - house red). For signed / z-scored data.
DIVERGING_RAMP <- c(low = "#2166AC", mid = "#F7F7F7", high = "#B2182B")

#' Developmental direction (sign of effect): accelerator rises toward later
#' stages, decelerator declines. Names cover the +/- encodings used in the data.
DIRECTION_COLORS <- c(
  "+" = "#D55E00", "Accelerator" = "#D55E00", "up" = "#D55E00",
  "-" = "#0072B2", "Decelerator" = "#0072B2", "down" = "#0072B2"
)

#' Transfer substrate identity for the cross-species figure (Fig 4):
#' design-matched human dGTEx (warm) vs the Cardoso 7-species atlas (cool).
SUBSTRATE_COLORS <- c("dGTEx" = "#E6863C", "Cardoso" = "#2C6FB0")

#' Reference / status colours shared across figures.
CHANCE_COLOR <- "#B2182B"   # dashed chance / null reference lines
SIG_COLOR    <- "#08519C"   # significant / observed statistic
NULL_COLOR   <- "#9E9E9E"   # permutation-null band / non-significant
NA_FILL      <- "#E8E8E8"   # missing / not-applicable heatmap cells
NEUTRAL_GREY <- "#6E6E6E"   # neutral annotation / unannotated loci

#' Qualitative palette for many-level categoricals (e.g. up to ~56 BioProjects).
#' Returns n visually distinct colours; intended for fills where the legend is
#' suppressed and only group separation matters.
#' @param n Number of colours required.
#' @export
categorical_palette <- function(n) {
  if (requireNamespace("pals", quietly = TRUE)) {
    cols <- pals::glasbey(max(n, 1))
  } else {
    # Fallback: evenly-spaced hues (distinct for many levels; ggplot default).
    cols <- scales::hue_pal(l = 55, c = 90)(max(n, 1))
  }
  unname(cols[seq_len(n)])
}

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

#' Performance / score gradient scale (red-yellow-green; bad -> good).
#' The single source for 0-1 performance heatmaps (precision/recall/F1/BA).
#' @param name Legend title (default "Score")
#' @param midpoint Value mapped to yellow (default 0.5)
#' @param limits Fill limits (default c(0, 1))
#' @param ... Additional arguments passed to scale_fill_gradient2
#' @export
scale_fill_performance <- function(name = "Score", midpoint = 0.5,
                                   limits = c(0, 1), ...) {
  scale_fill_gradient2(
    low = "#D73027", mid = "#FEE090", high = "#1A9850",
    midpoint = midpoint, limits = limits, name = name, ...
  )
}

#' Developmental-direction colour scale (accelerator vs decelerator).
#' @param ... Additional arguments passed to scale_fill_manual
#' @export
scale_fill_direction <- function(...) {
  scale_fill_manual(values = DIRECTION_COLORS, name = "Direction", ...)
}

#' Developmental-direction colour scale for points/lines.
#' @param ... Additional arguments passed to scale_color_manual
#' @export
scale_color_direction <- function(...) {
  scale_color_manual(values = DIRECTION_COLORS, name = "Direction", ...)
}

#' House sequential fill (counts / magnitudes). Single-hue blue ramp.
#' @param name Legend title (default "Value")
#' @param ... Additional arguments passed to scale_fill_gradient
#' @export
scale_fill_house_seq <- function(name = "Value", ...) {
  scale_fill_gradient(low = unname(SEQ_RAMP["low"]), high = unname(SEQ_RAMP["high"]),
                      name = name, na.value = NA_FILL, ...)
}

#' House diverging fill (signed / z-scored values). Blue-white-red.
#' @param name Legend title (default "Value")
#' @param midpoint Value mapped to white (default 0)
#' @param ... Additional arguments passed to scale_fill_gradient2
#' @export
scale_fill_house_div <- function(name = "Value", midpoint = 0, ...) {
  scale_fill_gradient2(low = unname(DIVERGING_RAMP["low"]), mid = unname(DIVERGING_RAMP["mid"]),
                       high = unname(DIVERGING_RAMP["high"]), midpoint = midpoint,
                       name = name, na.value = NA_FILL, ...)
}

#' Heatmap gradient scale (house sequential ramp; replaces viridis).
#' @param ... Additional arguments
#' @export
scale_fill_heatmap <- function(...) {
  scale_fill_house_seq(...)
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
  
  # Save PNG (raster, for preview/web). Pass type = "cairo" only when the active
  # device honours it; modern ggplot2 defaults to the ragg device, which ignores
  # `type`/`antialias` and would otherwise emit a warning.
  png_path <- file.path(output_dir, "png", paste0(filename, ".png"))
  png_args <- list(
    filename = png_path, plot = plot,
    width = width, height = height, units = "mm", dpi = dpi
  )
  uses_ragg <- requireNamespace("ragg", quietly = TRUE)
  if (!uses_ragg) {
    png_args$type <- "cairo"
  }
  do.call(ggsave, png_args)
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
