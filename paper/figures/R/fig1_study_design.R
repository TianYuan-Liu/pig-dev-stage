#!/usr/bin/env Rscript
# ==============================================================================
# Figure 1: Study Design and Methodology
# ==============================================================================
# Panels:
#   A - Developmental stage timeline
#   B - Machine learning pipeline flowchart (mermaid SVG)
#   C - Sample distribution heatmap (tissues × stages)
#   D - Classification schemes bar chart
# ==============================================================================

# ==============================================================================
# 1. SETUP
# ==============================================================================
suppressPackageStartupMessages({
  library(tidyverse)
  library(magick)
  library(patchwork)
})

# Load shared modules
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("--file=", "", file_arg))))
  }
  # Fallback for interactive use
  "/Users/tianyuan/Desktop/github_dev/pig-dev-stage/paper/figures/R"
}
script_dir <- get_script_dir()
source(file.path(script_dir, "_theme.R"))
source(file.path(script_dir, "_utils.R"))

cat("==============================================================================\n")
cat("Figure 1: Study Design and Methodology\n")
cat("==============================================================================\n\n")

# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
cat("Loading data...\n")

metadata <- load_metadata()
results_list <- load_all_ml_results()

cat(sprintf("  Metadata: %d samples\n", nrow(metadata)))
cat(sprintf("  ML results: %d tissues\n", length(results_list)))

# ==============================================================================
# 3. PANEL A: DEVELOPMENTAL TIMELINE
# ==============================================================================
create_panel_a <- function() {
  cat("Creating Panel A: Timeline (stage boundaries)...\n")

  # 6-stage developmental staging (boundaries [3, 28, 90, 165, 540] d):
  #   Neonatal       0-3 d
  #   Suckling       4-28 d    (pre-weaning)
  #   Juvenile       29-90 d
  #   Pre-pubertal   91-165 d  (puberty onset)
  #   Post-pubertal  166-540 d
  #   Adult          >540 d
  # Plotted on a sqrt(day) axis so the short perinatal stages (Neonatal/Suckling)
  # remain legible alongside the long post-pubertal/adult window.
  x_max <- 720
  sq <- function(d) sqrt(d)
  timeline_data <- tibble(
    stage = factor(STAGE_ORDER, levels = STAGE_ORDER),
    display = c("Neonatal", "Suckling", "Juvenile", "Pre-\npubertal",
                "Post-\npubertal", "Adult"),
    start = c(0, 4, 29, 91, 166, 541),
    end   = c(3, 28, 90, 165, 540, x_max),
    range_label = c("0–3 d", "4–28 d", "29–90 d", "91–165 d", "166–540 d", ">540 d"),
    y = 1
  ) %>%
    mutate(xs = sq(start), xe = sq(end), xmid = (sq(start) + sq(end)) / 2)

  # Biological transition markers placed on the boundaries (sqrt-day positions)
  markers <- tibble(
    x = sq(c(3, 28, 165)),
    label = c("birth", "weaning", "puberty")
  )

  ggplot(timeline_data) +
    geom_rect(
      aes(xmin = xs, xmax = xe, ymin = 0.8, ymax = 1.2, fill = stage),
      color = "white", linewidth = 0.5
    ) +
    geom_text(
      aes(x = xmid, y = 1.7, label = display),
      size = 2.4, fontface = "bold", color = "black", lineheight = 0.85
    ) +
    geom_text(
      aes(x = xmid, y = 0.5, label = range_label),
      size = 1.8, color = "gray40"
    ) +
    # Biological transition cues at the boundaries (claim-first annotation)
    geom_segment(
      data = markers,
      aes(x = x, xend = x, y = 1.22, yend = 2.05),
      inherit.aes = FALSE, linewidth = 0.3, color = "gray30"
    ) +
    geom_text(
      data = markers,
      aes(x = x, y = 2.28, label = label),
      inherit.aes = FALSE, size = 1.9, fontface = "italic", color = "gray30"
    ) +
    scale_x_continuous(
      breaks = sq(c(0, 28, 90, 165, 540, x_max)),
      labels = c("Birth", "28 d", "90 d", "165 d", "540 d", ">540 d"),
      expand = c(0, 0),
      limits = c(0, sq(x_max))
    ) +
    scale_fill_manual(values = STAGE_COLORS) +
    coord_cartesian(ylim = c(0, 2.5), clip = "off") +
    labs(x = NULL, y = NULL) +
    nature_theme() +
    theme(
      legend.position = "none",
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      axis.line.y = element_blank(),
      axis.text.x = element_text(margin = margin(t = 0), size = 5),
      axis.line.x = element_blank(),
      axis.ticks.x = element_blank(),
      plot.margin = margin(2, 0, 1, 0, "mm")
    )
}

# ==============================================================================
# 4. PANEL B: ML PIPELINE FLOWCHART (mermaid PNG)
# ==============================================================================
create_panel_b <- function() {
  cat("Creating Panel B: ML pipeline flowchart...\n")

  png_path <- file.path(dirname(script_dir), "output", "png", "figure1b.png")

  if (!file.exists(png_path)) {
    warning("Mermaid PNG not found at: ", png_path, "\nReturning empty panel.")
    return(ggplot() + theme_void())
  }

  img <- magick::image_read(png_path)
  img <- magick::image_trim(img)
  img_raster <- as.raster(img)

  img_h <- nrow(img_raster)
  img_w <- ncol(img_raster)

  ggplot() +
    annotation_raster(img_raster, xmin = 0, xmax = img_w, ymin = 0, ymax = img_h) +
    coord_fixed(ratio = 1, xlim = c(0, img_w), ylim = c(0, img_h), expand = FALSE) +
    theme_void(base_size = 7) +
    theme(plot.margin = margin(2, 2, 2, 2, "mm"))
}

# ==============================================================================
# 5. PANEL C: SAMPLE DISTRIBUTION HEATMAP (formerly Panel B)
# ==============================================================================
create_panel_c <- function(metadata, results_list) {
  cat("Creating Panel C: Heatmap...\n")

  tissues_to_show <- names(results_list)

  stage_counts <- metadata %>%
    filter(Tissue %in% tissues_to_show) %>%
    group_by(Tissue, Stage) %>%
    summarise(n = n(), .groups = "drop") %>%
    mutate(
      Stage = factor(Stage, levels = STAGE_ORDER),
      Tissue = factor(Tissue, levels = tissues_to_show)
    ) %>%
    complete(Tissue, Stage, fill = list(n = 0))

  # Order tissues by total count
  tissue_order <- stage_counts %>%
    group_by(Tissue) %>%
    summarise(total = sum(n)) %>%
    arrange(desc(total)) %>%
    pull(Tissue)

  stage_counts$Tissue <- factor(stage_counts$Tissue, levels = tissue_order)

  ggplot(stage_counts, aes(x = Stage, y = Tissue, fill = n)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(
      aes(label = n, color = n > 150),
      size = 2.5, fontface = "bold"
    ) +
    scale_color_manual(values = c("FALSE" = "grey15", "TRUE" = "white"), guide = "none") +
    scale_fill_house_seq(name = "Samples") +
    scale_x_discrete(labels = STAGE_LABELS) +
    labs(x = NULL, y = "Tissue") +
    nature_theme() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "right",
      legend.key.height = unit(8, "mm")
    )
}

# ==============================================================================
# 6. PANEL D: CLASSIFICATION SCHEMES (formerly Panel C)
# ==============================================================================
create_panel_d <- function(results_list) {
  cat("Creating Panel D: Classification schemes (kept stages, 6-stage)...\n")

  # Per-tissue number of retained (kept) 6-stage developmental classes after
  # dropping stages with < 10 samples, from the canonical within-pig 6-stage
  # artifact (no hard-coded scheme). n_samples = modelled (kept-stage) samples.
  wp <- jsonlite::fromJSON(
    project_path("paper/figures/data/within_pig_6stage.json"),
    simplifyVector = FALSE)
  classification_data <- map_dfr(names(results_list), function(tissue) {
    w <- wp[[tissue]]
    if (is.null(w)) return(NULL)
    n_kept <- as.integer(w$n_class)
    tibble(
      Tissue = tissue,
      n_classes = n_kept,
      Scheme = paste0(n_kept, "-class"),
      n_samples = sum(unlist(w$kept6))
    )
  })

  scheme_levels <- c("6-class", "5-class", "4-class", "3-class", "2-class")
  classification_data$Scheme <- factor(classification_data$Scheme, levels = scheme_levels)

  # Scheme palette comes from the central colour contract (SCHEME_COLORS),
  # a sequential blue that matches the panel-c sample heatmap.
  ggplot(classification_data, aes(x = reorder(Tissue, -n_samples), y = n_samples, fill = Scheme)) +
    geom_col(width = 0.7) +
    geom_text(aes(label = n_samples), vjust = -0.5, size = 2.5, fontface = "bold") +
    scale_fill_manual(values = SCHEME_COLORS, name = "Scheme", drop = FALSE) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(x = "Tissue", y = "Samples (modelled)") +
    nature_theme() +
    theme(
      legend.position = c(0.83, 0.78),
      legend.background = element_rect(fill = "white", color = "gray80", linewidth = 0.3)
    )
}


# ==============================================================================
# 7. FIGURE ASSEMBLY
# ==============================================================================
cat("\nAssembling figure...\n")

panel_a <- create_panel_a()
panel_b <- create_panel_b()
panel_c <- create_panel_c(metadata, results_list)
panel_d <- create_panel_d(results_list)

# Layout: A (timeline), B (ML pipeline centered), C (heatmap), D (schemes)
design <- "
  AAAA
  BBBB
  BBBB
  BBBB
  CCDD
  CCDD
  CCDD
"

fig1 <- wrap_plots(
  A = panel_a,
  B = panel_b,
  C = panel_c,
  D = panel_d,
  design = design
) +
  plot_annotation(
    tag_levels = list(c("a", "b", "c", "d")),
    theme = theme(
      plot.background = element_rect(fill = "white", color = NA),
      plot.tag = element_text(size = 8, face = "bold")
    )
  )

# ==============================================================================
# 8. SAVE FIGURE
# ==============================================================================
cat("\nSaving figure...\n")
save_figure(fig1, "fig1_study_design", width = 183, height = 180)

# ==============================================================================
# 9. SUMMARY STATISTICS
# ==============================================================================
cat("\nWriting summary...\n")

metadata_filtered <- metadata %>% filter(Tissue %in% names(results_list))

summary_text <- sprintf(
  "
FIGURE 1: STUDY DESIGN AND METHODOLOGY
======================================
Generated: %s

SAMPLE STATISTICS
-----------------
Total samples (modeled tissues): %d
Tissues with models: %s

Samples per tissue:
%s

Samples per stage:
%s

CLASSIFICATION SCHEMES
----------------------
%s

Figure dimensions: 183mm × 160mm
",
  format(Sys.time(), "%%Y-%%m-%%d %%H:%%M"),
  nrow(metadata_filtered),
  paste(names(results_list), collapse = ", "),
  paste(capture.output(
    metadata_filtered %>% count(Tissue, sort = TRUE) %>% print(n = Inf)
  ), collapse = "\n"),
  paste(capture.output(
    metadata_filtered %>%
      mutate(Stage = factor(Stage, levels = STAGE_ORDER)) %>%
      count(Stage) %>% print(n = Inf)
  ), collapse = "\n"),
  paste(map_chr(names(results_list), function(t) {
    r <- results_list[[t]]
    sprintf("  %s: %s (%d samples)", t, r$scheme, r$n_samples)
  }), collapse = "\n")
)

write_summary(summary_text, "fig1_summary.txt")

cat("\n✓ Figure 1 completed successfully!\n")
