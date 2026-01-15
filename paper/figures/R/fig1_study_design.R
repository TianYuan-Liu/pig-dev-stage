#!/usr/bin/env Rscript
# ==============================================================================
# Figure 1: Study Design and Methodology
# ==============================================================================
# Panels:
#   A - Developmental stage timeline
#   B - Sample distribution heatmap (tissues × stages)
#   C - Classification schemes bar chart
#   D - Machine learning workflow diagram
# ==============================================================================

# ==============================================================================
# 1. SETUP
# ==============================================================================
suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(viridis)
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
  cat("Creating Panel A: Timeline...\n")

  timeline_data <- tibble(
    stage = STAGE_ORDER,
    display = c("Infant", "Early\nchildhood", "Pre-\npubertal", "Post-\npubertal", "Adult"),
    start = c(0, 21, 60, 150, 365),
    end = c(20, 59, 149, 365, 500),
    y = 1
  )

  ggplot(timeline_data) +
    geom_rect(
      aes(xmin = start, xmax = end, ymin = 0.8, ymax = 1.2, fill = stage),
      color = "white", linewidth = 0.5
    ) +
    geom_text(
      aes(x = (start + end) / 2, y = 1.7, label = display),
      size = 2.5, fontface = "bold", color = "black"
    ) +
    geom_text(
      aes(
        x = (start + end) / 2, y = 0.5,
        label = paste0(start, "-", ifelse(end == 500, ">365", end), "d")
      ),
      size = 2, color = "gray40"
    ) +
    scale_x_continuous(
      breaks = c(0, 60, 150, 365, 500),
      labels = c("Birth", "2 mo", "5 mo", "1 yr", ">1 yr"),
      expand = c(0, 0),
      limits = c(0, 500)
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
      axis.text.x = element_text(margin = margin(t = 0)),
      axis.line.x = element_blank(),
      axis.ticks.x = element_blank(),
      plot.margin = margin(2, 0, 1, 0, "mm")
    )
}

# ==============================================================================
# 4. PANEL B: EMPTY PLACEHOLDER
# ==============================================================================
create_panel_b <- function() {
  cat("Creating Panel B: Empty placeholder...\n")

  # Create an empty plot
  ggplot() +
    theme_void() +
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
      aes(label = n, color = n > 300),
      size = 2.5, fontface = "bold"
    ) +
    scale_color_manual(values = c("FALSE" = "white", "TRUE" = "black"), guide = "none") +
    scale_fill_viridis(name = "Samples", option = "C") +
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
  cat("Creating Panel D: Classification schemes...\n")

  classification_data <- map_dfr(names(results_list), function(tissue) {
    res <- results_list[[tissue]]
    tibble(
      Tissue = res$tissue,
      Scheme = res$scheme,
      n_samples = res$n_samples
    )
  }) %>%
    filter(!is.na(Tissue))

  classification_data$Scheme <- factor(
    classification_data$Scheme,
    levels = c("4-class", "3-class", "2-class")
  )

  ggplot(classification_data, aes(x = reorder(Tissue, -n_samples), y = n_samples, fill = Scheme)) +
    geom_col(width = 0.7) +
    geom_text(aes(label = n_samples), vjust = -0.5, size = 2.5, fontface = "bold") +
    scale_fill_manual(values = SCHEME_COLORS, name = "Scheme") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(x = "Tissue", y = "Samples") +
    nature_theme() +
    theme(
      legend.position = c(0.85, 0.85),
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

# Layout: A (timeline), B (SVG workflow), C (heatmap), D (schemes)
# C and D should be on the same line
# Give B more space so text is legible
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
save_figure(fig1, "fig1_study_design", width = 183, height = 160)

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
