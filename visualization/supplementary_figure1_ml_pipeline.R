#!/usr/bin/env Rscript
# Supplementary Figure 1: Machine Learning Pipeline Details
# Detailed visualization of the ML pipeline used for stage classification

library(tidyverse)
library(patchwork)
library(viridis)
library(scales)

# Set publication theme
source("theme_configs/nature_theme.R")

# Output directory
output_dir <- "."
dir.create(output_dir, showWarnings = FALSE)

# Machine learning pipeline - expanded version with more details
create_ml_pipeline_figure <- function() {
  pipeline_steps <- tibble::tibble(
    stage = factor(
      c("PigGTEx cohorts", "QC & normalization", "Feature selection", "Ordinal model", "Validation & interpretation"),
      levels = c("PigGTEx cohorts", "QC & normalization", "Feature selection", "Ordinal model", "Validation & interpretation")
    ),
    step = seq_along(stage),
    headline = c(
      "PigGTEx cohorts",
      "QC & normalization",
      "Feature selection",
      "Ordinal model",
      "Validation & interpretation"
    ),
    detail = c(
      "2467 transcriptomes\n8 tissues; Stage labels",
      "Outlier removal\nLog2(TPM+1) + centering",
      "Top 20% variance\nMutual information (2k genes)",
      "Elastic net ordinal\nStratified by tissue",
      "70:30 temporal split\n1000 bootstraps; Shapley genes"
    )
  )

  connector_data <- tibble::tibble(
    x = pipeline_steps$step[-length(pipeline_steps$step)] + 0.45,
    xend = pipeline_steps$step[-1] - 0.45
  )

  main_pipeline <- ggplot(pipeline_steps, aes(x = step, y = 1)) +
    geom_rect(aes(xmin = step - 0.48, xmax = step + 0.48, ymin = 0.7, ymax = 1.3),
              fill = "#eef5fb", color = "#2c7fb8", linewidth = 0.8) +
    geom_segment(data = connector_data,
                 aes(x = x, xend = xend, y = 1, yend = 1),
                 inherit.aes = FALSE,
                 color = "#2c7fb8",
                 linewidth = 1,
                 arrow = arrow(type = "closed", length = grid::unit(0.16, "inches"))) +
    geom_text(aes(label = headline), y = 1.22, size = 4.5, fontface = "bold", color = "#1f2933") +
    geom_text(aes(label = detail), y = 0.88, size = 3.5, color = "#1f2933", lineheight = 1.15) +
    annotate("text", x = 1, y = 0.5, label = "Seed = 42", color = "#475569", size = 3.5, fontface = "italic") +
    coord_cartesian(xlim = c(0.3, 5.7), ylim = c(0.4, 1.5), expand = FALSE) +
    theme_void()

  # Additional details panel showing specific parameters
  details_data <- tibble::tibble(
    category = c("Data Processing", "Data Processing", "Data Processing",
                 "Feature Engineering", "Feature Engineering", "Feature Engineering",
                 "Model Training", "Model Training", "Model Training",
                 "Validation", "Validation", "Validation"),
    parameter = c("Sample filtering", "Expression normalization", "Batch correction",
                  "Variance filter", "Mutual information", "Gene selection",
                  "Ordinal regression", "Elastic net penalty", "Cross-validation",
                  "Bootstrap iterations", "Temporal split", "Performance metrics"),
    value = c("Z-score < 3", "Log2(TPM+1)", "None",
              "Top 20%", "MI threshold 0.1", "2000 genes",
              "Cumulative logit", "α=0.5, λ=CV", "5-fold stratified",
              "1000 iterations", "70:30 by date", "AUC, accuracy, κ")
  )

  details_data$category <- factor(details_data$category,
                                  levels = c("Data Processing", "Feature Engineering",
                                            "Model Training", "Validation"))

  details_panel <- ggplot(details_data, aes(x = parameter, y = value)) +
    geom_tile(aes(fill = category), color = "white", size = 0.5) +
    geom_text(aes(label = value), size = 3, color = "white", fontface = "bold") +
    facet_wrap(~ category, scales = "free", ncol = 2) +
    scale_fill_manual(values = c("Data Processing" = "#5f9ed1",
                                 "Feature Engineering" = "#8bbde5",
                                 "Model Training" = "#b9d6eb",
                                 "Validation" = "#dae9f6")) +
    labs(title = "Pipeline Parameters and Settings",
         x = NULL, y = NULL) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 9, color = "#1f2933"),
          axis.text.y = element_blank(),
          axis.ticks = element_blank(),
          legend.position = "none",
          strip.text = element_text(size = 10, face = "bold", color = "#1f2933"),
          plot.title = element_text(size = 12, face = "bold", hjust = 0.5, color = "#1f2933"),
          panel.grid = element_blank(),
          strip.background = element_rect(fill = "#f8fafc", color = "#cbd5e1"))

  # Combine panels (performance panel removed)
  final_figure <- main_pipeline / details_panel +
    plot_layout(heights = c(1, 1.5))

  final_figure
}

# Generate and save the figure
supp_figure <- create_ml_pipeline_figure()

ggsave(file.path(output_dir, "supplementary_figure1_ml_pipeline.pdf"),
       supp_figure, width = 14, height = 10, dpi = 300)
ggsave(file.path(output_dir, "supplementary_figure1_ml_pipeline.png"),
       supp_figure, width = 14, height = 10, dpi = 300)

cat("Supplementary Figure 1 saved to:", output_dir, "\n")