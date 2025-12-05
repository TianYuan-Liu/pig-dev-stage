#!/usr/bin/env Rscript
# Figure 1: Study Framework and Methodology
# Publication-quality figure for porcine developmental stage classifier

library(tidyverse)
library(patchwork)
library(viridis)
library(jsonlite)
library(ggalluvial)
library(scales)

# Set publication theme
source("theme_configs/nature_theme.R")

# Output directory
output_dir <- "."
dir.create(output_dir, showWarnings = FALSE)

# Load data
metadata <- read_csv("../data/full_metadata.csv")
summary_data <- fromJSON("../data/analysis_summary.json")
pipeline_summary <- fromJSON("../machine_learning/model_outputs/pipeline_summary.json")

# Define stage colors
stage_colors <- c(
  "Infant_0_20d" = "#f5fbff",
  "Early childhood_21_59d" = "#dae9f6",
  "Pre_pubertal_60_149d" = "#b9d6eb",
  "Post_pubertal_150_365d" = "#8bbde5",
  "Adult_>365d" = "#5f9ed1"
)

# Panel A: Developmental stage definitions and sample distribution
create_panel_a <- function() {
  # Prepare data for heatmap
  stage_counts <- metadata %>%
    group_by(Tissue, Stage) %>%
    summarise(n = n(), .groups = 'drop') %>%
    mutate(
      Stage = factor(Stage, levels = names(stage_colors)),
      Tissue = factor(Tissue)
    )

  # Calculate tissue totals for ordering
  tissue_totals <- stage_counts %>%
    group_by(Tissue) %>%
    summarise(total = sum(n)) %>%
    arrange(desc(total))

  stage_counts$Tissue <- factor(stage_counts$Tissue, levels = tissue_totals$Tissue)

  # Create timeline
  timeline_data <- tibble::tibble(
    stage = names(stage_colors),
    display = c("Infant", "Early\nchildhood", "Pre-\npubertal", "Post-\npubertal", "Adult"),
    start = c(0, 21, 60, 150, 365),
    end = c(20, 59, 149, 365, 500),
    y = 1
  )

  timeline_plot <- ggplot(timeline_data) +
    geom_rect(aes(xmin = start, xmax = end, ymin = 0.8, ymax = 1.2,
                  fill = stage), color = "white") +
    geom_text(aes(x = (start + end)/2, y = 1.5, label = display),
              size = 3.5, fontface = "bold", color = "#1f2933") +
    geom_text(aes(x = (start + end)/2, y = 0.5,
                  label = paste0(start, "-", ifelse(end == 500, ">365", end), "d")),
              size = 3, color = "#1f2933") +
    scale_x_continuous(breaks = c(0, 60, 150, 365, 500),
                       labels = c("Birth", "2 mo", "5 mo", "1 yr", ">1 yr")) +
    scale_fill_manual(values = stage_colors) +
    labs(x = "Age", y = NULL) +
    theme_minimal() +
    theme(legend.position = "none",
          axis.text.x = element_text(color = "#1f2933"),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          axis.title.x = element_text(color = "#1f2933"),
          panel.grid = element_blank(),
          panel.background = element_rect(fill = "white", color = NA),
          plot.background = element_rect(fill = "white", color = NA)) +
    coord_cartesian(ylim = c(0.3, 1.8))

  # Create heatmap
  heatmap_plot <- ggplot(stage_counts, aes(x = Stage, y = Tissue, fill = n)) +
    geom_tile(color = "white", size = 0.5) +
    geom_text(aes(label = n), size = 3, color = "white", fontface = "bold") +
    scale_fill_viridis(name = "Samples", option = "C") +
    scale_x_discrete(labels = c("Infant\n0-20d", "Early\n21-59d",
                                "Pre-pub\n60-149d", "Post-pub\n150-365d",
                                "Adult\n>365d")) +
    labs(x = "Developmental Stage", y = "Tissue",
         title = "Sample Distribution Across Tissues and Stages") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 9, color = "#1f2933"),
          axis.text.y = element_text(size = 10, color = "#1f2933"),
          axis.title.x = element_text(size = 10, color = "#1f2933"),
          axis.title.y = element_text(size = 10, color = "#1f2933"),
          legend.position = "right",
          legend.text = element_text(color = "#1f2933"),
          legend.title = element_text(color = "#1f2933"),
          plot.title = element_text(size = 12, face = "bold", color = "#1f2933"),
          panel.grid = element_blank(),
          panel.background = element_rect(fill = "white", color = NA),
          plot.background = element_rect(fill = "white", color = NA))

  timeline_plot / heatmap_plot + plot_layout(heights = c(1, 3))
}

# Panel B: Tissue classification summary
create_panel_b <- function() {
  # Define classification schemes
  classification_data <- data.frame(
    Tissue = c("Muscle", "Brain", "Liver", "Blood", "Small intestine",
               "Lung", "Adipose", "Testis"),
    Scheme = c("4-class", "4-class", "4-class", "3-class", "3-class",
               "3-class", "2-class", "2-class"),
    n_samples = c(914, 400, 329, 284, 180, 147, 144, 69),
    min_per_class = c(92, 79, 33, 30, 30, 26, 69, 29)
  )

  classification_data$Scheme <- factor(classification_data$Scheme,
                                       levels = c("4-class", "3-class", "2-class"))

  tissue_plot <- ggplot(classification_data, aes(x = reorder(Tissue, -n_samples),
                                                y = n_samples, fill = Scheme)) +
    geom_col(alpha = 0.85) +
    geom_text(aes(label = n_samples), vjust = -0.5, size = 3.5, color = "#1f2933") +
    geom_hline(yintercept = c(25, 30, 40), linetype = "dashed",
               color = c("#ef4444", "#f59e0b", "#22c55e"), alpha = 0.5) +
    scale_fill_manual(values = c("4-class" = "#8bbde5",
                                 "3-class" = "#5f9ed1",
                                 "2-class" = "#2c7fb8")) +
    labs(x = "Tissue", y = "Total samples",
         title = "Tissue Classification Schemes") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, color = "#1f2933"),
          axis.text.y = element_text(color = "#1f2933"),
          axis.title = element_text(color = "#1f2933"),
          legend.title = element_text(color = "#1f2933"),
          legend.text = element_text(color = "#1f2933"),
          plot.title = element_text(face = "bold", color = "#1f2933"))

  tissue_plot
}

# Combine all panels
panel_a <- create_panel_a()
panel_b <- create_panel_b()

# Create final figure with only two panels
final_figure <- patchwork::wrap_plots(
  panel_a,
  panel_b,
  ncol = 1,
  heights = c(2, 1.5)
) +
  plot_annotation(
    tag_levels = 'A',
    theme = theme()
  )

# Save figure
ggsave(file.path(output_dir, "figure1_methodology.pdf"),
       final_figure, width = 14, height = 11.5, dpi = 300)
ggsave(file.path(output_dir, "figure1_methodology.png"),
       final_figure, width = 14, height = 11.5, dpi = 300)

cat("Figure 1 saved to:", output_dir, "\n")
