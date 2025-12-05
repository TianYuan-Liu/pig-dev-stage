#!/usr/bin/env Rscript
# Figure 5: Biomarker Validation - Conserved Developmental Markers
# Shows that pig biomarkers follow similar developmental trends as in humans

library(tidyverse)
library(patchwork)
library(viridis)
library(ComplexHeatmap)
library(circlize)
library(RColorBrewer)
library(ggrepel)

# Set publication theme
source("theme_configs/nature_theme.R")

# Output directory
output_dir <- "."
dir.create(output_dir, showWarnings = FALSE)

# Load biomarker analysis results
results_dir <- "../results/biomarker_analysis"
stage_stats <- read_csv(file.path(results_dir, "biomarker_stage_statistics.csv"))
validation_summary <- read_csv(file.path(results_dir, "biomarker_validation_summary.csv"))
expression_data <- read_csv(file.path(results_dir, "biomarker_expression_data.csv"))

# Define stage order and labels
stage_order <- c("Infant_0_20d", "Early childhood_21_59d", "Pre_pubertal_60_149d",
                 "Post_pubertal_150_365d", "Adult_>365d")
stage_labels <- c("Infant\n(0-20d)", "Early\n(21-59d)", "Pre-pub\n(60-149d)",
                  "Post-pub\n(150-365d)", "Adult\n(>365d)")

# Panel A: Expression heatmap showing developmental trajectories
create_panel_a <- function() {
  # Prepare matrix for heatmap
  heatmap_data <- stage_stats %>%
    select(Biomarker, Gene_Name, Tissue, Stage, mean_expr) %>%
    mutate(Stage = factor(Stage, levels = stage_order)) %>%
    pivot_wider(names_from = Stage, values_from = mean_expr)

  # Create expression matrix
  expr_matrix <- as.matrix(heatmap_data[, stage_order])
  rownames(expr_matrix) <- paste0(heatmap_data$Gene_Name, " (", heatmap_data$Tissue, ")")

  # Z-score normalize by row
  expr_matrix_scaled <- t(scale(t(expr_matrix)))

  # Define tissue colors
  tissue_colors <- c(
    "Adipose" = "#FFD700",
    "Blood" = "#DC143C",
    "Brain" = "#4169E1",
    "Liver" = "#8B4513",
    "Lung" = "#32CD32",
    "Muscle" = "#FF69B4",
    "Small intestine" = "#FFA500",
    "Testis" = "#9370DB"
  )

  # Create tissue annotation
  tissue_anno <- HeatmapAnnotation(
    Tissue = heatmap_data$Tissue,
    col = list(Tissue = tissue_colors),
    show_legend = TRUE,
    annotation_name_side = "left"
  )

  # Create trend validation annotation
  validation_colors <- c("TRUE" = "#2ECC71", "FALSE" = "#E74C3C")
  validation_anno <- HeatmapAnnotation(
    Validated = validation_summary$Validation_Pass,
    col = list(Validated = validation_colors),
    show_legend = TRUE,
    annotation_name_side = "left"
  )

  # Color scheme for heatmap
  col_fun <- colorRamp2(c(-2, 0, 2), c("#313695", "#FFFFBF", "#A50026"))

  # Create heatmap
  ht <- Heatmap(
    expr_matrix_scaled,
    name = "Expression\n(Z-score)",
    col = col_fun,
    column_title = "Developmental Stage",
    column_title_side = "bottom",
    row_title = "Biomarker",
    row_names_side = "left",
    column_names_rot = 0,
    column_labels = stage_labels,
    left_annotation = rowAnnotation(
      df = data.frame(
        Tissue = heatmap_data$Tissue,
        Validated = validation_summary$Validation_Pass
      ),
      col = list(
        Tissue = tissue_colors,
        Validated = validation_colors
      ),
      width = unit(1, "cm")
    ),
    cluster_rows = FALSE,
    cluster_columns = FALSE,
    row_names_gp = gpar(fontsize = 10),
    column_names_gp = gpar(fontsize = 10),
    heatmap_legend_param = list(
      title_gp = gpar(fontsize = 10),
      labels_gp = gpar(fontsize = 9)
    )
  )

  # Save as PDF
  pdf(file.path(output_dir, "figure5_panel_a.pdf"), width = 10, height = 8)
  draw(ht, heatmap_legend_side = "right")
  dev.off()

  # Also create ggplot version for consistency
  gg_heatmap_data <- stage_stats %>%
    mutate(
      Stage = factor(Stage, levels = stage_order, labels = stage_labels),
      Biomarker_Label = paste0(Gene_Name, "\n(", Tissue, ")")
    ) %>%
    group_by(Biomarker) %>%
    mutate(scaled_expr = scale(mean_expr)[,1]) %>%
    ungroup()

  p <- ggplot(gg_heatmap_data, aes(x = Stage, y = Biomarker_Label, fill = scaled_expr)) +
    geom_tile(color = "white", size = 0.5) +
    scale_fill_gradient2(
      low = "#313695", mid = "#FFFFBF", high = "#A50026",
      midpoint = 0, limits = c(-2, 2),
      name = "Expression\n(Z-score)"
    ) +
    labs(
      title = "Biomarker Expression Trajectories Across Development",
      subtitle = "Z-score normalized expression showing developmental patterns",
      x = "Developmental Stage",
      y = NULL
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 0, hjust = 0.5, size = 10),
      axis.text.y = element_text(size = 10),
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 11),
      legend.position = "right"
    )

  return(p)
}

# Panel B: Line plots showing developmental trends with confidence intervals
create_panel_b <- function() {
  # Prepare data for line plots
  plot_data <- stage_stats %>%
    mutate(
      Stage = factor(Stage, levels = stage_order, labels = stage_labels),
      lower = mean_expr - sem_expr,
      upper = mean_expr + sem_expr,
      Biomarker_Label = paste0(Gene_Name, " (", Biomarker, ")")
    )

  # Define colors by expected trend
  trend_colors <- c(
    "increase" = "#2ECC71",
    "decrease" = "#E74C3C",
    "stable_high" = "#3498DB",
    "complex" = "#9B59B6"
  )

  # Create faceted line plot
  p <- ggplot(plot_data, aes(x = stage_numeric, y = mean_expr)) +
    geom_ribbon(aes(ymin = lower, ymax = upper, fill = Expected_Trend), alpha = 0.3) +
    geom_line(aes(color = Expected_Trend), size = 1.2) +
    geom_point(aes(color = Expected_Trend, size = n_samples), alpha = 0.8) +
    facet_wrap(~ Biomarker_Label, scales = "free_y", ncol = 4) +
    scale_x_continuous(
      breaks = 0:4,
      labels = c("Infant", "Early", "Pre-pub", "Post-pub", "Adult")
    ) +
    scale_color_manual(values = trend_colors, name = "Expected Trend") +
    scale_fill_manual(values = trend_colors, guide = "none") +
    scale_size_continuous(range = c(2, 5), name = "Sample Size") +
    labs(
      title = "Developmental Expression Patterns of Key Biomarkers",
      subtitle = "Lines show mean expression ± SEM across developmental stages",
      x = "Developmental Stage",
      y = "Log2 Expression"
    ) +
    theme_minimal() +
    theme(
      strip.text = element_text(size = 9, face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 11),
      legend.position = "bottom"
    )

  # Add validation status
  validation_data <- plot_data %>%
    group_by(Biomarker_Label, Trend_Valid) %>%
    summarise(
      x_pos = 4,
      y_pos = max(upper) * 0.95,
      .groups = "drop"
    ) %>%
    mutate(
      validation_symbol = ifelse(Trend_Valid, "✓", "✗"),
      validation_color = ifelse(Trend_Valid, "#2ECC71", "#E74C3C")
    )

  p <- p + geom_text(
    data = validation_data,
    aes(x = x_pos, y = y_pos, label = validation_symbol),
    color = validation_data$validation_color,
    size = 6,
    fontface = "bold"
  )

  return(p)
}

# Panel C: Comparison with human developmental patterns
create_panel_c <- function() {
  # Create comparison data showing alignment with human patterns
  comparison_data <- validation_summary %>%
    mutate(
      Biomarker_Label = paste0(Gene_Name, "\n(", Tissue, ")"),
      Alignment = ifelse(Validation_Pass, "Aligned", "Not aligned"),
      Correlation_Abs = abs(Correlation)
    )

  # Create lollipop plot showing correlation strength and validation
  p <- ggplot(comparison_data, aes(x = reorder(Biomarker_Label, Correlation), y = Correlation)) +
    geom_segment(aes(x = Biomarker_Label, xend = Biomarker_Label, y = 0, yend = Correlation,
                     color = Alignment), size = 1.5) +
    geom_point(aes(color = Alignment, size = -log10(P_Value)), alpha = 0.8) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    geom_hline(yintercept = c(-0.3, 0.3), linetype = "dotted", color = "gray70") +
    coord_flip() +
    scale_color_manual(
      values = c("Aligned" = "#2ECC71", "Not aligned" = "#E74C3C"),
      name = "Human Pattern"
    ) +
    scale_size_continuous(
      range = c(3, 8),
      name = "-log10(p-value)",
      breaks = c(1, 2, 3, 4),
      labels = c("1", "2", "3", "4")
    ) +
    labs(
      title = "Alignment with Human Developmental Patterns",
      subtitle = "Correlation between biomarker expression and developmental stage",
      x = NULL,
      y = "Spearman Correlation with Developmental Stage"
    ) +
    theme_minimal() +
    theme(
      axis.text.y = element_text(size = 10),
      axis.text.x = element_text(size = 10),
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 11),
      legend.position = "right"
    ) +
    annotate("text", x = 0.5, y = 0.35, label = "Positive\ntrend",
             hjust = 0, vjust = 0, size = 3, color = "gray60") +
    annotate("text", x = 0.5, y = -0.35, label = "Negative\ntrend",
             hjust = 1, vjust = 1, size = 3, color = "gray60")

  return(p)
}

# Panel D: Summary table with human references
create_panel_d <- function() {
  # Create summary table
  summary_table <- validation_summary %>%
    select(Gene_Name, Tissue, Function, Expected_Trend, Observed_Trend, Validation_Pass) %>%
    mutate(
      Validation = ifelse(Validation_Pass, "✓", "✗"),
      Expected_Trend = str_replace_all(Expected_Trend, "_", " ")
    ) %>%
    select(-Validation_Pass)

  # Create table plot using ggplot
  table_data <- summary_table %>%
    mutate(
      y_pos = row_number(),
      Function_Short = str_trunc(Function, 40)
    )

  p <- ggplot(table_data, aes(y = y_pos)) +
    geom_text(aes(x = 1, label = Gene_Name), hjust = 0, size = 3.5) +
    geom_text(aes(x = 2, label = Tissue), hjust = 0, size = 3.5) +
    geom_text(aes(x = 3, label = Function_Short), hjust = 0, size = 3) +
    geom_text(aes(x = 4, label = Expected_Trend), hjust = 0.5, size = 3.5) +
    geom_text(aes(x = 5, label = Validation,
                  color = ifelse(Validation == "✓", "#2ECC71", "#E74C3C")),
              hjust = 0.5, size = 5, fontface = "bold") +
    scale_color_identity() +
    scale_x_continuous(
      limits = c(0.8, 5.5),
      breaks = 1:5,
      labels = c("Biomarker", "Tissue", "Function", "Expected", "Valid")
    ) +
    scale_y_reverse() +
    labs(
      title = "Biomarker Validation Summary",
      subtitle = "Comparison with known human developmental patterns"
    ) +
    theme_void() +
    theme(
      axis.text.x = element_text(size = 10, face = "bold", vjust = 0),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 11, hjust = 0.5),
      plot.margin = margin(20, 20, 20, 20)
    )

  return(p)
}

# Main execution
main <- function() {
  print("Creating biomarker validation figure panels...")

  # Create individual panels
  panel_a <- create_panel_a()
  panel_b <- create_panel_b()
  panel_c <- create_panel_c()
  panel_d <- create_panel_d()

  # Combine panels
  combined_figure <- (panel_a + panel_c) / panel_b +
    plot_annotation(
      title = "Figure 5: Validation of Conserved Developmental Biomarkers",
      subtitle = "Pig biomarkers show similar developmental trends as observed in human development",
      theme = theme(
        plot.title = element_text(size = 16, face = "bold"),
        plot.subtitle = element_text(size = 12)
      )
    )

  # Save combined figure
  ggsave(
    filename = file.path(output_dir, "figure5_biomarker_validation.pdf"),
    plot = combined_figure,
    width = 16,
    height = 12,
    dpi = 300
  )

  ggsave(
    filename = file.path(output_dir, "figure5_biomarker_validation.png"),
    plot = combined_figure,
    width = 16,
    height = 12,
    dpi = 300
  )

  # Save individual panels
  ggsave(file.path(output_dir, "figure5_panel_b.pdf"), panel_b, width = 12, height = 8)
  ggsave(file.path(output_dir, "figure5_panel_c.pdf"), panel_c, width = 8, height = 6)
  ggsave(file.path(output_dir, "figure5_panel_d.pdf"), panel_d, width = 10, height = 6)

  print("Biomarker validation figures created successfully!")
  print(paste("Output saved to:", output_dir))
}

# Run if executed directly
if (!interactive()) {
  main()
}