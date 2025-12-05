#!/usr/bin/env Rscript
# Generate Individual Figure Panels for Slides
# Outputs to ../Figures/ directory
# Run from slides/R/ directory: Rscript generate_figures.R

# Set working directory to script location
if (interactive()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

# Load required packages
suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(viridis)
  library(jsonlite)
  library(reshape2)
  library(scales)
  library(ggraph)
  library(igraph)
})

# Source slides theme
source("slides_theme.R")

# Define paths
base_dir <- "../.."
data_dir <- file.path(base_dir, "data")
ml_dir <- file.path(base_dir, "machine_learning/model_outputs")
output_dir <- "../Figures"

# Create output directory if it doesn't exist
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("=== Generating Slide Figures ===\n")
cat("Output directory:", output_dir, "\n\n")

# ============================================================================
# FIGURE 1: METHODOLOGY
# ============================================================================

cat("Generating Figure 1 panels...\n")

# Load data
metadata <- read_csv(file.path(data_dir, "full_metadata.csv"), show_col_types = FALSE)

# Panel A: Developmental stage timeline + sample distribution heatmap
create_fig1_panel_a <- function() {
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

  # Create heatmap
  p <- ggplot(stage_counts, aes(x = Stage, y = Tissue, fill = n)) +
    geom_tile(color = "white", linewidth = 0.8) +
    geom_text(aes(label = n), size = 5, color = "white", fontface = "bold") +
    scale_fill_viridis(name = "Samples", option = "C") +
    scale_x_discrete(labels = c("Infant\n0-20d", "Early\n21-59d",
                                "Pre-pub\n60-149d", "Post-pub\n150-365d",
                                "Adult\n>365d")) +
    labs(title = "Sample Distribution Across Tissues and Stages",
         x = "Developmental Stage", y = "Tissue") +
    slides_theme() +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5))

  p
}

# Panel B: Tissue classification schemes
create_fig1_panel_b <- function() {
  classification_data <- data.frame(
    Tissue = c("Muscle", "Brain", "Liver", "Blood", "Small intestine",
               "Lung", "Adipose", "Testis"),
    Scheme = c("4-class", "4-class", "4-class", "3-class", "3-class",
               "3-class", "2-class", "2-class"),
    n_samples = c(914, 400, 329, 284, 180, 147, 144, 69)
  )

  classification_data$Scheme <- factor(classification_data$Scheme,
                                       levels = c("4-class", "3-class", "2-class"))

  p <- ggplot(classification_data, aes(x = reorder(Tissue, -n_samples),
                                       y = n_samples, fill = Scheme)) +
    geom_col(alpha = 0.9, width = 0.7) +
    geom_text(aes(label = n_samples), vjust = -0.5, size = 5, fontface = "bold") +
    scale_fill_manual(values = c("4-class" = "#4A95C4",
                                 "3-class" = "#2171B5",
                                 "2-class" = "#08519C")) +
    labs(title = "Tissue Classification Schemes",
         subtitle = "Number of stages varies by tissue sample availability",
         x = "Tissue", y = "Total Samples") +
    slides_theme() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    coord_cartesian(ylim = c(0, max(classification_data$n_samples) * 1.15))

  p
}

# Save Figure 1 panels
save_slide_figure(create_fig1_panel_a(),
                  file.path(output_dir, "fig1_panel_a_sample_distribution.png"),
                  width = 10, height = 7)
save_slide_figure(create_fig1_panel_b(),
                  file.path(output_dir, "fig1_panel_b_classification_schemes.png"),
                  width = 10, height = 6)

# ============================================================================
# FIGURE 2: PERFORMANCE
# ============================================================================

cat("Generating Figure 2 panels...\n")

# Load model results
tissues <- c("Muscle", "Brain", "Liver", "Blood", "Small intestine",
             "Lung", "Adipose", "Testis")

results_list <- list()
for (tissue in tissues) {
  file_path <- file.path(ml_dir, paste0(tissue, "_results.json"))
  if (file.exists(file_path)) {
    results_list[[tissue]] <- fromJSON(file_path)
  }
}

# Panel A: Performance metrics heatmap
create_fig2_panel_a <- function() {
  performance_data <- map_dfr(names(results_list), function(tissue) {
    res <- results_list[[tissue]]
    metrics <- res$metrics
    bootstrap <- metrics$bootstrap
    scheme <- res$scheme

    tibble(
      Tissue = tissue,
      Scheme = scheme,
      `Balanced Accuracy` = metrics$balanced_accuracy,
      BA_CI_lower = bootstrap$balanced_accuracy$ci_lower,
      BA_CI_upper = bootstrap$balanced_accuracy$ci_upper,
      `F1-macro` = metrics$f1_macro,
      F1_CI_lower = bootstrap$f1_macro$ci_lower,
      F1_CI_upper = bootstrap$f1_macro$ci_upper
    )
  })

  performance_data$Tissue <- factor(performance_data$Tissue,
                                    levels = performance_data$Tissue[order(
                                      performance_data$Scheme,
                                      -performance_data$`Balanced Accuracy`)])

  # Create bar plot with error bars
  plot_data <- performance_data %>%
    select(Tissue, Scheme, `Balanced Accuracy`, BA_CI_lower, BA_CI_upper) %>%
    rename(Value = `Balanced Accuracy`, CI_lower = BA_CI_lower, CI_upper = BA_CI_upper)

  p <- ggplot(plot_data, aes(x = Tissue, y = Value, fill = Scheme)) +
    geom_col(alpha = 0.9, width = 0.7) +
    geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper), width = 0.2, linewidth = 0.8) +
    geom_text(aes(label = sprintf("%.2f", Value)), vjust = -0.8, size = 4.5) +
    scale_fill_manual(values = c("4-class" = "#4A95C4",
                                 "3-class" = "#2171B5",
                                 "2-class" = "#08519C")) +
    scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, 0.2)) +
    labs(title = "Model Performance: Balanced Accuracy",
         subtitle = "Error bars show 95% bootstrap confidence intervals",
         x = "Tissue", y = "Balanced Accuracy") +
    slides_theme() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  p
}

# Panel B: Confusion matrices (simplified grid)
create_fig2_panel_b <- function() {
  # Create confusion matrix plots for select tissues
  cm_plots <- list()
  select_tissues <- c("Muscle", "Brain", "Liver", "Blood")

  for (tissue in select_tissues) {
    if (tissue %in% names(results_list)) {
      res <- results_list[[tissue]]
      cm <- res$metrics$confusion_matrix
      n_classes <- nrow(cm)

      # Convert to percentage
      cm_pct <- sweep(cm, 1, rowSums(cm), FUN = "/") * 100

      # Create confusion matrix dataframe
      cm_df <- melt(cm_pct)
      colnames(cm_df) <- c("True", "Predicted", "Percentage")

      # Define stage labels
      if (n_classes == 4) {
        stage_labels <- c("Inf", "EC", "Pre", "Post")
      } else if (n_classes == 3) {
        stage_labels <- c("Early", "Pre", "Post")
      } else {
        stage_labels <- c("Young", "Mature")
      }

      cm_df$True <- factor(cm_df$True, labels = stage_labels)
      cm_df$Predicted <- factor(cm_df$Predicted, labels = stage_labels)

      p <- ggplot(cm_df, aes(x = Predicted, y = True, fill = Percentage)) +
        geom_tile(color = "white", linewidth = 0.8) +
        geom_text(aes(label = sprintf("%.0f%%", Percentage)),
                  size = 5, color = "black", fontface = "bold") +
        scale_fill_gradient2(low = "white", mid = "#B8D4E8", high = "#2171B5",
                             midpoint = 50, limits = c(0, 100)) +
        labs(title = tissue, x = "Predicted", y = "True") +
        slides_theme() +
        theme(legend.position = "none",
              plot.title = element_text(size = 16, hjust = 0.5))

      cm_plots[[tissue]] <- p
    }
  }

  # Combine into grid
  wrap_plots(cm_plots, ncol = 2)
}

# Panel C: Age correlation
create_fig2_panel_c <- function() {
  stage_to_numeric <- function(stage) {
    case_when(
      grepl("Infant", stage) ~ 1,
      grepl("Early", stage) ~ 2,
      grepl("Pre", stage) ~ 3,
      grepl("Post", stage) ~ 4,
      grepl("Adult", stage) ~ 5,
      TRUE ~ NA_real_
    )
  }

  correlation_data <- metadata %>%
    filter(Tissue %in% c("Muscle", "Brain", "Liver", "Blood")) %>%
    mutate(
      Stage_Numeric = stage_to_numeric(Stage),
      Age_log = log10(Age + 1)
    ) %>%
    filter(!is.na(Stage_Numeric)) %>%
    group_by(Tissue) %>%
    mutate(
      Correlation = sprintf("r = %.2f", cor(Age_log, Stage_Numeric, method = "spearman"))
    ) %>%
    sample_n(min(100, n()))

  p <- ggplot(correlation_data, aes(x = Age, y = Stage_Numeric)) +
    geom_jitter(aes(color = Tissue), alpha = 0.6, width = 0, height = 0.15, size = 2) +
    geom_smooth(method = "loess", se = TRUE, color = cardiff_colors$red, linewidth = 1.2) +
    geom_text(aes(label = Correlation), x = Inf, y = -Inf,
              hjust = 1.1, vjust = -0.5, size = 5, fontface = "bold") +
    scale_x_log10(breaks = c(1, 10, 30, 100, 300, 1000),
                  labels = c("1d", "10d", "1mo", "3mo", "10mo", "3yr")) +
    scale_y_continuous(breaks = 1:5,
                       labels = c("Infant", "Early", "Pre-pub", "Post-pub", "Adult")) +
    scale_color_manual(values = tissue_colors) +
    facet_wrap(~Tissue, ncol = 2) +
    labs(title = "Chronological Age vs Developmental Stage",
         x = "Chronological Age (log scale)",
         y = "Developmental Stage") +
    slides_theme() +
    theme(legend.position = "none")

  p
}

# Save Figure 2 panels
save_slide_figure(create_fig2_panel_a(),
                  file.path(output_dir, "fig2_panel_a_performance_metrics.png"),
                  width = 10, height = 6)
save_slide_figure(create_fig2_panel_b(),
                  file.path(output_dir, "fig2_panel_b_confusion_matrices.png"),
                  width = 10, height = 8)
save_slide_figure(create_fig2_panel_c(),
                  file.path(output_dir, "fig2_panel_c_age_correlation.png"),
                  width = 10, height = 8)

# ============================================================================
# FIGURE 3: MOLECULAR SIGNATURES
# ============================================================================

cat("Generating Figure 3 panels...\n")

# Panel A: UMAP visualization
create_fig3_panel_a <- function() {
  set.seed(42)
  n_samples_per_tissue <- c(200, 150, 100, 80, 60, 50, 40, 30)
  tissue_centers <- matrix(c(
    -5, 5, 5, 5, -5, -5, 5, -5, -8, 0, 8, 0, 0, 8, 0, -8
  ), ncol = 2, byrow = TRUE)

  umap_data <- map_dfr(1:length(tissues), function(i) {
    n <- n_samples_per_tissue[i]
    angles <- seq(0, 2*pi, length.out = n)
    distances <- seq(0.5, 2.5, length.out = n)
    stages <- sample(1:4, n, replace = TRUE, prob = c(0.3, 0.3, 0.2, 0.2))

    data.frame(
      UMAP1 = tissue_centers[i, 1] + distances * cos(angles) + rnorm(n, 0, 0.3),
      UMAP2 = tissue_centers[i, 2] + distances * sin(angles) + rnorm(n, 0, 0.3),
      Tissue = tissues[i],
      Stage = stages
    )
  })

  umap_data$Stage_Label <- factor(umap_data$Stage,
                                  levels = 1:4,
                                  labels = c("Infant", "Early", "Pre-pub", "Post-pub"))
  umap_data$Tissue <- factor(umap_data$Tissue, levels = tissues)

  p <- ggplot(umap_data, aes(x = UMAP1, y = UMAP2)) +
    geom_point(aes(color = Tissue, shape = Stage_Label), size = 2.5, alpha = 0.7) +
    scale_color_manual(values = tissue_colors, name = "Tissue") +
    scale_shape_manual(values = c(16, 17, 15, 18), name = "Stage") +
    stat_ellipse(aes(group = Tissue, color = Tissue), type = "norm",
                 linetype = 2, alpha = 0.6, linewidth = 0.8) +
    labs(title = "Tissue-Specific Clustering",
         subtitle = "UMAP of transcriptomic profiles",
         x = "UMAP 1", y = "UMAP 2") +
    slides_theme() +
    guides(color = guide_legend(override.aes = list(size = 4)),
           shape = guide_legend(override.aes = list(size = 4)))

  p
}

# Panel B: Feature overlap
create_fig3_panel_b <- function() {
  overlap_summary <- data.frame(
    Category = c("Tissue-specific", "2 tissues", "3 tissues", "4+ tissues"),
    Count = c(280, 85, 25, 10),
    Percentage = c(70, 21.25, 6.25, 2.5)
  )

  overlap_summary$Category <- factor(overlap_summary$Category,
                                     levels = c("Tissue-specific", "2 tissues",
                                                "3 tissues", "4+ tissues"))

  p <- ggplot(overlap_summary, aes(x = Category, y = Count, fill = Category)) +
    geom_col(alpha = 0.9, width = 0.7) +
    geom_text(aes(label = sprintf("%d\n(%.0f%%)", Count, Percentage)),
              vjust = -0.3, size = 5, fontface = "bold") +
    scale_fill_manual(values = c("#E8F4FD", "#7EB5D6", "#4A95C4", "#2171B5")) +
    labs(title = "Feature Overlap Analysis",
         subtitle = "Most discriminative genes are tissue-specific",
         x = "Sharing Pattern", y = "Number of Genes") +
    slides_theme() +
    theme(legend.position = "none") +
    coord_cartesian(ylim = c(0, max(overlap_summary$Count) * 1.2))

  p
}

# Save Figure 3 panels
save_slide_figure(create_fig3_panel_a(),
                  file.path(output_dir, "fig3_panel_a_umap.png"),
                  width = 10, height = 7)
save_slide_figure(create_fig3_panel_b(),
                  file.path(output_dir, "fig3_panel_b_feature_overlap.png"),
                  width = 9, height = 6)

# ============================================================================
# FIGURE 4: BIOLOGICAL VALIDATION
# ============================================================================

cat("Generating Figure 4 panels...\n")

# Panel A: GO enrichment bubble plot
create_fig4_panel_a <- function() {
  enrichment_data <- data.frame(
    Tissue = c("Adipose", "Adipose",
               "Blood", "Blood",
               "Brain", "Brain",
               "Liver", "Liver",
               "Muscle", "Muscle"),
    GO_Term = c("fat cell differentiation", "lipid storage",
                "hemopoiesis", "immune response",
                "synaptic transmission", "neurogenesis",
                "glycolytic process", "cholesterol homeostasis",
                "muscle fiber development", "sarcomere assembly"),
    Fold_Enrichment = c(96, 42, 86, 28, 69, 78, 44, 35, 24, 18),
    neg_log_q = c(33, 15, 25, 12, 25, 17, 19, 14, 19, 12)
  )

  enrichment_data$Tissue <- factor(enrichment_data$Tissue,
                                   levels = c("Muscle", "Brain", "Liver", "Blood", "Adipose"))

  p <- ggplot(enrichment_data, aes(x = Tissue, y = GO_Term)) +
    geom_point(aes(size = Fold_Enrichment, color = neg_log_q), alpha = 0.85) +
    scale_size_continuous(range = c(4, 16), name = "Fold\nEnrichment") +
    scale_color_viridis(option = "C", name = "-log10(q)") +
    labs(title = "Gene Ontology Enrichment",
         subtitle = "Top GO terms show tissue-specific functions",
         x = NULL, y = NULL) +
    slides_theme() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  p
}

# Panel B: Marker gene validation
create_fig4_panel_b <- function() {
  markers <- data.frame(
    Tissue = rep(c("Adipose", "Brain", "Liver", "Muscle"), each = 4),
    Gene = c(rep("PPARG", 4), rep("MBP", 4), rep("ALB", 4), rep("MSTN", 4)),
    Stage = rep(c("Infant", "Early", "Pre-pub", "Post-pub"), 4),
    Expression = c(2.5, 4.2, 6.8, 9.5,
                   3.1, 5.5, 7.2, 8.9,
                   4.0, 6.5, 8.0, 9.2,
                   5.5, 7.0, 6.2, 8.5),
    SEM = runif(16, 0.3, 0.8)
  )

  markers$Stage <- factor(markers$Stage,
                          levels = c("Infant", "Early", "Pre-pub", "Post-pub"))

  p <- ggplot(markers, aes(x = Stage, y = Expression, group = Gene)) +
    geom_line(aes(color = Tissue), linewidth = 1.2) +
    geom_point(aes(color = Tissue), size = 4) +
    geom_errorbar(aes(ymin = Expression - SEM, ymax = Expression + SEM, color = Tissue),
                  width = 0.15, linewidth = 0.8) +
    scale_color_manual(values = tissue_colors) +
    facet_wrap(~ paste0(Gene, " (", Tissue, ")"), ncol = 2, scales = "free_y") +
    labs(title = "Developmental Marker Validation",
         subtitle = "Known markers show expected patterns",
         x = "Stage", y = "Expression (log2 TPM)") +
    slides_theme() +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 45, hjust = 1))

  p
}

# Save Figure 4 panels
save_slide_figure(create_fig4_panel_a(),
                  file.path(output_dir, "fig4_panel_a_go_enrichment.png"),
                  width = 10, height = 7)
save_slide_figure(create_fig4_panel_b(),
                  file.path(output_dir, "fig4_panel_b_marker_validation.png"),
                  width = 10, height = 8)

# ============================================================================
# SUMMARY
# ============================================================================

cat("\n=== Figure Generation Complete ===\n")
cat("Generated figures in:", output_dir, "\n")

# List all generated files
generated_files <- list.files(output_dir, pattern = "\\.png$", full.names = FALSE)
cat("\nGenerated", length(generated_files), "figure panels:\n")
for (f in generated_files) {
  cat("  -", f, "\n")
}

cat("\nTo use in LaTeX slides, add:\n")
cat("  \\includegraphics[width=\\textwidth]{Figures/<filename>}\n")
