#!/usr/bin/env Rscript
# ==============================================================================
# Supplementary Figures S1-S3
# ==============================================================================
# S1: Cross-species sensitivity analysis
# S2: Tissue specificity and feature stability
# S3: Extended performance metrics
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(ggrepel)
  library(viridis)
  library(umap)
  library(jsonlite)
})

# Load shared modules
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("--file=", "", file_arg))))
  }
  "/Users/tianyuan/Desktop/github_dev/pig-dev-stage/paper/figures/R"
}
script_dir <- get_script_dir()
source(file.path(script_dir, "_theme.R"))
source(file.path(script_dir, "_utils.R"))

cat("==============================================================================\n")
cat("Supplementary Figures S1-S3\n")
cat("==============================================================================\n\n")

# ==============================================================================
# FIGURE S1: CROSS-SPECIES SENSITIVITY ANALYSIS
# ==============================================================================

create_fig_s1 <- function() {
  cat("Creating Figure S1: Cross-species sensitivity...\n")
  
  # Load sensitivity data
  sensitivity_file <- project_path("paper/figures/output/stats/cross_species_sensitivity.csv")
  bootstrap_file <- project_path("paper/figures/output/stats/cross_species_sensitivity.json")
  
  if (!file.exists(sensitivity_file) || !file.exists(bootstrap_file)) {
    stop("Sensitivity analysis files not found. Run cross_species_sensitivity.py first.")
  }
  
  sensitivity <- read_csv(sensitivity_file, show_col_types = FALSE)
  bootstrap_data <- fromJSON(bootstrap_file)
  
  # Panel A: Correlation vs. p-value threshold
  panel_a <- sensitivity %>%
    filter(n_genes >= 15, n_genes <= 50) %>%
    group_by(p_threshold) %>%
    summarise(
      mean_r = mean(correlation, na.rm = TRUE),
      max_r = max(correlation, na.rm = TRUE),
      min_r = min(correlation, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    ggplot(aes(x = p_threshold, y = mean_r)) +
    geom_line(color = PRIMARY_COLORS[2], linewidth = 0.5) +
    geom_ribbon(aes(ymin = min_r, ymax = max_r), alpha = 0.2, fill = PRIMARY_COLORS[2]) +
    geom_point(color = PRIMARY_COLORS[2], size = 1.5) +
    labs(
      x = "P-value Threshold",
      y = "Mean Pearson Correlation (R)",
      title = "a"
    ) +
    nature_theme() +
    theme(
      plot.title = element_text(size = 8, face = "bold", hjust = 0),
      aspect.ratio = 0.6
    )
  
  # Panel B: Bootstrap distribution
  bootstrap_dist <- bootstrap_data$bootstrap_results$bootstrap_distribution
  bootstrap_df <- tibble(correlation = bootstrap_dist)
  
  panel_b <- ggplot(bootstrap_df, aes(x = correlation)) +
    geom_histogram(bins = 50, fill = PRIMARY_COLORS[1], alpha = 0.7, color = "white", linewidth = 0.2) +
    geom_vline(
      xintercept = bootstrap_data$bootstrap_results$correlation,
      linetype = "dashed",
      color = PRIMARY_COLORS[2],
      linewidth = 0.5
    ) +
    geom_vline(
      xintercept = c(
        bootstrap_data$bootstrap_results$ci_lower,
        bootstrap_data$bootstrap_results$ci_upper
      ),
      linetype = "dotted",
      color = PRIMARY_COLORS[2],
      linewidth = 0.3
    ) +
    annotate(
      "text",
      x = Inf, y = Inf,
      label = sprintf(
        "R = %.3f\n95%% CI: [%.3f, %.3f]",
        bootstrap_data$bootstrap_results$correlation,
        bootstrap_data$bootstrap_results$ci_lower,
        bootstrap_data$bootstrap_results$ci_upper
      ),
      hjust = 1.1, vjust = 1.3, size = 2.5
    ) +
    labs(
      x = "Bootstrap Correlation (R)",
      y = "Frequency",
      title = "b"
    ) +
    nature_theme() +
    theme(
      plot.title = element_text(size = 8, face = "bold", hjust = 0),
      aspect.ratio = 0.6
    )
  
  fig_s1 <- panel_a | panel_b
  
  return(fig_s1)
}

# ==============================================================================
# FIGURE S2: TISSUE SPECIFICITY AND FEATURE STABILITY
# ==============================================================================

create_fig_s2 <- function() {
  cat("Creating Figure S2: Tissue specificity and stability...\n")
  
  # Load data
  metadata <- load_metadata()
  results_list <- load_all_ml_results()
  stability_file <- project_path("machine_learning/analysis/results/feature_stability_results.json")
  
  if (!file.exists(stability_file)) {
    stop("Feature stability results not found")
  }
  
  stability_data <- fromJSON(stability_file)
  
  # Panel A: UMAP showing tissue clustering
  # Load expression data for UMAP
  tissues <- c("Muscle", "Brain", "Liver", "Blood", "Lung")
  
  # Try to load UMAP data, or compute it
  umap_file <- project_path("paper/figures/output/stats/umap_tissue_clustering.csv")
  
  if (file.exists(umap_file)) {
    umap_df <- read_csv(umap_file, show_col_types = FALSE)
    umap_df$Tissue <- factor(umap_df$Tissue)
  } else {
    # Sample a subset for UMAP (to avoid memory issues)
    expr_list <- list()
    tissue_labels <- c()
    
    for (tissue in tissues) {
      tryCatch({
        expr <- load_expression(tissue, log_transform = TRUE)
        # Sample up to 150 samples per tissue
        if (ncol(expr) > 150) {
          set.seed(42)
          expr <- expr[, sample(ncol(expr), 150)]
        }
        expr_list[[tissue]] <- expr
        tissue_labels <- c(tissue_labels, rep(tissue, ncol(expr)))
      }, error = function(e) {
        warning("Could not load expression for ", tissue, ": ", e$message)
      })
    }
    
    if (length(expr_list) == 0) {
      stop(
        "Could not load expression data for any tissue. ",
        "Please ensure expression files exist in data/pigGTEx/ directory.\n",
        "Expected files: Muscle.expr_tpm.txt.gz, Brain.expr_tpm.txt.gz, etc.\n",
        "See data/README.md for data download instructions."
      )
    }
    
    # Combine and transpose (samples x genes)
    combined_expr <- do.call(cbind, expr_list)
    combined_expr_t <- t(combined_expr)
    
    # Select top variable genes for UMAP
    gene_vars <- apply(combined_expr_t, 2, var, na.rm = TRUE)
    top_genes <- names(sort(gene_vars, decreasing = TRUE))[1:min(500, length(gene_vars))]
    combined_expr_subset <- combined_expr_t[, top_genes, drop = FALSE]
    
    # Run UMAP
    set.seed(42)
    umap_result <- umap(combined_expr_subset, n_neighbors = 15, min_dist = 0.1)
    umap_df <- as_tibble(umap_result$layout, .name_repair = ~ c("UMAP1", "UMAP2"))
    umap_df$Tissue <- factor(tissue_labels)
    
    # Save for future use
    write_csv(umap_df, umap_file)
    cat(sprintf("  Computed and saved UMAP for %d samples from %d tissues\n", 
                nrow(umap_df), length(expr_list)))
  }
  
  panel_a <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = Tissue)) +
    geom_point(size = 0.5, alpha = 0.6) +
    scale_color_manual(values = TISSUE_COLORS) +
    labs(
      x = "UMAP 1",
      y = "UMAP 2",
      title = "a"
    ) +
    nature_theme() +
    theme(
      plot.title = element_text(size = 8, face = "bold", hjust = 0),
      legend.position = "bottom",
      aspect.ratio = 1
    )
  
  # Panel B: Within-tissue feature stability
  stability_summary <- map_dfr(names(stability_data), function(tissue) {
    data <- stability_data[[tissue]]
    tibble(
      Tissue = tissue,
      Mean_Jaccard = data$jaccard_similarity$mean,
      Stable_Genes = sum(data$gene_frequency >= 4),  # >= 80% of 5 seeds
      Total_Genes = length(data$gene_frequency)
    )
  })
  
  panel_b <- stability_summary %>%
    mutate(Tissue = factor(Tissue, levels = TISSUE_COLORS %>% names())) %>%
    ggplot(aes(x = Tissue, y = Mean_Jaccard, fill = Tissue)) +
    geom_col(width = 0.7, show.legend = FALSE) +
    scale_fill_manual(values = TISSUE_COLORS) +
    labs(
      x = NULL,
      y = "Mean Jaccard Similarity\n(across seeds)",
      title = "b"
    ) +
    nature_theme() +
    theme(
      plot.title = element_text(size = 8, face = "bold", hjust = 0),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
      aspect.ratio = 0.8
    )
  
  # Panel C: Stage correlation of top features (from existing analysis)
  correlation_file <- project_path("machine_learning/analysis/results/expression_correlation_summary.csv")
  if (file.exists(correlation_file)) {
    corr_data <- read_csv(correlation_file, show_col_types = FALSE)
    
    # Get mean correlation per tissue (handle column name with spaces/pipes)
    # The CSV has "Mean |r|" but we'll rename it for consistency
    if ("Mean |r|" %in% names(corr_data)) {
      corr_data <- corr_data %>% rename(Mean_abs_r = `Mean |r|`)
    }
    
    corr_summary <- corr_data %>%
      group_by(Tissue) %>%
      summarise(Mean_abs_r = mean(Mean_abs_r, na.rm = TRUE), .groups = "drop") %>%
      mutate(Tissue = factor(Tissue, levels = names(TISSUE_COLORS)))
    
    panel_c <- corr_summary %>%
      ggplot(aes(x = Tissue, y = Mean_abs_r, fill = Tissue)) +
      geom_col(width = 0.7, show.legend = FALSE) +
      scale_fill_manual(values = TISSUE_COLORS) +
      labs(
        x = NULL,
        y = "Mean |Correlation| with Stage",
        title = "c"
      ) +
      nature_theme() +
      theme(
        plot.title = element_text(size = 8, face = "bold", hjust = 0),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
        aspect.ratio = 0.8
      )
  } else {
    # Fallback: simple placeholder
    panel_c <- ggplot() +
      annotate("text", x = 0.5, y = 0.5, label = "Correlation data not available", size = 3) +
      labs(title = "c") +
      nature_theme() +
      theme(plot.title = element_text(size = 8, face = "bold", hjust = 0))
  }
  
  fig_s2 <- (panel_a) / (panel_b | panel_c) +
    plot_layout(heights = c(1.2, 1))
  
  return(fig_s2)
}

# ==============================================================================
# FIGURE S3: EXTENDED PERFORMANCE METRICS
# ==============================================================================

create_fig_s3 <- function() {
  cat("Creating Figure S3: Extended performance metrics...\n")
  
  # Load data
  results_list <- load_all_ml_results()
  per_class_file <- project_path("paper/figures/output/stats/supplementary_per_class_metrics.csv")
  
  if (!file.exists(per_class_file)) {
    stop("Per-class metrics file not found. Run extract_supplementary_metrics.py first.")
  }
  
  per_class <- read_csv(per_class_file, show_col_types = FALSE)
  
  # Panel A: Per-class precision/recall/F1
  panel_a <- per_class %>%
    pivot_longer(cols = c(Precision, Recall, F1), names_to = "Metric", values_to = "Value") %>%
    mutate(
      Tissue = factor(Tissue, levels = TISSUE_COLORS %>% names()),
      Metric = factor(Metric, levels = c("Precision", "Recall", "F1"))
    ) %>%
    ggplot(aes(x = Stage, y = Value, fill = Tissue)) +
    geom_col(position = "dodge", width = 0.7) +
    facet_wrap(~ Metric, nrow = 1) +
    scale_fill_manual(values = TISSUE_COLORS) +
    labs(
      x = "Developmental Stage",
      y = "Score",
      title = "a"
    ) +
    nature_theme() +
    theme(
      plot.title = element_text(size = 8, face = "bold", hjust = 0),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
      legend.position = "bottom",
      strip.text = element_text(size = 6)
    )
  
  # Panel B: Test performance summary (already in main figure, but show here for completeness)
  perf_data <- extract_performance_metrics(results_list)
  
  panel_b <- perf_data %>%
    mutate(Tissue = factor(Tissue, levels = TISSUE_COLORS %>% names())) %>%
    select(Tissue, Balanced_Accuracy, F1_macro) %>%
    pivot_longer(cols = -Tissue, names_to = "Metric", values_to = "Value") %>%
    ggplot(aes(x = Tissue, y = Value, fill = Tissue)) +
    geom_col(position = "dodge", width = 0.7, show.legend = FALSE) +
    facet_wrap(~ Metric, nrow = 1, scales = "free_y") +
    scale_fill_manual(values = TISSUE_COLORS) +
    labs(
      x = NULL,
      y = "Score",
      title = "b"
    ) +
    nature_theme() +
    theme(
      plot.title = element_text(size = 8, face = "bold", hjust = 0),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
      strip.text = element_text(size = 6)
    )
  
  fig_s3 <- panel_a / panel_b +
    plot_layout(heights = c(1.2, 1))
  
  return(fig_s3)
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

cat("\nGenerating supplementary figures...\n\n")

# Create output directory
output_dir <- project_path("paper/figures/output/supplementary")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Generate figures
fig_s1 <- create_fig_s1()
fig_s2 <- create_fig_s2()
fig_s3 <- create_fig_s3()

  # Save figures
  cat("\nSaving figures...\n")
  save_figure(fig_s1, "figS1_cross_species_sensitivity", 
              width = 183, height = 90, output_dir = file.path(output_dir, ".."))
  save_figure(fig_s2, "figS2_tissue_specificity_stability", 
              width = 183, height = 200, output_dir = file.path(output_dir, ".."))
  save_figure(fig_s3, "figS3_extended_performance", 
              width = 183, height = 150, output_dir = file.path(output_dir, ".."))

cat("\n✓ All supplementary figures generated successfully!\n")
