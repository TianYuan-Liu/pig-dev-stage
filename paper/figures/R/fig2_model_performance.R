#!/usr/bin/env Rscript
# ==============================================================================
# Figure 2: Model Performance
# ==============================================================================
# Panels:
#   A - Performance metrics heatmap (with confidence intervals)
#   B - Confusion matrices grid
#   C - Age vs Stage correlation scatter plots
# ==============================================================================

# ==============================================================================
# 1. SETUP
# ==============================================================================
suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(cowplot)
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
cat("Figure 2: Model Performance\n")
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
# 3. PANEL A: PERFORMANCE METRICS HEATMAP
# ==============================================================================
create_panel_a <- function(results_list) {
  cat("Creating Panel A: Performance metrics...\n")

  # Extract metrics with confidence intervals
  performance_data <- extract_performance_metrics(results_list) %>%
    arrange(Scheme, desc(Balanced_Accuracy))

  performance_data$Tissue <- factor(
    performance_data$Tissue,
    levels = performance_data$Tissue
  )

  # Reshape for heatmap
  heatmap_data <- performance_data %>%
    select(Tissue, Balanced_Accuracy, F1_macro, MAE, Spearman_r) %>%
    rename(
      `Balanced\nAccuracy` = Balanced_Accuracy,
      `F1-macro` = F1_macro,
      `MAE` = MAE,
      `Spearman\nρ` = Spearman_r
    ) %>%
    pivot_longer(cols = -Tissue, names_to = "Metric", values_to = "Value")

  # Create labels with CI
  ci_labels <- performance_data %>%
    mutate(
      BA_label = sprintf(
        "%.2f\n(%.2f-%.2f)",
        Balanced_Accuracy,
        coalesce(BA_CI_lower, Balanced_Accuracy),
        coalesce(BA_CI_upper, Balanced_Accuracy)
      ),
      F1_label = sprintf(
        "%.2f\n(%.2f-%.2f)",
        F1_macro,
        coalesce(F1_CI_lower, F1_macro),
        coalesce(F1_CI_upper, F1_macro)
      ),
      MAE_label = ifelse(!is.na(MAE), sprintf("%.2f", MAE), "-"),
      Spearman_label = ifelse(!is.na(Spearman_r), sprintf("%.2f", Spearman_r), "-")
    ) %>%
    select(Tissue, BA_label, F1_label, MAE_label, Spearman_label) %>%
    pivot_longer(cols = -Tissue, names_to = "Metric", values_to = "Label")

  ci_labels$Metric <- factor(
    ci_labels$Metric,
    levels = c("BA_label", "F1_label", "MAE_label", "Spearman_label"),
    labels = c("Balanced\nAccuracy", "F1-macro", "MAE", "Spearman\nρ")
  )

  heatmap_data$Metric <- factor(
    heatmap_data$Metric,
    levels = c("Balanced\nAccuracy", "F1-macro", "MAE", "Spearman\nρ")
  )

  ggplot(heatmap_data, aes(x = Metric, y = Tissue, fill = Value)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(
      data = ci_labels,
      aes(x = Metric, y = Tissue, label = Label),
      inherit.aes = FALSE,
      size = 2.0, color = "black", lineheight = 0.85
    ) +
    scale_fill_performance(na.value = "gray90") +
    labs(x = NULL, y = NULL) +
    nature_theme() +
    theme(
      axis.text.x = element_text(angle = 0, hjust = 0.5, size = 5),
      legend.position = "bottom",
      legend.key.width = unit(10, "mm"),
      legend.key.height = unit(2.5, "mm")
    )
}

# ==============================================================================
# 4. PANEL B: CONFUSION MATRICES
# ==============================================================================
create_panel_b <- function(results_list) {
  cat("Creating Panel B: Confusion matrices...\n")

  confusion_plots <- list()

  for (tissue in names(results_list)) {
    res <- results_list[[tissue]]
    cm <- res$metrics$confusion_matrix
    n_classes <- nrow(cm)

    # Normalize by row
    cm_pct <- normalize_confusion_matrix(cm)
    adjacent_pct <- calc_adjacent_error_rate(cm)

    # Stage labels based on number of classes
    stage_labels <- if (n_classes == 4) {
      c("Inf", "EC", "Pre", "Post")
    } else if (n_classes == 3) {
      c("Early", "Pre", "Post")
    } else {
      c("Young", "Mature")
    }

    # Create dataframe
    cm_df <- as.data.frame(as.table(cm_pct))
    colnames(cm_df) <- c("True", "Predicted", "Percentage")
    cm_df$Percentage <- as.numeric(cm_df$Percentage)
    cm_df$True <- factor(cm_df$True, labels = stage_labels)
    cm_df$Predicted <- factor(cm_df$Predicted, labels = stage_labels)

    p <- ggplot(cm_df, aes(x = Predicted, y = True, fill = Percentage)) +
      geom_tile(color = "white", linewidth = 0.5) +
      geom_text(
        aes(label = sprintf("%.0f%%", Percentage)),
        size = 1.8, color = "black"
      ) +
      scale_fill_gradient2(
        low = "white", mid = "#FEE090", high = "#1A9850",
        midpoint = 50, limits = c(0, 100)
      ) +
      labs(
        title = tissue,
        subtitle = sprintf("Adj: %.0f%%", adjacent_pct),
        x = NULL, y = NULL
      ) +
      nature_theme() +
      theme(
        legend.position = "none",
        plot.title = element_text(size = 8, face = "bold"),
        plot.subtitle = element_text(size = 6, color = "gray40"),
        axis.text = element_text(size = 5),
        plot.margin = margin(1.5, 1.5, 1.5, 1.5, "mm")
      )

    confusion_plots[[tissue]] <- p
  }

  plot_grid(
    plotlist = confusion_plots,
    nrow = 2,
    ncol = 3,
    align = "hv"
  )
}

# ==============================================================================
# 5. PANEL C: AGE VS STAGE CORRELATION
# ==============================================================================
create_panel_c <- function(metadata, results_list) {
  cat("Creating Panel C: Age-stage correlation...\n")

  # Include all available tissues
  tissues_to_plot <- names(results_list)

  correlation_data <- map_dfr(tissues_to_plot, function(tissue) {
    tissue_data <- metadata %>%
      filter(Tissue == tissue) %>%
      mutate(
        Stage_Numeric = stage_to_numeric(Stage),
        Age_log = log10(Age_days + 1)
      ) %>%
      filter(!is.na(Stage_Numeric), !is.na(Age_log))

    if (nrow(tissue_data) > 0) {
      cor_val <- cor(tissue_data$Age_log, tissue_data$Stage_Numeric,
        method = "spearman", use = "complete.obs"
      )

      # Subsample for visualization
      if (nrow(tissue_data) > 100) {
        set.seed(42)
        tissue_data <- tissue_data %>% sample_n(100)
      }

      tissue_data %>%
        mutate(
          Tissue = tissue,
          Correlation = sprintf("ρ = %.2f", cor_val)
        )
    } else {
      tibble()
    }
  }) %>%
    filter(!is.na(Stage_Numeric))

  if (nrow(correlation_data) == 0) {
    return(ggplot() +
      theme_void() +
      labs(title = "No data"))
  }

  ggplot(correlation_data, aes(x = Age_days, y = Stage_Numeric)) +
    geom_jitter(alpha = 0.5, width = 0, height = 0.1, size = 1, color = PRIMARY_COLORS[1]) +
    geom_smooth(method = "loess", se = TRUE, color = PRIMARY_COLORS[2], fill = PRIMARY_COLORS[2], alpha = 0.2, linewidth = 0.5) +
    geom_text(
      aes(label = Correlation),
      x = Inf, y = -Inf,
      hjust = 1.1, vjust = -0.5, size = 2, fontface = "bold",
      check_overlap = TRUE
    ) +
    scale_x_log10(
      breaks = c(1, 10, 30, 100, 300, 1000),
      labels = c("1d", "10d", "1mo", "3mo", "10mo", "3yr")
    ) +
    scale_y_continuous(
      breaks = 1:5,
      labels = c("Infant", "Early", "Pre-pub", "Post-pub", "Adult")
    ) +
    facet_wrap(~Tissue, ncol = 3, scales = "free_x") +
    labs(
      x = "Chronological Age (log scale)",
      y = "Developmental Stage"
    ) +
    nature_theme() +
    theme(
      strip.text = element_text(size = 6, face = "bold")
    )
}

# ==============================================================================
# 6. FIGURE ASSEMBLY
# ==============================================================================
cat("\nAssembling figure...\n")

panel_a <- create_panel_a(results_list)
panel_b <- create_panel_b(results_list)
panel_c <- create_panel_c(metadata, results_list)

top_row <- plot_grid(
  panel_a,
  panel_b,
  nrow = 1,
  rel_widths = c(1.8, 2.2),
  labels = c("a", "b"),
  label_size = 6,
  label_fontface = "bold",
  label_x = c(0, 0),
  label_y = c(1, 1),
  hjust = 0,
  vjust = 1
)

fig2 <- plot_grid(
  top_row,
  panel_c,
  ncol = 1,
  rel_heights = c(1.1, 1.2),
  labels = c("", "c"),
  label_size = 6,
  label_fontface = "bold",
  label_x = c(0, 0),
  label_y = c(1, 1),
  hjust = 0,
  vjust = 1
)

# ==============================================================================
# 7. SAVE FIGURE
# ==============================================================================
cat("\nSaving figure...\n")
save_figure(fig2, "fig2_model_performance", width = 183, height = 170)

# ==============================================================================
# 8. SUMMARY STATISTICS
# ==============================================================================
cat("\nWriting summary...\n")

metrics_df <- extract_performance_metrics(results_list)

summary_text <- sprintf(
  "
FIGURE 2: MODEL PERFORMANCE
===========================
Generated: %s

PERFORMANCE METRICS
-------------------
%s

Mean Balanced Accuracy: %.3f
Best: %s (%.3f)
Worst: %s (%.3f)

ADJACENT ERROR RATES
--------------------
%s

AGE-STAGE CORRELATIONS
----------------------
%s

Figure dimensions: 183mm × 180mm
",
  format(Sys.time(), "%%Y-%%m-%%d %%H:%%M"),
  paste(capture.output(print(metrics_df, n = Inf)), collapse = "\n"),
  mean(metrics_df$Balanced_Accuracy),
  metrics_df$Tissue[which.max(metrics_df$Balanced_Accuracy)],
  max(metrics_df$Balanced_Accuracy),
  metrics_df$Tissue[which.min(metrics_df$Balanced_Accuracy)],
  min(metrics_df$Balanced_Accuracy),
  paste(map_chr(names(results_list), function(t) {
    cm <- results_list[[t]]$metrics$confusion_matrix
    sprintf("  %s: %.1f%%", t, calc_adjacent_error_rate(cm))
  }), collapse = "\n"),
  paste(map_chr(names(results_list)[1:min(4, length(results_list))], function(t) {
    tissue_data <- metadata %>%
      filter(Tissue == t, !is.na(Age_days), !is.na(Stage)) %>%
      mutate(Stage_Numeric = stage_to_numeric(Stage))
    if (nrow(tissue_data) > 0) {
      r <- cor(log10(tissue_data$Age_days + 1), tissue_data$Stage_Numeric,
        method = "spearman", use = "complete.obs"
      )
      sprintf("  %s: ρ = %.3f", t, r)
    } else {
      sprintf("  %s: N/A", t)
    }
  }), collapse = "\n")
)

write_summary(summary_text, "fig2_summary.txt")

cat("\n✓ Figure 2 completed successfully!\n")
