#!/usr/bin/env Rscript
# Figure 2: Classification Performance
# Publication-quality figure showing model performance metrics

library(tidyverse)
library(patchwork)
library(viridis)
library(jsonlite)
library(reshape2)
library(scales)

# Set publication theme
args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg)) else getwd()
base_dir <- dirname(script_dir)
source(file.path(script_dir, "theme_configs", "nature_theme.R"))
source(file.path(script_dir, "utils", "metadata_utils.R"))

# Output directory
output_dir <- "."
dir.create(output_dir, showWarnings = FALSE)

# Load model results
tissues <- c(
  "Muscle", "Brain", "Liver", "Blood", "Small intestine",
  "Lung", "Adipose", "Testis"
)

results_list <- list()
for (tissue in tissues) {
  file_path <- file.path(base_dir, "machine_learning", "model_outputs", paste0(tissue, "_results.json"))
  if (file.exists(file_path)) {
    results_list[[tissue]] <- fromJSON(file_path)
  }
}

# Load metadata for age correlation
metadata <- load_pig_metadata() %>%
  filter(!is.na(Stage), !is.na(Age_days), Tissue != "Unknown")

# Panel A: Performance metrics heatmap with confidence intervals
create_panel_a <- function() {
  # Extract performance metrics
  performance_data <- map_dfr(names(results_list), function(tissue) {
    res <- results_list[[tissue]]
    metrics <- res$metrics
    bootstrap <- metrics$bootstrap

    # Get classification scheme
    scheme <- res$scheme

    tibble(
      Tissue = tissue,
      Scheme = scheme,
      `Balanced Accuracy` = metrics$balanced_accuracy,
      BA_CI_lower = bootstrap$balanced_accuracy$ci_lower,
      BA_CI_upper = bootstrap$balanced_accuracy$ci_upper,
      `F1-macro` = metrics$f1_macro,
      F1_CI_lower = bootstrap$f1_macro$ci_lower,
      F1_CI_upper = bootstrap$f1_macro$ci_upper,
      MAE = ifelse(!is.null(metrics$mae), metrics$mae, NA),
      MAE_CI_lower = ifelse(!is.null(bootstrap$mae), bootstrap$mae$ci_lower, NA),
      MAE_CI_upper = ifelse(!is.null(bootstrap$mae), bootstrap$mae$ci_upper, NA),
      `Spearman ρ` = ifelse(!is.null(metrics$spearman_r), metrics$spearman_r, NA)
    )
  })

  # Order by scheme and performance
  performance_data$Tissue <- factor(performance_data$Tissue,
    levels = performance_data$Tissue[order(
      performance_data$Scheme,
      -performance_data$`Balanced Accuracy`
    )]
  )

  # Create heatmap data
  heatmap_data <- performance_data %>%
    select(Tissue, `Balanced Accuracy`, `F1-macro`, MAE, `Spearman ρ`) %>%
    melt(id.vars = "Tissue", variable.name = "Metric", value.name = "Value")

  # Add confidence intervals as labels
  ci_labels <- performance_data %>%
    mutate(
      BA_label = sprintf(
        "%.2f\n(%.2f-%.2f)",
        `Balanced Accuracy`, BA_CI_lower, BA_CI_upper
      ),
      F1_label = sprintf(
        "%.2f\n(%.2f-%.2f)",
        `F1-macro`, F1_CI_lower, F1_CI_upper
      ),
      MAE_label = ifelse(!is.na(MAE),
        sprintf("%.3f\n(%.3f-%.3f)", MAE, MAE_CI_lower, MAE_CI_upper),
        ""
      ),
      Spearman_label = ifelse(!is.na(`Spearman ρ`),
        sprintf("%.3f", `Spearman ρ`), ""
      )
    ) %>%
    select(Tissue, BA_label, F1_label, MAE_label, Spearman_label) %>%
    melt(id.vars = "Tissue", variable.name = "Metric", value.name = "Label")

  ci_labels$Metric <- factor(ci_labels$Metric,
    levels = c("BA_label", "F1_label", "MAE_label", "Spearman_label"),
    labels = c("Balanced Accuracy", "F1-macro", "MAE", "Spearman ρ")
  )

  # Create heatmap
  ggplot(heatmap_data, aes(x = Metric, y = Tissue, fill = Value)) +
    geom_tile(color = "white", size = 0.5) +
    geom_text(
      data = ci_labels,
      aes(x = Metric, y = Tissue, label = Label),
      inherit.aes = FALSE,
      size = 3
    ) +
    scale_fill_gradient2(
      low = "#d73027", mid = "#fee090", high = "#1a9850",
      midpoint = 0.5, limits = c(0, 1),
      name = "Score",
      na.value = "gray90"
    ) +
    labs(
      title = "Model Performance Metrics",
      subtitle = "Values show mean (95% CI from bootstrap)",
      x = NULL, y = NULL
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(size = 12, face = "bold"),
      plot.subtitle = element_text(size = 10)
    ) +
    facet_grid(
      rows = vars(performance_data$Scheme[match(Tissue, performance_data$Tissue)]),
      scales = "free_y", space = "free_y"
    )
}

# Panel B: Confusion matrices showing ordinal errors
create_panel_b <- function() {
  # Extract confusion matrices for all tissues
  confusion_plots <- list()

  for (i in 1:length(tissues)) {
    tissue <- tissues[i]
    if (tissue %in% names(results_list)) {
      res <- results_list[[tissue]]
      cm <- res$metrics$confusion_matrix
      n_classes <- nrow(cm)

      # Convert to percentage
      cm_pct <- sweep(cm, 1, rowSums(cm), FUN = "/") * 100

      # Calculate adjacent error rate
      adjacent_errors <- 0
      total_errors <- 0
      for (row in 1:n_classes) {
        for (col in 1:n_classes) {
          if (row != col) {
            total_errors <- total_errors + cm[row, col]
            if (abs(row - col) == 1) {
              adjacent_errors <- adjacent_errors + cm[row, col]
            }
          }
        }
      }
      adjacent_pct <- ifelse(total_errors > 0,
        adjacent_errors / total_errors * 100, 100
      )

      # Create confusion matrix plot
      cm_df <- melt(cm_pct)
      colnames(cm_df) <- c("True", "Predicted", "Percentage")

      # Define stage labels based on number of classes
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
        geom_tile(color = "white", size = 1.2) +
        geom_text(aes(label = sprintf("%.0f%%", Percentage)),
          size = 5, color = "black", fontface = "bold"
        ) +
        scale_fill_gradient2(
          low = "white", mid = "#fee090", high = "#1a9850",
          midpoint = 50, limits = c(0, 100)
        ) +
        labs(
          title = tissue,
          subtitle = sprintf("±1 stage: %.1f%%", adjacent_pct),
          x = "Predicted", y = "True"
        ) +
        theme_minimal() +
        theme(
          legend.position = "none",
          plot.title = element_text(size = 14, face = "bold"),
          plot.subtitle = element_text(size = 11),
          axis.text = element_text(size = 11, face = "bold"),
          axis.title = element_text(size = 12, face = "bold"),
          panel.grid = element_blank(),
          plot.margin = margin(10, 10, 10, 10)
        )

      confusion_plots[[i]] <- p
    }
  }

  confusion_plots <- purrr::compact(confusion_plots)
  if (length(confusion_plots) == 0) {
    stop("No confusion matrices available to plot.")
  }

  # Combine confusion matrices in 2x4 grid
  wrap_plots(confusion_plots, ncol = min(4, length(confusion_plots)))
}

# Panel C: Correlation with chronological age
create_panel_c <- function() {
  # Map stage labels to numeric values
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

  # Prepare data for correlation plots
  correlation_data <- map_dfr(tissues[1:4], function(tissue) { # Show 4 representative tissues
    # Get predictions from results
    if (tissue %in% names(results_list)) {
      # Use metadata to get age-stage relationships
      tissue_data <- metadata %>%
        filter(Tissue == tissue) %>%
        mutate(
          Stage_Numeric = stage_to_numeric(Stage),
          Age_log = log10(Age_days + 1)
        ) %>%
        filter(!is.na(Stage_Numeric))

      if (nrow(tissue_data) > 0) {
        # Calculate correlation
        cor_val <- cor(tissue_data$Age_log, tissue_data$Stage_Numeric,
          method = "spearman", use = "complete.obs"
        )

        tissue_data %>%
          mutate(
            Tissue = tissue,
            Correlation = sprintf("ρ = %.3f", cor_val)
          ) %>%
          sample_n(min(100, n())) # Subsample for clarity
      }
    }
  }) %>%
    filter(!is.na(Stage_Numeric))

  # Create scatter plots
  ggplot(correlation_data, aes(x = Age_days, y = Stage_Numeric)) +
    geom_jitter(alpha = 0.5, width = 0, height = 0.1, size = 1) +
    geom_smooth(method = "loess", se = TRUE, color = "#809BCE") +
    geom_text(aes(label = Correlation),
      x = Inf, y = -Inf,
      hjust = 1.1, vjust = -0.5, size = 3.5
    ) +
    scale_x_log10(
      breaks = c(1, 10, 30, 100, 300, 1000),
      labels = c("1d", "10d", "1mo", "3mo", "10mo", "3yr")
    ) +
    scale_y_continuous(
      breaks = 1:5,
      labels = c("Infant", "Early", "Pre-pub", "Post-pub", "Adult")
    ) +
    facet_wrap(~Tissue, ncol = 2, scales = "free_x") +
    labs(
      title = "Chronological Age vs Developmental Stage",
      subtitle = "Showing Spearman correlation coefficients",
      x = "Chronological Age (log scale)",
      y = "Developmental Stage"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 12, face = "bold"),
      plot.subtitle = element_text(size = 10)
    )
}

# Combine all panels
panel_a <- create_panel_a()
panel_b <- create_panel_b()
panel_c <- create_panel_c()

# Create final figure
final_figure <- panel_a / panel_b / panel_c +
  plot_layout(heights = c(1, 1.8, 1)) +
  plot_annotation(
    tag_levels = "A",
    theme = theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 12)
    )
  )

# Save figure
ggsave(file.path(output_dir, "figure2_performance.pdf"),
  final_figure,
  width = 14, height = 14, dpi = 300
)
ggsave(file.path(output_dir, "figure2_performance.png"),
  final_figure,
  width = 14, height = 14, dpi = 300
)

cat("Figure 2 saved to:", output_dir, "\n")
