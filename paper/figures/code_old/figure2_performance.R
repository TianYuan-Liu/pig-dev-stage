#!/usr/bin/env Rscript
# ==============================================================================
# Figure 2: Classification Performance
# Standalone script - no external source() dependencies
# ==============================================================================

# ==============================================================================
# 1. PACKAGE LOADING
# ==============================================================================
required_packages <- c(
  "tidyverse", "patchwork", "viridis", "jsonlite",
  "reshape2", "scales", "readxl", "grid"
)

load_required_packages <- function(packages) {
  missing <- packages[!sapply(packages, requireNamespace, quietly = TRUE)]
  if (length(missing) > 0) {
    stop(
      "Missing required packages: ", paste(missing, collapse = ", "),
      "\nInstall with: install.packages(c('", paste(missing, collapse = "', '"), "'))"
    )
  }
  invisible(lapply(packages, library, character.only = TRUE, warn.conflicts = FALSE))
}

load_required_packages(required_packages)

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
BASE_DIR <- "/Users/tianyuan/Desktop/github_dev/pig-dev-stage"
OUTPUT_DIR <- file.path(BASE_DIR, "paper/figures/out")
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Tissues with model results
TISSUES_WITH_MODELS <- c("Muscle", "Brain", "Liver", "Blood", "Lung")

# Stage definitions
STAGE_ORDER <- c(
  "Infant_0_20d", "Early childhood_21_59d", "Pre_pubertal_60_149d",
  "Post_pubertal_150_365d", "Adult_>365d"
)

# ==============================================================================
# 3. INLINED UTILITY FUNCTIONS
# ==============================================================================

# --- Nature theme ---
nature_theme <- function(base_size = 8, base_family = "sans") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      plot.title = element_text(size = base_size + 2, face = "bold", color = "black"),
      plot.subtitle = element_text(size = base_size, color = "black"),
      axis.text = element_text(size = base_size, color = "black"),
      axis.title = element_text(size = base_size, color = "black"),
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.ticks = element_line(color = "black", linewidth = 0.5),
      legend.text = element_text(size = base_size - 1),
      legend.title = element_text(size = base_size, face = "bold"),
      legend.key.size = unit(0.8, "lines"),
      legend.background = element_blank(),
      legend.key = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(size = base_size, face = "bold"),
      plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")
    )
}

# --- Metadata loader ---
load_pig_metadata <- function() {
  metadata_path <- file.path(BASE_DIR, "data", "PigGTEx_v0.MetaTable.xlsx")

  if (!file.exists(metadata_path)) {
    stop("Metadata file not found: ", metadata_path)
  }

  raw <- readxl::read_excel(metadata_path)

  parse_age_days <- function(age_value) {
    if (is.na(age_value)) {
      return(NA_real_)
    }
    if (is.numeric(age_value)) {
      return(as.numeric(age_value))
    }

    age_str <- tolower(trimws(as.character(age_value)))
    if (age_str == "" || grepl("unknown", age_str)) {
      return(NA_real_)
    }

    value <- suppressWarnings(as.numeric(gsub("[^0-9.]+", "", age_str)))
    if (is.na(value)) {
      return(NA_real_)
    }

    if (grepl("day", age_str)) {
      return(value)
    }
    if (grepl("week", age_str)) {
      return(value * 7)
    }
    if (grepl("month", age_str)) {
      return(value * 30)
    }
    if (grepl("year", age_str)) {
      return(value * 365)
    }

    NA_real_
  }

  assign_stage <- function(age_days) {
    if (is.na(age_days)) {
      return(NA_character_)
    }
    if (age_days <= 20) {
      return("Infant_0_20d")
    }
    if (age_days <= 59) {
      return("Early childhood_21_59d")
    }
    if (age_days <= 149) {
      return("Pre_pubertal_60_149d")
    }
    if (age_days <= 365) {
      return("Post_pubertal_150_365d")
    }
    "Adult_>365d"
  }

  metadata <- raw %>%
    dplyr::rename(
      Sample_ID = BioSample,
      Tissue = `Main categories`,
      Tissue_detail = `Sub categories`,
      Tissue_class = `Tissue class`,
      Sex = Sex,
      Age_raw = Age
    ) %>%
    dplyr::mutate(
      Age_days = vapply(Age_raw, parse_age_days, numeric(1)),
      Stage = vapply(Age_days, assign_stage, character(1))
    ) %>%
    dplyr::select(Sample_ID, Tissue, Tissue_detail, Tissue_class, Sex, Age_raw, Age_days, Stage)

  metadata
}

# ==============================================================================
# 4. DATA LOADING WITH ERROR HANDLING
# ==============================================================================
load_data <- function() {
  cat("Loading data...\n")

  # Load metadata
  metadata <- tryCatch(
    load_pig_metadata() %>%
      filter(!is.na(Stage), !is.na(Age_days), Tissue != "Unknown"),
    error = function(e) {
      warning("Failed to load metadata: ", e$message)
      NULL
    }
  )

  if (is.null(metadata)) {
    stop("Cannot proceed without metadata")
  }

  cat(sprintf("  Loaded %d samples from metadata\n", nrow(metadata)))

  # Load model results
  results_list <- list()
  for (tissue in TISSUES_WITH_MODELS) {
    path <- file.path(BASE_DIR, "machine_learning/model_outputs", paste0(tissue, "_results.json"))
    if (file.exists(path)) {
      results_list[[tissue]] <- fromJSON(path)
      cat(sprintf("  Loaded results for %s\n", tissue))
    }
  }

  list(metadata = metadata, results = results_list)
}

# ==============================================================================
# 5. PANEL CREATION FUNCTIONS
# ==============================================================================

# Panel A: Performance Metrics Heatmap with Confidence Intervals
create_panel_a <- function(results_list) {
  cat("Creating Panel A: Performance metrics heatmap...\n")

  # Extract performance metrics
  performance_data <- map_dfr(names(results_list), function(tissue) {
    res <- results_list[[tissue]]
    metrics <- res$metrics
    bootstrap <- metrics$bootstrap

    tibble(
      Tissue = tissue,
      Scheme = res$scheme,
      `Balanced Accuracy` = metrics$balanced_accuracy,
      BA_CI_lower = if (!is.null(bootstrap$balanced_accuracy)) bootstrap$balanced_accuracy$ci_lower else NA,
      BA_CI_upper = if (!is.null(bootstrap$balanced_accuracy)) bootstrap$balanced_accuracy$ci_upper else NA,
      `F1-macro` = metrics$f1_macro,
      F1_CI_lower = if (!is.null(bootstrap$f1_macro)) bootstrap$f1_macro$ci_lower else NA,
      F1_CI_upper = if (!is.null(bootstrap$f1_macro)) bootstrap$f1_macro$ci_upper else NA,
      MAE = if (!is.null(metrics$mae)) metrics$mae else NA,
      MAE_CI_lower = if (!is.null(bootstrap$mae)) bootstrap$mae$ci_lower else NA,
      MAE_CI_upper = if (!is.null(bootstrap$mae)) bootstrap$mae$ci_upper else NA,
      `Spearman` = if (!is.null(metrics$spearman_r)) metrics$spearman_r else NA
    )
  })

  # Order tissues by scheme and performance
  performance_data <- performance_data %>%
    arrange(Scheme, desc(`Balanced Accuracy`))

  performance_data$Tissue <- factor(performance_data$Tissue,
    levels = performance_data$Tissue
  )

  # Create heatmap data (long format)
  heatmap_data <- performance_data %>%
    select(Tissue, `Balanced Accuracy`, `F1-macro`, MAE, Spearman) %>%
    pivot_longer(cols = -Tissue, names_to = "Metric", values_to = "Value")

  # Create labels with CI
  ci_labels <- performance_data %>%
    mutate(
      BA_label = sprintf(
        "%.3f\n(%.3f-%.3f)", `Balanced Accuracy`,
        coalesce(BA_CI_lower, `Balanced Accuracy`),
        coalesce(BA_CI_upper, `Balanced Accuracy`)
      ),
      F1_label = sprintf(
        "%.3f\n(%.3f-%.3f)", `F1-macro`,
        coalesce(F1_CI_lower, `F1-macro`),
        coalesce(F1_CI_upper, `F1-macro`)
      ),
      MAE_label = ifelse(!is.na(MAE),
        sprintf(
          "%.3f\n(%.3f-%.3f)", MAE,
          coalesce(MAE_CI_lower, MAE),
          coalesce(MAE_CI_upper, MAE)
        ), ""
      ),
      Spearman_label = ifelse(!is.na(Spearman), sprintf("%.3f", Spearman), "")
    ) %>%
    select(Tissue, BA_label, F1_label, MAE_label, Spearman_label) %>%
    pivot_longer(cols = -Tissue, names_to = "Metric", values_to = "Label")

  ci_labels$Metric <- factor(ci_labels$Metric,
    levels = c("BA_label", "F1_label", "MAE_label", "Spearman_label"),
    labels = c("Balanced Accuracy", "F1-macro", "MAE", "Spearman")
  )

  heatmap_data$Metric <- factor(heatmap_data$Metric,
    levels = c("Balanced Accuracy", "F1-macro", "MAE", "Spearman")
  )

  # Create heatmap
  ggplot(heatmap_data, aes(x = Metric, y = Tissue, fill = Value)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(
      data = ci_labels,
      aes(x = Metric, y = Tissue, label = Label),
      inherit.aes = FALSE,
      size = 2.5, color = "black"
    ) +
    scale_fill_gradient2(
      low = "#d73027", mid = "#fee090", high = "#1a9850",
      midpoint = 0.5, limits = c(0, 1),
      name = "Score",
      na.value = "gray90"
    ) +
    labs(
      x = NULL, y = NULL
    ) +
    nature_theme() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
      axis.text.y = element_text(size = 10),
      plot.title = element_text(size = 10, face = "bold"),
      plot.subtitle = element_text(size = 8)
    )
}

# Panel B: Confusion Matrices Grid
create_panel_b <- function(results_list) {
  cat("Creating Panel B: Confusion matrices...\n")

  confusion_plots <- list()

  for (tissue in names(results_list)) {
    res <- results_list[[tissue]]
    cm <- res$metrics$confusion_matrix
    n_classes <- nrow(cm)

    # Convert to percentage by row
    cm_pct <- sweep(cm, 1, rowSums(cm), FUN = "/") * 100
    cm_pct[is.nan(cm_pct)] <- 0

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
    adjacent_pct <- ifelse(total_errors > 0, adjacent_errors / total_errors * 100, 100)

    # Define stage labels based on number of classes
    if (n_classes == 4) {
      stage_labels <- c("Inf", "EC", "Pre", "Post")
    } else if (n_classes == 3) {
      stage_labels <- c("Early", "Pre", "Post")
    } else {
      stage_labels <- c("Young", "Mature")
    }

    # Create confusion matrix dataframe
    cm_df <- melt(cm_pct)
    colnames(cm_df) <- c("True", "Predicted", "Percentage")
    cm_df$True <- factor(cm_df$True, labels = stage_labels)
    cm_df$Predicted <- factor(cm_df$Predicted, labels = stage_labels)

    # Create plot
    p <- ggplot(cm_df, aes(x = Predicted, y = True, fill = Percentage)) +
      geom_tile(color = "white", linewidth = 0.5) +
      geom_text(aes(label = sprintf("%.0f%%", Percentage)),
        size = 2.5, color = "black"
      ) +
      scale_fill_gradient2(
        low = "white", mid = "#fee090", high = "#1a9850",
        midpoint = 50, limits = c(0, 100)
      ) +
      labs(
        title = tissue,
        subtitle = sprintf("Adj err: %.0f%%", adjacent_pct),
        x = "Predicted", y = "True"
      ) +
      nature_theme() +
      theme(
        legend.position = "none",
        plot.title = element_text(size = 9, face = "bold"),
        plot.subtitle = element_text(size = 7),
        axis.text = element_text(size = 7),
        axis.title = element_text(size = 8)
      )

    confusion_plots[[tissue]] <- p
  }

  # Combine in grid
  wrap_plots(confusion_plots, ncol = 3) +
    plot_annotation(
      theme = theme(
        plot.title = element_blank(),
        plot.subtitle = element_blank()
      )
    )
}

# Panel C: Age vs Stage Correlation Scatter Plots
create_panel_c <- function(metadata, results_list) {
  cat("Creating Panel C: Age vs stage correlation...\n")

  # Map stage to numeric
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

  # Select 4 representative tissues
  tissues_to_plot <- names(results_list)[1:min(4, length(results_list))]

  # Prepare correlation data
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
          Correlation = sprintf("rho = %.3f", cor_val)
        )
    } else {
      tibble()
    }
  }) %>%
    filter(!is.na(Stage_Numeric))

  if (nrow(correlation_data) == 0) {
    warning("No correlation data available for Panel C")
    return(ggplot() +
      theme_void() +
      labs(title = "No data available"))
  }

  # Create scatter plots
  ggplot(correlation_data, aes(x = Age_days, y = Stage_Numeric)) +
    geom_jitter(alpha = 0.5, width = 0, height = 0.1, size = 1.5, color = "#0173B2") +
    geom_smooth(method = "loess", se = TRUE, color = "#DE8F05", fill = "#DE8F05", alpha = 0.2) +
    geom_text(aes(label = Correlation),
      x = Inf, y = -Inf,
      hjust = 1.1, vjust = -0.5, size = 3, fontface = "bold"
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
      x = "Chronological Age (log scale)",
      y = "Developmental Stage"
    ) +
    nature_theme() +
    theme(
      plot.title = element_text(size = 10, face = "bold"),
      plot.subtitle = element_text(size = 8),
      strip.text = element_text(size = 9, face = "bold")
    )
}

# ==============================================================================
# 6. FIGURE ASSEMBLY AND SAVE
# ==============================================================================
create_figure <- function(data) {
  cat("Assembling figure...\n")

  panel_a <- create_panel_a(data$results)
  panel_b <- create_panel_b(data$results)
  panel_c <- create_panel_c(data$metadata, data$results)

  # Combine panels
  # Wrap panel_b to ensure it gets a single tag (B) instead of tagging individual plots
  panel_b_wrapped <- wrap_elements(panel_b)

  final_figure <- (panel_a | panel_b_wrapped) / panel_c +
    plot_layout(heights = c(1.2, 1)) +
    plot_annotation(
      tag_levels = "A",
      theme = theme(
        plot.background = element_rect(fill = "white", color = NA)
      )
    )

  # Save figure
  output_path <- file.path(OUTPUT_DIR, "figure2_performance")

  ggsave(paste0(output_path, ".pdf"), final_figure,
    width = 250, height = 200, units = "mm", dpi = 300
  )
  cat(sprintf("Saved: %s.pdf\n", output_path))

  ggsave(paste0(output_path, ".png"), final_figure,
    width = 250, height = 200, units = "mm", dpi = 300
  )
  cat(sprintf("Saved: %s.png\n", output_path))

  final_figure
}

# ==============================================================================
# 7. SUMMARY STATISTICS OUTPUT
# ==============================================================================
write_summary <- function(data) {
  cat("Writing summary statistics...\n")

  results <- data$results
  metadata <- data$metadata

  # Extract metrics for each tissue
  metrics_summary <- map_dfr(names(results), function(tissue) {
    res <- results[[tissue]]
    m <- res$metrics
    b <- m$bootstrap

    # Calculate adjacent error rate
    cm <- m$confusion_matrix
    n_classes <- nrow(cm)
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
    adj_err_pct <- ifelse(total_errors > 0, adjacent_errors / total_errors * 100, 100)

    tibble(
      Tissue = tissue,
      Scheme = res$scheme,
      Balanced_Accuracy = m$balanced_accuracy,
      BA_CI = sprintf(
        "(%.3f-%.3f)",
        if (!is.null(b$balanced_accuracy)) b$balanced_accuracy$ci_lower else m$balanced_accuracy,
        if (!is.null(b$balanced_accuracy)) b$balanced_accuracy$ci_upper else m$balanced_accuracy
      ),
      F1_macro = m$f1_macro,
      MAE = if (!is.null(m$mae)) m$mae else NA,
      Spearman_rho = if (!is.null(m$spearman_r)) m$spearman_r else NA,
      Adjacent_Error_Pct = adj_err_pct
    )
  })

  # Calculate correlations from metadata
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

  correlations <- metadata %>%
    filter(Tissue %in% names(results)) %>%
    mutate(Stage_Numeric = stage_to_numeric(Stage)) %>%
    filter(!is.na(Stage_Numeric), !is.na(Age_days)) %>%
    group_by(Tissue) %>%
    summarise(
      Age_Stage_Correlation = cor(log10(Age_days + 1), Stage_Numeric, method = "spearman"),
      .groups = "drop"
    )

  # Build summary text
  summary_text <- sprintf(
    "FIGURE 2: CLASSIFICATION PERFORMANCE
=====================================
Generated: %s
Script: figure2_performance.R

================================================================================
INPUT DATA SOURCES
================================================================================
- Model results: %d tissues (%s)
- Metadata: %d samples with age information

================================================================================
KEY STATISTICS
================================================================================

PANEL A: PERFORMANCE METRICS
----------------------------
%s

Mean Balanced Accuracy: %.3f
Best performing tissue: %s (%.3f)
Worst performing tissue: %s (%.3f)

PANEL B: CONFUSION MATRICES
---------------------------
Adjacent error rates (errors within +/-1 stage as %% of all errors):
%s

Mean adjacent error rate: %.1f%%

PANEL C: AGE-STAGE CORRELATIONS
-------------------------------
Spearman correlation between log(age) and developmental stage:
%s

================================================================================
FIGURE SPECIFICATIONS
================================================================================
- Output file: figure2_performance.pdf
- Dimensions: 250mm x 200mm
- DPI: 300
- Panels: A (metrics heatmap), B (confusion matrices), C (age correlation)
",
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    length(results),
    paste(names(results), collapse = ", "),
    nrow(metadata %>% filter(!is.na(Age_days))),
    paste(
      sprintf(
        "  %s (%s): BA=%.3f %s, F1=%.3f, MAE=%s, Spearman=%s",
        metrics_summary$Tissue,
        metrics_summary$Scheme,
        metrics_summary$Balanced_Accuracy,
        metrics_summary$BA_CI,
        metrics_summary$F1_macro,
        ifelse(is.na(metrics_summary$MAE), "N/A", sprintf("%.3f", metrics_summary$MAE)),
        ifelse(is.na(metrics_summary$Spearman_rho), "N/A", sprintf("%.3f", metrics_summary$Spearman_rho))
      ),
      collapse = "\n"
    ),
    mean(metrics_summary$Balanced_Accuracy),
    metrics_summary$Tissue[which.max(metrics_summary$Balanced_Accuracy)],
    max(metrics_summary$Balanced_Accuracy),
    metrics_summary$Tissue[which.min(metrics_summary$Balanced_Accuracy)],
    min(metrics_summary$Balanced_Accuracy),
    paste(sprintf("  %s: %.1f%%", metrics_summary$Tissue, metrics_summary$Adjacent_Error_Pct), collapse = "\n"),
    mean(metrics_summary$Adjacent_Error_Pct),
    paste(sprintf("  %s: rho = %.3f", correlations$Tissue, correlations$Age_Stage_Correlation), collapse = "\n")
  )

  output_path <- file.path(OUTPUT_DIR, "figure2_summary.txt")
  writeLines(summary_text, output_path)
  cat(sprintf("Saved: %s\n", output_path))
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
main <- function() {
  cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
  cat("Figure 2: Classification Performance\n")
  cat("=" %>% rep(70) %>% paste(collapse = ""), "\n\n")

  # Load data
  data <- load_data()

  # Create figure
  create_figure(data)

  # Write summary
  write_summary(data)

  cat("\nFigure 2 completed successfully!\n")
}

# Run
main()
