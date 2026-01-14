#!/usr/bin/env Rscript
# ==============================================================================
# Figure 1: Study Framework and Methodology
# Standalone script - no external source() dependencies
# ==============================================================================

# ==============================================================================
# 1. PACKAGE LOADING
# ==============================================================================
required_packages <- c(
  "tidyverse", "patchwork", "viridis", "jsonlite",
  "scales", "readxl", "grid"
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

# Only tissues with model results
TISSUES_WITH_MODELS <- c("Muscle", "Liver", "Lung", "Blood", "Brain")

# Stage definitions
STAGE_ORDER <- c(
  "Infant_0_20d", "Early childhood_21_59d", "Pre_pubertal_60_149d",
  "Post_pubertal_150_365d", "Adult_>365d"
)

STAGE_COLORS <- c(
  "Infant_0_20d" = "#f5fbff",
  "Early childhood_21_59d" = "#dae9f6",
  "Pre_pubertal_60_149d" = "#b9d6eb",
  "Post_pubertal_150_365d" = "#8bbde5",
  "Adult_>365d" = "#5f9ed1"
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
      filter(!is.na(Stage), !is.na(Tissue), Tissue != "Unknown"),
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
    } else {
      warning(sprintf("Results file not found for %s: %s", tissue, path))
    }
  }

  list(metadata = metadata, results = results_list)
}

# ==============================================================================
# 5. PANEL CREATION FUNCTIONS
# ==============================================================================

# Panel A1: Variable Width Timeline
create_timeline_plot <- function() {
  cat("Creating Panel A1: Timeline...\n")
  # Create timeline with VARIABLE WIDTH days
  timeline_data <- tibble::tibble(
    stage = STAGE_ORDER,
    display = c("Infant", "Early\nchildhood", "Pre-\npubertal", "Post-\npubertal", "Adult"),
    start = c(0, 21, 60, 150, 365),
    end = c(20, 59, 149, 365, 500),
    y = 1
  )

  ggplot(timeline_data) +
    geom_rect(aes(
      xmin = start, xmax = end, ymin = 0.8, ymax = 1.2,
      fill = stage
    ), color = "white") +
    geom_text(aes(x = (start + end) / 2, y = 1.5, label = display),
      size = 3.5, fontface = "bold", color = "black"
    ) +
    geom_text(
      aes(
        x = (start + end) / 2, y = 0.5,
        label = paste0(start, "-", ifelse(end == 500, ">365", end), "d")
      ),
      size = 3, color = "black"
    ) +
    scale_x_continuous(
      breaks = c(0, 60, 150, 365, 500),
      labels = c("Birth", "2 mo", "5 mo", "1 yr", ">1 yr")
    ) +
    scale_fill_manual(values = STAGE_COLORS) +
    labs(x = NULL, y = NULL) +
    nature_theme(base_size = 10) +
    theme(
      legend.position = "none",
      axis.text.x = element_text(color = "black", size = 9),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      axis.line.y = element_blank(),
      axis.title.y = element_blank(),
      panel.grid = element_blank(),
      plot.margin = margin(10, 10, 5, 10)
    ) +
    coord_cartesian(ylim = c(0, 3.0), clip = "off")
}

# Panel A2: Sample Distribution Heatmap
create_heatmap_plot <- function(metadata, results_list) {
  cat("Creating Panel A2: Heatmap...\n")
  # FILTER to only tissues with model results
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

  # Order tissues by total sample count
  tissue_totals <- stage_counts %>%
    group_by(Tissue) %>%
    summarise(total = sum(n)) %>%
    arrange(desc(total))

  stage_counts$Tissue <- factor(stage_counts$Tissue, levels = tissue_totals$Tissue)

  ggplot(stage_counts, aes(x = Stage, y = Tissue, fill = n)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = n, color = n > 300), size = 3, fontface = "bold") +
    scale_color_manual(values = c("FALSE" = "white", "TRUE" = "black"), guide = "none") +
    scale_fill_viridis(name = "Samples", option = "C") +
    scale_x_discrete(labels = c(
      "Infant\n0-20d", "Early\n21-59d",
      "Pre-pub\n60-149d", "Post-pub\n150-365d",
      "Adult\n>365d"
    )) +
    labs(x = NULL, y = "Tissue") +
    nature_theme(base_size = 10) +
    theme(
      axis.text.x = element_text(angle = 0, hjust = 0.5, size = 9),
      axis.text.y = element_text(size = 10),
      legend.position = "right",
      plot.margin = margin(0, 5, 5, 5)
    )
}

# Panel B: Classification Schemes Bar Chart
create_panel_b <- function(results_list) {
  cat("Creating Panel B: Classification schemes...\n")

  if (length(results_list) == 0) {
    stop("No model results available for Panel B")
  }

  # Extract classification data from results
  classification_data <- map_dfr(names(results_list), function(tissue) {
    res <- results_list[[tissue]]
    tibble(
      Tissue = res$tissue,
      Scheme = res$scheme,
      n_samples = res$n_samples
    )
  }) %>%
    filter(!is.na(Tissue))

  if (nrow(classification_data) == 0) {
    stop("No valid classification data extracted from model results")
  }

  classification_data$Scheme <- factor(classification_data$Scheme,
    levels = c("4-class", "3-class", "2-class")
  )

  # Create bar plot
  ggplot(classification_data, aes(
    x = reorder(Tissue, -n_samples),
    y = n_samples, fill = Scheme
  )) +
    geom_col(width = 0.7) +
    geom_text(aes(label = n_samples), vjust = -0.5, size = 4, fontface = "bold") +
    scale_fill_manual(
      values = c("4-class" = "#2166AC", "3-class" = "#67A9CF", "2-class" = "#D1E5F0"),
      name = "Classification\nScheme"
    ) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(
      x = "Tissue", y = "Number of Samples"
    ) +
    nature_theme(base_size = 10) +
    theme(
      axis.text.x = element_text(angle = 0, hjust = 0.5, size = 11),
      axis.text.y = element_text(size = 10),
      legend.position = c(0.8, 0.8),
      plot.margin = margin(10, 10, 10, 10)
    )
}

# Panel C: Machine Learning Workflow (Professional Design)
create_panel_c <- function() {
  cat("Creating Panel C: ML Workflow...\n")

  # Define nodes with coordinates
  nodes <- tibble(
    id = 1:6,
    x = c(0, 0, 0, 0, 0, 0),
    y = c(6, 5, 4, 3, 2, 1),
    label = c(
      "Pig GTEx Data\n(Gene Exp + Metadata)",
      "Stage Harmonization\n(Adaptive Granularity)",
      "Data Splitting\n(70% Train / 30% Test)",
      "Preprocessing & QC\n(Log2, Filter, Z-score)",
      "Model Training\n(LightGBM + Feat. Sel.)",
      "Evaluation\n(Metrics on Test Set)"
    ),
    type = c("Data", "Process", "Process", "Process", "Model", "Output"),
    xmin = -0.4, xmax = 0.4,
    ymin = c(5.6, 4.6, 3.6, 2.6, 1.6, 0.6),
    ymax = c(6.4, 5.4, 4.4, 3.4, 2.4, 1.4)
  )

  # Define arrows
  arrows <- tibble(
    x = 0, xend = 0,
    y = c(5.6, 4.6, 3.6, 2.6, 1.6),
    yend = c(5.4, 4.4, 3.4, 2.4, 1.4)
  )

  ggplot() +
    # Draw arrows
    geom_segment(
      data = arrows,
      aes(x = x, xend = xend, y = y, yend = yend),
      arrow = arrow(length = unit(0.2, "cm"), type = "closed"),
      color = "grey60", linewidth = 0.6
    ) +
    # Draw simple icons (shapes) - Left aligned
    geom_point(
      data = nodes,
      aes(x = -0.35, y = y, shape = type, color = type),
      size = 5, stroke = 1.2
    ) +
    # Add text - Left aligned next to icons
    geom_text(
      data = nodes,
      aes(x = -0.28, y = y, label = label),
      size = 3.2, fontface = "bold", color = "black", hjust = 0
    ) +
    scale_shape_manual(values = c(
      "Data" = 16, # Circle
      "Process" = 15, # Square
      "Model" = 17, # Triangle
      "Output" = 18 # Diamond
    ), guide = "none") +
    scale_color_manual(values = c(
      "Data" = "#1565C0",
      "Process" = "#2E7D32",
      "Model" = "#EF6C00",
      "Output" = "#6A1B9A"
    ), guide = "none") +
    scale_x_continuous(limits = c(-0.5, 0.5)) +
    scale_y_continuous(limits = c(0.5, 6.5)) +
    labs(x = NULL, y = NULL) +
    nature_theme(base_size = 10) +
    theme(
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      axis.line = element_blank(),
      panel.grid = element_blank(),
      plot.margin = margin(5, 5, 5, 5)
    )
}

# ==============================================================================
# 6. FIGURE ASSEMBLY AND SAVE
# ==============================================================================
create_figure <- function(data) {
  cat("Assembling figure...\n")

  panel_timeline <- create_timeline_plot()
  panel_heatmap <- create_heatmap_plot(data$metadata, data$results)
  panel_b <- create_panel_b(data$results)
  panel_c <- create_panel_c()

  # Define layout design (A=Timeline, B=Heatmap, C=Bar, D=Workflow)
  # A is taller than standard to show labels clearly
  # B is large for heatmap visibility
  design <- "
    AAAA
    BBBB
    BBBB
    BBBB
    CCDD
    CCDD
  "

  # Combine plots with explicit design
  final_figure <- wrap_plots(
    A = panel_timeline,
    B = panel_heatmap,
    C = panel_b,
    D = panel_c,
    design = design
  ) +
    plot_annotation(
      tag_levels = "A",
      theme = theme(
        plot.background = element_rect(fill = "white", color = NA),
        plot.margin = margin(10, 10, 10, 10)
      )
    )

  # Save figure
  output_path <- file.path(OUTPUT_DIR, "figure1_methodology")

  ggsave(paste0(output_path, ".pdf"), final_figure,
    width = 220, height = 200, units = "mm", dpi = 300
  )
  cat(sprintf("Saved: %s.pdf\n", output_path))

  ggsave(paste0(output_path, ".png"), final_figure,
    width = 220, height = 200, units = "mm", dpi = 300
  )
  cat(sprintf("Saved: %s.png\n", output_path))

  final_figure
}

# ==============================================================================
# 7. SUMMARY STATISTICS OUTPUT
# ==============================================================================
write_summary <- function(data) {
  cat("Writing summary statistics...\n")

  metadata <- data$metadata
  results <- data$results

  # Filter metadata to only tissues with models
  metadata_filtered <- metadata %>% filter(Tissue %in% names(results))

  total_samples <- nrow(metadata_filtered)
  n_tissues_models <- length(results)

  # Samples per tissue (only modeled tissues)
  tissue_counts <- metadata_filtered %>%
    group_by(Tissue) %>%
    summarise(n = n()) %>%
    arrange(desc(n))

  # Samples per stage (only modeled tissues)
  stage_counts <- metadata_filtered %>%
    group_by(Stage) %>%
    summarise(n = n()) %>%
    mutate(Stage = factor(Stage, levels = STAGE_ORDER)) %>%
    arrange(Stage)

  # Classification schemes
  scheme_info <- map_dfr(names(results), function(tissue) {
    res <- results[[tissue]]
    tibble(
      Tissue = tissue,
      Scheme = res$scheme,
      N_samples = res$n_samples,
      N_classes = nrow(res$metrics$confusion_matrix)
    )
  })

  summary_text <- sprintf(
    "FIGURE 1: STUDY FRAMEWORK AND METHODOLOGY
==========================================
Generated: %s
Script: figure1_methodology.R

================================================================================
INPUT DATA SOURCES
================================================================================
- Metadata: data/PigGTEx_v0.MetaTable.xlsx
- Model results: %d tissues (%s)

================================================================================
KEY STATISTICS
================================================================================

PANEL A: SAMPLE DISTRIBUTION (Tissues with Models Only)
-------------------------------------------------------
Total samples in modeled tissues: %d
Tissues with models: %d

Samples per tissue:
%s

Samples per developmental stage:
%s

PANEL B: CLASSIFICATION SCHEMES
-------------------------------
%s

================================================================================
FIGURE SPECIFICATIONS
================================================================================
- Output file: figure1_methodology.pdf
- Dimensions: 220mm x 200mm
- DPI: 300
- Panels: A (timeline + heatmap), B (classification bar chart), C (ML workflow)
",
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    n_tissues_models,
    paste(names(results), collapse = ", "),
    total_samples,
    n_tissues_models,
    paste(sprintf("  %s: %d", tissue_counts$Tissue, tissue_counts$n), collapse = "\n"),
    paste(sprintf("  %s: %d", stage_counts$Stage, stage_counts$n), collapse = "\n"),
    paste(sprintf(
      "  %s: %s (%d samples, %d classes)",
      scheme_info$Tissue, scheme_info$Scheme,
      scheme_info$N_samples, scheme_info$N_classes
    ), collapse = "\n")
  )

  output_path <- file.path(OUTPUT_DIR, "figure1_summary.txt")
  writeLines(summary_text, output_path)
  cat(sprintf("Saved: %s\n", output_path))
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
main <- function() {
  cat("======================================================================\n")
  cat("Figure 1: Study Framework and Methodology\n")
  cat("======================================================================\n\n")

  # Load data
  data <- load_data()

  # Create figure
  create_figure(data)

  # Write summary
  write_summary(data)

  cat("\nFigure 1 completed successfully!\n")
}

# Run
main()
