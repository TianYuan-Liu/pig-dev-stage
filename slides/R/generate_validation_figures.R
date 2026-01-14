#!/usr/bin/env Rscript
# Generate Validation Figures for Slides
# Shows evidence that top genes capture real biological signal
#
# Figures generated:
# - fig4_panel_a_expression_correlation.png - % genes correlated with stage
# - fig4_panel_b_correlation_heatmap.png - Top gene correlation heatmap
# - fig4_panel_c_top_gene_trajectories.png - Expression trajectories
# - fig4_panel_d_feature_stability.png - Stability analysis summary

# Set working directory to script location
if (interactive()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

# Load required packages
suppressPackageStartupMessages({
  library(tidyverse)
  library(jsonlite)
  library(viridis)
  library(patchwork)
  library(scales)
  library(data.table)
})

# Source slides theme
source("slides_theme.R")

# Define paths
base_dir <- "../.."
analysis_dir <- file.path(base_dir, "machine_learning/analysis/results")
ml_dir <- file.path(base_dir, "machine_learning/model_outputs")
data_dir <- file.path(base_dir, "data/pigGTEx")
output_dir <- "../Figures"

# Standard Figure Dimensions
FIG_WIDTH <- 10
FIG_HEIGHT <- 6
FIG_DPI <- 200

cat("=== Generating Validation Figures ===\n")
cat("Output directory:", output_dir, "\n\n")

# ============================================================================
# LOAD DATA
# ============================================================================

cat("Loading analysis results...\n")

# Load expression correlation results
corr_file <- file.path(analysis_dir, "expression_correlation_results.json")
if (file.exists(corr_file)) {
  corr_results <- fromJSON(corr_file)
  cat("  Loaded expression correlation results\n")
} else {
  stop("Expression correlation results not found. Run expression_correlation.py first.")
}

# Load feature stability results
stability_file <- file.path(analysis_dir, "feature_stability_results.json")
if (file.exists(stability_file)) {
  stability_results <- fromJSON(stability_file)
  cat("  Loaded feature stability results\n")
} else {
  cat("  Warning: Feature stability results not found\n")
  stability_results <- NULL
}

# ============================================================================
# FIGURE 4A: Expression vs Age Scatter Plot (Up/Down Regulated Genes)
# ============================================================================

cat("\nGenerating fig4_panel_a_expression_correlation.png...\n")

# Load metadata to get actual ages
cat("  Loading metadata for age information...\n")
library(readxl)
metadata <- read_excel(file.path(base_dir, "data/PigGTEx_v0.MetaTable.xlsx"))

# Parse age to days
parse_age_to_days <- function(age_str) {
  if (is.na(age_str) || tolower(trimws(age_str)) == "unknown") {
    return(NA)
  }
  age_str <- tolower(trimws(age_str))

  # Match patterns like "91 days", "6 months", "4 weeks", "1 year"
  match <- regmatches(age_str, regexec("([0-9.]+)\\s*(day|days|week|weeks|month|months|year|years)", age_str))[[1]]
  if (length(match) < 3) {
    return(NA)
  }

  value <- as.numeric(match[2])
  unit <- match[3]

  if (grepl("day", unit)) {
    return(value)
  }
  if (grepl("week", unit)) {
    return(value * 7)
  }
  if (grepl("month", unit)) {
    return(value * 30)
  }
  if (grepl("year", unit)) {
    return(value * 365)
  }
  return(NA)
}

metadata$Age_Days <- sapply(metadata$Age, parse_age_to_days)
metadata <- metadata %>% filter(!is.na(Age_Days), Age_Days > 0)
cat("  Found", nrow(metadata), "samples with valid age data\n")

# Function to load expression data for a tissue
load_tissue_expression <- function(tissue_name) {
  file_path <- file.path(base_dir, "data/pigGTEx", paste0(tissue_name, ".expr_tpm.txt.gz"))
  if (!file.exists(file_path)) {
    # Try with underscore
    file_path <- file.path(base_dir, "data/pigGTEx", paste0(gsub(" ", "_", tissue_name), ".expr_tpm.txt.gz"))
  }
  if (!file.exists(file_path)) {
    return(NULL)
  }

  expr <- fread(file_path, sep = "\t", header = TRUE, check.names = FALSE)
  gene_col <- names(expr)[1]
  genes <- expr[[gene_col]]
  expr <- as.data.frame(expr[, -1])
  rownames(expr) <- genes
  return(expr)
}

# Collect scatter plot data for selected tissues and a few representative genes
scatter_data <- list()
tissues_to_plot <- c("Muscle", "Brain") # Focus on 2 tissues for clarity

for (tissue in tissues_to_plot) {
  cat("  Processing", tissue, "...\n")

  # Get top genes for this tissue from correlation results
  if (!tissue %in% names(corr_results)) next

  gene_corrs <- corr_results[[tissue]]$gene_correlations %>% as.data.frame()

  # Get top 3 up-regulated and top 3 down-regulated genes
  top_up <- gene_corrs %>%
    filter(direction == "increasing", significant == TRUE) %>%
    arrange(desc(spearman_rho)) %>%
    head(3)

  top_down <- gene_corrs %>%
    filter(direction == "decreasing", significant == TRUE) %>%
    arrange(spearman_rho) %>%
    head(3)

  top_genes <- bind_rows(top_up, top_down)

  if (nrow(top_genes) == 0) next

  # Load expression data
  expr_data <- load_tissue_expression(tissue)
  if (is.null(expr_data)) {
    cat("    Could not load expression data\n")
    next
  }

  # Get tissue samples from metadata
  tissue_meta <- metadata %>%
    filter(`Tissue class` == tissue) %>%
    filter(BioSample %in% colnames(expr_data))

  if (nrow(tissue_meta) < 10) {
    cat("    Not enough samples with age data\n")
    next
  }

  # Extract expression for top genes
  for (i in seq_len(nrow(top_genes))) {
    gene <- top_genes$gene[i]
    direction <- top_genes$direction[i]
    rho <- top_genes$spearman_rho[i]

    if (!gene %in% rownames(expr_data)) next

    expr_values <- as.numeric(expr_data[gene, tissue_meta$BioSample])

    scatter_data <- c(scatter_data, list(data.frame(
      Tissue = tissue,
      Gene = gene,
      Gene_Short = substr(gene, nchar(gene) - 4, nchar(gene)),
      Expression = expr_values,
      Age_Days = tissue_meta$Age_Days,
      Direction = ifelse(direction == "increasing", "Up-regulated", "Down-regulated"),
      Correlation = rho
    )))
  }
}

if (length(scatter_data) > 0) {
  scatter_df <- bind_rows(scatter_data)

  # Remove zeros and negative values for log scale
  scatter_df <- scatter_df %>%
    filter(Expression > 0, Age_Days > 0)

  # Define colors
  direction_colors <- c(
    "Up-regulated" = "#E74C3C", # Red
    "Down-regulated" = "#3498DB" # Blue
  )

  # Create scatter plot: Expression (x) vs Age (y, log scale)
  fig4a <- ggplot(scatter_df, aes(x = Expression, y = Age_Days, color = Direction)) +
    geom_point(alpha = 0.5, size = 2) +
    geom_smooth(method = "lm", se = FALSE, linewidth = 1.2) +
    facet_grid(Tissue ~ Direction, scales = "free_x") +
    scale_y_log10(
      labels = scales::comma,
      breaks = c(1, 7, 30, 100, 365, 1000)
    ) +
    scale_x_continuous(labels = scales::comma) +
    scale_color_manual(values = direction_colors) +
    labs(
      title = "Gene Expression Correlates with Age",
      subtitle = "Top correlated genes: x-axis = expression (TPM), y-axis = age (days, log scale)",
      x = "Gene Expression (TPM)",
      y = "Age (days, log scale)"
    ) +
    slides_theme() +
    theme(
      legend.position = "none",
      strip.text = element_text(size = 11, face = "bold"),
      panel.spacing = unit(1, "lines")
    )

  save_slide_figure(fig4a, file.path(output_dir, "fig4_panel_a_expression_correlation.png"))
  cat("  Saved scatter plot with", nrow(scatter_df), "data points\n")
} else {
  cat("  Warning: Could not generate scatter plot, falling back to summary plot\n")
  # Fallback: create a simple summary plot
  fig4a <- ggplot(data.frame(x = 1, y = 1), aes(x, y)) +
    annotate("text", x = 1, y = 1, label = "Expression vs Age scatter plot\nrequires raw data", size = 5) +
    slides_theme()
  save_slide_figure(fig4a, file.path(output_dir, "fig4_panel_a_expression_correlation.png"))
}

# Create summary statistics for reference
corr_summary <- data.frame(
  Tissue = names(corr_results),
  pct_significant = sapply(corr_results, function(x) x$pct_significant),
  mean_correlation = sapply(corr_results, function(x) x$mean_abs_correlation),
  n_strong = sapply(corr_results, function(x) x$n_strong_correlation)
)

# ============================================================================
# FIGURE 4B: Correlation Coefficient Summary
# ============================================================================

cat("Generating fig4_panel_b_correlation_heatmap.png...\n")

# Create summary of mean correlation per tissue
corr_summary2 <- corr_summary %>%
  mutate(
    label = sprintf("%.2f", mean_correlation),
    category = case_when(
      mean_correlation >= 0.5 ~ "Strong",
      mean_correlation >= 0.3 ~ "Moderate",
      TRUE ~ "Weak"
    )
  )

fig4b <- ggplot(corr_summary2, aes(x = Tissue, y = 1)) +
  geom_tile(aes(fill = mean_correlation), color = "white", linewidth = 2) +
  geom_text(aes(label = label), size = 6, fontface = "bold", color = "white") +
  scale_fill_viridis_c(
    option = "plasma", limits = c(0.3, 0.6),
    name = "Mean |r|", oob = squish
  ) +
  labs(
    title = "Mean Correlation Strength by Tissue",
    subtitle = "Average |Spearman correlation| between top gene expression and developmental stage",
    x = NULL,
    y = NULL
  ) +
  slides_theme() +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid = element_blank(),
    legend.position = "right"
  )

save_slide_figure(fig4b, file.path(output_dir, "fig4_panel_b_correlation_heatmap.png"))

# ============================================================================
# FIGURE 4C: Top Gene Trajectories
# ============================================================================

cat("Generating fig4_panel_c_top_gene_trajectories.png...\n")

# Extract top 5 genes per tissue with their stage means
trajectory_data <- list()

for (tissue in names(corr_results)) {
  gene_corrs <- corr_results[[tissue]]$gene_correlations

  # Get top 5 genes by absolute correlation
  top_genes <- gene_corrs %>%
    as.data.frame() %>%
    arrange(desc(abs_rho)) %>%
    head(5)

  for (i in 1:nrow(top_genes)) {
    gene_row <- top_genes[i, ]
    stage_means <- gene_row$stage_means

    if (!is.null(stage_means) && length(stage_means) > 0) {
      for (stage_name in names(stage_means)) {
        trajectory_data <- c(trajectory_data, list(data.frame(
          Tissue = tissue,
          Gene = gene_row$gene,
          Stage = as.integer(stage_name),
          Expression = stage_means[[stage_name]],
          Correlation = gene_row$spearman_rho,
          Direction = gene_row$direction
        )))
      }
    }
  }
}

trajectory_df <- bind_rows(trajectory_data)

# Map stage numbers to labels
stage_labels <- c("0" = "Infant", "1" = "Early", "2" = "Pre-pub", "3" = "Post-pub", "4" = "Adult")
trajectory_df <- trajectory_df %>%
  mutate(Stage_Label = stage_labels[as.character(Stage)])

# Select just 2 tissues for clarity
tissues_to_plot <- c("Muscle", "Brain")
trajectory_subset <- trajectory_df %>%
  filter(Tissue %in% tissues_to_plot) %>%
  # Get top 3 genes per tissue
  group_by(Tissue) %>%
  filter(Gene %in% head(unique(Gene[order(-abs(Correlation))]), 3)) %>%
  ungroup()

# Shorten gene names for display
trajectory_subset <- trajectory_subset %>%
  mutate(Gene_Short = substr(Gene, nchar(Gene) - 4, nchar(Gene)))

fig4c <- ggplot(
  trajectory_subset,
  aes(x = Stage, y = Expression, color = Gene_Short, group = Gene)
) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  facet_wrap(~Tissue, scales = "free_y", ncol = 2) +
  scale_color_viridis_d(option = "turbo", name = "Gene") +
  scale_x_continuous(breaks = 0:4, labels = c("Infant", "Early", "Pre-pub", "Post-pub", "Adult")) +
  labs(
    title = "Top Gene Expression Across Developmental Stages",
    subtitle = "Top 3 genes by correlation strength per tissue (showing clear monotonic trends)",
    x = "Developmental Stage",
    y = "Mean Expression (log2 TPM)"
  ) +
  slides_theme() +
  theme(
    legend.position = "right",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

save_slide_figure(fig4c, file.path(output_dir, "fig4_panel_c_top_gene_trajectories.png"))

# ============================================================================
# FIGURE 4D: Feature Stability Summary
# ============================================================================

cat("Generating fig4_panel_d_feature_stability.png...\n")

if (!is.null(stability_results)) {
  # Extract stability data
  stability_summary <- data.frame(
    Tissue = names(stability_results),
    mean_jaccard = sapply(stability_results, function(x) x$jaccard_similarity$mean),
    std_jaccard = sapply(stability_results, function(x) x$jaccard_similarity$std),
    n_stable = sapply(stability_results, function(x) x$n_stable_genes),
    n_core = sapply(stability_results, function(x) x$n_core_genes)
  )

  # Order by mean jaccard
  stability_summary <- stability_summary %>%
    arrange(desc(mean_jaccard)) %>%
    mutate(Tissue = factor(Tissue, levels = Tissue))

  # Plot 1: Jaccard similarity
  p1 <- ggplot(stability_summary, aes(x = Tissue, y = mean_jaccard, fill = Tissue)) +
    geom_col(width = 0.7, show.legend = FALSE) +
    geom_errorbar(
      aes(
        ymin = mean_jaccard - std_jaccard,
        ymax = mean_jaccard + std_jaccard
      ),
      width = 0.2, linewidth = 0.8
    ) +
    geom_hline(yintercept = 0.3, linetype = "dashed", color = cardiff_colors$red) +
    scale_fill_manual(values = tissue_colors) +
    scale_y_continuous(limits = c(0, 0.6), expand = c(0, 0)) +
    labs(
      title = "Feature Stability Across Seeds",
      x = NULL,
      y = "Mean Jaccard Similarity"
    ) +
    annotate("text",
      x = 0.7, y = 0.32, label = "Stability threshold",
      color = cardiff_colors$red, size = 3, hjust = 0
    ) +
    slides_theme()

  # Plot 2: Number of stable genes
  p2 <- ggplot(stability_summary, aes(x = Tissue, y = n_stable, fill = Tissue)) +
    geom_col(width = 0.7, show.legend = FALSE) +
    geom_text(aes(label = n_stable), vjust = -0.5, size = 4, fontface = "bold") +
    scale_fill_manual(values = tissue_colors) +
    scale_y_continuous(limits = c(0, 40), expand = c(0, 0, 0.1, 0)) +
    labs(
      title = "Stable Genes (>80% across seeds)",
      x = NULL,
      y = "Number of Genes"
    ) +
    slides_theme()

  # Combine plots
  fig4d <- p1 + p2 +
    plot_annotation(
      title = "Feature Selection Stability Analysis",
      subtitle = "Testing if same genes are selected across different random seeds"
    ) &
    theme(plot.title = element_text(size = 14, face = "bold"))
} else {
  # Create placeholder if no stability data
  fig4d <- ggplot() +
    annotate("text",
      x = 0.5, y = 0.5, label = "Stability analysis not available",
      size = 6, color = cardiff_colors$dark_grey
    ) +
    labs(title = "Feature Selection Stability Analysis") +
    slides_theme() +
    theme(
      axis.text = element_blank(),
      axis.title = element_blank()
    )
}

save_slide_figure(fig4d, file.path(output_dir, "fig4_panel_d_feature_stability.png"))

# ============================================================================
# FIGURE 4E: Correlation Distribution (Optional)
# ============================================================================

cat("Generating fig4_panel_e_correlation_distribution.png...\n")

# Extract all correlation values
all_corrs <- list()
for (tissue in names(corr_results)) {
  gene_corrs <- corr_results[[tissue]]$gene_correlations %>%
    as.data.frame() %>%
    mutate(Tissue = tissue)
  all_corrs <- c(all_corrs, list(gene_corrs))
}
all_corrs_df <- bind_rows(all_corrs)

# Order tissues by mean correlation
tissue_order <- all_corrs_df %>%
  group_by(Tissue) %>%
  summarize(mean_r = mean(abs(spearman_rho), na.rm = TRUE)) %>%
  arrange(desc(mean_r)) %>%
  pull(Tissue)

all_corrs_df <- all_corrs_df %>%
  mutate(Tissue = factor(Tissue, levels = tissue_order))

fig4e <- ggplot(all_corrs_df, aes(x = Tissue, y = abs(spearman_rho), fill = Tissue)) +
  geom_violin(scale = "width", alpha = 0.7, show.legend = FALSE) +
  geom_boxplot(width = 0.2, fill = "white", outlier.size = 1) +
  geom_hline(yintercept = 0.3, linetype = "dashed", color = cardiff_colors$red) +
  scale_fill_manual(values = tissue_colors) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(
    title = "Distribution of Stage Correlations",
    subtitle = "Absolute Spearman correlation for top 50 genes per tissue",
    x = NULL,
    y = "|Spearman r|"
  ) +
  annotate("text",
    x = 5.5, y = 0.32, label = "Significance threshold (0.3)",
    color = cardiff_colors$red, size = 3, hjust = 1
  ) +
  slides_theme()

save_slide_figure(fig4e, file.path(output_dir, "fig4_panel_e_correlation_distribution.png"))

# ============================================================================
# SUMMARY
# ============================================================================

cat("\n=== Validation Figures Generated ===\n")
cat("Output directory:", output_dir, "\n")
cat("Figures created:\n")
cat("  - fig4_panel_a_expression_correlation.png\n")
cat("  - fig4_panel_b_correlation_heatmap.png\n")
cat("  - fig4_panel_c_top_gene_trajectories.png\n")
cat("  - fig4_panel_d_feature_stability.png\n")
cat("  - fig4_panel_e_correlation_distribution.png\n")
