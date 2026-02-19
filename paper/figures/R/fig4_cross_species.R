#!/usr/bin/env Rscript
# ==============================================================================
# Figure 4: Cross-Species Comparison (Pig vs Human)
# ==============================================================================
# Panels:
#   A - Cross-species fold change correlation scatter
#   B - Feature importance bar chart (top conserved markers)
#   C - Paired fold change comparison (lollipop/dotplot)
# ==============================================================================

# ==============================================================================
# 1. SETUP
# ==============================================================================
suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(ggrepel)
  library(viridis)
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
cat("Figure 4: Cross-Species Comparison\n")
cat("==============================================================================\n\n")

# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
cat("Loading data...\n")

stats_file <- project_path("paper/figures/output/stats/fig4_expression_stats.csv")

if (!file.exists(stats_file)) {
  cat("Expression stats not found. Run python script first:\n")
  cat("  python paper/figures/python/fig4_data_prep.py\n")
  stop("Missing data file: ", stats_file)
}

df <- read_csv(stats_file, show_col_types = FALSE)
cat(sprintf("  Loaded %d genes\n", nrow(df)))

# ==============================================================================
# 3. DATA PREPARATION
# ==============================================================================
cat("Preparing data...\n")

# Pre-specified criteria: FDR < 0.10 (BH-corrected) and |log2FC| > 0.5 in both species
df_clean <- df %>%
  filter(!is.na(log2fc_pig), !is.na(log2fc_human)) %>%
  filter(!is.na(gene_symbol), gene_symbol != "") %>%
  filter(fdr_pig < 0.10, fdr_human < 0.10) %>%
  filter(abs(log2fc_pig) > 0.5, abs(log2fc_human) > 0.5) %>%
  arrange(desc(importance))

cat(sprintf("  Selected %d genes for visualization\n", nrow(df_clean)))

# ==============================================================================
# 4. PANEL A: CROSS-SPECIES CORRELATION
# ==============================================================================
create_panel_a <- function(df_clean) {
  cat("Creating Panel A: Correlation scatter...\n")

  cor_test <- cor.test(df_clean$log2fc_pig, df_clean$log2fc_human)
  r_val <- round(cor_test$estimate, 3)
  p_val <- format.pval(cor_test$p.value, digits = 3)

  ggplot(df_clean, aes(x = log2fc_pig, y = log2fc_human)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray70", linewidth = 0.3) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray70", linewidth = 0.3) +
    geom_smooth(
      method = "lm", color = PRIMARY_COLORS[2], fill = PRIMARY_COLORS[2],
      alpha = 0.2, linewidth = 0.5, se = TRUE
    ) +
    geom_point(alpha = 0.7, color = PRIMARY_COLORS[1], size = 2) +
    geom_text_repel(aes(label = gene_symbol),
      size = 2, max.overlaps = 15,
      segment.size = 0.2, segment.color = "gray50"
    ) +
    annotate("text",
      x = Inf, y = -Inf,
      label = sprintf("R = %s\np = %s", r_val, p_val),
      hjust = 1.1, vjust = -0.3, size = 2.5, fontface = "bold"
    ) +
    labs(
      x = "Pig Log2 Fold Change (Adult/Infant)",
      y = "Human Log2 Fold Change (Adult/Infant)"
    ) +
    nature_theme() +
    theme(
      aspect.ratio = 0.5
    )
}

# ==============================================================================
# 5. PANEL B: FEATURE IMPORTANCE
# ==============================================================================
create_panel_b <- function(df_clean) {
  cat("Creating Panel B: Feature importance...\n")

  plot_data <- df_clean %>%
    mutate(gene_symbol = factor(gene_symbol,
      levels = gene_symbol[order(importance)]
    ))

  ggplot(plot_data, aes(x = gene_symbol, y = log10(importance + 1))) +
    geom_col(aes(fill = log10(importance + 1)), width = 0.7, show.legend = FALSE) +
    coord_flip() +
    scale_fill_viridis_c(option = "magma") +
    labs(
      x = NULL,
      y = "Log10(Feature Importance)"
    ) +
    nature_theme() +
    theme(
      axis.text.y = element_text(size = 5)
    )
}

# ==============================================================================
# 6. PANEL C: PAIRED FOLD CHANGE COMPARISON
# ==============================================================================
create_panel_c <- function(df_clean) {
  cat("Creating Panel C: Fold change comparison...\n")

  # Reshape for paired plotting
  plot_data <- df_clean %>%
    select(gene_symbol, log2fc_pig, log2fc_human) %>%
    pivot_longer(
      cols = c(log2fc_pig, log2fc_human),
      names_to = "species", values_to = "log2fc"
    ) %>%
    mutate(
      species = ifelse(species == "log2fc_pig", "Pig", "Human"),
      gene_symbol = factor(gene_symbol,
        levels = df_clean$gene_symbol[order(df_clean$importance)]
      )
    )

  ggplot(plot_data, aes(x = log2fc, y = gene_symbol, color = species)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray70", linewidth = 0.3) +
    geom_line(aes(group = gene_symbol), color = "gray80", linewidth = 0.3) +
    geom_point(size = 2, alpha = 0.8) +
    scale_color_manual(
      values = c("Pig" = PRIMARY_COLORS[2], "Human" = PRIMARY_COLORS[1]),
      name = "Species"
    ) +
    labs(
      x = "Log2 Fold Change (Adult/Infant)",
      y = NULL
    ) +
    nature_theme() +
    theme(
      axis.text.y = element_text(size = 5),
      legend.position = "bottom",
      panel.border = element_rect(color = "gray80", fill = NA, linewidth = 0.3)
    )
}

# ==============================================================================
# 7. FIGURE ASSEMBLY
# ==============================================================================
cat("\nAssembling figure...\n")

panel_a <- create_panel_a(df_clean)
panel_b <- create_panel_b(df_clean)
panel_c <- create_panel_c(df_clean)

# Layout: A on top, B and C below (A is wide/short, B/C get more space)
fig4 <- panel_a / (panel_b | panel_c) +
  plot_layout(heights = c(0.8, 1.4)) +
  plot_annotation(
    tag_levels = list(c("a", "b", "c")),
    theme = theme(
      plot.background = element_rect(fill = "white", color = NA),
      plot.tag = element_text(size = 8, face = "bold")
    )
  )

# ==============================================================================
# 8. SAVE FIGURE
# ==============================================================================
cat("\nSaving figure...\n")
save_figure(fig4, "fig4_cross_species", width = 183, height = 250)

# ==============================================================================
# 9. SUMMARY STATISTICS
# ==============================================================================
cat("\nWriting summary...\n")

cor_test <- cor.test(df_clean$log2fc_pig, df_clean$log2fc_human)

# Direction conservation
direction_match <- sum(sign(df_clean$log2fc_pig) == sign(df_clean$log2fc_human))

summary_text <- sprintf(
  "
FIGURE 4: CROSS-SPECIES COMPARISON
==================================
Generated: %s

DATA SELECTION
--------------
FDR threshold: < 0.10 (Benjamini-Hochberg)
|log2FC| threshold: > 0.5
Number of genes: %d
Selection criteria: Pre-specified (FDR < 0.10 AND |log2FC| > 0.5 in both species)

CORRELATION ANALYSIS
--------------------
Pearson R: %.3f
P-value: %s
95%% CI: [%.3f, %.3f]

DIRECTIONAL CONSERVATION
------------------------
Genes with conserved direction: %d / %d (%.1f%%)

TOP CONSERVED MARKERS
---------------------
%s

Figure dimensions: 183mm × 250mm
",
  format(Sys.time(), "%%Y-%%m-%%d %%H:%%M"),
  nrow(df_clean),
  cor_test$estimate,
  format.pval(cor_test$p.value, digits = 3),
  cor_test$conf.int[1],
  cor_test$conf.int[2],
  direction_match,
  nrow(df_clean),
  direction_match / nrow(df_clean) * 100,
  paste(sprintf(
    "  %s: Pig=%.2f, Human=%.2f, Imp=%.1f",
    df_clean$gene_symbol,
    df_clean$log2fc_pig,
    df_clean$log2fc_human,
    df_clean$importance
  ), collapse = "\n")
)

write_summary(summary_text, "fig4_summary.txt")

cat("\n✓ Figure 4 completed successfully!\n")
