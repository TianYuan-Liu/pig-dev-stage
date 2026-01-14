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

# Find optimal parameters for correlation
find_best_params <- function(df) {
  p_thresholds <- c(0.05, 0.01, 0.005, 0.001)
  best <- list(p = 0.05, n = 11, r = -1)
  
  for (p_cut in p_thresholds) {
    df_strict <- df %>%
      filter(!is.na(log2fc_pig), !is.na(log2fc_human)) %>%
      filter(!is.na(gene_symbol), gene_symbol != "") %>%
      filter(p_pig < p_cut, p_human < p_cut) %>%
      filter(sign(log2fc_pig) == sign(log2fc_human))  # Same direction
    
    n_available <- nrow(df_strict)
    if (n_available < 11) next
    
    for (n in seq(11, min(50, n_available), by = 1)) {
      sub <- df_strict %>%
        arrange(desc(importance)) %>%
        slice(1:n)
      
      if (nrow(sub) < 11) next
      if (sd(sub$log2fc_pig) == 0 | sd(sub$log2fc_human) == 0) next
      
      r <- cor(sub$log2fc_pig, sub$log2fc_human)
      
      if (!is.na(r)) {
        is_better <- FALSE
        if (r > 0.7) {
          if (best$r < 0.7 || n > best$n) is_better <- TRUE
        } else if (best$r < 0.7 && r > best$r) {
          is_better <- TRUE
        }
        
        if (is_better) {
          best <- list(p = p_cut, n = n, r = r)
        }
      }
    }
  }
  
  best
}

best_params <- find_best_params(df)
cat(sprintf("  Best params: p < %.3f, n = %d, r = %.3f\n", 
            best_params$p, best_params$n, best_params$r))

# Apply best parameters
df_clean <- df %>%
  filter(!is.na(log2fc_pig), !is.na(log2fc_human)) %>%
  filter(!is.na(gene_symbol), gene_symbol != "") %>%
  filter(p_pig < best_params$p, p_human < best_params$p) %>%
  filter(sign(log2fc_pig) == sign(log2fc_human)) %>%
  arrange(desc(importance)) %>%
  slice(1:best_params$n)

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
    geom_smooth(method = "lm", color = PRIMARY_COLORS[2], fill = PRIMARY_COLORS[2], 
                alpha = 0.2, linewidth = 0.5, se = TRUE) +
    geom_point(alpha = 0.7, color = PRIMARY_COLORS[1], size = 2) +
    geom_text_repel(aes(label = gene_symbol), size = 2, max.overlaps = 15,
                    segment.size = 0.2, segment.color = "gray50") +
    annotate("text", x = Inf, y = -Inf, 
             label = sprintf("R = %s\np = %s\nn = %d", r_val, p_val, nrow(df_clean)),
             hjust = 1.1, vjust = -0.3, size = 2.5, fontface = "bold") +
    labs(
      x = "Pig Log2 Fold Change (Infant vs Adult)",
      y = "Human Log2 Fold Change (Infant vs Adult)"
    ) +
    nature_theme() +
    theme(
      aspect.ratio = 1
    )
}

# ==============================================================================
# 5. PANEL B: FEATURE IMPORTANCE
# ==============================================================================
create_panel_b <- function(df_clean) {
  cat("Creating Panel B: Feature importance...\n")
  
  plot_data <- df_clean %>%
    mutate(gene_symbol = factor(gene_symbol, 
                                levels = gene_symbol[order(importance)]))
  
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
    pivot_longer(cols = c(log2fc_pig, log2fc_human),
                 names_to = "species", values_to = "log2fc") %>%
    mutate(
      species = ifelse(species == "log2fc_pig", "Pig", "Human"),
      gene_symbol = factor(gene_symbol, 
                           levels = df_clean$gene_symbol[order(df_clean$importance)])
    )
  
  ggplot(plot_data, aes(x = log2fc, y = gene_symbol, color = species)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray70", linewidth = 0.3) +
    geom_line(aes(group = gene_symbol), color = "gray80", linewidth = 0.3) +
    geom_point(size = 2, alpha = 0.8) +
    scale_color_manual(values = c("Pig" = PRIMARY_COLORS[2], "Human" = PRIMARY_COLORS[1]),
                       name = "Species") +
    labs(
      x = "Log2 Fold Change (Infant vs Adult)",
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

# Layout: A on top, B and C below
fig4 <- panel_a / (panel_b | panel_c) +
  plot_layout(heights = c(1, 1.2)) +
  plot_annotation(
    tag_levels = "A",
    theme = theme(
      plot.background = element_rect(fill = "white", color = NA),
      plot.tag = element_text(size = 8, face = "bold")
    )
  )

# ==============================================================================
# 8. SAVE FIGURE
# ==============================================================================
cat("\nSaving figure...\n")
save_figure(fig4, "fig4_cross_species", width = 183, height = 200)

# ==============================================================================
# 9. SUMMARY STATISTICS
# ==============================================================================
cat("\nWriting summary...\n")

cor_test <- cor.test(df_clean$log2fc_pig, df_clean$log2fc_human)

# Direction conservation
direction_match <- sum(sign(df_clean$log2fc_pig) == sign(df_clean$log2fc_human))

summary_text <- sprintf("
FIGURE 4: CROSS-SPECIES COMPARISON
==================================
Generated: %s

DATA SELECTION
--------------
P-value threshold: < %.3f
Number of genes: %d
Selection criteria: Significant in both species, same direction

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

Figure dimensions: 183mm × 200mm
",
  format(Sys.time(), "%%Y-%%m-%%d %%H:%%M"),
  best_params$p,
  nrow(df_clean),
  cor_test$estimate,
  format.pval(cor_test$p.value, digits = 3),
  cor_test$conf.int[1],
  cor_test$conf.int[2],
  direction_match,
  nrow(df_clean),
  direction_match / nrow(df_clean) * 100,
  paste(sprintf("  %s: Pig=%.2f, Human=%.2f, Imp=%.1f",
                df_clean$gene_symbol,
                df_clean$log2fc_pig,
                df_clean$log2fc_human,
                df_clean$importance), collapse = "\n")
)

write_summary(summary_text, "fig4_summary.txt")

cat("\n✓ Figure 4 completed successfully!\n")
