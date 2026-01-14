#!/usr/bin/env Rscript
# Generate Pathway Figures for Slides
# Uses results from run_pathway_analysis.R
#
# Figures generated:
# - fig5_panel_a_kegg_reactome_enrichment.png - Top pathways bubble plot
# - fig5_panel_b_pathway_overlap.png - Cross-tissue pathway overlap heatmap
# - fig5_panel_c_gene_vs_pathway_overlap.png - Comparison showing pathways share even without genes

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
})

# Source slides theme
source("slides_theme.R")

# Define paths
data_dir <- "../data"
output_dir <- "../Figures"

# Standard Figure Dimensions
FIG_WIDTH <- 10
FIG_HEIGHT <- 6
FIG_DPI <- 200

cat("=== Generating Pathway Figures ===\n")
cat("Output directory:", output_dir, "\n\n")

# ============================================================================
# LOAD DATA
# ============================================================================

cat("Loading pathway analysis results...\n")

# Load pathway enrichment results
enrichment_file <- file.path(data_dir, "pathway_enrichment_all.csv")
if (!file.exists(enrichment_file)) {
  stop("Pathway enrichment results not found. Run run_pathway_analysis.R first.")
}
all_enrichment <- read_csv(enrichment_file, show_col_types = FALSE)
cat("  Loaded", nrow(all_enrichment), "pathway results\n")

# Load overlap matrix
overlap_file <- file.path(data_dir, "pathway_overlap_matrix.csv")
overlap_matrix <- read.csv(overlap_file, row.names = 1)
cat("  Loaded pathway overlap matrix\n")

# Load pathway summary
summary_file <- file.path(data_dir, "pathway_summary.csv")
pathway_summary <- read_csv(summary_file, show_col_types = FALSE)
cat("  Loaded pathway summary\n")

# Load shared pathway counts
shared_counts_file <- file.path(data_dir, "pathway_shared_counts.csv")
shared_counts <- read.csv(shared_counts_file, row.names = 1)
cat("  Loaded shared pathway counts\n")

# Load pathway tissue counts
tissue_counts_file <- file.path(data_dir, "pathway_tissue_counts.csv")
pathway_tissue_counts <- read_csv(tissue_counts_file, show_col_types = FALSE)
cat("  Loaded pathway tissue counts\n")

# ============================================================================
# FIGURE 5A: Top KEGG/Reactome Pathways Bubble Plot
# ============================================================================

cat("\nGenerating fig5_panel_a_kegg_reactome_enrichment.png...\n")

# Get top pathways by significance for each tissue (focus on KEGG and Reactome)
top_pathways <- all_enrichment %>%
  filter(source %in% c("KEGG", "REAC")) %>%
  group_by(tissue) %>%
  slice_min(order_by = p_value, n = 10) %>%
  ungroup() %>%
  mutate(
    neg_log_p = -log10(p_value),
    # Shorten long pathway names
    term_short = ifelse(nchar(term_name) > 40,
                        paste0(substr(term_name, 1, 37), "..."),
                        term_name)
  )

# For a cleaner plot, get most significant pathways across all tissues
top_shared_pathways <- pathway_tissue_counts %>%
  filter(source %in% c("KEGG", "REAC")) %>%
  filter(n_tissues >= 2) %>%
  slice_min(order_by = mean_p_value, n = 15)

# Filter enrichment to these top shared pathways
plot_data_a <- all_enrichment %>%
  filter(term_id %in% top_shared_pathways$term_id) %>%
  mutate(
    neg_log_p = -log10(p_value),
    term_short = ifelse(nchar(term_name) > 45,
                        paste0(substr(term_name, 1, 42), "..."),
                        term_name),
    source_label = case_when(
      source == "KEGG" ~ "KEGG",
      source == "REAC" ~ "Reactome",
      TRUE ~ source
    )
  )

# Order pathways by number of tissues they appear in, then by mean p-value
pathway_order <- plot_data_a %>%
  group_by(term_short) %>%
  summarize(n_tissues = n(), mean_p = mean(p_value)) %>%
  arrange(desc(n_tissues), mean_p) %>%
  pull(term_short)

plot_data_a <- plot_data_a %>%
  mutate(term_short = factor(term_short, levels = rev(pathway_order)))

fig5a <- ggplot(plot_data_a, aes(x = tissue, y = term_short)) +
  geom_point(aes(size = gene_ratio, color = neg_log_p), alpha = 0.8) +
  scale_size_continuous(name = "Gene Ratio", range = c(2, 8)) +
  scale_color_viridis_c(option = "plasma", name = "-log10(p)") +
  labs(
    title = "Shared Developmental Pathways Across Tissues",
    subtitle = "KEGG and Reactome pathways enriched in 2+ tissues (top genes from each tissue)",
    x = NULL,
    y = NULL
  ) +
  slides_theme() +
  theme(
    axis.text.y = element_text(size = 9),
    legend.position = "right",
    panel.grid.major.x = element_blank()
  )

save_slide_figure(fig5a, file.path(output_dir, "fig5_panel_a_kegg_reactome_enrichment.png"),
                  width = 12, height = 8)

# ============================================================================
# FIGURE 5B: Pathway Overlap Heatmap
# ============================================================================

cat("Generating fig5_panel_b_pathway_overlap.png...\n")

# Convert overlap matrix to long format
overlap_long <- overlap_matrix %>%
  rownames_to_column("tissue1") %>%
  pivot_longer(-tissue1, names_to = "tissue2", values_to = "jaccard")

# Order tissues consistently
tissue_order <- c("Muscle", "Brain", "Liver", "Lung", "Blood")
overlap_long <- overlap_long %>%
  mutate(
    tissue1 = factor(tissue1, levels = tissue_order),
    tissue2 = factor(tissue2, levels = tissue_order)
  )

# Also prepare shared counts for annotation
shared_long <- shared_counts %>%
  rownames_to_column("tissue1") %>%
  pivot_longer(-tissue1, names_to = "tissue2", values_to = "n_shared") %>%
  mutate(
    tissue1 = factor(tissue1, levels = tissue_order),
    tissue2 = factor(tissue2, levels = tissue_order)
  )

# Merge for annotation
overlap_annotated <- left_join(overlap_long, shared_long, by = c("tissue1", "tissue2")) %>%
  mutate(label = sprintf("%.2f\n(%d)", jaccard, n_shared))

fig5b <- ggplot(overlap_annotated, aes(x = tissue2, y = tissue1)) +
  geom_tile(aes(fill = jaccard), color = "white", linewidth = 1) +
  geom_text(aes(label = label), size = 4, color = "white", fontface = "bold") +
  scale_fill_viridis_c(option = "magma", limits = c(0, 1), name = "Jaccard\nSimilarity") +
  labs(
    title = "Pathway Overlap Is High Despite Low Gene Overlap",
    subtitle = "Jaccard similarity of enriched pathways between tissues (shared pathway count)",
    x = NULL,
    y = NULL
  ) +
  coord_fixed() +
  slides_theme() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank(),
    legend.position = "right"
  )

save_slide_figure(fig5b, file.path(output_dir, "fig5_panel_b_pathway_overlap.png"))

# ============================================================================
# FIGURE 5C: Gene vs Pathway Overlap Comparison
# ============================================================================

cat("Generating fig5_panel_c_gene_vs_pathway_overlap.png...\n")

# Create comparison data
# Gene overlap is ~0.003 (from previous analysis)
# Pathway overlap is ~0.25 (mean from our matrix)

comparison_data <- data.frame(
  Type = c("Genes", "Pathways"),
  Overlap = c(0.003, mean(overlap_matrix[lower.tri(overlap_matrix)])),
  Label = c("0.3%", sprintf("%.0f%%", mean(overlap_matrix[lower.tri(overlap_matrix)]) * 100))
)

# Add context
comparison_data$Description <- c(
  "Different genes in each tissue",
  "Shared biological processes"
)

fig5c_bar <- ggplot(comparison_data, aes(x = Type, y = Overlap, fill = Type)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = Label), vjust = -0.5, size = 6, fontface = "bold") +
  scale_fill_manual(values = c("Genes" = cardiff_colors$red, "Pathways" = "#0072B2")) +
  scale_y_continuous(limits = c(0, 0.35), labels = percent, expand = c(0, 0, 0.1, 0)) +
  labs(
    title = "Gene vs Pathway Cross-Tissue Overlap",
    subtitle = "Mean Jaccard similarity across all tissue pairs",
    x = NULL,
    y = "Cross-Tissue Overlap"
  ) +
  slides_theme()

# Create text annotation panel
annotation_text <- paste0(
  "KEY FINDING:\n\n",
  "While tissues use DIFFERENT GENES\n",
  "to predict developmental stage,\n",
  "they share SIMILAR PATHWAYS.\n\n",
  "This confirms:\n",
  "1. Biology is real (not noise)\n",
  "2. Tissues have parallel development\n",
  "3. Same processes, different genes"
)

fig5c_text <- ggplot() +
  annotate("text", x = 0.5, y = 0.5, label = annotation_text,
           size = 4.5, hjust = 0.5, vjust = 0.5, lineheight = 1.2,
           family = "sans") +
  xlim(0, 1) + ylim(0, 1) +
  theme_void() +
  theme(
    panel.background = element_rect(fill = "#f8f9fa", color = NA)
  )

# Combine
fig5c <- fig5c_bar + fig5c_text +
  plot_layout(widths = c(1, 1)) +
  plot_annotation(
    theme = theme(plot.margin = margin(10, 10, 10, 10))
  )

save_slide_figure(fig5c, file.path(output_dir, "fig5_panel_c_gene_vs_pathway_overlap.png"))

# ============================================================================
# FIGURE 5D: Tissue-Specific Pathways Summary
# ============================================================================

cat("Generating fig5_panel_d_tissue_pathways_summary.png...\n")

# Prepare data for stacked bar
pathway_counts <- pathway_summary %>%
  select(tissue, n_kegg, n_reactome) %>%
  pivot_longer(-tissue, names_to = "source", values_to = "count") %>%
  mutate(
    source = case_when(
      source == "n_kegg" ~ "KEGG",
      source == "n_reactome" ~ "Reactome",
      TRUE ~ source
    ),
    tissue = factor(tissue, levels = c("Muscle", "Brain", "Liver", "Lung", "Blood"))
  )

# Total counts for text labels
totals <- pathway_summary %>%
  mutate(
    total = n_kegg + n_reactome,
    tissue = factor(tissue, levels = c("Muscle", "Brain", "Liver", "Lung", "Blood"))
  )

fig5d <- ggplot(pathway_counts, aes(x = tissue, y = count, fill = source)) +
  geom_col(position = "stack", width = 0.7) +
  geom_text(data = totals, aes(x = tissue, y = total, label = total, fill = NULL),
            vjust = -0.5, size = 4, fontface = "bold") +
  scale_fill_manual(values = c("KEGG" = "#0072B2", "Reactome" = "#56B4E9"),
                    name = "Database") +
  scale_y_continuous(expand = c(0, 0, 0.1, 0)) +
  labs(
    title = "Enriched Pathways by Tissue and Database",
    subtitle = "KEGG and Reactome pathways enriched in top 50 developmental genes",
    x = NULL,
    y = "Number of Enriched Pathways"
  ) +
  slides_theme() +
  theme(legend.position = "right")

save_slide_figure(fig5d, file.path(output_dir, "fig5_panel_d_tissue_pathways_summary.png"))

# ============================================================================
# FIGURE 5E: Key Shared Pathways Table
# ============================================================================

cat("Generating fig5_panel_e_shared_pathways_table.png...\n")

# Get top shared pathways
top_shared <- pathway_tissue_counts %>%
  filter(n_tissues >= 3) %>%
  filter(source %in% c("KEGG", "REAC", "GO:BP")) %>%
  slice_min(order_by = mean_p_value, n = 12) %>%
  mutate(
    term_short = ifelse(nchar(term_name) > 40,
                        paste0(substr(term_name, 1, 37), "..."),
                        term_name),
    source_label = case_when(
      source == "KEGG" ~ "KEGG",
      source == "REAC" ~ "Reactome",
      source == "GO:BP" ~ "GO",
      TRUE ~ source
    ),
    row_num = row_number()
  )

# Create table-like plot
fig5e <- ggplot(top_shared, aes(y = reorder(term_short, -row_num))) +
  geom_segment(aes(x = 0, xend = n_tissues, yend = reorder(term_short, -row_num)),
               color = cardiff_colors$dark_grey, linewidth = 0.5) +
  geom_point(aes(x = n_tissues, color = source_label), size = 5) +
  geom_text(aes(x = n_tissues, label = n_tissues), color = "white", size = 3, fontface = "bold") +
  scale_color_manual(values = c("KEGG" = "#0072B2",
                                "Reactome" = cardiff_colors$red,
                                "GO" = cardiff_colors$black),
                     name = "Source") +
  scale_x_continuous(limits = c(0, 5.5), breaks = 1:5) +
  labs(
    title = "Key Pathways Shared Across Multiple Tissues",
    subtitle = "Top 12 developmental pathways (by p-value) found in 3+ tissues",
    x = "Number of Tissues",
    y = NULL
  ) +
  slides_theme() +
  theme(
    legend.position = "right",
    panel.grid.major.y = element_blank()
  )

save_slide_figure(fig5e, file.path(output_dir, "fig5_panel_e_shared_pathways_table.png"),
                  width = 11, height = 7)

# ============================================================================
# SUMMARY
# ============================================================================

cat("\n=== Pathway Figures Generated ===\n")
cat("Output directory:", output_dir, "\n")
cat("Figures created:\n")
cat("  - fig5_panel_a_kegg_reactome_enrichment.png\n")
cat("  - fig5_panel_b_pathway_overlap.png\n")
cat("  - fig5_panel_c_gene_vs_pathway_overlap.png\n")
cat("  - fig5_panel_d_tissue_pathways_summary.png\n")
cat("  - fig5_panel_e_shared_pathways_table.png\n")
