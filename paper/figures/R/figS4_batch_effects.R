#!/usr/bin/env Rscript
# ==============================================================================
# Supplementary Figure: Batch-effect evaluation across five tissues
# ==============================================================================
# R port of review/analyses/batch_effect_visualization.py (house style).
# Reads CACHED PCA/UMAP coordinates (no recompute):
#   review/analyses/results/batch_effect_embeddings_<Tissue>.csv
# Per tissue: UMAP coloured by BioProject (technical batch) and by Stage
# (biological signal). Output basename kept as figS4_batch_effects.
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
})

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) return(dirname(normalizePath(sub("--file=", "", file_arg))))
  "/Users/tianyuan/Desktop/github_dev/pig-dev-stage/paper/figures/R"
}
script_dir <- get_script_dir()
source(file.path(script_dir, "_theme.R"))
source(file.path(script_dir, "_utils.R"))

cat("Supplementary Figure: batch effects (R port)\n")

TISSUE_LV <- c("Muscle", "Brain", "Liver", "Lung", "Blood")

# ---- load cached embeddings -------------------------------------------------
# The cached CSVs carry a STALE 5-stage `Stage` column; re-derive the 6-stage
# Stage from the canonical metadata (assign_stage, boundaries [3,28,90,165,540])
# by Sample_ID so the colouring matches the rest of the paper. UMAP/PCA coords
# are stage-independent, so only the colour mapping changes.
meta6 <- load_metadata() %>% dplyr::select(Sample_ID, Stage6 = Stage)
emb <- map_dfr(TISSUE_LV, function(tis) {
  f <- project_path("review/analyses/results", sprintf("batch_effect_embeddings_%s.csv", tis))
  if (!file.exists(f)) { warning("missing ", f); return(NULL) }
  readr::read_csv(f, show_col_types = FALSE) %>%
    dplyr::select(-dplyr::any_of("Stage")) %>%   # drop stale 5-stage column
    mutate(tissue = tis)
}) %>%
  left_join(meta6, by = "Sample_ID") %>%
  mutate(Stage = factor(Stage6, levels = STAGE_ORDER)) %>%
  filter(!is.na(Stage))   # 6-level Stage scale, shared across panels

# ---- one tissue's two panels ------------------------------------------------
panel_pair <- function(tis, keep_legend = FALSE) {
  d <- emb %>% filter(tissue == tis)
  n_s <- nrow(d); n_bp <- dplyr::n_distinct(d$BioProject)
  n_st <- dplyr::n_distinct(d$Stage[!is.na(d$Stage)])
  bp_levels <- names(sort(table(d$BioProject), decreasing = TRUE))
  bp_pal <- setNames(categorical_palette(length(bp_levels)), bp_levels)

  base <- function(p, title, tagcol) {
    p + labs(title = NULL, x = "UMAP1", y = "UMAP2") +
      annotate("text", x = Inf, y = -Inf, label = title,
               hjust = 1.02, vjust = -0.6, size = 1.7, colour = "grey25") +
      nature_theme() +
      theme(plot.subtitle = element_text(size = 6, face = "bold"),
            axis.text = element_text(size = 5),
            plot.margin = margin(2, 2, 2, 2, "mm"))
  }

  p_bp <- ggplot(d, aes(UMAP1, UMAP2, colour = BioProject)) +
    geom_point(size = 0.7, alpha = 0.85) +
    scale_colour_manual(values = bp_pal, guide = "none") +
    labs(subtitle = sprintf("%s — BioProject", tis))
  p_bp <- base(p_bp, sprintf("n=%d / %d projects", n_s, n_bp))

  p_st <- ggplot(d, aes(UMAP1, UMAP2, fill = Stage)) +
    geom_point(size = 1.0, shape = 21, colour = "grey20", stroke = 0.15, alpha = 0.9) +
    scale_fill_manual(values = STAGE_COLORS, name = "Developmental stage",
                      drop = FALSE, breaks = STAGE_ORDER,
                      guide = if (keep_legend) "legend" else "none") +
    labs(subtitle = sprintf("%s — Stage", tis))
  p_st <- base(p_st, sprintf("n=%d / %d stages", n_s, n_st))

  list(bp = p_bp, st = p_st)
}

pairs <- lapply(seq_along(TISSUE_LV), function(i) panel_pair(TISSUE_LV[i], keep_legend = (i == 1)))
panels <- list()
for (pr in pairs) panels <- c(panels, list(pr$bp, pr$st))

fig <- wrap_plots(panels, ncol = 2, byrow = TRUE) +
  plot_layout(guides = "collect") +
  plot_annotation(theme = theme(plot.background = element_rect(fill = "white", colour = NA))) &
  theme(legend.position = "bottom")

save_figure(fig, "figS4_batch_effects", width = 183, height = 245)
cat("\n✓ figS4_batch_effects done\n")
