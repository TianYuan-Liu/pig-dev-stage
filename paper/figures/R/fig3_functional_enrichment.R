#!/usr/bin/env Rscript
# ==============================================================================
# Figure 3: Stage-informative biomarkers from the PLS model (VIP)
# ==============================================================================
# Reproducible: reads the committed PLS-VIP table only (no live g:Profiler,
# no pre-PLS model_outputs). Panels (single merged biomarker figure):
#   a - Top VIP genes per tissue, signed by developmental direction (lollipop)
#   b - Stage-expression trajectories of those SAME genes (row-z heatmap)
#       [absorbed from the former standalone Supp. Fig. "tissue features" to
#        remove the duplicate gene panel; shares the panel-a gene rows]
#   c - Annotation status of the top-VIP genes (verified / weak / novel)
# Data: review/analyses/results/pls_vip_features.csv (100 rows = top-20 x 5).
# ==============================================================================

# ==============================================================================
# 1. SETUP
# ==============================================================================
suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
})

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
cat("Figure 3: PLS-VIP developmental biomarkers (lollipop + trajectory + status)\n")
cat("==============================================================================\n\n")

set.seed(42)

TISSUE_LV <- c("Muscle", "Brain", "Liver", "Blood", "Lung")
# Blood is a sex-confound negative control (top-VIP set dominated by Y-chromosome
# genes), not a developmental-biomarker set: its facet strip is annotated and its
# lollipops are drawn in neutral grey rather than accelerator/decelerator colours
# so the panel cannot be read as a maturation direction (see Fig. 3 caption).
TISSUE_STRIP <- c(Muscle = "Muscle", Brain = "Brain", Liver = "Liver",
                  Blood = "Blood (sex control)", Lung = "Lung")
DIR_LABELS <- c("+" = "Accelerator", "-" = "Decelerator")
DIR_FILL   <- c("Accelerator" = unname(DIRECTION_COLORS[["+"]]),
                "Decelerator" = unname(DIRECTION_COLORS[["-"]]),
                "Sex-aliased (control)" = unname(NULL_COLOR))
STATUS_FILL <- c("verified" = SIG_COLOR, "weak" = NULL_COLOR, "novel" = "#E6863C")
STATUS_LAB  <- c("verified" = "Verified", "weak" = "Weak/partial", "novel" = "Novel locus")
TOP_N      <- 8
STAGE_TPM_COLS <- c("Neonatal_TPM", "Suckling_TPM", "Juvenile_TPM",
                    "Pre-pubertal_TPM", "Post-pubertal_TPM", "Adult_TPM")

# ==============================================================================
# 2. DATA
# ==============================================================================
cat("Loading PLS-VIP table...\n")
vip_path <- project_path("review/analyses/results", "pls_vip_features.csv")
vip <- readr::read_csv(vip_path, show_col_types = FALSE) %>%
  mutate(
    tissue = factor(tissue, levels = TISSUE_LV),
    dir    = recode(sign, !!!DIR_LABELS),
    label  = ifelse(is.na(symbol) | symbol == "", gene, symbol),
    is_novel = annotation_status == "novel"
  )
cat(sprintf("  %d biomarker rows across %d tissues\n", nrow(vip), nlevels(vip$tissue)))

sep <- "___"

# Shared top-N-per-tissue table used by BOTH panel a (lollipop) and panel b
# (trajectory heatmap) so the two panels show the IDENTICAL genes in the
# IDENTICAL within-tissue order (ascending VIP -> strongest gene at top).
d_top <- vip %>%
  group_by(tissue) %>%
  slice_min(rank, n = TOP_N, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(
    signed_vip = vip * ifelse(sign == "+", 1, -1),
    dir_display = factor(
      ifelse(as.character(tissue) == "Blood", "Sex-aliased (control)", as.character(dir)),
      levels = c("Accelerator", "Decelerator", "Sex-aliased (control)")
    ),
    ylab = paste(label, tissue, sep = sep),
    ylab = fct_reorder(ylab, vip)
  )

# ==============================================================================
# 3. PANEL A: top VIP genes per tissue, signed by direction (lollipop)
# ==============================================================================
create_panel_a <- function(d) {
  ggplot(d, aes(x = signed_vip, y = ylab, colour = dir_display)) +
    geom_vline(xintercept = 0, colour = "grey70", linewidth = 0.4) +
    geom_segment(aes(x = 0, xend = signed_vip, yend = ylab), linewidth = 0.7) +
    geom_point(size = 1.7) +
    facet_grid(rows = vars(tissue), scales = "free_y", space = "free_y") +
    scale_y_discrete(labels = function(x) sub(paste0(sep, ".*$"), "", x)) +
    scale_x_continuous(expand = expansion(mult = c(0.04, 0.04))) +
    scale_colour_manual(values = DIR_FILL, name = NULL) +
    labs(x = "Signed VIP (− decelerator   |   accelerator +)", y = NULL) +
    nature_theme() +
    theme(
      axis.text.y = element_text(size = 5.6, face = "italic"),
      legend.position = "top",
      legend.justification = "left",
      strip.text.y = element_blank(),     # tissue strip lives on the heatmap (panel b)
      panel.spacing.y = unit(1.4, "mm"),
      plot.margin = margin(2, 2, 2, 2, "mm")
    )
}

# ==============================================================================
# 4. PANEL B: stage-expression trajectories of the SAME genes (row-z heatmap)
# ==============================================================================
create_panel_b <- function(d) {
  long <- d %>%
    select(tissue, ylab, all_of(STAGE_TPM_COLS)) %>%
    pivot_longer(all_of(STAGE_TPM_COLS), names_to = "stage_raw", values_to = "expr") %>%
    mutate(
      stage = gsub("_TPM$", "", stage_raw),
      stage = gsub("_", " ", stage),
      stage = factor(stage, levels = STAGE_ORDER, labels = STAGE_LABELS[STAGE_ORDER])
    ) %>%
    group_by(ylab) %>%
    mutate(z = if (sd(expr, na.rm = TRUE) > 0)
      (expr - mean(expr, na.rm = TRUE)) / sd(expr, na.rm = TRUE) else 0) %>%
    ungroup()

  ggplot(long, aes(x = stage, y = ylab, fill = z)) +
    geom_tile(colour = "white", linewidth = 0.3) +
    facet_grid(rows = vars(tissue), scales = "free_y", space = "free_y",
               labeller = labeller(tissue = TISSUE_STRIP)) +
    scale_fill_house_div(name = "Row z-score\nlog2(TPM+1)") +
    scale_x_discrete(expand = c(0, 0)) +
    labs(x = NULL, y = NULL) +
    nature_theme() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 5.6),
      axis.text.y = element_blank(),       # gene labels shown once, on panel a
      axis.ticks.y = element_blank(),
      legend.position = "right",
      legend.key.height = unit(6, "mm"),
      legend.key.width = unit(2.4, "mm"),
      strip.text.y = element_text(angle = -90, face = "bold"),
      strip.clip = "off",
      panel.grid = element_blank(),
      panel.spacing.y = unit(1.4, "mm"),
      plot.margin = margin(2, 3, 2, 1, "mm")
    )
}

# ==============================================================================
# 5. PANEL C: annotation status of top-VIP genes
# ==============================================================================
create_panel_c <- function(vip) {
  d <- vip %>%
    count(tissue, annotation_status, name = "n") %>%
    mutate(annotation_status = factor(annotation_status,
                                      levels = c("verified", "weak", "novel")))

  ggplot(d, aes(x = tissue, y = n, fill = annotation_status)) +
    geom_col(width = 0.72) +
    scale_fill_manual(values = STATUS_FILL, labels = STATUS_LAB, name = NULL) +
    labs(x = NULL, y = "Top-VIP genes (per tissue)") +
    nature_theme() +
    theme(
      axis.text.x = element_text(angle = 30, hjust = 1),
      legend.position = "right",
      legend.key.size = unit(3, "mm"),
      plot.margin = margin(2, 2, 2, 2, "mm")
    )
}

# ==============================================================================
# 6. ASSEMBLY
# ==============================================================================
cat("Assembling figure...\n")
panel_a <- create_panel_a(d_top)
panel_b <- create_panel_b(d_top)
panel_c <- create_panel_c(vip)

# Top: lollipop (gene labels) | trajectory heatmap (shared rows). Bottom:
# annotation-status bar spanning the width.
fig3 <- (panel_a | panel_b) / panel_c +
  plot_layout(heights = c(3.4, 0.7), widths = c(1)) +
  plot_annotation(
    tag_levels = "a",
    theme = theme(
      plot.background = element_rect(fill = "white", color = NA),
      plot.tag = element_text(size = 9, face = "bold")
    )
  ) &
  theme(plot.tag = element_text(size = 9, face = "bold"))

# ==============================================================================
# 7. SAVE
# ==============================================================================
cat("Saving figure...\n")
save_figure(fig3, "fig3_functional_enrichment", width = 183, height = 220)

# ==============================================================================
# 8. SUMMARY
# ==============================================================================
status_counts <- vip %>% count(annotation_status)
summary_text <- sprintf(
"
FIGURE 3: PLS-VIP DEVELOPMENTAL BIOMARKERS
==========================================
Generated: %s
Source: review/analyses/results/pls_vip_features.csv (no live enrichment query)

Panel a: top %d VIP genes per tissue, signed by developmental direction.
Panel b: row-z-scored stage-expression trajectories of those same genes
         (absorbed from the former standalone tissue-feature supp. figure).
Panel c: annotation status of the 100 top-VIP genes.
Per-gene per-stage values remain tabulated in Supplementary Table S4.

Annotation status (all 100): %s
Figure dimensions: 183mm x 220mm
",
  format(Sys.time(), "%%Y-%%m-%%d %%H:%%M"),
  TOP_N,
  paste(sprintf("%s=%d", status_counts$annotation_status, status_counts$n), collapse = ", ")
)
write_summary(summary_text, "fig3_summary.txt")

cat("\n✓ Figure 3 completed successfully!\n")
