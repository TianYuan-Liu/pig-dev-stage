#!/usr/bin/env Rscript
# ==============================================================================
# Supplementary Figure S1: Extended model evaluation (COMPOSITE)
# ==============================================================================
# Merges what were two separate supplementary figures into one, removing the
# internal heatmap-vs-bars duplication of the old per-class figure:
#   a - In-distribution (stratified, batch-inclusive) separability balanced
#       accuracy per tissue, plus the cross-study (grouped, whole-study-held-out)
#       deployment BA as a hollow bar for muscle and liver (the two tissues whose
#       stages span multiple studies); cross-species developmental validation
#       status annotated per tissue.
#   b - Within-tissue feature-selection stability (mean Jaccard of top-50 across
#       10 seeds; Random = analytical expectation).
#   c - Mean |Spearman r| of top features with developmental stage, per tissue.
# No fabricated numbers: a <- paper/figures/data/within_pig_6stage.json
#   + review/analyses/results/liver_loco_stage_restricted.json (liver cross-study);
# b <- machine_learning/analysis/results/feature_stability_results.json;
# c <- machine_learning/analysis/results/expression_correlation_summary.csv.
# ==============================================================================

set.seed(42)

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(jsonlite)
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

cat("==============================================================================\n")
cat("Supplementary Figure S1: extended model evaluation (composite)\n")
cat("==============================================================================\n\n")

TISSUES    <- c("Muscle", "Lung", "Liver", "Brain", "Blood")
STAGE_LV   <- c("Neonatal", "Suckling", "Juvenile", "Pre-pubertal",
                "Post-pubertal", "Adult")
STAGE_ABBR <- c("Neonatal" = "Neo", "Suckling" = "Suck", "Juvenile" = "Juv",
                "Pre-pubertal" = "Pre", "Post-pubertal" = "Post", "Adult" = "Adult")

# ==============================================================================
# PANEL A: in-distribution separability vs cross-study deployment (per-tissue BA)
# ==============================================================================
# One balanced-accuracy summary per tissue. Every tissue shows its
# in-distribution (stratified, batch-inclusive) separability BA (solid bar);
# muscle alone also shows the cross-study (grouped, whole-study-held-out)
# deployment BA (hollow bar), the only tissue for which that quantity is
# identifiable. The grouped BA for the other four tissues is deliberately NOT
# shown: it is unidentifiable (stage aliased with study) and degenerate for the
# binary tissues. Cross-species developmental validation status (from the
# main-text transfer analysis) is annotated under each tissue.
wp <- jsonlite::fromJSON(
  project_path("paper/figures/data/within_pig_6stage.json"),
  simplifyVector = FALSE)

# liver's restricted cross-study estimate (its four study-spanning stages, after
# excluding the single-cohort neonatal/adult stages; grouped CV, 5 folds x 5 seeds)
liv_cs <- jsonlite::fromJSON(
  project_path("review/analyses/results/liver_loco_stage_restricted.json"),
  simplifyVector = FALSE)[["drop Neonatal + Adult"]]

# facet order for panel a: lead with the two cross-study-identifiable tissues
PANEL_A_ORDER <- c("Muscle", "Liver", "Lung", "Brain", "Blood")

read_ba <- function(tissue, slot, metric) {
  s <- wp[[tissue]][[slot]]
  if (is.null(s) || is.null(s$BA_mean)) return(NULL)
  tibble(Tissue = tissue, Metric = metric,
         BA = as.numeric(s$BA_mean),
         SD = as.numeric(s$BA_sd %||% 0),
         chance = as.numeric(wp[[tissue]]$chance),
         seeds = list(as.numeric(unlist(s$per_seed_ba))))
}

# liver cross-study row (computed from per-seed BAs so the displayed mean and the
# %.2f label match the manuscript value, 0.39, exactly)
liver_seeds <- as.numeric(unlist(liv_cs$per_seed_ba))
liver_cross <- tibble(Tissue = "Liver", Metric = "Cross-study",
                      BA = mean(liver_seeds), SD = as.numeric(liv_cs$BA_sd),
                      chance = as.numeric(liv_cs$chance),
                      seeds = list(liver_seeds))

# in-distribution separability for all tissues; cross-study for muscle & liver
ba_bars <- bind_rows(
  map_dfr(PANEL_A_ORDER, read_ba, slot = "stratified_6stage", metric = "In-dist."),
  read_ba("Muscle", "grouped_6stage", "Cross-study"),
  liver_cross
) %>%
  mutate(Tissue = factor(Tissue, levels = PANEL_A_ORDER),
         Metric = factor(Metric, levels = c("In-dist.", "Cross-study")),
         lab_y  = pmin(1.10, BA + SD + 0.05))
ba_pts    <- ba_bars %>% tidyr::unnest(seeds) %>% rename(val = seeds)
# single-stage-study fraction per tissue (the ordering / identifiability variable)
ss_pct <- setNames(
  round(sapply(PANEL_A_ORDER, function(t) as.numeric(wp[[t]]$single_stage_pct))),
  PANEL_A_ORDER)
# Compact cross-species conservation tag per tissue, framed as a graded axis (NOT a
# pass/fail glyph): conserved organs transfer to independent atlases, the labile liver
# transfers weakly (a finding, not a failure), and blood is a by-design negative
# control with no cross-species substrate. Full detail in the caption.
XSP_GLYPH <- c(Muscle = "conserved", Liver = "labile", Lung = "conserved",
               Brain = "conserved", Blood = "neg. ctrl")

create_panel_a <- function() {
  cat("Panel a: separability -> cross-study deployment (dumbbell)...\n")
  ord  <- rev(PANEL_A_ORDER)                    # Muscle at top, Blood at bottom
  yidx <- setNames(seq_along(ord), ord)
  d <- ba_bars %>% mutate(t = as.character(Tissue), yy = yidx[t])
  indist <- dplyr::filter(d, Metric == "In-dist.")
  cross  <- dplyr::filter(d, Metric == "Cross-study")
  arr <- indist %>% dplyr::select(t, yy, x_in = BA) %>%
    dplyr::inner_join(dplyr::select(cross, t, x_cs = BA), by = "t")
  ni  <- dplyr::filter(indist, !t %in% cross$t)        # cross-study unidentifiable
  chance_pts <- d %>% dplyr::distinct(t, yy, Metric, chance)
  ylabs <- sprintf("%s %s\n%d%% single-stage", ord, XSP_GLYPH[ord], ss_pct[ord])

  ggplot() +
    # chance reference: each measured point at its own 1/K (liver 1/6 vs 1/4 differ)
    geom_segment(data = chance_pts,
                 aes(x = chance, xend = chance, y = yy - 0.18, yend = yy + 0.18),
                 colour = CHANCE_COLOR, linewidth = 0.4, linetype = "22") +
    # deployment-drop arrow (in-distribution -> cross-study): muscle & liver
    geom_segment(data = arr, aes(x = x_in, xend = x_cs, y = yy, yend = yy),
                 colour = "grey55", linewidth = 0.5,
                 arrow = grid::arrow(length = unit(1.6, "mm"), type = "closed")) +
    # +/- s.d. whiskers
    geom_segment(data = d, aes(x = pmax(0, BA - SD), xend = pmin(1, BA + SD),
                               y = yy, yend = yy),
                 colour = "grey45", linewidth = 0.35) +
    # in-distribution (filled) and cross-study (open) markers
    geom_point(data = indist, aes(BA, yy, fill = t), shape = 21, size = 3.1,
               colour = "white", stroke = 0.4) +
    geom_point(data = cross, aes(BA, yy, colour = t), shape = 21, size = 3.1,
               fill = "white", stroke = 1.2) +
    # value labels above each marker
    geom_text(data = dplyr::bind_rows(indist, cross),
              aes(BA, yy + 0.31, label = sprintf("%.2f", BA)),
              size = 1.95, family = "Arial", colour = "grey15") +
    # 'not identifiable' note on the 100%-single-stage rows
    geom_text(data = ni, aes(x = chance + 0.04, y = yy,
                             label = "cross-study not identifiable"),
              hjust = 0, vjust = 0.5, size = 1.7, family = "Arial",
              fontface = "italic", colour = "grey60") +
    scale_fill_manual(values = TISSUE_COLORS, guide = "none") +
    scale_colour_manual(values = TISSUE_COLORS, guide = "none") +
    scale_x_continuous(limits = c(0, 1.0), breaks = seq(0, 1, 0.25),
                       expand = expansion(mult = c(0.01, 0.03))) +
    scale_y_continuous(breaks = seq_along(ord), labels = ylabs,
                       limits = c(0.4, length(ord) + 0.65),
                       expand = expansion(mult = c(0, 0))) +
    labs(subtitle = "Filled = in-distribution (batch-inclusive)  •  open = cross-study (whole-study-held-out)  •  arrow = deployment drop  •  dashed = chance (1/K)",
         x = "Balanced accuracy", y = NULL) +
    nature_theme() +
    theme(plot.subtitle = element_text(size = 5.0, colour = "grey35"),
          axis.text.y = element_text(size = 6, lineheight = 0.92),
          panel.grid.major.x = element_line(colour = "grey92", linewidth = 0.3))
}

# ==============================================================================
# PANEL B: feature-selection stability (mean Jaccard of top-50 across seeds)
# ==============================================================================
create_panel_b <- function() {
  cat("Panel b: feature-selection stability (Jaccard)...\n")
  stability_data <- fromJSON(
    project_path("machine_learning/analysis/results/feature_stability_results.json"))

  stability_summary <- map_dfr(names(stability_data), function(tissue) {
    d <- stability_data[[tissue]]
    tibble(Tissue = tissue,
           Mean_Jaccard = d$jaccard_similarity$mean,
           SD_Jaccard = d$jaccard_similarity$std)
  })

  # Analytical random expectation for selecting k of N features
  n_genes_total <- 11000; k_features <- 50
  expected_intersection <- (k_features^2) / n_genes_total
  random_jaccard <- expected_intersection / (2 * k_features - expected_intersection)

  stability_summary <- stability_summary %>%
    bind_rows(tibble(Tissue = "Random", Mean_Jaccard = random_jaccard, SD_Jaccard = 0)) %>%
    filter(Tissue %in% c(names(TISSUE_COLORS), "Random")) %>%
    arrange(Mean_Jaccard) %>%
    mutate(Tissue = factor(Tissue, levels = Tissue))
  plot_colors <- c(TISSUE_COLORS, "Random" = "grey70")

  ggplot(stability_summary, aes(Mean_Jaccard, Tissue)) +
    geom_segment(aes(x = 0, xend = Mean_Jaccard, yend = Tissue, colour = Tissue),
                 linewidth = 0.6, show.legend = FALSE) +
    geom_segment(aes(x = pmax(0, Mean_Jaccard - SD_Jaccard),
                     xend = Mean_Jaccard + SD_Jaccard, yend = Tissue),
                 colour = "grey40", linewidth = 0.35) +
    geom_point(aes(colour = Tissue), size = 2.5, show.legend = FALSE) +
    geom_text(aes(label = ifelse(Mean_Jaccard < 0.01, sprintf("%.3f", Mean_Jaccard),
                                 sprintf("%.2f", Mean_Jaccard)),
                  x = Mean_Jaccard + SD_Jaccard + 0.025),
              hjust = 0, size = 1.95, family = "Arial", colour = "grey15") +
    scale_colour_manual(values = plot_colors) +
    scale_x_continuous(limits = c(0, 0.78), breaks = seq(0, 0.75, 0.25),
                       expand = expansion(mult = c(0, 0.06))) +
    labs(x = "Mean Jaccard similarity (top-50, 10 seeds)", y = NULL) +
    nature_theme() +
    theme(panel.grid.major.x = element_line(colour = "grey92", linewidth = 0.3))
}

# ==============================================================================
# PANEL C: feature-stage correlation (mean |Spearman r|)
# ==============================================================================
create_panel_c <- function() {
  cat("Panel c: feature-stage correlation (Spearman)...\n")
  corr_data <- read_csv(
    project_path("machine_learning/analysis/results/expression_correlation_summary.csv"),
    show_col_types = FALSE)
  if ("Mean |r|" %in% names(corr_data)) {
    corr_data <- corr_data %>% rename(Mean_abs_r = `Mean |r|`)
  }
  corr_summary <- corr_data %>%
    mutate(Mean_abs_r = as.numeric(Mean_abs_r)) %>%
    filter(Tissue %in% names(TISSUE_COLORS)) %>%
    arrange(Mean_abs_r) %>%
    mutate(Tissue = factor(Tissue, levels = Tissue))

  ggplot(corr_summary, aes(Mean_abs_r, Tissue)) +
    geom_segment(aes(x = 0, xend = Mean_abs_r, yend = Tissue, colour = Tissue),
                 linewidth = 0.6, show.legend = FALSE) +
    geom_point(aes(colour = Tissue), size = 2.5, show.legend = FALSE) +
    geom_text(aes(label = sprintf("%.2f", Mean_abs_r), x = Mean_abs_r + 0.025),
              hjust = 0, size = 1.95, family = "Arial", colour = "grey15") +
    scale_colour_manual(values = TISSUE_COLORS) +
    scale_x_continuous(limits = c(0, 0.78), breaks = seq(0, 0.75, 0.25),
                       expand = expansion(mult = c(0, 0.06))) +
    labs(x = "Mean |Spearman r| with stage", y = NULL) +
    nature_theme() +
    theme(panel.grid.major.x = element_line(colour = "grey92", linewidth = 0.3))
}

# ==============================================================================
# ASSEMBLY + SAVE
# ==============================================================================
cat("\nAssembling figure...\n")
pa <- create_panel_a()
pb <- create_panel_b()
pc <- create_panel_c()

figS1 <- pa / (pb | pc) +
  plot_layout(heights = c(1.25, 1.0)) +
  plot_annotation(
    tag_levels = "a",
    theme = theme(
      plot.background = element_rect(fill = "white", color = NA),
      plot.tag = element_text(size = 9, face = "bold")
    )
  )

cat("Saving figure...\n")
save_figure(figS1, "figS1_model_evaluation", width = 183, height = 150)

cat("\n✓ Supplementary Figure S1 (model evaluation composite) completed!\n")
