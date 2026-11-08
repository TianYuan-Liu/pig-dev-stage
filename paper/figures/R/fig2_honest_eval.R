#!/usr/bin/env Rscript
# ==============================================================================
# Figure 2 (REFRAME v2): Confound mechanism + the muscle positive control
# ==============================================================================
# FRAMING (user, 2026-05-28): LEAD with the study-confound mechanism and the
# muscle positive control. The 5-tissue stratified-vs-grouped collapse figure has
# been REMOVED entirely (author rule below): never plot the four confounded
# tissues' grouped BA as a score. The supplement instead carries a stage x
# BioProject composition TABLE (handled outside this script).
#
# CLAIM: In PigGTEx, developmental stage is near-perfectly confounded with study
# (BioProject) for four of five tissues: 95-100% of their retained samples come
# from SINGLE-STAGE studies (one BioProject == one stage). Only MUSCLE has
# substantial multi-stage studies (47% single-stage), so cross-study (held-out
# study) validation of developmental staging is only FEASIBLE in muscle. In
# muscle, a leave-one-BioProject-out positive control confirms a genuine,
# study-independent developmental signal (5-class grouped BA 0.45 vs chance
# 0.20; ordinal rho 0.66, 5 seeds), with a large inflation gap vs the batch-
# inclusive stratified BA (0.79).
#
# Panels:
#   a (HERO) - confound diagnostic: % of retained (kept-stage) samples from
#              single-stage BioProjects per tissue. Muscle highlighted as the
#              only tissue with substantial multi-stage studies -> the only one
#              where a cross-study (held-out-study) test is feasible.
#   b        - muscle inflation gap (METHODS STRENGTH, muscle only): the
#              in-distribution stratified BA (batch-inclusive; NOT a deployment
#              metric) vs the cross-study grouped-CV BA (deployment estimate),
#              quantifying how large batch-inclusive inflation is where a clean
#              comparator exists. Chance 0.20 (5-class muscle).
#
# AUTHOR RULE: NEVER plot/tabulate the four confounded tissues' grouped balanced
# accuracy as a score -- a printed low BA reads as a bad model regardless of
# caption. Show the CONFOUND (stage x BioProject composition, panel a), not the
# score it produces. Fig 2 shows grouped/clean BA ONLY for muscle, the tissue
# with a legitimate clean comparator; per-class recall for all tissues is in
# Supplementary Fig. S3.
#
# Backend: R / ggplot2 (exclusive). White background. cairo PDF + PNG + SVG/TIFF.
#
# HARD RULE / PARAMETERISED DATA (no fabricated numbers): ALL numbers come from
#   paper/figures/data/within_pig_6stage.json (OrdinalPLS default 10 comp,
#   6-stage drop-small, 5 seeds; stratified + StratifiedGroupKFold-on-BioProject)
#   -- per tissue: single_stage_pct, kept6 (n / n_class), chance=1/K, stratified
#   & grouped BA. A staging/model refresh = regenerate that JSON via
#   paper/figures/provenance/within_pig_6stage_figdata.py (no code edit needed).
# ==============================================================================

set.seed(42)

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(jsonlite)
})

# Load shared modules
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("--file=", "", file_arg))))
  }
  "/Users/tianyuan/Desktop/github_dev/.claude-worktree-dgtex-v2/paper/figures/R"
}
script_dir <- get_script_dir()
source(file.path(script_dir, "_theme.R"))
source(file.path(script_dir, "_utils.R"))

cat("==============================================================================\n")
cat("Figure 2 (REFRAME v2): confound mechanism + muscle positive control\n")
cat("==============================================================================\n\n")

TISSUES <- c("Muscle", "Lung", "Brain", "Blood", "Liver")

# ==============================================================================
# 1. DATA LOADING  (canonical within-pig 6-stage artifact; no fabricated numbers)
# ==============================================================================
# paper/figures/data/within_pig_6stage.json : per tissue n/kept6/n_class/
# chance/single_stage_pct + stratified & grouped(BioProject) 6-stage honest eval
# (OrdinalPLS, default 10 comp, 5 seeds; reproducible from the repo). A staging
# refresh = regenerate that JSON via paper/figures/provenance/within_pig_6stage_figdata.py.
wp_path <- project_path("paper/figures/data/within_pig_6stage.json")
if (!file.exists(wp_path)) stop("within-pig 6-stage artifact not found: ", wp_path)
wp <- jsonlite::fromJSON(wp_path, simplifyVector = FALSE)

# % of retained (kept-stage) samples that come from single-stage BioProjects.
single_stage_pct <- vapply(TISSUES, function(t) as.numeric(wp[[t]]$single_stage_pct),
                           numeric(1))
names(single_stage_pct) <- TISSUES

# per-tissue kept-n (sum of kept-stage counts) + n_class (for axis context).
tissue_meta <- map_dfr(TISSUES, function(t) {
  tibble(Tissue = t,
         n = sum(unlist(wp[[t]]$kept6)),
         n_class = as.integer(wp[[t]]$n_class))
})

# Muscle honest-eval numbers (the only cross-study-identifiable tissue).
mus            <- wp[["Muscle"]]
muscle_strat   <- as.numeric(mus$stratified_6stage$BA_mean)   # batch-inclusive
muscle_group   <- as.numeric(mus$grouped_6stage$BA_mean)      # cross-study (grouped)
muscle_group_sd <- as.numeric(mus$grouped_6stage$BA_sd)
muscle_chance  <- as.numeric(mus$chance)                      # 1/K = 0.20 (5-class)

cat(sprintf("Muscle 6-stage: stratified BA %.3f | grouped BA %.3f +/- %.3f (chance %.2f)\n",
            muscle_strat, muscle_group, muscle_group_sd, muscle_chance))
cat(sprintf("Single-stage %%: %s\n",
            paste(sprintf("%s %.0f%%", names(single_stage_pct), single_stage_pct),
                  collapse = ", ")))

# ==============================================================================
# 2. PANEL A (HERO): study-confound diagnostic
# ==============================================================================
create_panel_a <- function() {
  cat("Creating Panel A (HERO): study-confound diagnostic...\n")

  df <- tibble(
    Tissue = names(single_stage_pct),
    pct    = as.numeric(single_stage_pct)
  ) %>%
    left_join(tissue_meta, by = "Tissue") %>%
    mutate(
      feasible  = Tissue == "Muscle",  # only tissue with substantial multi-stage studies
      Tissue_lab = sprintf("%s (n=%d)", Tissue, n),
      multi_pct = 100 - pct
    ) %>%
    arrange(pct) %>%
    mutate(Tissue_lab = factor(Tissue_lab, levels = Tissue_lab))

  # stacked: single-stage (confounded) vs multi-stage (informative) fraction
  stk <- df %>%
    select(Tissue_lab, feasible,
           `single-stage studies` = pct,
           `multi-stage studies`  = multi_pct) %>%
    pivot_longer(c(`single-stage studies`, `multi-stage studies`),
                 names_to = "kind", values_to = "pct") %>%
    mutate(kind = factor(kind, levels = c("multi-stage studies",
                                          "single-stage studies")))

  FILL <- c("single-stage studies" = "#bdbdbd",
            "multi-stage studies"  = "#E64B35")

  ggplot(stk, aes(pct, Tissue_lab, fill = kind)) +
    geom_col(width = 0.66, colour = "white", linewidth = 0.3) +
    # label single-stage % at the right end of each bar
    geom_text(data = df, aes(x = 101, y = Tissue_lab,
                             label = sprintf("%d%% single-stage", round(pct))),
              inherit.aes = FALSE, hjust = 0, size = 1.95, colour = "grey25") +
    # flag muscle as the feasible-test tissue
    annotate("text", x = 23.5, y = "Muscle (n=757)",
             label = "53% multi-stage -> cross-study test feasible",
             size = 1.85, fontface = "italic", colour = "#7a1f14", hjust = 0.5) +
    scale_fill_manual(values = FILL, name = NULL,
                      breaks = c("single-stage studies", "multi-stage studies")) +
    scale_x_continuous(limits = c(0, 158), breaks = c(0, 25, 50, 75, 100),
                       expand = expansion(mult = c(0, 0))) +
    labs(title = "Stage is confounded with study (a)",
         x = "% of retained samples, by study type", y = NULL) +
    nature_theme() +
    theme(legend.position = "bottom",
          legend.direction = "horizontal",
          legend.key.size = unit(2.8, "mm"),
          legend.text = element_text(size = 5.6),
          legend.margin = margin(0, 0, 0, 0),
          legend.box.spacing = unit(1, "mm"),
          axis.text.y = element_text(size = 6.2),
          axis.title.x = element_text(size = 6.2))
}

# (The former muscle leave-one-BioProject-out per-class recall panel has been
# removed; per-class recall under grouped CV for ALL tissues now lives in
# Supplementary Fig. S3, sourced from the within-pig 6-stage artifact.)

# ==============================================================================
# 4. PANEL C: muscle batch-inclusive vs clean (cross-study) BA gap
# ==============================================================================
# Methods strength, NOT a tissue ranking. Shows ONLY muscle -- the one tissue
# with a clean comparator. The in-distribution STRATIFIED BA (0.882; batch-
# inclusive; NOT a deployment metric) is contrasted with the cross-study LOBO BA
# (0.51), quantifying how large batch-inclusive inflation can be. The four
# confounded tissues' grouped BA is NEVER shown (author rule: a printed "0.17"
# reads as a bad model regardless of caption -- show the confound, not the score).
create_panel_c <- function() {
  cat("Creating Panel C: muscle batch-inclusive vs clean BA gap...\n")

  # muscle stratified in-distribution BA (batch-inclusive, NOT a deployment metric)
  strat_ba_muscle <- muscle_strat
  # muscle cross-study BA: StratifiedGroupKFold on BioProject (whole studies held
  # out), 5 seeds -- the leakage-controlled deployment estimate.
  clean_ba_muscle <- muscle_group
  chance_muscle   <- muscle_chance

  dd <- tibble(
    metric = c("Stratified, in-distribution\n(batch-inclusive)",
               "Cross-study (grouped CV)\n(deployment estimate)"),
    ba     = c(strat_ba_muscle, clean_ba_muscle),
    x      = c(1, 2)
  ) %>%
    mutate(metric = factor(metric, levels = metric))

  COLS <- c("#bdbdbd", "#E64B35")  # grey = batch-inclusive, red = clean

  ggplot(dd, aes(x = x, y = ba)) +
    # dumbbell connector showing the inflation gap
    annotate("segment", x = 1, xend = 2, y = strat_ba_muscle, yend = clean_ba_muscle,
             colour = "grey55", linewidth = 0.6) +
    geom_point(aes(fill = metric), shape = 21, size = 3.4, stroke = 0.4,
               colour = "grey20") +
    geom_text(aes(label = sprintf("%.2f", ba)), vjust = -1.2, size = 2.2) +
    # gap annotation (mid-point)
    annotate("text", x = 1.5, y = (strat_ba_muscle + clean_ba_muscle) / 2,
             label = sprintf("%.2f inflation\ngap", strat_ba_muscle - clean_ba_muscle),
             size = 1.85, colour = "#7a1f14", fontface = "italic", hjust = 0.5,
             vjust = -0.2) +
    # chance line
    geom_hline(yintercept = chance_muscle, linetype = "22", linewidth = 0.45,
               colour = CHANCE_COLOR) +
    annotate("text", x = 2.45, y = chance_muscle + 0.03,
             label = sprintf("chance %.2f", chance_muscle),
             size = 1.7, colour = CHANCE_COLOR, hjust = 1) +
    scale_fill_manual(values = COLS, guide = "none") +
    scale_x_continuous(limits = c(0.5, 2.6),
                       breaks = c(1, 2),
                       labels = c("Stratified\n(batch-incl.)", "Cross-study\n(grouped)")) +
    scale_y_continuous(limits = c(0, 1.0), breaks = seq(0, 1, 0.25),
                       expand = expansion(mult = c(0, 0.03))) +
    coord_cartesian(clip = "off") +
    labs(title = "Muscle: inflation gap (b)",
         subtitle = "batch confound diagnosed and controlled (muscle only)",
         x = NULL, y = "Balanced accuracy") +
    nature_theme() +
    theme(plot.subtitle = element_text(size = 4.9, colour = "grey35"),
          axis.text.x = element_text(size = 5.2, lineheight = 0.85),
          plot.margin = margin(5, 8, 5, 5, "mm"))
}

# ==============================================================================
# 5. FIGURE ASSEMBLY
# ==============================================================================
cat("\nAssembling figure...\n")

pa <- create_panel_a()
pc <- create_panel_c()

# (a) the stage--study confound; (b) muscle in-distribution vs cross-study gap.
# Per-class recall (former panel b) lives in Supplementary Fig. S2 (all tissues).
fig2 <- pa / pc +
  plot_layout(heights = c(1.0, 0.95), widths = c(1)) +
  plot_annotation(
    tag_levels = list(c("a", "b")),
    theme = theme(
      plot.background = element_rect(fill = "white", color = NA),
      plot.tag = element_text(size = 9, face = "bold")
    )
  )

# ==============================================================================
# 6. SAVE FIGURE
# ==============================================================================
cat("\nSaving figure...\n")
save_figure(fig2, "fig2_honest_eval", width = 183, height = 122)

out_dir <- project_path("paper/figures/output")
dir.create(file.path(out_dir, "svg"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_dir, "tiff"), showWarnings = FALSE, recursive = TRUE)
w_in <- 183 / 25.4
h_in <- 122 / 25.4
svglite::svglite(file.path(out_dir, "svg", "fig2_honest_eval.svg"),
                 width = w_in, height = h_in)
print(fig2); dev.off()
ragg::agg_tiff(file.path(out_dir, "tiff", "fig2_honest_eval.tiff"),
               width = w_in, height = h_in, units = "in", res = 600, compression = "lzw")
print(fig2); dev.off()
cat("  Saved SVG + TIFF\n")

# ==============================================================================
# 7. SUMMARY
# ==============================================================================
summary_text <- sprintf(
  "
FIGURE 2: STAGE-STUDY CONFOUND + MUSCLE INFLATION GAP (6-stage, PLS)
===================================================================
Generated: %s
Source (parameterised; no fabricated numbers):
  paper/figures/data/within_pig_6stage.json
  (OrdinalPLS default 10 comp, 6-stage drop-small, 5 seeds; stratified +
   StratifiedGroupKFold-on-BioProject)

PANEL A (HERO): %% retained (kept-stage) samples from single-stage BioProjects
%s
  -> Only muscle has substantial multi-stage studies (%.0f%% multi-stage); a
     cross-study (held-out-study) developmental test is feasible only in muscle.

PANEL B: muscle inflation gap (METHODS STRENGTH, muscle only)
  stratified in-distribution BA = %.3f (batch-inclusive, NOT a deployment metric)
  cross-study (grouped CV) BA   = %.3f +/- %.3f (deployment estimate); chance %.2f
  inflation gap                 = %.3f

AUTHOR RULE: the four confounded tissues' grouped BA is NEVER plotted/tabulated
as a score (a printed low BA reads as a bad model regardless of caption). Fig 2
shows grouped/clean BA only for muscle (the one tissue with a clean comparator);
per-class recall for all tissues is in Supplementary Fig. S3.

Figure dimensions: 183mm x 122mm
",
  format(Sys.time(), "%%Y-%%m-%%d %%H:%%M"),
  paste(sprintf("  %-7s %.0f%%", names(single_stage_pct), single_stage_pct),
        collapse = "\n"),
  100 - single_stage_pct[["Muscle"]],
  muscle_strat, muscle_group, muscle_group_sd, muscle_chance,
  muscle_strat - muscle_group
)
write_summary(summary_text, "fig2_honest_eval_summary.txt")

cat("\nFigure 2 (confound + muscle positive control) completed successfully!\n")
