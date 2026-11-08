#!/usr/bin/env Rscript
# ==============================================================================
# Figure 5: The linear-conservation trade-off (nonlinearity sweep, Exp B)
# ==============================================================================
# THESIS (the paper's central finding): cross-species developmental conservation
# resides in the LINEAR / additive component of the pig program. As a stage model
# is made more nonlinear (linear PLS/Ridge -> gradient-boosted trees of growing
# depth), its WITHIN-PIG muscle accuracy RISES but its CROSS-SPECIES transfer
# COLLAPSES -- except for MUSCLE, which uniquely carries a conserved low-order
# interaction (its transfer survives, even improves, with depth >= 2).
#
# Panels:
#   a (HERO) - the scissors: within-pig muscle grouped-CV BA (rises) vs mean
#              cross-species transfer rho over the 4 non-muscle targets (falls),
#              across the model-complexity sweep. Both are in [0,1], one axis.
#   b        - per-target transfer rho across the sweep: every non-muscle target
#              (Lung/Adipose/Heart/Testis) collapses at the linear->tree boundary,
#              while MUSCLE is maintained/recovered -- the lone exception.
#
# Backend: R / ggplot2 (exclusive). White background. cairo PDF + PNG + SVG/TIFF.
#
# HARD RULE / PARAMETERISED DATA (no fabricated numbers):
#   paper/figures/data/transfer_6stage/expB_sweep.csv
#   (per model: muscle_BA_mean/sd from the within-pig 6-stage grouped CV; mean
#    transfer rho ex-Muscle; per-target dGTEx/Cardoso rho + scramble p.)
# ==============================================================================

set.seed(42)

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
cat("Figure 5: linear-conservation trade-off (nonlinearity sweep)\n")
cat("==============================================================================\n\n")

# ==============================================================================
# 1. DATA
# ==============================================================================
sweep_csv <- project_path("paper/figures/data/transfer_6stage/expB_sweep.csv")
if (!file.exists(sweep_csv)) stop("Exp B sweep CSV not found: ", sweep_csv)
sw <- readr::read_csv(sweep_csv, show_col_types = FALSE)

# Model complexity order: linear (PLS, Ridge) -> gradient-boosted trees by depth.
MODEL_LV  <- c("pls", "ridge", "lgb_depth1", "lgb_depth2", "lgb_depth4", "lgb_unlimited")
MODEL_LAB <- c(pls = "PLS", ridge = "Ridge", lgb_depth1 = "GBM\nd1",
               lgb_depth2 = "GBM\nd2", lgb_depth4 = "GBM\nd4",
               lgb_unlimited = "GBM\nd∞")
sw <- sw %>%
  filter(model %in% MODEL_LV) %>%
  mutate(model = factor(model, levels = MODEL_LV),
         xi = as.integer(model),
         kind = ifelse(model %in% c("pls", "ridge"), "linear", "tree"))

MUSCLE_CHANCE <- 0.20           # 5-class muscle (within_pig_6stage.json)
LINEAR_TREE_X <- 2.5            # boundary between Ridge (2) and GBM d1 (3)

WITHIN_COL <- "#6A51A3"         # within-pig (purple; matches Fig 4 panel d)
TRANSF_COL <- "#1B7837"         # cross-species transfer (green)

cat(sprintf("  Models: %s\n", paste(levels(sw$model), collapse = ", ")))
cat(sprintf("  Within-pig muscle BA: %.3f (PLS) -> %.3f (GBM dInf)\n",
            sw$muscle_BA_mean[sw$model == "pls"],
            sw$muscle_BA_mean[sw$model == "lgb_unlimited"]))
cat(sprintf("  Mean transfer rho ex-Muscle: %.3f (PLS) -> %.3f (GBM dInf)\n",
            sw$mean_rho_exMuscle4[sw$model == "pls"],
            sw$mean_rho_exMuscle4[sw$model == "lgb_unlimited"]))

# ==============================================================================
# 2. PANEL A (HERO): the scissors -- within-pig BA up, transfer rho down
# ==============================================================================
create_panel_a <- function() {
  cat("Creating Panel A (HERO): within-pig BA vs transfer rho scissors...\n")

  long <- bind_rows(
    sw %>% transmute(xi, model,
                     value = muscle_BA_mean, sd = muscle_BA_sd,
                     series = "within-pig muscle BA (grouped CV)"),
    sw %>% transmute(xi, model,
                     value = mean_rho_exMuscle4, sd = NA_real_,
                     series = "cross-species transfer ρ (ex-muscle mean)")
  ) %>%
    mutate(series = factor(series,
             levels = c("within-pig muscle BA (grouped CV)",
                        "cross-species transfer ρ (ex-muscle mean)")))

  SER_COL <- setNames(c(WITHIN_COL, TRANSF_COL), levels(long$series))

  ggplot(long, aes(xi, value, colour = series, group = series)) +
    # linear | tree regime divider
    annotate("rect", xmin = 0.5, xmax = LINEAR_TREE_X, ymin = -Inf, ymax = Inf,
             fill = "grey95", colour = NA) +
    annotate("text", x = 1.5, y = 0.97, label = "linear", size = 2.0,
             colour = "grey45", fontface = "italic") +
    annotate("text", x = 4.5, y = 0.97, label = "gradient-boosted trees",
             size = 2.0, colour = "grey45", fontface = "italic") +
    # chance line for the within-pig BA
    geom_hline(yintercept = MUSCLE_CHANCE, linetype = "22", linewidth = 0.4,
               colour = CHANCE_COLOR) +
    annotate("text", x = 6.3, y = MUSCLE_CHANCE + 0.03, label = "BA chance 0.20",
             size = 1.7, colour = CHANCE_COLOR, hjust = 1) +
    geom_errorbar(aes(ymin = value - sd, ymax = value + sd), width = 0.12,
                  linewidth = 0.35, na.rm = TRUE) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 1.9) +
    scale_colour_manual(values = SER_COL, name = NULL) +
    scale_x_continuous(breaks = sw$xi, labels = MODEL_LAB[levels(sw$model)],
                       expand = expansion(mult = c(0.02, 0.02))) +
    scale_y_continuous(limits = c(0, 1.0), breaks = seq(0, 1, 0.25),
                       expand = expansion(mult = c(0, 0.02))) +
    labs(title = "The linear-conservation trade-off (a)",
         subtitle = paste0("more nonlinear -> within-pig accuracy rises, ",
                           "cross-species transfer collapses"),
         x = NULL, y = "Balanced accuracy  /  transfer ρ") +
    nature_theme() +
    theme(legend.position = "top", legend.justification = "left",
          legend.text = element_text(size = 5.4),
          legend.key.size = unit(3, "mm"),
          plot.subtitle = element_text(size = 5.6, colour = "grey35"),
          axis.text.x = element_text(size = 5.6, lineheight = 0.82))
}

# ==============================================================================
# 3. PANEL B: per-target transfer rho across the sweep -- the muscle exception
# ==============================================================================
create_panel_b <- function() {
  cat("Creating Panel B: per-target transfer rho (muscle exception)...\n")

  tgt <- sw %>%
    transmute(xi, model,
              `Muscle (dGTEx)`  = rho_dG_Muscle,
              `Lung (dGTEx)`    = rho_dG_Lung,
              `Adipose (dGTEx)` = rho_dG_Adipose,
              `Heart (Cardoso)` = rho_Card_Heart,
              `Testis (Cardoso)`= rho_Card_Testis) %>%
    pivot_longer(-c(xi, model), names_to = "target", values_to = "rho") %>%
    filter(!is.na(rho)) %>%
    mutate(is_muscle = grepl("Muscle", target))

  TGT_COL <- c("Muscle (dGTEx)"  = unname(TISSUE_COLORS["Muscle"]),
               "Lung (dGTEx)"    = "#3a7dba",
               "Adipose (dGTEx)" = "#E6863C",
               "Heart (Cardoso)" = "#1b7837",
               "Testis (Cardoso)"= "#9970ab")

  ggplot(tgt, aes(xi, rho, colour = target, group = target)) +
    annotate("rect", xmin = 0.5, xmax = LINEAR_TREE_X, ymin = -Inf, ymax = Inf,
             fill = "grey95", colour = NA) +
    geom_hline(yintercept = 0, colour = "grey80", linewidth = 0.4) +
    # muscle drawn thick (the exception); others thinner
    geom_line(aes(linewidth = is_muscle)) +
    geom_point(aes(size = is_muscle)) +
    scale_linewidth_manual(values = c(`TRUE` = 1.1, `FALSE` = 0.5), guide = "none") +
    scale_size_manual(values = c(`TRUE` = 2.0, `FALSE` = 1.2), guide = "none") +
    scale_colour_manual(values = TGT_COL, name = NULL) +
    scale_x_continuous(breaks = sw$xi, labels = MODEL_LAB[levels(sw$model)],
                       expand = expansion(mult = c(0.02, 0.02))) +
    scale_y_continuous(limits = c(-0.3, 1.0), breaks = c(0, 0.5, 1.0)) +
    labs(title = "Muscle is the exception (b)",
         subtitle = "non-muscle targets collapse at the linear→tree boundary; muscle is maintained",
         x = NULL, y = expression(transfer~rho)) +
    nature_theme() +
    theme(legend.position = "right",
          legend.text = element_text(size = 5.0),
          legend.key.size = unit(2.6, "mm"),
          plot.subtitle = element_text(size = 5.4, colour = "grey35"),
          axis.text.x = element_text(size = 5.6, lineheight = 0.82))
}

# ==============================================================================
# 4. ASSEMBLY + SAVE
# ==============================================================================
cat("\nAssembling figure...\n")
pa <- create_panel_a()
pb <- create_panel_b()

fig5 <- pa / pb +
  plot_layout(heights = c(1.0, 0.9)) +
  plot_annotation(
    tag_levels = list(c("a", "b")),
    theme = theme(
      plot.background = element_rect(fill = "white", color = NA),
      plot.tag = element_text(size = 9, face = "bold")
    )
  )

cat("\nSaving figure...\n")
save_figure(fig5, "fig5_nonlinearity_sweep", width = 140, height = 150)

out_dir <- project_path("paper/figures/output")
dir.create(file.path(out_dir, "svg"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_dir, "tiff"), showWarnings = FALSE, recursive = TRUE)
w_in <- 140 / 25.4; h_in <- 150 / 25.4
svglite::svglite(file.path(out_dir, "svg", "fig5_nonlinearity_sweep.svg"),
                 width = w_in, height = h_in)
print(fig5); dev.off()
ragg::agg_tiff(file.path(out_dir, "tiff", "fig5_nonlinearity_sweep.tiff"),
               width = w_in, height = h_in, units = "in", res = 600, compression = "lzw")
print(fig5); dev.off()
cat("  Saved SVG + TIFF\n")

# ==============================================================================
# 5. SUMMARY
# ==============================================================================
summary_text <- sprintf(
"
FIGURE 5: LINEAR-CONSERVATION TRADE-OFF (NONLINEARITY SWEEP, Exp B)
==================================================================
Generated: %s
Source: paper/figures/data/transfer_6stage/expB_sweep.csv

PANEL A (HERO): within-pig muscle grouped-CV BA vs mean cross-species transfer
rho (4 non-muscle targets), across the model-complexity sweep.
  within-pig BA   : %.3f (PLS) -> %.3f (GBM dInf)   [RISES]
  transfer rho    : %.3f (PLS) -> %.3f (GBM dInf)   [FALLS]

PANEL B: per-target transfer rho across the sweep. Non-muscle targets collapse
at the linear->tree boundary; MUSCLE is maintained/recovered (depth>=2), the
lone exception carrying a conserved low-order interaction.

Figure dimensions: 140mm x 150mm
",
  format(Sys.time(), "%%Y-%%m-%%d %%H:%%M"),
  sw$muscle_BA_mean[sw$model == "pls"], sw$muscle_BA_mean[sw$model == "lgb_unlimited"],
  sw$mean_rho_exMuscle4[sw$model == "pls"], sw$mean_rho_exMuscle4[sw$model == "lgb_unlimited"]
)
write_summary(summary_text, "fig5_nonlinearity_sweep_summary.txt")

cat("\nFigure 5 (nonlinearity sweep) completed successfully!\n")
