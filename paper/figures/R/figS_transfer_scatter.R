#!/usr/bin/env Rscript
# ==============================================================================
# Supplementary Figure: what each cross-species transfer rho actually LOOKS like
# ==============================================================================
# CLAIM: the single pig->target Spearman rho reported per cell in Fig 4 is a
# faithful summary of a genuine monotonic correlation between the pig-trained PLS
# developmental score and the target species' true developmental ordering. This
# figure shows that correlation directly, one scatter per (organ x species) cell.
#
# Each panel: x = target developmental progression (scaled 0-1 from the ordinal
# developmental rank), y = projected pig PLS score (min-max scaled 0-1 within the
# cell). Both transforms are strictly monotone, so the annotated Spearman rho is
# IDENTICAL to the published Fig-4 value (verified upstream in
# paper/figures/provenance/dump_transfer_scatter.py, worst |drift| = 0.0000 over 47 cells).
#
# Panels:
#   a - Cardoso-Moreira 2019 grid: 6 organs (rows, by descending mean rho) x
#       7 amniote species (cols, by divergence from pig). Empty cells = organ
#       not sampled for that species (Ovary in Human/Macaque).
#   b - Human dGTEx grid: 7 organs (AGECOHORT 1-4 ordinal), the independent
#       human-cohort confirmation substrate.
#
# rho annotation: bold + "*" where the 1000x ortholog-scramble null is passed
# (p < 0.05); grey + "n.s." otherwise. n = target-sample count per cell.
#
# DATA (parameterised; a staging refresh is a data swap, not a code edit):
#   per-sample scores : paper/figures/data/transfer_scatter_6stage.csv
#   Cardoso rho + p   : paper/figures/data/determinants_6stage.json
#   dGTEx rho + p     : paper/figures/data/supp_table_dGTEx_only_6stage.csv
#
# Backend: R / ggplot2 (exclusive). White background. Seed 42.
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
cat("Supplementary Figure: per-cell cross-species transfer correlation scatter\n")
cat("==============================================================================\n\n")

# ==============================================================================
# 1. DATA
# ==============================================================================
scatter <- readr::read_csv(
  project_path("paper/figures/data/transfer_scatter_6stage.csv"),
  show_col_types = FALSE)

# Per-cell rho + scramble-null pass flag + n_target.
det <- jsonlite::fromJSON(
  project_path("paper/figures/data/determinants_6stage.json"),
  simplifyVector = FALSE)
cardoso_stats <- map_dfr(det$points, function(p) tibble(
  substrate = "Cardoso", organ = p$organ, species = p$species,
  rho = as.numeric(p$rho), scramble_p = as.numeric(p$scramble_p_systemA),
  n_target = as.numeric(p$n_target)))

dgtex_stats <- readr::read_csv(
  project_path("paper/figures/data/supp_table_dGTEx_only_6stage.csv"),
  show_col_types = FALSE) %>%
  transmute(substrate = "dGTEx", organ = organ, species = "Human",
            rho = as.numeric(rho), scramble_p = as.numeric(scramble_p),
            n_target = as.numeric(n_target))

stats_all <- bind_rows(cardoso_stats, dgtex_stats) %>%
  mutate(
    sig = !is.na(scramble_p) & scramble_p < 0.05,
    rho_lab = ifelse(sig, sprintf("bold(rho == %.2f ~ '*')", rho),
                          sprintf("rho == %.2f", rho)),
    n_lab = sprintf("n == %d", as.integer(n_target))
  )

# ==============================================================================
# 2. SCALING + ORDERING (monotone -> rho unchanged)
# ==============================================================================
# Per-cell min-max scaling so every panel shares one 0-1 box; x from the ordinal
# developmental rank, y from the projected score.
rescale01 <- function(v) {
  rng <- range(v, na.rm = TRUE)
  if (diff(rng) == 0) return(rep(0.5, length(v)))
  (v - rng[1]) / diff(rng)
}
scatter <- scatter %>%
  group_by(substrate, organ, species) %>%
  mutate(x = rescale01(dev_rank), y = rescale01(score)) %>%
  ungroup()

# Organ row order: by descending Cardoso per-organ mean rho.
organ_mean <- cardoso_stats %>% group_by(organ) %>%
  summarise(m = mean(rho), .groups = "drop") %>% arrange(desc(m))
CARDOSO_ORGAN_LV <- organ_mean$organ                          # Brain..Ovary
SPP_DIV <- c(Human = 94, Macaque = 94, Mouse = 94, Rat = 94, Rabbit = 94,
             Opossum = 159, Chicken = 319)
SPP_LV <- names(sort(SPP_DIV))

# dGTEx organ order: by descending dGTEx rho.
DGTEX_ORGAN_LV <- dgtex_stats %>% arrange(desc(rho)) %>% pull(organ)

# Organ accent colours (shared with Fig 4 for cross-figure consistency).
ORGAN_COL <- c(Heart = "#b2182b", Testis = "#762a83", Kidney = "#1b7837",
               Brain = "#2166ac", Liver = "#e08214", Ovary = "#999999",
               Lung = "#0F8C79", Muscle = "#9970ab", Adipose = "#C2853B",
               Spleen = "#777777")

# ==============================================================================
# 3. PANEL BUILDER
# ==============================================================================
# Builds a faceted scatter grid (organ rows x species cols) for one substrate.
build_grid <- function(sub, organ_lv, species_lv, title, x_title, wrap = FALSE) {
  dat <- scatter %>% filter(substrate == sub) %>%
    mutate(organ = factor(organ, levels = organ_lv),
           species = factor(species, levels = species_lv))
  st <- stats_all %>% filter(substrate == sub) %>%
    mutate(organ = factor(organ, levels = organ_lv),
           species = factor(species, levels = species_lv))

  # rho/n annotation; scramble significance encoded by bold+'*' and text colour.
  # Split into sig / non-sig layers so the text colour stays OUTSIDE the organ
  # colour scale used by the points (avoids a two-scale clash).
  st_sig <- st %>% filter(sig)
  st_ns  <- st %>% filter(!sig)

  ggplot(dat, aes(x, y)) +
    geom_smooth(aes(colour = organ), method = "lm", se = FALSE,
                linewidth = 0.45, alpha = 0.9) +
    geom_point(aes(colour = organ), size = 0.62, alpha = 0.8, stroke = 0) +
    # rho (top-left): significant in blue, non-significant in grey
    geom_text(data = st_sig, aes(x = 0.03, y = 1.12, label = rho_lab),
              colour = SIG_COLOR, hjust = 0, vjust = 1, size = 1.95,
              parse = TRUE, inherit.aes = FALSE) +
    geom_text(data = st_ns, aes(x = 0.03, y = 1.12, label = rho_lab),
              colour = NEUTRAL_GREY, hjust = 0, vjust = 1, size = 1.95,
              parse = TRUE, inherit.aes = FALSE) +
    geom_text(data = st, aes(x = 0.03, y = 0.90, label = n_lab),
              hjust = 0, vjust = 1, size = 1.5, colour = NEUTRAL_GREY,
              parse = TRUE, inherit.aes = FALSE) +
    (if (wrap) facet_wrap(~ organ, nrow = 1)
     else facet_grid(organ ~ species, drop = FALSE, switch = "y")) +
    scale_colour_manual(values = ORGAN_COL, guide = "none") +
    scale_x_continuous(breaks = c(0, 1), labels = c("young", "old"), expand = c(0, 0)) +
    scale_y_continuous(breaks = c(0, 1), expand = c(0, 0)) +
    coord_cartesian(xlim = c(-0.04, 1.06), ylim = c(-0.03, 1.16), clip = "on") +
    labs(title = title, x = x_title, y = "pig PLS developmental score (scaled)") +
    nature_theme() +
    theme(
      panel.spacing = unit(2.2, "mm"),
      panel.border = element_rect(colour = "grey85", fill = NA, linewidth = 0.3),
      axis.line = element_blank(), axis.ticks = element_line(linewidth = 0.25),
      axis.text = element_text(size = 4.4, colour = "grey45"),
      strip.text.x = element_text(size = 6, face = "bold", margin = margin(1, 0, 1, 0)),
      strip.text.y.left = element_text(size = 6, face = "bold", angle = 0,
                                       margin = margin(0, 1, 0, 1)),
      strip.background = element_rect(fill = "grey96", colour = NA),
      plot.title = element_text(size = 8, face = "bold")
    )
}

# Annotate the not-sampled Cardoso cells (Ovary Human/Macaque) so the grid gap
# reads as "absent in the atlas", not "we dropped it".
missing_cardoso <- tidyr::crossing(organ = CARDOSO_ORGAN_LV, species = SPP_LV) %>%
  anti_join(cardoso_stats, by = c("organ", "species")) %>%
  mutate(organ = factor(organ, levels = CARDOSO_ORGAN_LV),
         species = factor(species, levels = SPP_LV),
         lab = "not\nsampled")

pa <- build_grid("Cardoso", CARDOSO_ORGAN_LV, SPP_LV,
                 "a  Cross-species transfer (Cardoso-Moreira 2019, 7 amniotes)",
                 "target developmental progression") +
  geom_text(data = missing_cardoso, aes(x = 0.5, y = 0.5, label = lab),
            hjust = 0.5, vjust = 0.5, size = 1.5, colour = "grey70",
            lineheight = 0.9, inherit.aes = FALSE)

pb <- build_grid("dGTEx", DGTEX_ORGAN_LV, "Human",
                 "b  Independent human confirmation (dGTEx, AGECOHORT 1-4)",
                 "target developmental progression (AGECOHORT)", wrap = TRUE)

# ==============================================================================
# 4. ASSEMBLE + SAVE
# ==============================================================================
fig <- pa / pb +
  plot_layout(heights = c(6, 1.5)) +
  plot_annotation(theme = theme(plot.background = element_rect(fill = "white", colour = NA)))

cat("Saving figure...\n")
save_figure(fig, "figS_transfer_scatter", width = 183, height = 215)

out_dir <- project_path("paper/figures/output")
dir.create(file.path(out_dir, "svg"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_dir, "tiff"), showWarnings = FALSE, recursive = TRUE)
w_in <- 183 / 25.4; h_in <- 215 / 25.4
svglite::svglite(file.path(out_dir, "svg", "figS_transfer_scatter.svg"),
                 width = w_in, height = h_in); print(fig); dev.off()
ragg::agg_tiff(file.path(out_dir, "tiff", "figS_transfer_scatter.tiff"),
               width = w_in, height = h_in, units = "in", res = 600,
               compression = "lzw"); print(fig); dev.off()
cat("  Saved SVG + TIFF\n")

# ==============================================================================
# 5. SUMMARY
# ==============================================================================
summ <- sprintf(
"SUPPLEMENTARY FIGURE: per-cell cross-species transfer correlation scatter
=========================================================================
Source (parameterised): paper/figures/data/transfer_scatter_6stage.csv
  Cardoso rho/p : determinants_6stage.json   dGTEx rho/p : supp_table_dGTEx_only_6stage.csv
Cells: Cardoso %d (6 organs x 7 species, minus Ovary Human/Macaque) + dGTEx %d organs.
Total target samples plotted: %d.
Each panel annotates the published Spearman rho (bold* = scramble p<0.05) over
min-max-scaled axes (monotone -> rho unchanged). Worst rho drift vs Fig 4 = 0.0000.
",
  nrow(cardoso_stats), nrow(dgtex_stats), nrow(scatter))
write_summary(summ, "figS_transfer_scatter_summary.txt")

cat("\nSupplementary transfer-scatter figure completed.\n")
