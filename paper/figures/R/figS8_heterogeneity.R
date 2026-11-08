#!/usr/bin/env Rscript
# ==============================================================================
# Supplementary Figure: Within-stage subgroup heterogeneity
# ==============================================================================
# R port of heterogeneity_figure.py (house style). Panels:
#   a - dominance heatmaps (tissue x stage) for Breed / Sex / BioProject
#   b - per-subgroup balanced accuracy vs the marginal BA (point size ~ n)
#   c - observed std(BA) vs permutation-null band (CSV-backed; the raw 1000-value
#       null is not stored, so each pair shows observed vs null mean->q95 + p)
# Output basename kept as figS8_heterogeneity.
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

cat("Supplementary Figure: heterogeneity (R port)\n")

TISSUES <- c("Muscle", "Brain", "Liver", "Lung", "Blood")
DIMS    <- c("Breed", "Sex", "BioProject")
rp <- function(f) project_path("review/analyses/results", f)

# ---- Panel a: dominance heatmaps -------------------------------------------
strat <- readr::read_csv(rp("heterogeneity_stratification.csv"), show_col_types = FALSE)
dom <- strat %>%
  filter(dimension %in% DIMS) %>%
  group_by(dimension, tissue, stage) %>%
  summarise(total = sum(n_samples), dominance = max(n_samples) / sum(n_samples),
            .groups = "drop") %>%
  mutate(tissue = factor(tissue, levels = rev(TISSUES)),
         stage  = factor(stage, levels = STAGE_ORDER, labels = STAGE_LABELS[STAGE_ORDER]),
         dimension = factor(dimension, levels = DIMS))

# Per-cell label colour: white on dark (high-dominance) cells, dark grey on light
# cells, so the annotated value stays legible across the full sequential ramp.
# Reproduce the fill interpolation (same ramp + limits 0.2-1) and switch on the
# cell's relative luminance.
.seq_ramp <- scales::seq_gradient_pal(unname(SEQ_RAMP["low"]), unname(SEQ_RAMP["high"]),
                                      space = "Lab")
dom <- dom %>%
  mutate(
    .cell_hex = .seq_ramp(scales::rescale(pmin(pmax(dominance, 0.2), 1),
                                          to = c(0, 1), from = c(0.2, 1))),
    label_col = {
      rgb <- grDevices::col2rgb(.cell_hex) / 255
      lum <- 0.2126 * rgb[1, ] + 0.7152 * rgb[2, ] + 0.0722 * rgb[3, ]
      ifelse(lum < 0.5, "white", "grey15")
    }
  )

p_a <- ggplot(dom, aes(stage, tissue, fill = dominance)) +
  geom_tile(colour = "white", linewidth = 0.4) +
  geom_text(aes(label = sprintf("%.2f", dominance), colour = label_col), size = 1.7) +
  scale_colour_identity() +
  facet_wrap(~dimension, nrow = 1) +
  scale_fill_house_seq(name = "Dominance\n(max level / n)", limits = c(0.2, 1)) +
  scale_x_discrete(expand = c(0, 0)) +
  labs(x = NULL, y = NULL) +
  nature_theme() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
        strip.text = element_text(size = 6.2, face = "bold"),
        panel.grid = element_blank(), legend.key.height = unit(5, "mm"))

# ---- Panel b: subgroup BA vs marginal --------------------------------------
perf <- readr::read_csv(rp("heterogeneity_subgroup_performance.csv"), show_col_types = FALSE) %>%
  mutate(tissue = factor(tissue, levels = TISSUES))
marg <- perf %>% filter(dimension == "Overall") %>%
  transmute(tissue, ba = balanced_accuracy, x = as.integer(tissue))
sub  <- perf %>% filter(dimension != "Overall") %>%
  mutate(dimension = factor(dimension, levels = DIMS))
dim_pal <- setNames(categorical_palette(length(DIMS)), DIMS)

p_b <- ggplot(sub, aes(x = as.integer(tissue), y = balanced_accuracy)) +
  geom_hline(yintercept = 0.5, linetype = "22", colour = "grey70", linewidth = 0.3) +
  geom_segment(data = marg, aes(x = x - 0.36, xend = x + 0.36, y = ba, yend = ba),
               colour = "black", linewidth = 0.5, inherit.aes = FALSE) +
  geom_point(aes(size = n_samples, colour = dimension),
             position = position_jitter(width = 0.16, height = 0, seed = 1), alpha = 0.85) +
  scale_colour_manual(values = dim_pal, name = "Subgroup") +
  scale_size_continuous(range = c(0.6, 3.2), name = "n") +
  scale_x_continuous(breaks = seq_along(TISSUES), labels = TISSUES) +
  labs(x = NULL, y = "Balanced accuracy",
       subtitle = "Per-subgroup BA (points, n>=10); marginal BA = black bar") +
  nature_theme() +
  theme(plot.subtitle = element_text(size = 6, colour = "grey35"),
        legend.position = "right", legend.box = "vertical")

# ---- Panel c: observed std(BA) vs permutation-null band --------------------
pv <- readr::read_csv(rp("heterogeneity_permutation_pvalues.csv"), show_col_types = FALSE) %>%
  mutate(pair = sprintf("%s / %s", tissue, dimension),
         sig  = pvalue < 0.05,
         pair = fct_reorder(pair, observed_std_ba))

p_c <- ggplot(pv, aes(y = pair)) +
  geom_segment(aes(x = perm_null_mean_std, xend = perm_null_q95_std,
                   yend = pair), colour = NULL_COLOR, linewidth = 1.6, alpha = 0.7) +
  geom_point(aes(x = observed_std_ba, colour = sig), size = 2.2) +
  geom_text(aes(x = observed_std_ba, label = sprintf("p=%.3f", pvalue)),
            vjust = -0.9, size = 1.7, colour = "grey25") +
  scale_colour_manual(values = c("TRUE" = SIG_COLOR, "FALSE" = NULL_COLOR),
                      labels = c("TRUE" = "p<0.05", "FALSE" = "n.s."), name = NULL) +
  labs(x = "std(subgroup BA): observed point vs null mean->95th-pctile band",
       y = NULL) +
  nature_theme() +
  theme(legend.position = "right", plot.margin = margin(2, 4, 2, 2, "mm"))

fig <- p_a / p_b / p_c +
  plot_layout(heights = c(1.0, 1.05, 1.0)) +
  plot_annotation(tag_levels = "a",
    theme = theme(plot.background = element_rect(fill = "white", colour = NA),
                  plot.tag = element_text(size = 9, face = "bold")))

save_figure(fig, "figS8_heterogeneity", width = 183, height = 195)
cat("\n✓ figS8_heterogeneity done\n")
