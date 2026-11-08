#!/usr/bin/env Rscript
# ==============================================================================
# Figure 4: Cross-species transfer of the porcine developmental program
#           (COMBINED two-substrate view)
# ==============================================================================
# CLAIM: On the staging basis, a pig-trained transcriptomic
# developmental-stage model generalises (i) within human across organs
# (dGTEx) and (ii) across amniotes (Cardoso-Moreira 2019) in a TISSUE-
# dependent, not phylogenetic-distance-dependent, manner.
#
# This figure UNIFIES BOTH transfer substrates in one hero heatmap, visually
# differentiated by colour ramp + header band + axis labels:
#   * dGTEx (human)         -> warm (orange) ramp, left block
#   * Cardoso-Moreira 2019  -> cool (blue)  ramp, right block (7 amniotes)
#
# Panels:
#   a - HERO: organ x target transfer heatmap (Spearman rho), two substrates
#       side by side, colour-differentiated. dGTEx-human | Cardoso 7 species.
#   b - H2 tissue effect: per-organ mean Cardoso transfer rho (ranked).
#   c - H1 distance effect (stacked pair on one shared divergence axis):
#       top    = principled species-level test (species-mean rho vs divergence,
#                n=7, the significant/load-bearing phylogeny statistic);
#       bottom = pseudoreplicated cell-level boxplots (n=40, non-significant).
#   d - batch-immune rescue: within-pig grouped CV vs cross-species transfer.
#
# The muscle gene-level conservation scatter (Cardoso-Moreira r=0.69 + paired
# fold-change) is NOT part of this figure; it lives as a SEPARATE SUPPLEMENTARY
# figure (gene-level, not model-transfer) to keep this figure a clean
# model-transfer matrix. See reply for caption guidance.
#
# Backend: R / ggplot2 (exclusive). White background.
# Editable SVG + cairo PDF + 600-dpi TIFF + PNG preview.
#
# HARD RULE / PARAMETERISED DATA (no hard-coded rho):
#   dGTEx-human rho  <- paper/figures/data/supp_table_dGTEx_only.csv
#                       (column `rho`)  [staging canonical]
#   dGTEx n samples  <- paper/figures/data/transfer_results/dgtex_family_<organ>.json
#                       (`Human_dGTEx$n_target_samples`)  [context only]
#   Cardoso rho/tests<- paper/figures/data/transfer_results/determinants_systemB.json
#   A staging-model refresh = swap those data files; no code edit needed.
# NHP-dGTEx (macaque/marmoset dGTEx) is DROPPED. Cardoso macaque is KEPT.
# Missing cells (Ovary Human/Macaque) shown as grey "NA". Seed 42.
# ==============================================================================

set.seed(42)

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(ggrepel)
  library(jsonlite)
  library(scales)
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
cat("Figure 4: Cross-Species Transfer -- COMBINED two-substrate view\n")
cat("==============================================================================\n\n")

# ==============================================================================
# 1. DATA LOADING (authoritative staging artefacts; parameterised)
# ==============================================================================

# ---- (i) Cardoso 7-species transfer + hypothesis tests (6-stage) -------------
det_path <- project_path("paper/figures/data/determinants_6stage.json")
if (!file.exists(det_path)) stop("determinants file not found: ", det_path)
det <- jsonlite::fromJSON(det_path, simplifyVector = FALSE)
cat(sprintf("  Loaded determinants_6stage.json: %d Cardoso cells\n",
            length(det$points)))

cardoso <- map_dfr(det$points, function(p) {
  tibble(
    organ          = p$organ,
    species        = p$species,
    rho            = if (is.null(p$rho)) NA_real_ else as.numeric(p$rho),
    divergence_mya = as.numeric(p$divergence_mya),
    n_target       = as.numeric(p$n_target),
    n_orthologs    = as.numeric(p$n_orthologs),
    scramble_p     = as.numeric(p$scramble_p_systemA)
  )
}) %>%
  mutate(substrate = "Cardoso",
         significant = !is.na(scramble_p) & scramble_p < 0.05)

tests    <- det$tests
H1       <- tests$H1_rho_vs_divergence_spearman
H2c      <- tests$H2_rho_vs_complexity
h2_means <- tests$H2_tissue_mean_rho
h2_breadth <- tests$H2_pig_breadth

# ---- (ii) dGTEx-human transfer (staging rho from canonical CSV, 6-stage) -----
dgtex_csv <- project_path("paper/figures/data/supp_table_dGTEx_only_6stage.csv")
if (!file.exists(dgtex_csv)) stop("dGTEx staging CSV not found: ", dgtex_csv)
dgtex_raw <- readr::read_csv(dgtex_csv, show_col_types = FALSE)
if (!"rho" %in% names(dgtex_raw)) {
  stop("Expected column `rho` in ", dgtex_csv)
}

# dGTEx target-sample n for context (staging-independent), from family JSONs.
# Read defensively; missing -> NA (label simply omits the n).
read_dgtex_n <- function(organ) {
  jf <- project_path(sprintf("paper/figures/data/transfer_results/dgtex_family_%s.json",
                             tolower(organ)))
  if (!file.exists(jf)) return(NA_real_)
  j <- tryCatch(jsonlite::fromJSON(jf, simplifyVector = FALSE),
                error = function(e) NULL)
  if (is.null(j)) return(NA_real_)
  hk <- grep("Human|dGTEx", names(j), value = TRUE)
  if (length(hk) == 0) return(NA_real_)
  n <- j[[hk[1]]]$n_target_samples
  if (is.null(n)) NA_real_ else as.numeric(n)
}

# PLS revision: the regenerated CSV carries per-cell n_target, scramble_p and a
# `flag` column ("underpowered" where n_target < 15). Heart (n=7) and Testis
# (n=10) additionally FAIL the PLS ortholog-scramble null (scramble_p >= 0.05):
# they ride a global-magnitude / library-size confound, not cardiac/testis genes.
# Per the locked framing (number_map_PLS.md) heart/testis conservation must read
# from the well-powered Cardoso columns, NOT these underpowered dGTEx points.
has_flag    <- "flag" %in% names(dgtex_raw)
has_n_csv   <- "n_target" %in% names(dgtex_raw)
has_scr_csv <- "scramble_p" %in% names(dgtex_raw)

dgtex <- dgtex_raw %>%
  transmute(
    organ      = organ,
    species    = "Human",
    rho        = as.numeric(rho),
    substrate  = "dGTEx",
    n_target   = if (has_n_csv)   as.numeric(n_target) else NA_real_,
    scramble_p = if (has_scr_csv) as.numeric(scramble_p) else NA_real_,
    flag       = if (has_flag)    as.character(flag) else NA_character_
  ) %>%
  mutate(
    underpowered  = !is.na(flag) & flag == "underpowered",
    # scramble-fail among the underpowered cells (null not beaten at p<0.05)
    scramble_fail = !is.na(scramble_p) & scramble_p >= 0.05
  )

# If the CSV lacked n_target, fall back to the family-JSON reader (context only).
if (!has_n_csv) {
  dgtex <- dgtex %>% rowwise() %>%
    mutate(n_target = read_dgtex_n(organ)) %>% ungroup()
}

# Cells to mask in the hero heatmap: underpowered (n<15) dGTEx cells that ALSO
# have a Cardoso column, so their conservation is read from the well-powered
# Cardoso block rather than the n<15 dGTEx point. At 6-stage this is Heart,
# Testis and Liver (all underpowered + scramble-failing + present in Cardoso).
# Spleen (underpowered but dGTEx-only) is NOT masked: it has no Cardoso fallback.
DGTEX_MASK <- intersect(unique(cardoso$organ), dgtex$organ[dgtex$underpowered])

cat(sprintf("  Loaded dGTEx (PLS) rho for %d organs (column rho)\n",
            nrow(dgtex)))
cat(sprintf("  dGTEx underpowered (n<15): %s\n",
            paste(dgtex$organ[dgtex$underpowered], collapse = ", ")))
cat(sprintf("  dGTEx masked in hero (underpowered + scramble-fail): %s\n",
            paste(DGTEX_MASK, collapse = ", ")))

# Headline dGTEx set per number_map.md: 6 organs (Spleen excluded from headline).
DGTEX_HEADLINE <- c("Heart", "Testis", "Adipose", "Lung", "Muscle", "Liver")

# ==============================================================================
# 2. PALETTE / ORDERING
# ==============================================================================
# Two distinct sequential ramps so the substrates read apart instantly.
SEQ_CARDOSO <- c("#f7fbff", "#c9dcef", "#7fb0d6", "#3a7dba", "#185594", "#08306b") # blues
SEQ_DGTEX   <- c("#fff5eb", "#fdd9b4", "#fdae6b", "#f16913", "#d94801", "#8c2d04") # oranges
RHO_LIMS    <- c(-0.2, 1)
NA_FILL     <- "grey88"

# Substrate banner colours (header band + label)
SUB_COL <- c(dGTEx = "#d94801", Cardoso = "#185594")

# Organ display order: by descending Cardoso per-organ mean (H2), with dGTEx-only
# organs (Adipose, Lung, Spleen) appended in dGTEx-rho order beneath.
cardoso_order <- names(sort(unlist(h2_means), decreasing = TRUE)) # Heart..Ovary
dgtex_only    <- setdiff(dgtex$organ, cardoso_order)
dgtex_only    <- dgtex %>% filter(organ %in% dgtex_only) %>%
  arrange(desc(rho)) %>% pull(organ)
ORGAN_LV <- c(cardoso_order, dgtex_only)

# Cardoso species ordered by divergence from pig (closest -> farthest)
SPP_DIV <- c(Human = 94, Macaque = 94, Mouse = 94, Rat = 94, Rabbit = 94,
             Opossum = 159, Chicken = 319)
spp_order <- names(sort(SPP_DIV))

# Organ accent colours (for H1/H2 panels)
ORGAN_COL <- setNames(
  c("#b2182b", "#762a83", "#1b7837", "#2166ac", "#e08214", "#999999"),
  c("Heart", "Testis", "Kidney", "Brain", "Liver", "Ovary")
)

# ==============================================================================
# 3. PANEL A: unified two-substrate transfer heatmap (HERO)
# ==============================================================================
# Dependency-free two-ramp heatmap: each cell's fill is PRECOMPUTED from the
# substrate-specific palette (dGTEx -> oranges; Cardoso -> blues) and drawn with
# scale_fill_identity(); two compact manual colourbars are drawn as inset tile
# strips so the figure needs no ggnewscale. Differentiation = colour ramp +
# header band + per-substrate colourbar labels + a vertical divider.

# Map a rho in RHO_LIMS to a hex colour on a given palette (NA -> NA_FILL).
rho_to_hex <- function(rho, palette) {
  pal <- scales::gradient_n_pal(palette)
  out <- rep(NA_FILL, length(rho))
  ok  <- !is.na(rho)
  z   <- scales::rescale(pmax(pmin(rho[ok], RHO_LIMS[2]), RHO_LIMS[1]),
                         from = RHO_LIMS, to = c(0, 1))
  out[ok] <- pal(z)
  out
}

create_panel_a <- function(dgtex, cardoso) {
  cat("Creating Panel A: unified two-substrate transfer heatmap...\n")

  # Column = target. dGTEx-human first (1 col), then 7 Cardoso species.
  # Use NUMERIC x/y positions throughout so the header band, divider and tiles
  # share one continuous coordinate system (avoids discrete/continuous clashes).
  col_lv <- c("Human\n(dGTEx)", spp_order)  # spp_order: Human..Chicken (Cardoso)
  organ_lv_btt <- rev(ORGAN_LV)             # bottom-to-top plotting order
  xmap <- setNames(seq_along(col_lv), col_lv)
  ymap <- setNames(seq_along(organ_lv_btt), organ_lv_btt)

  long <- bind_rows(
    dgtex   %>% transmute(organ, substrate, species, rho, col = "Human\n(dGTEx)"),
    cardoso %>% transmute(organ, substrate, species, rho, col = species)
  ) %>%
    mutate(
      # Mask underpowered + scramble-failing dGTEx cells (Heart, Testis): grey
      # them out and replace the value with an asterisk. Their conservation must
      # be read from the Cardoso columns, not these n=7/n=10 dGTEx points.
      masked = substrate == "dGTEx" & organ %in% DGTEX_MASK,
      xi = xmap[col],
      yi = ymap[organ],
      fillhex = ifelse(masked, NA_FILL,
                ifelse(substrate == "dGTEx",
                       rho_to_hex(rho, SEQ_DGTEX),
                       rho_to_hex(rho, SEQ_CARDOSO))),
      lab    = ifelse(masked, "*",
                ifelse(is.na(rho), "NA", sprintf("%.2f", rho))),
      txtcol = ifelse(masked, "grey45",
                ifelse(is.na(rho), "grey55",
                      ifelse(rho > 0.55, "white", "grey15"))),
      txtsize = ifelse(masked, 3.4, 1.95)
    )

  ncol <- length(col_lv)
  band <- tibble(
    substrate = c("dGTEx", "Cardoso"),
    xmin = c(0.5, 1.5),
    xmax = c(1.5, ncol + 0.5),
    label = c("dGTEx (human)", "Cardoso-Moreira 2019")
  )
  ytop <- length(organ_lv_btt) + 0.5
  x_labels <- gsub("\\\\n", "\n", col_lv)

  band_lo <- ytop + 0.55
  band_hi <- ytop + 1.35

  ggplot() +
    # substrate header band (raised above the column labels)
    geom_rect(data = band,
              aes(xmin = xmin, xmax = xmax, ymin = band_lo, ymax = band_hi),
              fill = SUB_COL[band$substrate], colour = NA) +
    geom_text(data = band,
              aes(x = (xmin + xmax) / 2, y = (band_lo + band_hi) / 2, label = label),
              colour = "white", fontface = "bold", size = 2.0) +
    # heatmap tiles (precomputed per-substrate fill)
    geom_tile(data = long, aes(xi, yi, fill = fillhex),
              colour = "white", linewidth = 0.6, width = 0.98, height = 0.98) +
    geom_text(data = long, aes(xi, yi, label = lab, colour = txtcol, size = txtsize)) +
    scale_fill_identity() +
    scale_colour_identity() +
    scale_size_identity() +
    # vertical divider between substrates
    annotate("segment", x = 1.5, xend = 1.5, y = 0.5, yend = band_hi,
             colour = "grey20", linewidth = 0.5) +
    # horizontal divider separating Cardoso-measured organs (upper) from the
    # dGTEx-only organs (lower), so the unmeasured-cell greys read as structure.
    annotate("segment", x = 0.5, xend = ncol + 0.5,
             y = length(dgtex_only) + 0.5, yend = length(dgtex_only) + 0.5,
             colour = "grey55", linewidth = 0.4, linetype = "22") +
    scale_x_continuous(position = "top", breaks = seq_along(col_lv),
                       labels = x_labels, expand = c(0, 0),
                       limits = c(0.4, ncol + 0.6)) +
    scale_y_continuous(breaks = seq_along(organ_lv_btt),
                       labels = organ_lv_btt, expand = c(0, 0)) +
    coord_cartesian(ylim = c(0.4, band_hi + 0.15), clip = "off") +
    labs(
      title = "Cross-species transfer (PLS)",
      subtitle = paste0("dGTEx human (orange) vs Cardoso-Moreira 7 amniotes (blue).  ",
                        "Grey = not measured;  * = dGTEx underpowered (n<15), read from Cardoso"),
      x = NULL, y = NULL
    ) +
    nature_theme() +
    theme(
      axis.text.x = element_text(angle = 0, face = "bold", size = 5.4,
                                 margin = margin(b = 1)),
      axis.line = element_blank(), axis.ticks = element_blank(),
      legend.position = "none",
      plot.subtitle = element_text(size = 5.2, colour = "grey35"),
      plot.margin = margin(7, 5, 2, 5, "mm")
    )
}

# ------------------------------------------------------------------------------
# 3b. Dependency-free dual colourbar legend strip (for the two rho ramps)
# ------------------------------------------------------------------------------
create_legend_strip <- function() {
  nseg <- 80
  zgrid <- seq(0, 1, length.out = nseg)
  rho_grid <- scales::rescale(zgrid, from = c(0, 1), to = RHO_LIMS)
  ticks <- tibble(
    z = scales::rescale(c(-0.2, 0.2, 0.6, 1.0), from = RHO_LIMS, to = c(0, 1)),
    lab = c("-0.2", "0.2", "0.6", "1.0")
  )
  # Two bars on a shared 0-1 x scale, stacked vertical positions.
  bars <- bind_rows(
    tibble(z = zgrid, y = 2, fillhex = rho_to_hex(rho_grid, SEQ_DGTEX)),
    tibble(z = zgrid, y = 1, fillhex = rho_to_hex(rho_grid, SEQ_CARDOSO))
  )
  ggplot() +
    geom_tile(data = bars, aes(z, y, fill = fillhex),
              width = 1 / nseg * 1.04, height = 0.55) +
    scale_fill_identity() +
    # tick labels under the lower bar
    geom_text(data = ticks, aes(z, 0.55, label = lab), size = 1.5,
              colour = "grey35") +
    annotate("text", x = -0.06, y = 2, hjust = 1, size = 1.85,
             colour = SUB_COL["dGTEx"], fontface = "bold",
             label = "dGTEx") +
    annotate("text", x = -0.06, y = 1, hjust = 1, size = 1.85,
             colour = SUB_COL["Cardoso"], fontface = "bold",
             label = "Cardoso") +
    annotate("text", x = 0.5, y = 2.65, size = 1.9, colour = "grey20",
             label = expression(Spearman~rho)) +
    scale_x_continuous(limits = c(-0.35, 1.05), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0.2, 2.95), expand = c(0, 0)) +
    coord_cartesian(clip = "off") +
    theme_void() +
    theme(plot.margin = margin(1, 4, 1, 4, "mm"))
}

# ==============================================================================
# 4. PANEL B: per-organ mean Cardoso transfer rho ranking (H2 tissue effect)
# ==============================================================================
create_panel_b <- function() {
  cat("Creating Panel B: tissue effect (H2)...\n")
  tm <- tibble(
    organ = names(h2_means),
    mean_rho = as.numeric(unlist(h2_means))
  ) %>%
    mutate(organ = factor(organ, levels = rev(names(sort(unlist(h2_means))))))

  h2_lab <- sprintf("complexity~n.s.~(rho==%.2f*','~p==%.2f)",
                    H2c$spearman_r, H2c$p)

  ggplot(tm, aes(mean_rho, organ)) +
    geom_segment(aes(x = 0, xend = mean_rho, yend = organ),
                 color = "grey75", linewidth = 0.5) +
    geom_point(aes(fill = mean_rho), shape = 21, size = 2.6, stroke = 0.3,
               color = "grey20") +
    geom_text(aes(label = sprintf("%.2f", mean_rho)), hjust = -0.4, size = 1.85) +
    scale_fill_gradientn(colours = SEQ_CARDOSO, limits = RHO_LIMS, guide = "none",
                         na.value = NA_FILL) +
    scale_x_continuous(limits = c(0, 1.05), breaks = c(0, 0.5, 1.0)) +
    annotate("text", x = 0.02, y = 6.35, hjust = 0, vjust = 1, size = 1.75,
             color = "grey35", parse = TRUE, label = h2_lab) +
    coord_cartesian(clip = "off") +
    labs(title = "Tissue effect (H2, Cardoso)",
         x = expression(mean~Cardoso~rho), y = NULL) +
    nature_theme()
}

# ==============================================================================
# 5. PANEL C: H1 distance effect, shown as a stacked pair
#    c-top    = the principled SPECIES-level test (each species' mean transfer
#               vs divergence; n=7, the load-bearing phylogeny statistic).
#    c-bottom = the pseudoreplicated CELL-level view (per-cell boxplots; n=40).
#    Both share the divergence (MYA-from-pig) x-axis; only the bottom panel
#    carries the x labels. Keeping BOTH is deliberate: the significant claim
#    rests on the species-level test, but the non-significant cell-level test is
#    shown in full so nothing is selectively reported. sp-level rho/p recomputed
#    here (not hard-coded) so a staging refresh stays a data swap.
# ==============================================================================
create_panel_c_species <- function(cardoso) {
  cat("Creating Panel C (top): species-level distance test...\n")
  ce <- cardoso %>% filter(!is.na(rho))
  sp_data <- ce %>%
    group_by(species) %>%
    summarise(mean_rho = mean(rho, na.rm = TRUE),
              divergence_mya = first(divergence_mya),
              .groups = "drop") %>%
    mutate(deep = divergence_mya > 100)
  sp_test <- cor.test(sp_data$mean_rho, sp_data$divergence_mya,
                      method = "spearman", exact = FALSE)
  lab <- sprintf("species~rho[s]==%.2f*','~p==%.3f*','~n==7",
                 sp_test$estimate, sp_test$p.value)

  ggplot(sp_data, aes(divergence_mya, mean_rho)) +
    geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
                colour = "grey55", linewidth = 0.5, linetype = "22") +
    geom_point(aes(fill = deep), shape = 21, size = 1.9, stroke = 0.3,
               colour = "grey20") +
    scale_fill_manual(values = c(`FALSE` = "#4393c3", `TRUE` = "#d6604d"),
                      guide = "none") +
    ggrepel::geom_text_repel(
      data = dplyr::filter(sp_data, deep), aes(label = species),
      size = 1.7, colour = "grey25", min.segment.length = 0,
      segment.size = 0.2, box.padding = 0.3, seed = 42) +
    annotate("text", x = 338, y = 0.90, hjust = 1, size = 1.9, color = "grey15",
             label = lab, parse = TRUE) +
    annotate("text", x = 94, y = 0.685, hjust = 0.5, size = 1.55,
             color = "#2166ac", label = "5 eutherians") +
    scale_x_continuous(limits = c(70, 340), breaks = c(94, 159, 319)) +
    scale_y_continuous(limits = c(0.5, 0.92)) +
    labs(title = "Distance effect (H1): species-level",
         x = NULL, y = expression("species-mean"~rho)) +
    nature_theme() +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
}

create_panel_c_cell <- function(cardoso) {
  cat("Creating Panel C (bottom): cell-level boxplots...\n")
  ce <- cardoso %>% filter(!is.na(rho)) %>%
    mutate(organ = factor(organ, levels = c("Heart", "Testis", "Kidney",
                                             "Brain", "Liver", "Ovary")))
  h1_lab_cell <- sprintf("cell~rho[s]==%.2f*','~p==%.3f", H1$spearman_r, H1$p)

  ggplot(ce, aes(factor(divergence_mya), rho)) +
    geom_boxplot(outlier.shape = NA, fill = "grey96", colour = "grey50", width = 0.4, lwd = 0.35) +
    geom_jitter(aes(color = organ), width = 0.15, height = 0, size = 1.0, alpha = 0.85) +
    scale_color_manual(values = ORGAN_COL, name = "organ") +
    scale_x_discrete(labels = c("94" = "94\nEutheria", "159" = "159\nMarsupial", "319" = "319\nBird")) +
    scale_y_continuous(limits = c(-0.25, 1.18), breaks = c(0, 0.4, 0.8)) +
    annotate("text", x = 2, y = 1.12, hjust = 0.5, size = 1.85, color = "grey25",
             label = h1_lab_cell, parse = TRUE) +
    labs(title = "Cell-level (pseudoreplicated)", x = "MYA from pig",
         y = expression(Cardoso~rho)) +
    nature_theme() +
    theme(legend.position = c(0.52, 0.20), legend.direction = "horizontal",
          legend.key.size = unit(2.0, "mm"), legend.title = element_text(size = 5.6),
          legend.text = element_text(size = 5.0),
          axis.text.x = element_text(size = 5.2)) +
    guides(color = guide_legend(nrow = 2, override.aes = list(size = 1.3)))
}

# ==============================================================================
# 5b. PANEL D: within-pig grouped-CV rho (IDENTIFIABLE tissues only) vs
#     cross-species transfer rho (6-stage)
# ==============================================================================
# The batch-immune rescue. A within-pig grouped (leave-whole-study-out) estimate
# is IDENTIFIABLE for only two tissues (Supplementary Table S1; Methods):
#   * Muscle -- all five retained stages span multiple studies (47% single-stage);
#               grouped ordinal rho = 0.66.
#   * Liver  -- identifiable over its four multi-cohort stages (neonatal + adult
#               are each one dominant BioProject and are dropped); grouped ordinal
#               rho = +0.44. This is the SAME number behind the Table S1 / Fig S1a
#               liver cell (BA 0.39, rho 0.44, chance 0.25), NOT the degenerate
#               full-six-stage value in within_pig_6stage.json (-0.15).
# For Lung / Brain / Blood every retained stage is confined to a single study
# (100% single-stage): a held-out study presents an essentially unseen stage, so
# the grouped quantity is NOT identifiable. The manuscript reports no cross-study
# estimate for them, so we plot NO within-pig point and label it "n.i." here.
#
# Cross-species transfer fits on ALL pig data and projects onto a FOREIGN-batch
# atlas, so it cannot inherit the pig batch confound; it is the only valid
# developmental read-out for the three confounded tissues. Lung (dGTEx human
# rho 0.67) and Brain (Cardoso human rho 0.86) both clear the scramble null;
# Liver transfers weakly and non-significantly (rho 0.33, n.s.); Blood has no
# cross-species substrate (sex-confound negative control).
#
# within-pig grouped rho  <- paper/figures/data/within_pig_6stage.json (Muscle)
#                            review/analyses/results/liver_loco_stage_restricted.json
#                            ("drop Neonatal + Adult" -> Liver, the Table S1 value)
# cross-species rho        : design-matched human target, already loaded above --
#   Muscle/Lung/Liver -> dGTEx human (organ-matched);  Brain -> Cardoso human
#   (no dGTEx brain column);  Blood -> none. Filled point = passes the 1000x
#   ortholog-scramble null (p<0.05); open point = not significant.
create_panel_d <- function() {
  cat("Creating Panel D: within-pig grouped rho (identifiable tissues) vs cross-species transfer rho...\n")

  tissues <- c("Muscle", "Lung", "Brain", "Liver", "Blood")

  # within-pig grouped-CV ordinal rho: identifiable for Muscle & Liver only.
  wp <- jsonlite::fromJSON(
    project_path("paper/figures/data/within_pig_6stage.json"),
    simplifyVector = FALSE)
  liver_loco <- jsonlite::fromJSON(
    project_path("review/analyses/results/liver_loco_stage_restricted.json"),
    simplifyVector = FALSE)
  # Liver's manuscript value = grouped CV over its four multi-cohort stages.
  liver_rho <- as.numeric(liver_loco[["drop Neonatal + Adult"]]$rho)
  within_rho <- function(t) {
    if (t == "Muscle") return(as.numeric(wp[["Muscle"]]$grouped_6stage$rho_mean))
    if (t == "Liver")  return(liver_rho)
    NA_real_   # Lung / Brain / Blood: grouped estimate NOT identifiable (single-stage)
  }

  # cross-species design-matched human rho + scramble p (from data loaded above).
  dgtex_lk <- function(org) {
    r <- dgtex %>% filter(organ == org)
    if (nrow(r) == 0) return(c(rho = NA_real_, p = NA_real_))
    c(rho = as.numeric(r$rho[1]), p = as.numeric(r$scramble_p[1]))
  }
  brain_h <- cardoso %>% filter(organ == "Brain", species == "Human")
  cross_of <- function(t) {
    if (t == "Brain") {
      if (nrow(brain_h) == 0) return(c(rho = NA_real_, p = NA_real_))
      return(c(rho = as.numeric(brain_h$rho[1]), p = as.numeric(brain_h$scramble_p[1])))
    }
    if (t == "Blood") return(c(rho = NA_real_, p = NA_real_))
    dgtex_lk(t)
  }

  dd <- map_dfr(tissues, function(t) {
    cs <- cross_of(t)
    tibble(tissue = t, within_rho = within_rho(t),
           cross_rho = unname(cs["rho"]), cross_p = unname(cs["p"]))
  }) %>%
    mutate(cross_pass = !is.na(cross_p) & cross_p < 0.05,
           tissue = factor(tissue, levels = tissues))

  # Long form: within-pig + cross-species point per tissue (NA points dropped, so
  # only Muscle/Liver carry a within-pig point and a connecting line). `sig` =
  # always TRUE for within-pig; for the cross point = passes the scramble null.
  dl <- dd %>%
    transmute(tissue, within_rho, cross_rho, cross_pass) %>%
    pivot_longer(c(within_rho, cross_rho), names_to = "kind", values_to = "rho") %>%
    mutate(
      sig  = ifelse(kind == "within_rho", TRUE, dd$cross_pass[match(tissue, dd$tissue)]),
      kind = recode(kind, within_rho = "within-pig",
                          cross_rho  = "cross-species"),
      kind = factor(kind, levels = c("within-pig", "cross-species"))
    ) %>%
    filter(!is.na(rho))
  ns_pts <- dl %>% filter(!sig)   # non-significant cross-species points (Liver)

  # "Not identifiable" within-pig annotation for the single-stage tissues.
  ni <- tibble(
    tissue = factor(c("Lung", "Brain", "Blood"), levels = tissues),
    rho    = 0.03,
    label  = c("within-pig n.i.", "within-pig n.i.", "within-pig n.i. (no target)")
  )

  KIND_COL <- c("within-pig"    = "#9970ab",
                "cross-species" = "#1b7837")

  ggplot(dl, aes(rho, tissue, group = tissue)) +
    geom_vline(xintercept = 0, colour = "grey80", linewidth = 0.4) +
    # connecting segment within-pig <-> cross-species (drawn only where both exist)
    geom_line(aes(group = tissue), colour = "grey70", linewidth = 0.5,
              na.rm = TRUE) +
    # Plot significant points (filled circles with grey20 border)
    geom_point(data = filter(dl, sig), aes(fill = kind), shape = 21, size = 2.4, stroke = 0.3,
               colour = "grey20", na.rm = TRUE) +
    # Plot non-significant points (open circles: white fill, grey20 border)
    geom_point(data = filter(dl, !sig), shape = 21, fill = "white", colour = "grey20", size = 2.4, stroke = 0.3,
               na.rm = TRUE) +
    # mark non-significant cross-species points (do not beat the scramble null)
    geom_text(data = ns_pts, aes(rho, tissue, label = "n.s."),
              vjust = -1.1, size = 1.7, colour = "grey35", na.rm = TRUE) +
    # within-pig not-identifiable labels (Lung / Brain / Blood)
    geom_text(data = ni, aes(rho, tissue, label = label), inherit.aes = FALSE,
              hjust = 0, vjust = 0.5, size = 1.7, fontface = "italic",
              colour = "grey45") +
    scale_fill_manual(values = KIND_COL, name = NULL) +
    scale_y_discrete(drop = FALSE) +
    scale_x_continuous(limits = c(-0.05, 1.0), breaks = c(0, 0.25, 0.5, 0.75, 1.0)) +
    labs(title = "Batch-immune rescue",
         x = expression(Spearman~rho), y = NULL) +
    nature_theme() +
    theme(legend.position = c(0.80, 0.88), legend.direction = "vertical",
          legend.key.size = unit(2.2, "mm"),
          legend.text = element_text(size = 4.8),
          legend.background = element_rect(fill = scales::alpha("white", 0.6),
                                           colour = NA),
          axis.text.y = element_text(size = 6)) +
    guides(fill = guide_legend(override.aes = list(shape = 21, colour = "grey20", stroke = 0.3)))
}

# ==============================================================================
# 6. FIGURE ASSEMBLY
# ==============================================================================
cat("\nAssembling figure...\n")

pa      <- create_panel_a(dgtex, cardoso)
pleg    <- create_legend_strip()
pb      <- create_panel_b()
pc_sp   <- create_panel_c_species(cardoso)
pc_cell <- create_panel_c_cell(cardoso)
# Panel c is a stacked pair: species-level test (top) over cell-level view
# (bottom). Sharing one column keeps the a/b/c/d lettering (c = the composite).
pc      <- pc_sp / pc_cell + plot_layout(heights = c(0.85, 1.0))
pd      <- create_panel_d()

# Legend strip carries no panel tag; tag only a/b/c/d on the real panels. The
# nested c-composite contributes two leaves, so the c-bottom leaf is left untagged.
fig4 <- pa / pleg / (pb | pc | pd) +
  plot_layout(heights = c(1.05, 0.13, 0.95)) +
  plot_annotation(
    tag_levels = list(c("a", "", "b", "c", "", "d")),
    theme = theme(
      plot.background = element_rect(fill = "white", color = NA),
      plot.tag = element_text(size = 9, face = "bold")
    )
  )

# ==============================================================================
# 7. SAVE FIGURE (PDF + PNG via save_figure; plus SVG + 600-dpi TIFF)
# ==============================================================================
cat("\nSaving figure...\n")
save_figure(fig4, "fig4_cross_species", width = 183, height = 186)

out_dir <- project_path("paper/figures/output")
dir.create(file.path(out_dir, "svg"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_dir, "tiff"), showWarnings = FALSE, recursive = TRUE)
w_in <- 183 / 25.4
h_in <- 186 / 25.4
svglite::svglite(file.path(out_dir, "svg", "fig4_cross_species.svg"),
                 width = w_in, height = h_in)
print(fig4); dev.off()
ragg::agg_tiff(file.path(out_dir, "tiff", "fig4_cross_species.tiff"),
               width = w_in, height = h_in, units = "in", res = 600, compression = "lzw")
print(fig4); dev.off()
cat("  Saved SVG + TIFF\n")

# ==============================================================================
# 8. SUMMARY STATISTICS
# ==============================================================================
cat("\nWriting summary...\n")

dgtex_rank <- dgtex %>% arrange(desc(rho))
headline_rng <- dgtex %>% filter(organ %in% DGTEX_HEADLINE) %>%
  summarise(lo = min(rho), hi = max(rho))

tissue_rank <- tibble(
  organ = names(h2_means),
  mean_rho = as.numeric(unlist(h2_means))
) %>% arrange(desc(mean_rho))

summary_text <- sprintf(
  "
FIGURE 4: CROSS-SPECIES TRANSFER -- COMBINED TWO-SUBSTRATE VIEW
=========================================================================
Generated: %s
Sources (parameterised; a staging refresh is a data swap, not a code edit):
  dGTEx rho : paper/figures/data/supp_table_dGTEx_only.csv (rho)
  Cardoso   : paper/figures/data/transfer_results/determinants_systemB.json
Basis: %s  Seed: %d  Cardoso cells: %d

PANEL A (HERO): unified organ x target transfer heatmap.
  Left block  = dGTEx human (orange ramp), 1 target column.
  Right block = Cardoso-Moreira 2019 (blue ramp), 7 amniote columns.

dGTEx-HUMAN transfer rho, ranked:
%s
  Headline 6-organ range = %.2f - %.2f (Spleen excluded from headline).

CARDOSO per-organ mean transfer rho (H2), ranked:
%s
  rho vs pig transcriptomic complexity: rho = %.3f, p = %.3f (n=%d; n.s.)

H1 DISTANCE EFFECT (Cardoso rho vs MYA from pig):
  Spearman rho = %.3f, p = %.4f, n = %d
  Eutherian mean = %.3f (n=%d); Opossum = %.3f (n=%d); Chicken = %.3f (n=%d)

DEEPEST SPLIT (pig -> chicken, Cardoso):
  Heart %.3f | Testis %.3f | Kidney %.3f | Liver %.3f | Brain %.3f | Ovary %.3f

NOTE: muscle gene-level conservation scatter (Cardoso-Moreira r=0.69 + paired
FC) is NOT in this figure; it is a separate SUPPLEMENTARY (gene-level) figure.
Figure dimensions: 183mm x 168mm
",
  format(Sys.time(), "%%Y-%%m-%%d %%H:%%M"),
  det$basis, det$seed, length(det$points),
  paste(sprintf("  %-7s rho = %.2f", dgtex_rank$organ, dgtex_rank$rho),
        collapse = "\n"),
  headline_rng$lo, headline_rng$hi,
  paste(sprintf("  %-7s mean rho = %.3f", tissue_rank$organ, tissue_rank$mean_rho),
        collapse = "\n"),
  H2c$spearman_r, H2c$p, H2c$n_tissues,
  H1$spearman_r, H1$p, H1$n,
  tests$H1_eutherian_mean_rho, tests$H1_eutherian_n,
  tests$H1_opossum_mean_rho, tests$H1_opossum_n,
  tests$H1_chicken_mean_rho, tests$H1_chicken_n,
  cardoso %>% filter(organ == "Heart",  species == "Chicken") %>% pull(rho),
  cardoso %>% filter(organ == "Testis", species == "Chicken") %>% pull(rho),
  cardoso %>% filter(organ == "Kidney", species == "Chicken") %>% pull(rho),
  cardoso %>% filter(organ == "Liver",  species == "Chicken") %>% pull(rho),
  cardoso %>% filter(organ == "Brain",  species == "Chicken") %>% pull(rho),
  cardoso %>% filter(organ == "Ovary",  species == "Chicken") %>% pull(rho)
)

write_summary(summary_text, "fig4_summary.txt")

cat("\nFigure 4 (combined two-substrate) completed successfully!\n")
