#!/usr/bin/env Rscript
# Slides Theme Configuration
# Optimized for Cardiff University beamer presentations (16:9 aspect ratio)

library(ggplot2)
library(scales)

# Cardiff University Color Palette
cardiff_colors <- list(
  red = "#D3374A",
  black = "#22211F",
  grey = "#D3D3D2",
  dark_grey = "#545351",
  white = "#FFFFFF",
  light_grey = "#F2F2F2"
)

# Stage colors (consistent with paper but slightly more vibrant for slides)
stage_colors <- c(
  "Infant_0_20d" = "#E8F4FD",
  "Early childhood_21_59d" = "#B8D4E8",
  "Pre_pubertal_60_149d" = "#7EB5D6",
  "Post_pubertal_150_365d" = "#4A95C4",
  "Adult_>365d" = "#2171B5"
)

# Simplified stage colors for 4-class
stage_colors_4class <- c(
  "Infant" = "#E8F4FD",
  "Early" = "#95B8D1",
  "Pre-pub" = "#5A9BD4",
  "Post-pub" = "#2171B5"
)

# Tissue colors (colorblind-friendly)
tissue_colors <- c(
  "Muscle" = "#E69F00",
  "Brain" = "#56B4E9",
  "Liver" = "#009E73",
  "Blood" = "#D55E00",
  "Small intestine" = "#CC79A7",
  "Lung" = "#0072B2",
  "Adipose" = "#F0E442",
  "Testis" = "#999999"
)

# Slides theme function - minimalist and clean
# Reduced base_size from 14 to 12 for better density on slides
slides_theme <- function(base_size = 12, base_family = "sans") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      # Titles - clean and bold
      plot.title = element_text(
        size = base_size + 4,
        face = "bold",
        color = cardiff_colors$black,
        margin = margin(b = 8)
      ),
      plot.subtitle = element_text(
        size = base_size - 1,
        color = cardiff_colors$dark_grey,
        margin = margin(b = 12)
      ),

      # Axis text - minimalist
      axis.text = element_text(
        size = base_size - 1,
        color = cardiff_colors$dark_grey
      ),
      axis.title = element_text(
        size = base_size,
        color = cardiff_colors$black,
        face = "bold",
        margin = margin(t = 10, r = 10)
      ),
      axis.line.x = element_line(color = cardiff_colors$black, linewidth = 0.5),
      axis.line.y = element_blank(),
      axis.ticks.x = element_line(color = cardiff_colors$black),
      axis.ticks.y = element_blank(),

      # Legend - clean placement
      legend.position = "top",
      legend.justification = "left",
      legend.text = element_text(size = base_size - 1, color = cardiff_colors$dark_grey),
      legend.title = element_text(size = base_size - 1, face = "bold", color = cardiff_colors$black),
      legend.key.size = unit(0.8, "lines"),
      legend.background = element_blank(),
      legend.key = element_blank(),

      # Panel - very subtle grid
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major.y = element_line(color = cardiff_colors$light_grey, linewidth = 0.3),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),

      # Strip (for facets) - clean
      strip.background = element_blank(),
      strip.text = element_text(
        size = base_size,
        face = "bold",
        color = cardiff_colors$black,
        hjust = 0
      ),

      # Plot background - white
      plot.background = element_rect(fill = "white", color = NA),

      # Margins
      plot.margin = margin(15, 15, 15, 15)
    )
}

# Save function for slide figures
# Increased default DPI from 150 to 200 for sharper text
save_slide_figure <- function(plot, filename, width = 10, height = 6, dpi = 200) {
  ggsave(
    filename,
    plot = plot,
    width = width,
    height = height,
    dpi = dpi,
    bg = "white"
  )
  cat("Saved:", filename, "\n")
}

# Set slides theme as default
theme_set(slides_theme())

cat("Slides theme loaded successfully.\n")
cat("Cardiff colors available: cardiff_colors$red, cardiff_colors$black, etc.\n")
cat("Stage colors available: stage_colors, stage_colors_4class\n")
cat("Tissue colors available: tissue_colors\n")
