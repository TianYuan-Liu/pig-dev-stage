#!/usr/bin/env Rscript
#
# Nature Journal Theme Configuration for R Plots
#
# This file defines the theme and styling requirements for Nature journal publications
# All figures should use this theme to ensure consistency with Nature's standards
#

library(ggplot2)
library(scales)

# Nature journal theme function
nature_theme <- function(base_size = 8, base_family = "sans") {
  # Try to use Arial if available, otherwise fallback to sans
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      # Remove titles (will use figure captions in manuscript)
      plot.title = element_blank(),
      plot.subtitle = element_blank(),

      # Axis text
      axis.text = element_text(size = base_size, color = "black"),
      axis.title = element_text(size = base_size, color = "black"),

      # Axis lines
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.ticks = element_line(color = "black", linewidth = 0.5),

      # Legend
      legend.text = element_text(size = base_size - 1),
      legend.title = element_text(size = base_size, face = "bold"),
      legend.key.size = unit(0.8, "lines"),
      legend.background = element_blank(),
      legend.key = element_blank(),

      # Panel
      panel.background = element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),

      # Strip (for facets)
      strip.background = element_blank(),
      strip.text = element_text(size = base_size, face = "bold"),

      # Spacing
      plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")
    )
}

# Color palette for Nature (colorblind-friendly)
nature_colors <- c(
  "#0173B2",  # Blue
  "#DE8F05",  # Orange
  "#029E73",  # Green
  "#CC78BC",  # Light purple
  "#ECE133",  # Yellow
  "#56B4E9",  # Light blue
  "#F0E442",  # Light yellow
  "#949494",  # Gray
  "#000000"   # Black
)

# Function to add subplot labels (a, b, c, d)
add_subplot_label <- function(plot, label, x = 0.02, y = 0.98) {
  plot +
    annotation_custom(
      grid::textGrob(
        label,
        x = unit(x, "npc"),
        y = unit(y, "npc"),
        just = c("left", "top"),
        gp = grid::gpar(fontsize = 10, fontface = "bold", fontfamily = "sans")
      )
    )
}

# Function to convert bar chart to dot plot with error bars
bar_to_dot_plot <- function(data, x_var, y_var, group_var = NULL,
                            error_var = NULL, dodge_width = 0.3) {
  p <- ggplot(data, aes_string(x = x_var, y = y_var))

  if (!is.null(group_var)) {
    p <- p + aes_string(color = group_var)
  }

  # Add points
  if (!is.null(group_var)) {
    p <- p + geom_point(position = position_dodge(width = dodge_width),
                       size = 2.5, shape = 16)
  } else {
    p <- p + geom_point(size = 2.5, shape = 16)
  }

  # Add error bars if specified
  if (!is.null(error_var)) {
    if (!is.null(group_var)) {
      p <- p + geom_errorbar(
        aes_string(ymin = paste(y_var, "-", error_var),
                  ymax = paste(y_var, "+", error_var)),
        position = position_dodge(width = dodge_width),
        width = 0.2
      )
    } else {
      p <- p + geom_errorbar(
        aes_string(ymin = paste(y_var, "-", error_var),
                  ymax = paste(y_var, "+", error_var)),
        width = 0.2
      )
    }
  }

  p + nature_theme()
}

# Function to create box plot with individual points
nature_boxplot <- function(data, x_var, y_var, point_alpha = 0.5) {
  ggplot(data, aes_string(x = x_var, y = y_var)) +
    geom_boxplot(outlier.shape = NA, width = 0.5) +
    geom_jitter(width = 0.2, alpha = point_alpha, size = 1) +
    nature_theme()
}

# Nature-compliant color palette
nature_colors <- function() {
  scale_color_manual(values = c(
    "#2E86C1",  # Blue
    "#E74C3C",  # Red (avoid if possible for accessibility)
    "#48C9B0",  # Teal
    "#F39C12",  # Orange
    "#8E44AD",  # Purple
    "#27AE60",  # Green
    "#34495E",  # Dark gray
    "#F1C40F"   # Yellow
  ))
}

# Export figure function with Nature standards
save_nature_figure <- function(plot, filename, width = 89, height = 89,
                             units = "mm", dpi = 300, format = "pdf") {
  # Nature single column width: 89mm
  # Nature double column width: 183mm
  # Maximum height: 247mm

  if (format == "pdf") {
    ggsave(
      paste0(filename, ".pdf"),
      plot = plot,
      width = width,
      height = height,
      units = units,
      dpi = dpi,
      device = cairo_pdf
    )
  } else if (format == "eps") {
    ggsave(
      paste0(filename, ".eps"),
      plot = plot,
      width = width,
      height = height,
      units = units,
      dpi = dpi,
      device = "eps"
    )
  }

  # Also save PNG for review
  ggsave(
    paste0(filename, "_preview.png"),
    plot = plot,
    width = width,
    height = height,
    units = units,
    dpi = 150
  )
}

# Set default theme
theme_set(nature_theme())

# Export functions for use in other scripts
return(list(
  nature_theme = nature_theme,
  nature_colors = nature_colors,
  add_subplot_label = add_subplot_label,
  bar_to_dot_plot = bar_to_dot_plot,
  nature_boxplot = nature_boxplot,
  save_nature_figure = save_nature_figure
))