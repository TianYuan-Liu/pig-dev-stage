#!/usr/bin/env Rscript
# Generate Next Steps Figure for Slides
#
# Creates a simple visual showing future directions

# Set working directory to script location
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) > 0) {
    setwd(dirname(script_path))
}

# Load theme
source("slides_theme.R")

library(ggplot2)
library(dplyr)

# Output directory
output_dir <- "../Figures"
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

cat("=== Generating Next Steps Figure ===\n")

# Create next steps data
next_steps <- data.frame(
    step = c("Cross-Species Integration", "Clinical Translation", "Biomarker Panel"),
    description = c(
        "Map pig developmental genes\nto human orthologs",
        "Validate age biomarkers\nacross species",
        "Develop diagnostic panel\nfor developmental staging"
    ),
    icon = c("🧬", "🏥", "📊"),
    x = c(1, 2, 3),
    color = c(cardiff_colors$red, cardiff_colors$black, "#0072B2")
)

# Create the figure
fig <- ggplot(next_steps, aes(x = x, y = 0.5)) +
    # Main circles
    geom_point(aes(color = step), size = 40, show.legend = FALSE) +
    scale_color_manual(values = setNames(next_steps$color, next_steps$step)) +

    # Step numbers
    geom_text(aes(label = x), color = "white", size = 12, fontface = "bold") +

    # Step titles above
    geom_text(aes(y = 0.75, label = step),
        size = 6, fontface = "bold", color = cardiff_colors$black
    ) +

    # Descriptions below
    geom_text(aes(y = 0.25, label = description),
        size = 4, color = cardiff_colors$dark_grey, lineheight = 0.9
    ) +

    # Connecting arrows
    annotate("segment",
        x = 1.3, xend = 1.7, y = 0.5, yend = 0.5,
        arrow = arrow(length = unit(0.3, "cm"), type = "closed"),
        color = cardiff_colors$dark_grey, linewidth = 1.5
    ) +
    annotate("segment",
        x = 2.3, xend = 2.7, y = 0.5, yend = 0.5,
        arrow = arrow(length = unit(0.3, "cm"), type = "closed"),
        color = cardiff_colors$dark_grey, linewidth = 1.5
    ) +

    # Attribution
    annotate("text",
        x = 1, y = 0.05, label = "Lead: Rujing",
        size = 3.5, color = cardiff_colors$dark_grey, fontface = "italic"
    ) +
    coord_cartesian(xlim = c(0.3, 3.7), ylim = c(0, 1)) +
    theme_void() +
    theme(
        plot.background = element_rect(fill = "white", color = NA),
        plot.margin = margin(20, 20, 20, 20)
    )

# Save figure
ggsave(
    file.path(output_dir, "fig6_next_steps.png"),
    fig,
    width = 12,
    height = 5,
    dpi = 300,
    bg = "white"
)

cat("Saved:", file.path(output_dir, "fig6_next_steps.png"), "\n")
cat("\n=== Next Steps Figure Generated ===\n")
