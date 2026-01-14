#!/usr/bin/env Rscript
# Install packages required by visualization figure scripts

cran_packages <- c(
  "tidyverse",
  "data.table",
  "patchwork",
  "viridis",
  "jsonlite",
  "ggalluvial",
  "scales",
  "reshape2",
  "umap",
  "UpSetR",
  "circlize",
  "RColorBrewer",
  "ggraph",
  "igraph",
  "readxl",
  "ggrepel"
)

bioc_packages <- c(
  "ComplexHeatmap"
)

install_missing <- function(packages) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    install.packages(missing, repos = "https://cloud.r-project.org")
  }
}

install_missing(cran_packages)

if (length(bioc_packages) > 0) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
  }
  missing_bioc <- bioc_packages[!vapply(bioc_packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_bioc) > 0) {
    BiocManager::install(missing_bioc, ask = FALSE, update = FALSE)
  }
}

cat("Package installation complete.\n")
