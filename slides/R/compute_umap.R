#!/usr/bin/env Rscript
# Compute UMAP Coordinates from Real Expression Data
# Outputs to ../data/umap_coordinates.csv
# Run from slides/R/ directory: Rscript compute_umap.R
#
# Uses real developmental stage data from PigGTEx_v0.MetaTable.xlsx metadata

cat("=== Computing UMAP from Real Expression Data ===\n\n")

# Set working directory to script location
if (interactive()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

# Load required packages
suppressPackageStartupMessages({
  library(data.table)
  library(uwot)
  library(jsonlite)
  library(dplyr)
  library(readxl)
})

# Define paths
base_dir <- "../.."
data_dir <- file.path(base_dir, "data")
ml_dir <- file.path(base_dir, "machine_learning/model_outputs")
expression_dir <- file.path(data_dir, "pigGTEx")
output_file <- "../data/umap_coordinates.csv"

# Tissues to include (matching file names)
tissues <- c(
  "Muscle", "Brain", "Liver", "Blood",
  "Small_intestine", "Lung", "Adipose", "Testis"
)

# Maximum samples per tissue (to balance and speed up computation)
MAX_SAMPLES_PER_TISSUE <- 200

# Load real metadata for stage assignments
cat("Loading metadata for real developmental stages...\n")
metadata_file <- file.path(data_dir, "PigGTEx_v0.MetaTable.xlsx")
if (!file.exists(metadata_file)) {
  stop("ERROR: Metadata file not found: ", metadata_file)
}
metadata <- read_excel(metadata_file)
metadata <- metadata %>%
  rename(Sample_ID = BioSample, Tissue = `Tissue class`)
cat("  Loaded metadata for", nrow(metadata), "samples\n")

# Function to parse age string to developmental stage
parse_age_to_stage <- function(age_str) {
  if (is.na(age_str) || age_str == "Unknown") return(NA_character_)
  age_str <- tolower(age_str)

  days <- NA_real_
  if (grepl("day", age_str)) {
    days <- as.numeric(gsub("\\s*days?", "", age_str))
  } else if (grepl("week", age_str)) {
    days <- as.numeric(gsub("\\s*weeks?", "", age_str)) * 7
  } else if (grepl("month", age_str)) {
    days <- as.numeric(gsub("\\s*months?", "", age_str)) * 30
  } else if (grepl("year", age_str)) {
    days <- as.numeric(gsub("\\s*years?", "", age_str)) * 365
  }

  if (is.na(days)) return(NA_character_)

  if (days <= 20) return("Infant_0_20d")
  if (days <= 59) return("Early childhood_21_59d")
  if (days <= 149) return("Pre_pubertal_60_149d")
  if (days <= 365) return("Post_pubertal_150_365d")
  return("Adult")
}

# Add Stage column to metadata
metadata <- metadata %>%
  mutate(Stage = sapply(Age, parse_age_to_stage))

cat("  Samples with valid stage data:", sum(!is.na(metadata$Stage)), "\n")

# Function to load expression data for a tissue
load_expression <- function(tissue) {
  file_path <- file.path(expression_dir, paste0(tissue, ".expr_tpm.txt.gz"))

  if (!file.exists(file_path)) {
    warning(paste("Expression file not found:", file_path))
    return(NULL)
  }

  cat("  Loading", tissue, "expression data...\n")
  expr <- fread(file_path)

  # First column is gene ID
  gene_ids <- expr[[1]]
  expr <- as.matrix(expr[, -1, with = FALSE])
  rownames(expr) <- gene_ids

  return(expr)
}

# Load top genes from ML outputs (for feature selection)
load_top_genes <- function() {
  all_genes <- c()

  for (tissue in tissues) {
    tissue_name <- gsub("_", " ", tissue)
    json_file <- file.path(ml_dir, paste0(tissue_name, "_results.json"))

    if (file.exists(json_file)) {
      results <- fromJSON(json_file)
      if (!is.null(results$top_genes)) {
        all_genes <- c(all_genes, results$top_genes)
      }
    }
  }

  unique(all_genes)
}

# Collect expression data from all tissues
cat("Loading expression data from all tissues...\n")
all_expr_list <- list()
sample_info <- data.frame()

for (tissue in tissues) {
  expr <- load_expression(tissue)

  if (!is.null(expr)) {
    n_samples <- ncol(expr)
    sample_ids <- colnames(expr)

    # Subsample if too many samples
    if (n_samples > MAX_SAMPLES_PER_TISSUE) {
      set.seed(42)
      sample_idx <- sample(1:n_samples, MAX_SAMPLES_PER_TISSUE)
      expr <- expr[, sample_idx]
      sample_ids <- sample_ids[sample_idx]
      n_samples <- MAX_SAMPLES_PER_TISSUE
    }

    # Create sample info using REAL stages from metadata
    tissue_label <- gsub("_", " ", tissue)

    # Match samples to metadata to get real developmental stages
    sample_meta <- data.frame(Sample_ID = sample_ids, stringsAsFactors = FALSE) %>%
      left_join(metadata %>% select(Sample_ID, Stage), by = "Sample_ID")

    sample_stages <- sample_meta$Stage
    n_with_stage <- sum(!is.na(sample_stages))
    cat("    ", tissue_label, ": matched", n_with_stage, "/", n_samples, "samples to real stages\n")

    all_expr_list[[tissue]] <- expr

    sample_info <- rbind(sample_info, data.frame(
      Sample_ID = sample_ids,
      Tissue = tissue_label,
      Stage = sample_stages,
      stringsAsFactors = FALSE
    ))

    cat("    ", tissue, ":", n_samples, "samples\n")
  }
}

cat("\nTotal samples for UMAP:", nrow(sample_info), "\n")

# Get top genes from ML pipeline for feature selection
cat("\nLoading top genes from ML results...\n")
top_genes <- load_top_genes()
cat("  Unique top genes across tissues:", length(top_genes), "\n")

# Find common genes across all tissues
cat("\nFinding common genes...\n")
common_genes <- Reduce(intersect, lapply(all_expr_list, rownames))
cat("  Common genes across all tissues:", length(common_genes), "\n")

# Filter to top genes that are also common
selected_genes <- intersect(top_genes, common_genes)
cat("  Selected genes (top genes & common):", length(selected_genes), "\n")

# If not enough genes from ML, use top variable genes
if (length(selected_genes) < 500) {
  cat("  Selecting top variable genes from common genes...\n")

  # Combine all expression data for common genes
  combined_expr <- do.call(cbind, lapply(all_expr_list, function(x) x[common_genes, ]))

  # Calculate variance per gene
  gene_vars <- apply(combined_expr, 1, var)

  # Select top 2000 variable genes
  selected_genes <- names(sort(gene_vars, decreasing = TRUE))[1:min(2000, length(gene_vars))]
  cat("  Using top", length(selected_genes), "variable genes\n")
}

# Build combined expression matrix with selected genes
cat("\nBuilding combined expression matrix...\n")
expr_matrices <- lapply(all_expr_list, function(x) {
  x[selected_genes, , drop = FALSE]
})
combined_expr <- do.call(cbind, expr_matrices)
cat("  Matrix dimensions:", nrow(combined_expr), "genes x", ncol(combined_expr), "samples\n")

# Transpose for UMAP (samples as rows)
expr_for_umap <- t(combined_expr)

# Log transform (adding small value to avoid log(0))
cat("\nLog-transforming expression data...\n")
expr_for_umap <- log2(expr_for_umap + 1)

# Remove any samples with NA/Inf values
valid_samples <- apply(expr_for_umap, 1, function(x) all(is.finite(x)))
expr_for_umap <- expr_for_umap[valid_samples, ]
sample_info <- sample_info[valid_samples, ]
cat("  Valid samples after filtering:", nrow(expr_for_umap), "\n")

# Run UMAP
cat("\nRunning UMAP (this may take a few minutes)...\n")
set.seed(42) # For reproducibility

umap_result <- umap(
  expr_for_umap,
  n_neighbors = 30,
  min_dist = 0.3,
  n_components = 2,
  metric = "cosine",
  n_threads = 4,
  verbose = TRUE
)

cat("\nUMAP completed!\n")

# Build output dataframe
cat("\nBuilding output dataframe...\n")
umap_df <- data.frame(
  Sample_ID = rownames(expr_for_umap),
  UMAP1 = umap_result[, 1],
  UMAP2 = umap_result[, 2],
  Tissue = sample_info$Tissue,
  Stage = sample_info$Stage,
  stringsAsFactors = FALSE
)

# Handle samples with missing stage information
na_stages <- is.na(umap_df$Stage)
if (any(na_stages)) {
  cat("WARNING:", sum(na_stages), "samples have unknown developmental stage (marker not found)\n")
  cat("  These samples will be labeled as 'Unknown' in the visualization\n")
  umap_df$Stage[na_stages] <- "Unknown"
}



# Save output
cat("\nSaving to", output_file, "...\n")
fwrite(umap_df, output_file)

cat("\n=== UMAP Computation Complete ===\n")
cat("Output file:", output_file, "\n")
cat("Total samples:", nrow(umap_df), "\n")
cat("Columns:", paste(names(umap_df), collapse = ", "), "\n")

# Summary by tissue
cat("\nSamples per tissue:\n")
print(table(umap_df$Tissue))
