#!/usr/bin/env Rscript
# Extract Real Marker Gene Expression Data
# Outputs to ../data/marker_expression.csv
# Run from slides/R/ directory: Rscript extract_marker_expression.R
#
# Uses real developmental stage data from PigGTEx_v0.MetaTable.xlsx metadata

cat("=== Extracting Real Marker Gene Expression ===\n\n")

# Set working directory to script location
if (interactive()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

# Load required packages
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(viridis)
  library(readxl)
})

# Source slides theme for consistent aesthetics
if (file.exists("slides_theme.R")) {
  source("slides_theme.R")
} else {
  warning("slides_theme.R not found. Plotting might fail or look different.")
}

# Define paths
base_dir <- "../.."
data_dir <- file.path(base_dir, "data")
expression_dir <- file.path(data_dir, "pigGTEx")
output_file <- "../data/marker_expression.csv"
figures_dir <- "../Figures"

# Create output folder if needed
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

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

cat("  Samples with valid stage data:", sum(!is.na(metadata$Stage)), "\n\n")

# Canonical marker genes and their target tissues
# Using known Ensembl IDs for pig (Sus scrofa)
markers <- data.frame(
  Gene_Symbol = c("PPARG", "MBP", "ALB", "MSTN", "HBB", "SFTPC", "LGR5", "DMRT1"),
  Target_Tissue = c("Adipose", "Brain", "Liver", "Muscle", "Blood", "Lung", "Small_intestine", "Testis"),
  # Known pig Ensembl gene IDs (from Ensembl 111)
  Ensembl_ID = c(
    "ENSSSCG00000004130", # PPARG - adipogenesis
    "ENSSSCG00000008574", # MBP - myelination
    "ENSSSCG00000005095", # ALB - liver function
    "ENSSSCG00000039058", # MSTN - muscle growth
    "ENSSSCG00000014727", # HBB - hemoglobin
    "ENSSSCG00000007979", # SFTPC - lung surfactant
    "ENSSSCG00000004244", # LGR5 - intestinal stem cells
    "ENSSSCG00000013313" # DMRT1 - testis development
  ),
  stringsAsFactors = FALSE
)

cat("Target markers:\n")
print(markers)

# Function to load expression for specific genes from a tissue
load_gene_expression <- function(tissue, gene_ids) {
  file_path <- file.path(expression_dir, paste0(tissue, ".expr_tpm.txt.gz"))

  if (!file.exists(file_path)) {
    warning(paste("Expression file not found:", file_path))
    return(NULL)
  }

  cat("  Loading", tissue, "...\n")

  # Read expression file
  expr <- fread(file_path)

  # First column is gene ID
  gene_col <- colnames(expr)[1]

  # Filter to genes of interest
  expr_filtered <- expr[get(gene_col) %in% gene_ids]

  if (nrow(expr_filtered) == 0) {
    # Try partial matching (gene IDs might have version numbers)
    gene_prefixes <- sub("\\..*", "", gene_ids)
    expr$gene_prefix <- sub("\\..*", "", expr[[gene_col]])
    expr_filtered <- expr[gene_prefix %in% gene_prefixes]
    expr_filtered$gene_prefix <- NULL
  }

  if (nrow(expr_filtered) == 0) {
    warning(paste("No matching genes found in", tissue))
    return(NULL)
  }

  return(expr_filtered)
}

# Collect expression data for all marker genes
cat("\nExtracting expression for marker genes...\n")
all_expression <- list()

for (i in 1:nrow(markers)) {
  gene_symbol <- markers$Gene_Symbol[i]
  target_tissue <- markers$Target_Tissue[i]
  ensembl_id <- markers$Ensembl_ID[i]

  cat("\nProcessing", gene_symbol, "(", ensembl_id, ") in", target_tissue, "\n")

  expr_data <- load_gene_expression(target_tissue, ensembl_id)

  if (!is.null(expr_data) && nrow(expr_data) > 0) {
    # Get expression values (all columns except gene ID)
    gene_col <- colnames(expr_data)[1]
    sample_ids <- colnames(expr_data)[-1]
    expr_values <- as.numeric(unlist(expr_data[1, -1, with = FALSE]))
    names(expr_values) <- sample_ids

    # Match samples to metadata to get REAL developmental stages
    tissue_label <- gsub("_", " ", target_tissue)
    sample_metadata <- data.frame(
      Sample_ID = sample_ids,
      Expression = expr_values,
      stringsAsFactors = FALSE
    ) %>%
      left_join(metadata %>% select(Sample_ID, Stage), by = "Sample_ID") %>%
      filter(!is.na(Stage), !is.na(Expression), Expression > 0)

    n_with_stage <- nrow(sample_metadata)
    cat("    Matched", n_with_stage, "/", length(sample_ids), "samples to real stages\n")

    if (n_with_stage >= 10) {
      # Calculate expression statistics per REAL developmental stage
      stage_data <- sample_metadata %>%
        group_by(Stage) %>%
        summarise(
          Expression_mean = mean(Expression, na.rm = TRUE),
          Expression_sd = sd(Expression, na.rm = TRUE),
          n_samples = n(),
          .groups = "drop"
        ) %>%
        mutate(
          Gene_Symbol = gene_symbol,
          Tissue = tissue_label,
          Expression_sem = Expression_sd / sqrt(n_samples)
        ) %>%
        select(Gene_Symbol, Tissue, Stage, Expression_mean, Expression_sd, n_samples, Expression_sem)

      all_expression[[paste(gene_symbol, target_tissue)]] <- stage_data
      cat("    Stages found:", paste(stage_data$Stage, collapse = ", "), "\n")
    } else {
      warning(paste("  Not enough samples with real stage data for", gene_symbol, "(n=", n_with_stage, ")"))
    }
  }
}

# Combine all expression data
if (length(all_expression) > 0) {
  combined_expr <- do.call(rbind, all_expression)
  cat("\nTotal gene-tissue-stage combinations:", nrow(combined_expr), "\n")
} else {
  stop("No expression data extracted!")
}

# Calculate log2 transformed values
combined_expr <- combined_expr %>%
  mutate(
    Log2_Expression_mean = log2(Expression_mean + 1),
    Log2_Expression_sem = Expression_sem / (Expression_mean + 1) / log(2)
  )

# Save output data
cat("\nSaving to", output_file, "...\n")
fwrite(combined_expr, output_file)

# Display summary
cat("\nExpression summary:\n")
print(as.data.frame(combined_expr[, c("Gene_Symbol", "Tissue", "Stage", "Log2_Expression_mean", "n_samples")]))


# ============================================================================
# GENERATE PLOT
# ============================================================================

cat("\n=== Generating Marker Validation Plot ===\n")

# Prepare data for plotting
plot_data <- combined_expr %>%
  mutate(
    Stage_short = case_when(
      grepl("Infant", Stage) ~ "Infant",
      grepl("Early", Stage) ~ "Early",
      grepl("Pre", Stage) ~ "Pre-pub",
      grepl("Post", Stage) ~ "Post-pub",
      grepl("Adult", Stage) ~ "Adult",
      TRUE ~ Stage
    ),
    Stage_short = factor(Stage_short, levels = c("Infant", "Early", "Pre-pub", "Post-pub", "Adult"))
  ) %>%
  filter(!is.na(Stage_short), !is.na(Log2_Expression_mean))

# Create Plot
p <- ggplot(plot_data, aes(x = Stage_short, y = Log2_Expression_mean, group = Gene_Symbol)) +
  geom_line(color = "#809BCE", linewidth = 1) +
  geom_point(size = 3, color = "#809BCE") +
  geom_errorbar(
    aes(
      ymin = Log2_Expression_mean - Log2_Expression_sem,
      ymax = Log2_Expression_mean + Log2_Expression_sem
    ),
    width = 0.2, alpha = 0.5
  ) +
  facet_wrap(~ paste0(Gene_Symbol, " (", Tissue, ")"), ncol = 4, scales = "free_y") +
  labs(
    title = "Marker Gene Validation",
    subtitle = "Real expression levels (Log2 TPM) across developmental stages",
    x = NULL,
    y = "Log2 TPM"
  ) +
  slides_theme() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    strip.text = element_text(size = 9)
  )

# Save Plot
plot_file <- file.path(figures_dir, "fig4_panel_b_marker_validation.png")
save_slide_figure(p, plot_file, width = 10, height = 6) # Standard dimensions

cat("\n=== Marker Expression Extraction Complete ===\n")
cat("Output file:", output_file, "\n")
cat("Plot saved to:", plot_file, "\n")
cat("Genes extracted:", length(unique(combined_expr$Gene_Symbol)), "\n")
