#!/usr/bin/env Rscript
# GO Enrichment Analysis for Developmental Stage Markers
# This script performs Gene Ontology enrichment analysis on the top marker genes
# identified by the machine learning pipeline for each tissue.

# Set working directory to script location if running interactively
if (interactive()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

# Load required packages
suppressPackageStartupMessages({
  library(jsonlite)
  library(gprofiler2)
  library(data.table)
  library(dplyr)
})

cat("=== GO Enrichment Analysis ===\n\n")

# Define paths
base_dir <- normalizePath(file.path("..", ".."))
ml_dir <- file.path(base_dir, "machine_learning", "model_outputs")
output_dir <- file.path(base_dir, "results", "enrichment_analysis")

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat("Created output directory:", output_dir, "\n")
}

# Tissues to analyze
tissues <- c("Adipose", "Blood", "Brain", "Liver", "Lung", "Muscle", "Small intestine", "Testis")

# Number of top genes to use for enrichment
N_TOP_GENES <- 50

# Run enrichment for each tissue
all_results <- list()

for (tissue in tissues) {
  cat("\nProcessing", tissue, "...\n")

  # Load model results
  json_file <- file.path(ml_dir, paste0(tissue, "_results.json"))
  if (!file.exists(json_file)) {
    warning(paste("Results file not found for", tissue, ":", json_file))
    next
  }

  results <- fromJSON(json_file)

  # Get top genes (Ensembl IDs)
  if (is.null(results$top_genes) || length(results$top_genes) == 0) {
    warning(paste("No top genes found for", tissue))
    next
  }

  top_genes <- head(results$top_genes, N_TOP_GENES)
  cat("  Using", length(top_genes), "top genes\n")

  # Run gprofiler2 enrichment analysis
  # Using pig (Sus scrofa) as organism
  tryCatch({
    gost_result <- gost(
      query = top_genes,
      organism = "sscrofa",  # Sus scrofa (pig)
      ordered_query = TRUE,  # Genes are ranked by importance
      multi_query = FALSE,
      significant = TRUE,
      exclude_iea = FALSE,  # Include electronically inferred annotations
      measure_underrepresentation = FALSE,
      evcodes = FALSE,
      user_threshold = 0.05,
      correction_method = "fdr",
      domain_scope = "annotated",
      sources = c("GO:BP", "GO:MF", "GO:CC")  # Gene Ontology terms only
    )

    if (!is.null(gost_result$result) && nrow(gost_result$result) > 0) {
      # Extract and format results
      enrichment_df <- gost_result$result %>%
        select(
          source,
          term_id,
          term_name,
          p_value,
          term_size,
          query_size,
          intersection_size,
          precision,
          recall
        ) %>%
        mutate(
          Tissue = tissue,
          fold_enrichment = (intersection_size / query_size) / (term_size / 20000),  # Approximate
          q_value = p_value  # Already FDR-corrected
        ) %>%
        arrange(p_value) %>%
        head(20)  # Top 20 enriched terms

      cat("  Found", nrow(enrichment_df), "significant GO terms\n")

      # Save tissue-specific results
      tissue_file <- file.path(output_dir, paste0(gsub(" ", "_", tissue), "_go_enrichment.csv"))
      fwrite(enrichment_df, tissue_file)
      cat("  Saved to:", tissue_file, "\n")

      all_results[[tissue]] <- enrichment_df
    } else {
      cat("  No significant GO terms found\n")
    }

  }, error = function(e) {
    warning(paste("Error running enrichment for", tissue, ":", e$message))
  })
}

# Combine all results
if (length(all_results) > 0) {
  combined_results <- bind_rows(all_results)
  combined_file <- file.path(output_dir, "all_tissues_go_enrichment.csv")
  fwrite(combined_results, combined_file)
  cat("\n\nCombined results saved to:", combined_file, "\n")

  # Summary statistics
  cat("\n=== Summary ===\n")
  summary_stats <- combined_results %>%
    group_by(Tissue) %>%
    summarise(
      n_terms = n(),
      top_term = first(term_name),
      top_fold_enrichment = first(fold_enrichment),
      top_q_value = first(q_value)
    )
  print(as.data.frame(summary_stats))
} else {
  cat("\nNo enrichment results generated.\n")
}

cat("\n=== GO Enrichment Analysis Complete ===\n")
