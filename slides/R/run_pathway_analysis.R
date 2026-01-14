#!/usr/bin/env Rscript
# Run Pathway Analysis for Top Developmental Genes
# Uses gprofiler2 for KEGG and Reactome enrichment
#
# Outputs:
# - pathway_enrichment_results.rds - Full enrichment results
# - pathway_overlap_matrix.csv - Cross-tissue pathway overlap
# - pathway_summary.csv - Summary statistics

# Set working directory to script location
if (interactive()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

# Load required packages
suppressPackageStartupMessages({
  library(tidyverse)
  library(gprofiler2)
  library(jsonlite)
})

# Define paths
base_dir <- "../.."
data_dir <- file.path(base_dir, "machine_learning/cross_tissue_analysis/overlap_analysis_results")
output_dir <- "../data"

# Create output directory if it doesn't exist
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("=== Running Pathway Analysis (KEGG + Reactome) ===\n\n")

# ============================================================================
# LOAD TOP GENES
# ============================================================================

cat("Loading top genes per tissue...\n")

# Load top genes with symbols
top_genes_file <- file.path(data_dir, "top_genes_with_symbols.csv")
if (!file.exists(top_genes_file)) {
  stop("Top genes file not found: ", top_genes_file)
}

top_genes <- read_csv(top_genes_file, show_col_types = FALSE)
cat("  Loaded", nrow(top_genes), "genes across", length(unique(top_genes$tissue)), "tissues\n")

# Get unique tissues
tissues <- unique(top_genes$tissue)
cat("  Tissues:", paste(tissues, collapse = ", "), "\n\n")

# ============================================================================
# RUN GPROFILER ENRICHMENT
# ============================================================================

cat("Running gProfiler enrichment analysis...\n")
cat("  Organism: Sus scrofa (pig)\n")
cat("  Sources: KEGG, Reactome (REAC)\n\n")

# Store results
enrichment_results <- list()

for (tissue in tissues) {
  cat(sprintf("  Processing %s...\n", tissue))

  # Get genes for this tissue
  tissue_genes <- top_genes %>%
    filter(tissue == !!tissue) %>%
    pull(ensembl_id)

  # Also get gene symbols for display
  tissue_symbols <- top_genes %>%
    filter(tissue == !!tissue) %>%
    filter(!is.na(symbol) & symbol != ensembl_id) %>%
    pull(symbol)

  cat(sprintf("    %d Ensembl IDs, %d with symbols\n", length(tissue_genes), length(tissue_symbols)))

  # Run gProfiler with both Ensembl IDs and symbols
  # Use Ensembl IDs as primary input (more reliable for pig)
  # Note: Pig has limited KEGG/Reactome annotations, so we also try GO:BP
  tryCatch({
    # First try with pig organism and multiple sources
    gost_result <- gost(
      query = tissue_genes,
      organism = "sscrofa",
      sources = c("GO:BP", "KEGG", "REAC"),  # Added GO:BP for better coverage
      evcodes = TRUE,
      correction_method = "fdr",
      significant = FALSE,  # Get all results, filter later
      user_threshold = 0.1  # Less stringent threshold
    )

    # If no results, try with gene symbols
    if (is.null(gost_result$result) || nrow(gost_result$result) == 0) {
      if (length(tissue_symbols) > 10) {
        cat("    Retrying with gene symbols...\n")
        gost_result <- gost(
          query = tissue_symbols,
          organism = "sscrofa",
          sources = c("GO:BP", "KEGG", "REAC"),
          evcodes = TRUE,
          correction_method = "fdr",
          significant = FALSE,
          user_threshold = 0.1
        )
      }
    }

    if (!is.null(gost_result$result) && nrow(gost_result$result) > 0) {
      result_df <- gost_result$result %>%
        mutate(
          tissue = tissue,
          gene_ratio = intersection_size / query_size,
          fold_enrichment = (intersection_size / query_size) / (term_size / effective_domain_size)
        ) %>%
        select(
          tissue,
          source,
          term_id,
          term_name,
          p_value,
          term_size,
          query_size,
          intersection_size,
          gene_ratio,
          fold_enrichment,
          intersection
        ) %>%
        arrange(p_value)

      enrichment_results[[tissue]] <- result_df
      cat(sprintf("    Found %d significant pathways\n", nrow(result_df)))
    } else {
      cat(sprintf("    No significant pathways found\n"))
      enrichment_results[[tissue]] <- NULL
    }
  }, error = function(e) {
    cat(sprintf("    Error: %s\n", e$message))
    enrichment_results[[tissue]] <- NULL
  })
}

# Combine all results
all_enrichment <- bind_rows(enrichment_results)
cat(sprintf("\nTotal significant pathways: %d\n", nrow(all_enrichment)))

# ============================================================================
# ANALYZE PATHWAY OVERLAP
# ============================================================================

cat("\nAnalyzing pathway overlap across tissues...\n")

# Get unique pathways per tissue
pathway_sets <- list()
for (tissue in tissues) {
  if (!is.null(enrichment_results[[tissue]])) {
    pathway_sets[[tissue]] <- unique(enrichment_results[[tissue]]$term_id)
  } else {
    pathway_sets[[tissue]] <- character(0)
  }
}

# Calculate overlap matrix (Jaccard similarity)
overlap_matrix <- matrix(0, nrow = length(tissues), ncol = length(tissues),
                         dimnames = list(tissues, tissues))

shared_pathways_matrix <- matrix(0, nrow = length(tissues), ncol = length(tissues),
                                 dimnames = list(tissues, tissues))

for (i in 1:length(tissues)) {
  for (j in 1:length(tissues)) {
    set_i <- pathway_sets[[tissues[i]]]
    set_j <- pathway_sets[[tissues[j]]]

    if (length(set_i) > 0 || length(set_j) > 0) {
      intersection <- length(intersect(set_i, set_j))
      union_size <- length(union(set_i, set_j))
      overlap_matrix[i, j] <- intersection / union_size
      shared_pathways_matrix[i, j] <- intersection
    }
  }
}

cat("  Pathway overlap matrix (Jaccard):\n")
print(round(overlap_matrix, 3))

# ============================================================================
# IDENTIFY SHARED PATHWAYS
# ============================================================================

cat("\nIdentifying pathways shared across multiple tissues...\n")

# Handle case with no enrichment results
if (nrow(all_enrichment) == 0) {
  cat("  WARNING: No enrichment results found.\n")
  cat("  This may be due to limited pig annotations in databases.\n")
  cat("  Creating placeholder results...\n")

  pathway_tissue_count <- data.frame(
    term_id = character(0),
    term_name = character(0),
    source = character(0),
    n_tissues = integer(0),
    tissues = character(0),
    mean_p_value = numeric(0),
    mean_fold_enrichment = numeric(0)
  )
  shared_pathways <- pathway_tissue_count
} else {
  # Count how many tissues each pathway appears in
  pathway_tissue_count <- all_enrichment %>%
    group_by(term_id, term_name, source) %>%
    summarize(
      n_tissues = n_distinct(tissue),
      tissues = paste(unique(tissue), collapse = ", "),
      mean_p_value = mean(p_value),
      mean_fold_enrichment = mean(fold_enrichment),
      .groups = "drop"
    ) %>%
    arrange(desc(n_tissues), mean_p_value)

  # Shared pathways (in 2+ tissues)
  shared_pathways <- pathway_tissue_count %>%
    filter(n_tissues >= 2)
}

cat(sprintf("  Pathways in 2+ tissues: %d\n", nrow(shared_pathways)))
cat(sprintf("  Pathways in 3+ tissues: %d\n", sum(pathway_tissue_count$n_tissues >= 3)))

if (nrow(shared_pathways) > 0) {
  cat("\n  Top shared pathways:\n")
  print(head(shared_pathways %>% select(term_name, source, n_tissues, tissues), 10))
}

# ============================================================================
# TISSUE-SPECIFIC PATHWAYS
# ============================================================================

cat("\nIdentifying tissue-specific pathways...\n")

if (nrow(all_enrichment) > 0) {
  tissue_specific <- pathway_tissue_count %>%
    filter(n_tissues == 1) %>%
    arrange(mean_p_value)

  for (tissue in tissues) {
    tissue_pathways <- all_enrichment %>%
      filter(tissue == !!tissue) %>%
      filter(!term_id %in% shared_pathways$term_id) %>%
      arrange(p_value)

    cat(sprintf("  %s: %d tissue-specific pathways\n", tissue, nrow(tissue_pathways)))
    if (nrow(tissue_pathways) > 0) {
      cat(sprintf("    Top 3: %s\n", paste(head(tissue_pathways$term_name, 3), collapse = "; ")))
    }
  }
} else {
  cat("  No tissue-specific pathways (no enrichment results)\n")
  tissue_specific <- data.frame()
}

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

cat("\nGenerating summary statistics...\n")

pathway_summary <- data.frame(
  tissue = tissues,
  n_pathways = sapply(tissues, function(t) {
    if (!is.null(enrichment_results[[t]])) nrow(enrichment_results[[t]]) else 0
  }),
  n_kegg = sapply(tissues, function(t) {
    if (!is.null(enrichment_results[[t]])) sum(enrichment_results[[t]]$source == "KEGG") else 0
  }),
  n_reactome = sapply(tissues, function(t) {
    if (!is.null(enrichment_results[[t]])) sum(enrichment_results[[t]]$source == "REAC") else 0
  }),
  n_shared = sapply(tissues, function(t) {
    if (!is.null(enrichment_results[[t]])) {
      sum(enrichment_results[[t]]$term_id %in% shared_pathways$term_id)
    } else 0
  }),
  n_tissue_specific = sapply(tissues, function(t) {
    if (!is.null(enrichment_results[[t]])) {
      sum(!enrichment_results[[t]]$term_id %in% shared_pathways$term_id)
    } else 0
  })
)

cat("\nSummary by tissue:\n")
print(pathway_summary)

# ============================================================================
# SAVE RESULTS
# ============================================================================

cat("\nSaving results...\n")

# Save full enrichment results
saveRDS(enrichment_results, file.path(output_dir, "pathway_enrichment_results.rds"))
cat("  Saved: pathway_enrichment_results.rds\n")

# Save combined results as CSV
write_csv(all_enrichment, file.path(output_dir, "pathway_enrichment_all.csv"))
cat("  Saved: pathway_enrichment_all.csv\n")

# Save overlap matrix
write.csv(overlap_matrix, file.path(output_dir, "pathway_overlap_matrix.csv"))
cat("  Saved: pathway_overlap_matrix.csv\n")

# Save shared pathways matrix (counts)
write.csv(shared_pathways_matrix, file.path(output_dir, "pathway_shared_counts.csv"))
cat("  Saved: pathway_shared_counts.csv\n")

# Save pathway tissue counts
write_csv(pathway_tissue_count, file.path(output_dir, "pathway_tissue_counts.csv"))
cat("  Saved: pathway_tissue_counts.csv\n")

# Save summary
write_csv(pathway_summary, file.path(output_dir, "pathway_summary.csv"))
cat("  Saved: pathway_summary.csv\n")

# Save as JSON for downstream use
pathway_json <- list(
  summary = pathway_summary,
  overlap_matrix = as.data.frame(overlap_matrix),
  shared_pathways = shared_pathways,
  tissue_counts = pathway_tissue_count
)
write_json(pathway_json, file.path(output_dir, "pathway_analysis_results.json"), pretty = TRUE)
cat("  Saved: pathway_analysis_results.json\n")

cat("\n=== Pathway Analysis Complete ===\n")
cat(sprintf("Total pathways found: %d\n", nrow(all_enrichment)))
cat(sprintf("Shared pathways (2+ tissues): %d\n", nrow(shared_pathways)))
cat(sprintf("Results saved to: %s\n", output_dir))
