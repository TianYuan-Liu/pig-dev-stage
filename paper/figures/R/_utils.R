# ==============================================================================
# Shared Utility Functions
# Paper: Pig Developmental Stage Classification
# ==============================================================================
# This module provides data loading and processing functions used across
# all figure scripts to ensure consistency and avoid code duplication.
# ==============================================================================

# Required packages
suppressPackageStartupMessages({
  library(tidyverse)
  library(readxl)
  library(jsonlite)
  library(data.table)
})

# Load theme module for config access
get_utils_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("--file=", "", file_arg))))
  }
  "/Users/tianyuan/Desktop/github_dev/pig-dev-stage/paper/figures/R"
}
source(file.path(get_utils_dir(), "_theme.R"))

# ==============================================================================
# PATH HELPERS
# ==============================================================================

#' Get project root directory
#' @return Character path to project root
PROJECT_ROOT <- get_project_root()

#' Construct path relative to project root
#' @param ... Path components
#' @return Full path
project_path <- function(...) {
  file.path(PROJECT_ROOT, ...)
}

# ==============================================================================
# METADATA LOADING
# ==============================================================================

#' Load and parse PigGTEx metadata
#'
#' Loads the metadata file, parses age values, and assigns developmental stages.
#'
#' @param filter_valid If TRUE, removes samples with missing Stage or Tissue
#' @return Tibble with standardized metadata columns
#' @export
load_metadata <- function(filter_valid = TRUE) {
  metadata_path <- project_path("data", "PigGTEx_v0.MetaTable.xlsx")
  
  if (!file.exists(metadata_path)) {
    stop("Metadata file not found: ", metadata_path)
  }
  
  raw <- readxl::read_excel(metadata_path)
  
  # Standardize column names and parse
  metadata <- raw %>%
    dplyr::rename(
      Sample_ID = BioSample,
      Tissue = `Main categories`,
      Tissue_detail = `Sub categories`,
      Tissue_class = `Tissue class`,
      Sex = Sex,
      Age_raw = Age
    ) %>%
    dplyr::mutate(
      Age_days = vapply(Age_raw, parse_age_days, numeric(1)),
      Stage = vapply(Age_days, assign_stage, character(1))
    ) %>%
    dplyr::select(Sample_ID, Tissue, Tissue_detail, Tissue_class, Sex, Age_raw, Age_days, Stage)
  
  if (filter_valid) {
    metadata <- metadata %>%
      filter(!is.na(Stage), !is.na(Tissue), Tissue != "Unknown")
  }
  
  message(sprintf("Loaded metadata: %d samples", nrow(metadata)))
  metadata
}

#' Parse age value to days
#'
#' Converts various age formats (e.g., "30 days", "2 weeks", "3 months")
#' to numeric days.
#'
#' @param age_value Age value (character or numeric)
#' @return Numeric age in days, or NA
parse_age_days <- function(age_value) {
  if (is.na(age_value)) return(NA_real_)
  if (is.numeric(age_value)) return(as.numeric(age_value))
  
  age_str <- tolower(trimws(as.character(age_value)))
  if (age_str == "" || grepl("unknown", age_str)) return(NA_real_)
  
  value <- suppressWarnings(as.numeric(gsub("[^0-9.]+", "", age_str)))
  if (is.na(value)) return(NA_real_)
  
  if (grepl("day", age_str)) return(value)
  if (grepl("week", age_str)) return(value * 7)
  if (grepl("month", age_str)) return(value * 30)
  if (grepl("year", age_str)) return(value * 365)
  
  NA_real_
}

#' Assign developmental stage based on age in days
#'
#' Uses the stage boundaries defined in configuration.
#'
#' @param age_days Numeric age in days
#' @return Character stage label
#' @export
assign_stage <- function(age_days) {
  if (is.na(age_days)) return(NA_character_)
  
  if (age_days <= 20) return("Infant_0_20d")
  if (age_days <= 59) return("Early childhood_21_59d")
  if (age_days <= 149) return("Pre_pubertal_60_149d")
  if (age_days <= 365) return("Post_pubertal_150_365d")
  
  "Adult_>365d"
}

#' Convert stage to numeric for correlation analysis
#' @param stage Stage label
#' @return Numeric value 1-5
stage_to_numeric <- function(stage) {
  dplyr::case_when(
    grepl("Infant", stage) ~ 1,
    grepl("Early", stage) ~ 2,
    grepl("Pre", stage) ~ 3,
    grepl("Post", stage) ~ 4,
    grepl("Adult", stage) ~ 5,
    TRUE ~ NA_real_
  )
}

# ==============================================================================
# ML RESULTS LOADING
# ==============================================================================

#' Load ML model results for a tissue
#'
#' @param tissue Tissue name (e.g., "Muscle", "Brain")
#' @return List containing model results
#' @export
load_ml_results <- function(tissue) {
  results_path <- project_path("machine_learning/model_outputs", paste0(tissue, "_results.json"))
  
  if (!file.exists(results_path)) {
    warning("Results not found for ", tissue, ": ", results_path)
    return(NULL)
  }
  
  jsonlite::fromJSON(results_path)
}

#' Load ML results for all tissues
#'
#' @param tissues Character vector of tissue names (default: all with models)
#' @return Named list of results
#' @export
load_all_ml_results <- function(tissues = c("Muscle", "Brain", "Liver", "Blood", "Lung")) {
  results <- list()
  
  for (tissue in tissues) {
    res <- load_ml_results(tissue)
    if (!is.null(res)) {
      results[[tissue]] <- res
      message(sprintf("  Loaded ML results: %s", tissue))
    }
  }
  
  if (length(results) == 0) {
    stop("No ML results found for any tissue")
  }
  
  results
}

#' Extract performance metrics from results
#'
#' @param results_list List of ML results
#' @return Tibble with performance metrics per tissue
#' @export
extract_performance_metrics <- function(results_list) {
  purrr::map_dfr(names(results_list), function(tissue) {
    res <- results_list[[tissue]]
    metrics <- res$metrics
    bootstrap <- metrics$bootstrap
    
    tibble::tibble(
      Tissue = tissue,
      Scheme = res$scheme,
      N_samples = res$n_samples,
      Balanced_Accuracy = metrics$balanced_accuracy,
      BA_CI_lower = if (!is.null(bootstrap$balanced_accuracy)) bootstrap$balanced_accuracy$ci_lower else NA,
      BA_CI_upper = if (!is.null(bootstrap$balanced_accuracy)) bootstrap$balanced_accuracy$ci_upper else NA,
      F1_macro = metrics$f1_macro,
      F1_CI_lower = if (!is.null(bootstrap$f1_macro)) bootstrap$f1_macro$ci_lower else NA,
      F1_CI_upper = if (!is.null(bootstrap$f1_macro)) bootstrap$f1_macro$ci_upper else NA,
      MAE = if (!is.null(metrics$mae)) metrics$mae else NA,
      Spearman_r = if (!is.null(metrics$spearman_r)) metrics$spearman_r else NA
    )
  })
}

# ==============================================================================
# EXPRESSION DATA LOADING
# ==============================================================================

#' Load expression data for a tissue
#'
#' @param tissue Tissue name
#' @param log_transform Apply log2(TPM + 1) transformation
#' @return Expression matrix (genes x samples)
#' @export
load_expression <- function(tissue, log_transform = FALSE) {
  # Map tissue name to file
  file_map <- c(
    "Muscle" = "Muscle.expr_tpm.txt.gz",
    "Brain" = "Brain.expr_tpm.txt.gz",
    "Liver" = "Liver.expr_tpm.txt.gz",
    "Blood" = "Blood.expr_tpm.txt.gz",
    "Lung" = "Lung.expr_tpm.txt.gz",
    "Adipose" = "Adipose.expr_tpm.txt.gz",
    "Small intestine" = "Small_intestine.expr_tpm.txt.gz",
    "Testis" = "Testis.expr_tpm.txt.gz"
  )
  
  filename <- file_map[tissue]
  if (is.na(filename)) {
    stop("Unknown tissue: ", tissue)
  }
  
  expr_path <- project_path("data/pigGTEx", filename)
  
  if (!file.exists(expr_path)) {
    stop("Expression file not found: ", expr_path)
  }
  
  # Read compressed file (.gz format)
  # Use gunzip -c or zcat with proper handling for .gz files
  if (grepl("\\.gz$", expr_path)) {
    expr <- data.table::fread(cmd = paste("gunzip -c", expr_path), header = TRUE)
  } else {
    expr <- data.table::fread(cmd = paste("zcat", expr_path), header = TRUE)
  }
  expr <- as.data.frame(expr)
  rownames(expr) <- expr[[1]]
  expr <- expr[, -1, drop = FALSE]
  
  if (log_transform) {
    expr <- log2(expr + 1)
  }
  
  message(sprintf("Loaded expression: %s (%d genes, %d samples)", 
                  tissue, nrow(expr), ncol(expr)))
  as.matrix(expr)
}

# ==============================================================================
# ENRICHMENT DATA LOADING
# ==============================================================================

#' Load GO enrichment results
#'
#' @return Tibble with GO enrichment results
#' @export
load_go_enrichment <- function() {
  go_path <- project_path("results/enrichment_analysis/all_tissues_go_enrichment.csv")
  
  if (!file.exists(go_path)) {
    warning("GO enrichment file not found: ", go_path)
    return(NULL)
  }
  
  readr::read_csv(go_path, show_col_types = FALSE)
}

# ==============================================================================
# CONFUSION MATRIX HELPERS
# ==============================================================================

#' Calculate adjacent error rate from confusion matrix
#'
#' Adjacent errors are misclassifications to immediately adjacent stages.
#'
#' @param cm Confusion matrix (square matrix)
#' @return Percentage of errors that are adjacent
#' @export
calc_adjacent_error_rate <- function(cm) {
  n_classes <- nrow(cm)
  adjacent_errors <- 0
  total_errors <- 0
  
  for (row in 1:n_classes) {
    for (col in 1:n_classes) {
      if (row != col) {
        total_errors <- total_errors + cm[row, col]
        if (abs(row - col) == 1) {
          adjacent_errors <- adjacent_errors + cm[row, col]
        }
      }
    }
  }
  
  if (total_errors > 0) {
    adjacent_errors / total_errors * 100
  } else {
    100
  }
}

#' Normalize confusion matrix by row (true class)
#'
#' @param cm Confusion matrix
#' @return Percentage confusion matrix
#' @export
normalize_confusion_matrix <- function(cm) {
  cm_pct <- sweep(cm, 1, rowSums(cm), FUN = "/") * 100
  cm_pct[is.nan(cm_pct)] <- 0
  cm_pct
}

# ==============================================================================
# SUMMARY STATISTICS WRITER
# ==============================================================================

#' Write summary statistics to text file
#'
#' @param summary_text Character string of summary
#' @param filename Output filename (without path)
#' @export
write_summary <- function(summary_text, filename) {
  output_path <- project_path("paper/figures/output/stats", filename)
  dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
  writeLines(summary_text, output_path)
  message("Saved summary: ", output_path)
}

# ==============================================================================
# PRINT LOADED MESSAGE
# ==============================================================================
message("Utility functions loaded successfully")
message("  - Project root: ", PROJECT_ROOT)
