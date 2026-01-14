#!/usr/bin/env Rscript
# Figure 3: Tissue-Specific Molecular Signatures
# UMAP visualization and feature analysis

library(tidyverse)
library(data.table)
library(patchwork)
library(viridis)
library(jsonlite)
library(umap)
library(UpSetR)
library(ComplexHeatmap)
library(circlize)
library(RColorBrewer)

# Set publication theme
args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg)) else getwd()
base_dir <- dirname(script_dir)
source(file.path(script_dir, "theme_configs", "nature_theme.R"))
source(file.path(script_dir, "utils", "metadata_utils.R"))

# Output directory
output_dir <- "."
dir.create(output_dir, showWarnings = FALSE)

# Set seed for reproducibility
set.seed(42)

# Load data
tissues <- c(
  "Muscle", "Brain", "Liver", "Blood", "Small intestine",
  "Lung", "Adipose", "Testis"
)

# Load expression data and metadata for UMAP
metadata <- load_pig_metadata() %>%
  filter(!is.na(Stage), Tissue != "Unknown")

# Helper utilities ---------------------------------------------------------

# Gene labels default to Ensembl IDs to avoid external annotation dependencies.
fetch_gene_symbols <- function(ensembl_ids) {
  unique_ids <- unique(na.omit(ensembl_ids))
  tibble(Ensembl_ID = unique_ids, Gene_Name = NA_character_)
}

# Load the top-N discriminative gene identifiers for a specific tissue.
load_top_gene_ids <- function(tissue, top_n = 5) {
  file_path <- file.path(
    base_dir, "machine_learning", "model_outputs",
    paste0(tissue, "_results.json")
  )

  if (!file.exists(file_path)) {
    warning(sprintf("Results file missing for %s (expected at %s).", tissue, file_path))
    return(tibble())
  }

  res <- fromJSON(file_path)

  gene_ids <- NULL
  if (!is.null(res$top_genes)) {
    gene_ids <- res$top_genes
  } else if (!is.null(res$top_features)) {
    gene_ids <- res$top_features
  }

  if (is.null(gene_ids) || length(gene_ids) == 0) {
    warning(sprintf("No top gene information available for %s.", tissue))
    return(tibble())
  }

  tibble::tibble(
    Tissue = tissue,
    Ensembl_ID = gene_ids[seq_len(min(length(gene_ids), top_n))]
  )
}

# Panel A: UMAP visualization with dual encoding
create_panel_a <- function() {
  # Load real expression data from TPM files and compute UMAP
  cat("Loading expression data for UMAP...\n")

  # Define file mappings for tissue names to filenames
  tissue_file_map <- c(
    "Muscle" = "Muscle.expr_tpm.txt.gz",
    "Brain" = "Brain.expr_tpm.txt.gz",
    "Liver" = "Liver.expr_tpm.txt.gz",
    "Blood" = "Blood.expr_tpm.txt.gz",
    "Lung" = "Lung.expr_tpm.txt.gz",
    "Adipose" = "Adipose.expr_tpm.txt.gz",
    "Small intestine" = "Small_intestine.expr_tpm.txt.gz",
    "Testis" = "Testis.expr_tpm.txt.gz"
  )

  # Load and combine expression data with aggressive subsampling for speed
  max_samples_per_tissue <- 30 # Reduced for faster UMAP computation
  all_expr_data <- list()
  all_sample_info <- list()

  for (tissue in names(tissue_file_map)) {
    file_path <- file.path(base_dir, "data", "pigGTEx", tissue_file_map[tissue])
    if (file.exists(file_path)) {
      cat(sprintf("  Loading %s...\n", tissue))
      # Read only first 500 genes for speed
      expr <- read.table(gzfile(file_path),
        header = TRUE, sep = "\t",
        row.names = 1, check.names = FALSE, nrows = 500
      )

      # Subsample if too many samples
      n_samples <- min(ncol(expr), max_samples_per_tissue)
      if (ncol(expr) > max_samples_per_tissue) {
        set.seed(42)
        sample_cols <- sample(1:ncol(expr), n_samples)
        expr <- expr[, sample_cols, drop = FALSE]
      }

      all_expr_data[[tissue]] <- expr
      all_sample_info[[tissue]] <- data.frame(
        Sample_ID = colnames(expr),
        Tissue = tissue,
        stringsAsFactors = FALSE
      )
    }
  }

  # Combine all expression data
  if (length(all_expr_data) == 0) {
    stop("No expression data files found for UMAP panel.")
  }

  # Find common genes across tissues
  common_genes <- Reduce(intersect, lapply(all_expr_data, rownames))
  cat(sprintf(
    "  Found %d common genes across %d tissues\n",
    length(common_genes), length(all_expr_data)
  ))

  # Use top highly variable genes for UMAP
  n_hvg <- min(1000, length(common_genes))
  combined_expr <- do.call(cbind, lapply(all_expr_data, function(x) x[common_genes, ]))

  # Calculate variance and select top genes
  gene_vars <- apply(combined_expr, 1, var, na.rm = TRUE)
  hvg <- names(sort(gene_vars, decreasing = TRUE))[1:n_hvg]
  expr_for_umap <- t(log1p(combined_expr[hvg, ])) # Log transform and transpose

  # Merge sample info
  sample_info <- do.call(rbind, all_sample_info)

  # Add stage information from metadata
  sample_info <- sample_info %>%
    left_join(metadata %>% select(Sample_ID, Stage), by = "Sample_ID") %>%
    mutate(
      Stage_Num = case_when(
        grepl("Infant", Stage) ~ 1,
        grepl("Early", Stage) ~ 2,
        grepl("Pre", Stage) ~ 3,
        grepl("Post", Stage) ~ 4,
        grepl("Adult", Stage) ~ 4,
        TRUE ~ NA_real_
      )
    ) %>%
    filter(!is.na(Stage_Num))

  expr_for_umap <- expr_for_umap[rownames(expr_for_umap) %in% sample_info$Sample_ID, , drop = FALSE]
  sample_info <- sample_info %>%
    filter(Sample_ID %in% rownames(expr_for_umap))
  sample_info <- sample_info[match(rownames(expr_for_umap), sample_info$Sample_ID), ]

  # Compute UMAP
  cat("  Computing UMAP...\n")
  umap_config <- umap.defaults
  umap_config$n_neighbors <- 15
  umap_config$min_dist <- 0.1
  umap_result <- umap(expr_for_umap, config = umap_config)

  # Create UMAP data frame
  umap_data <- data.frame(
    UMAP1 = umap_result$layout[, 1],
    UMAP2 = umap_result$layout[, 2],
    Tissue = sample_info$Tissue,
    Stage = sample_info$Stage_Num,
    Sample_ID = sample_info$Sample_ID
  )

  # Define shapes and colors
  tissue_shapes <- c(16, 17, 15, 18, 8, 10, 11, 12) # Different point shapes
  stage_colors <- c("#E8F4FD", "#95B8D1", "#809BCE", "#666A86")

  # Convert stage to factor with labels
  umap_data$Stage_Label <- factor(umap_data$Stage,
    levels = 1:4,
    labels = c("Infant", "Early", "Pre-pub", "Post-pub")
  )

  umap_data$Tissue <- factor(umap_data$Tissue, levels = tissues)

  # Create UMAP plot
  p <- ggplot(umap_data, aes(x = UMAP1, y = UMAP2)) +
    geom_point(aes(color = Stage_Label, shape = Tissue),
      size = 2, alpha = 0.7
    ) +
    scale_color_manual(
      values = stage_colors,
      name = "Developmental\nStage"
    ) +
    scale_shape_manual(
      values = tissue_shapes,
      name = "Tissue"
    ) +
    stat_ellipse(aes(group = Tissue),
      type = "norm",
      linetype = 2, color = "gray50", alpha = 0.5
    ) +
    labs(
      title = "Tissue-Specific Clustering with Developmental Gradients",
      subtitle = sprintf("UMAP of transcriptomic profiles (n = %s samples)", format(nrow(umap_data), big.mark = ",")),
      x = "UMAP 1", y = "UMAP 2"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 12, face = "bold"),
      plot.subtitle = element_text(size = 10),
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 9)
    ) +
    guides(
      shape = guide_legend(override.aes = list(size = 3)),
      color = guide_legend(override.aes = list(size = 3))
    )

  p
}

# Panel B: Feature overlap analysis (UpSet plot)
create_panel_b <- function() {
  # Load top features for each tissue
  feature_lists <- list()
  for (tissue in tissues) {
    file_path <- file.path(base_dir, "machine_learning", "model_outputs", paste0(tissue, "_results.json"))
    if (file.exists(file_path)) {
      res <- fromJSON(file_path)
      if (!is.null(res$top_genes)) {
        feature_lists[[tissue]] <- res$top_genes[1:min(50, length(res$top_genes))]
      } else if (!is.null(res$top_features)) {
        feature_lists[[tissue]] <- res$top_features[1:min(50, length(res$top_features))]
      }
    }
  }

  if (length(feature_lists) == 0) {
    stop("No top gene lists available for overlap analysis.")
  }

  # Calculate overlaps
  available_tissues <- names(feature_lists)
  all_genes <- unique(unlist(feature_lists))
  overlap_matrix <- matrix(0, nrow = length(all_genes), ncol = length(available_tissues))
  colnames(overlap_matrix) <- available_tissues

  for (i in seq_along(available_tissues)) {
    overlap_matrix[all_genes %in% feature_lists[[available_tissues[i]]], i] <- 1
  }

  # Calculate overlap percentages
  total_unique <- length(all_genes)
  tissue_specific <- sum(rowSums(overlap_matrix) == 1)
  shared_genes <- sum(rowSums(overlap_matrix) > 1)
  overlap_pct <- (shared_genes / total_unique) * 100

  # Create custom upset plot using ggplot
  # Simplified version for visualization
  overlap_summary <- data.frame(
    Category = c("Tissue-specific", "2 tissues", "3 tissues", "4+ tissues"),
    Count = c(
      sum(rowSums(overlap_matrix) == 1),
      sum(rowSums(overlap_matrix) == 2),
      sum(rowSums(overlap_matrix) == 3),
      sum(rowSums(overlap_matrix) >= 4)
    )
  )

  overlap_summary$Percentage <- overlap_summary$Count / sum(overlap_summary$Count) * 100

  p <- ggplot(overlap_summary, aes(x = reorder(Category, -Count), y = Count)) +
    geom_col(fill = "#809BCE", alpha = 0.8) +
    geom_text(aes(label = sprintf("%d\n(%.1f%%)", Count, Percentage)),
      vjust = -0.5, size = 3.5
    ) +
    labs(
      title = "Feature Overlap Analysis",
      subtitle = sprintf("Top 50 genes per tissue, %.1f%% overlap", overlap_pct),
      x = "Gene Sharing Pattern",
      y = "Number of Genes"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 12, face = "bold"),
      plot.subtitle = element_text(size = 10),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )

  p
}

# Panel C: Top discriminative genes heatmap
create_panel_c <- function(top_n = 5) {
  top_gene_ids <- map_dfr(tissues, load_top_gene_ids, top_n = top_n)

  if (nrow(top_gene_ids) == 0) {
    stop("No top gene identifiers available to build Panel C.")
  }

  top_gene_ids <- top_gene_ids %>%
    dplyr::mutate(Tissue = factor(Tissue, levels = tissues)) %>%
    dplyr::arrange(Tissue)

  gene_lookup <- fetch_gene_symbols(top_gene_ids$Ensembl_ID)

  top_gene_ids <- top_gene_ids %>%
    dplyr::left_join(gene_lookup, by = "Ensembl_ID") %>%
    dplyr::mutate(
      Gene_Label = if_else(!is.na(Gene_Name) & Gene_Name != "",
        Gene_Name,
        Ensembl_ID
      ),
      Gene_Display = paste0(Gene_Label, " (", Tissue, ")")
    )

  # Load real expression data for top genes
  stages <- c("Infant", "Early", "Pre-pub", "Post-pub")

  stage_mapping <- c(
    "Infant_0_20d" = "Infant",
    "Early childhood_21_59d" = "Early",
    "Pre_pubertal_60_149d" = "Pre-pub",
    "Post_pubertal_150_365d" = "Post-pub",
    "Adult_>365d" = "Post-pub" # Combine adult with post-pub for consistency
  )

  tissue_file_map <- c(
    "Muscle" = "Muscle.expr_tpm.txt.gz",
    "Brain" = "Brain.expr_tpm.txt.gz",
    "Liver" = "Liver.expr_tpm.txt.gz",
    "Blood" = "Blood.expr_tpm.txt.gz",
    "Lung" = "Lung.expr_tpm.txt.gz",
    "Adipose" = "Adipose.expr_tpm.txt.gz",
    "Small intestine" = "Small_intestine.expr_tpm.txt.gz",
    "Testis" = "Testis.expr_tpm.txt.gz"
  )

  top_genes_data <- map_dfr(seq_len(nrow(top_gene_ids)), function(i) {
    gene_id <- top_gene_ids$Ensembl_ID[i]
    tissue <- as.character(top_gene_ids$Tissue[i])
    gene_display <- top_gene_ids$Gene_Display[i]

    # Try to load expression for this tissue
    file_name <- tissue_file_map[tissue]
    if (is.na(file_name)) {
      return(tibble())
    }

    file_path <- file.path(base_dir, "data", "pigGTEx", file_name)
    if (!file.exists(file_path)) {
      return(tibble())
    }

    expr_data <- tryCatch(
      {
        expr <- data.table::fread(file_path)
        gene_col <- colnames(expr)[1]
        gene_row <- expr[get(gene_col) == gene_id]
        if (nrow(gene_row) == 0) {
          expr$gene_prefix <- sub("\\..*", "", expr[[gene_col]])
          gene_row <- expr[gene_prefix == sub("\\..*", "", gene_id)]
          expr$gene_prefix <- NULL
        }
        if (nrow(gene_row) == 0) {
          return(NULL)
        }
        sample_ids <- colnames(gene_row)[-1]
        values <- as.numeric(unlist(gene_row[1, -1, with = FALSE]))
        data.frame(Sample_ID = sample_ids, Expression = values, stringsAsFactors = FALSE)
      },
      error = function(e) NULL
    )

    if (is.null(expr_data)) {
      return(tibble())
    }

    # Merge with metadata for stage
    expr_with_stage <- expr_data %>%
      left_join(metadata %>% select(Sample_ID, Stage), by = "Sample_ID") %>%
      filter(!is.na(Stage)) %>%
      mutate(Stage_Label = stage_mapping[Stage]) %>%
      filter(!is.na(Stage_Label))

    if (nrow(expr_with_stage) == 0) {
      return(tibble())
    }

    # Calculate mean expression per stage
    stage_means <- expr_with_stage %>%
      group_by(Stage_Label) %>%
      summarise(Expression = mean(log1p(Expression), na.rm = TRUE), .groups = "drop")

    # Ensure all stages are present
    result <- data.frame(Stage = stages, stringsAsFactors = FALSE) %>%
      left_join(stage_means, by = c("Stage" = "Stage_Label")) %>%
      mutate(
        Gene_Display = gene_display,
        Tissue = tissue
      )

    result
  })

  if (nrow(top_genes_data) == 0) {
    stop("No expression data available for top discriminative genes.")
  }

  expr_matrix <- top_genes_data %>%
    dplyr::select(Gene_Display, Stage, Expression) %>%
    tidyr::pivot_wider(names_from = Stage, values_from = Expression) %>%
    column_to_rownames("Gene_Display") %>%
    as.matrix()

  expr_matrix <- expr_matrix[complete.cases(expr_matrix), , drop = FALSE]
  if (nrow(expr_matrix) == 0) {
    stop("No complete expression profiles available for heatmap.")
  }

  expr_matrix <- t(scale(t(expr_matrix)))

  gene_tissue_lookup <- top_gene_ids %>%
    dplyr::select(Gene_Display, Tissue) %>%
    dplyr::distinct()

  expr_df <- as.data.frame(expr_matrix) %>%
    tibble::rownames_to_column("Gene") %>%
    tidyr::pivot_longer(-Gene, names_to = "Stage", values_to = "Expression") %>%
    dplyr::left_join(gene_tissue_lookup, by = c("Gene" = "Gene_Display"))

  expr_df$Gene <- factor(expr_df$Gene, levels = rownames(expr_matrix))
  expr_df$Stage <- factor(expr_df$Stage, levels = stages)
  expr_df$Tissue <- factor(expr_df$Tissue, levels = tissues)

  p <- ggplot(expr_df, aes(x = Stage, y = Gene, fill = Expression)) +
    geom_tile(color = "white", size = 0.1) +
    scale_fill_gradient2(
      low = "blue", mid = "white", high = "red",
      midpoint = 0, limits = c(-2, 2),
      name = "Scaled\nExpression"
    ) +
    facet_grid(rows = vars(Tissue), scales = "free_y", space = "free_y") +
    labs(
      title = "Top Discriminative Genes",
      subtitle = sprintf("Top %d genes per tissue (row-scaled expression)", top_n),
      x = "Developmental Stage", y = NULL
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 12, face = "bold"),
      plot.subtitle = element_text(size = 10),
      axis.text.y = element_text(size = 7),
      axis.text.x = element_text(angle = 45, hjust = 1),
      strip.text = element_text(size = 8, face = "bold")
    )

  p
}

# Combine all panels
panel_a <- create_panel_a()
panel_b <- create_panel_b()
panel_c <- create_panel_c()

# Create final figure
final_figure <- panel_a / (panel_b | panel_c) +
  plot_layout(heights = c(1.2, 1)) +
  plot_annotation(
    tag_levels = "A",
    theme = theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 12)
    )
  )

# Save figure
ggsave(file.path(output_dir, "figure3_molecular_signatures.pdf"),
  final_figure,
  width = 14, height = 12, dpi = 300
)
ggsave(file.path(output_dir, "figure3_molecular_signatures.png"),
  final_figure,
  width = 14, height = 12, dpi = 300
)

cat("Figure 3 saved to:", output_dir, "\n")
