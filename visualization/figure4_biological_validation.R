#!/usr/bin/env Rscript
# Figure 4: Biological Validation
# GO enrichment and pathway analysis

library(tidyverse)
library(patchwork)
library(viridis)
library(ggraph)
library(igraph)
library(scales)
library(data.table)

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

# Panel A: GO enrichment bubble plot with exact values from enrichment.txt
create_panel_a <- function() {
  go_file <- file.path(base_dir, "results", "enrichment_analysis", "all_tissues_go_enrichment.csv")
  if (!file.exists(go_file)) {
    stop("GO enrichment results not found: ", go_file)
  }

  enrichment_raw <- read_csv(go_file, show_col_types = FALSE)

  enrichment_data <- enrichment_raw %>%
    filter(source %in% c("GO:BP", "GO:MF", "GO:CC")) %>%
    mutate(
      q_value = ifelse(is.na(q_value), p_value, q_value),
      Category = recode(source, `GO:BP` = "GO:BP", `GO:MF` = "GO:MF", `GO:CC` = "GO:CC")
    ) %>%
    filter(!is.na(q_value), q_value < 0.05) %>%
    group_by(Tissue) %>%
    arrange(q_value) %>%
    slice_head(n = 3) %>%
    ungroup() %>%
    transmute(
      Tissue,
      GO_Term = term_name,
      Fold_Enrichment = fold_enrichment,
      neg_log_q = -log10(q_value),
      Category
    )

  if (nrow(enrichment_data) == 0) {
    stop("No significant GO terms found at q < 0.05.")
  }

  enrichment_data$Category <- factor(
    enrichment_data$Category,
    levels = c("GO:BP", "GO:MF", "GO:CC")
  )

  # Order tissues by classification scheme
  enrichment_data$Tissue <- factor(enrichment_data$Tissue,
    levels = c(
      "Muscle", "Brain", "Liver",
      "Blood", "Small intestine", "Lung",
      "Adipose", "Testis"
    )
  )

  # Create bubble plot
  p <- ggplot(enrichment_data, aes(x = Tissue, y = GO_Term)) +
    geom_point(aes(size = Fold_Enrichment, color = neg_log_q),
      alpha = 0.8
    ) +
    scale_size_continuous(
      range = c(2, 12),
      name = "Fold\nEnrichment"
    ) +
    scale_color_viridis(
      option = "C",
      name = "-log10(q)"
    ) +
    facet_grid(rows = vars(Category), scales = "free_y", space = "free_y") +
    labs(
      title = "Gene Ontology Enrichment Analysis",
      subtitle = "Top 3 GO terms per tissue (q < 0.05)",
      x = NULL, y = NULL
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
      axis.text.y = element_text(size = 8),
      plot.title = element_text(size = 12, face = "bold"),
      plot.subtitle = element_text(size = 10),
      strip.text = element_text(size = 9, face = "bold"),
      legend.title = element_text(size = 9),
      legend.text = element_text(size = 8)
    )

  p
}

# Panel B: Tissue-specific developmental programs network
create_panel_b <- function() {
  go_file <- file.path(base_dir, "results", "enrichment_analysis", "all_tissues_go_enrichment.csv")
  if (!file.exists(go_file)) {
    stop("GO enrichment results not found: ", go_file)
  }

  go_data <- read_csv(go_file, show_col_types = FALSE) %>%
    mutate(q_value = ifelse(is.na(q_value), p_value, q_value))

  go_bp <- go_data %>%
    filter(source == "GO:BP", !is.na(q_value), q_value < 0.05)

  if (nrow(go_bp) == 0) {
    stop("No GO:BP enrichment terms found at q < 0.05 for network panel.")
  }

  tissue_terms <- go_bp %>%
    group_by(Tissue) %>%
    summarise(terms = list(unique(term_id)), .groups = "drop")

  tissues <- tissue_terms$Tissue
  edges <- expand.grid(from = tissues, to = tissues, stringsAsFactors = FALSE) %>%
    filter(from < to) %>%
    rowwise() %>%
    mutate(
      overlap = length(intersect(
        tissue_terms$terms[[match(from, tissue_terms$Tissue)]],
        tissue_terms$terms[[match(to, tissue_terms$Tissue)]]
      )),
      union = length(union(
        tissue_terms$terms[[match(from, tissue_terms$Tissue)]],
        tissue_terms$terms[[match(to, tissue_terms$Tissue)]]
      )),
      weight = ifelse(union > 0, overlap / union, 0)
    ) %>%
    ungroup() %>%
    filter(weight > 0) %>%
    select(from, to, weight)

  nodes <- tissue_terms %>%
    mutate(n_terms = lengths(terms)) %>%
    select(Tissue, n_terms) %>%
    rename(name = Tissue)

  g <- graph_from_data_frame(edges, directed = FALSE, vertices = nodes)

  tissue_colors <- c(
    "Adipose" = "#FFD700", "Blood" = "#DC143C",
    "Brain" = "#4169E1", "Liver" = "#8B4513",
    "Lung" = "#FF69B4", "Muscle" = "#FF6347",
    "Small intestine" = "#32CD32", "Testis" = "#9370DB"
  )

  set.seed(42)
  p <- ggraph(g, layout = "fr") +
    geom_edge_link(aes(alpha = weight, width = weight), color = "gray50") +
    geom_node_point(aes(size = n_terms, color = name)) +
    geom_node_text(aes(label = name), size = 3, repel = TRUE) +
    scale_color_manual(values = tissue_colors, name = "Tissue") +
    scale_size_continuous(range = c(3, 10), name = "Significant\nGO terms") +
    scale_edge_alpha_continuous(range = c(0.2, 0.8), guide = "none") +
    scale_edge_width_continuous(range = c(0.3, 1.2), guide = "none") +
    labs(
      title = "Tissue-Specific Developmental Programs",
      subtitle = "Edges show GO:BP overlap (Jaccard similarity)"
    ) +
    theme_void() +
    theme(
      plot.title = element_text(size = 12, face = "bold"),
      plot.subtitle = element_text(size = 10),
      legend.title = element_text(size = 9),
      legend.text = element_text(size = 8)
    )

  p
}

# Panel C: Key marker validation with real expression data
create_panel_c <- function() {
  metadata <- load_pig_metadata() %>%
    filter(!is.na(Stage))

  marker_info <- tibble(
    Gene = c("PPARG", "MBP", "ALB", "MSTN", "HBB", "SFTPC", "LGR5", "DMRT1"),
    Tissue = c("Adipose", "Brain", "Liver", "Muscle", "Blood", "Lung", "Small intestine", "Testis"),
    Ensembl_ID = c(
      "ENSSSCG00000004130",
      "ENSSSCG00000008574",
      "ENSSSCG00000005095",
      "ENSSSCG00000039058",
      "ENSSSCG00000014727",
      "ENSSSCG00000007979",
      "ENSSSCG00000004244",
      "ENSSSCG00000013313"
    )
  )

  tissue_file_map <- c(
    "Adipose" = "Adipose.expr_tpm.txt.gz",
    "Brain" = "Brain.expr_tpm.txt.gz",
    "Liver" = "Liver.expr_tpm.txt.gz",
    "Muscle" = "Muscle.expr_tpm.txt.gz",
    "Blood" = "Blood.expr_tpm.txt.gz",
    "Lung" = "Lung.expr_tpm.txt.gz",
    "Small intestine" = "Small_intestine.expr_tpm.txt.gz",
    "Testis" = "Testis.expr_tpm.txt.gz"
  )

  stage_mapping <- c(
    "Infant_0_20d" = "Infant",
    "Early childhood_21_59d" = "Early",
    "Pre_pubertal_60_149d" = "Pre-pub",
    "Post_pubertal_150_365d" = "Post-pub",
    "Adult_>365d" = "Post-pub"
  )

  stages <- c("Infant", "Early", "Pre-pub", "Post-pub")

  all_markers <- map_dfr(unique(marker_info$Tissue), function(tissue) {
    file_name <- tissue_file_map[tissue]
    file_path <- file.path(base_dir, "data", "pigGTEx", file_name)
    if (!file.exists(file_path)) {
      warning("Expression file missing for tissue: ", tissue)
      return(tibble())
    }

    tissue_markers <- marker_info %>%
      filter(Tissue == tissue)

    expr <- data.table::fread(file_path)
    gene_col <- colnames(expr)[1]
    expr_filtered <- expr[get(gene_col) %in% tissue_markers$Ensembl_ID]

    if (nrow(expr_filtered) == 0) {
      expr$gene_prefix <- sub("\\..*", "", expr[[gene_col]])
      marker_prefix <- sub("\\..*", "", tissue_markers$Ensembl_ID)
      expr_filtered <- expr[gene_prefix %in% marker_prefix]
      expr$gene_prefix <- NULL
    }

    if (nrow(expr_filtered) == 0) {
      warning("No marker genes found in expression file for tissue: ", tissue)
      return(tibble())
    }

    expr_long <- as.data.frame(expr_filtered) %>%
      mutate(Gene_ID = .data[[gene_col]]) %>%
      select(-all_of(gene_col)) %>%
      pivot_longer(cols = -Gene_ID, names_to = "Sample_ID", values_to = "Expression") %>%
      left_join(tissue_markers, by = c("Gene_ID" = "Ensembl_ID")) %>%
      select(Gene, Tissue, Sample_ID, Expression)

    expr_long %>%
      left_join(metadata %>% select(Sample_ID, Stage), by = "Sample_ID") %>%
      filter(!is.na(Stage)) %>%
      mutate(Stage_Label = stage_mapping[Stage]) %>%
      filter(!is.na(Stage_Label)) %>%
      group_by(Tissue, Gene, Stage_Label) %>%
      summarise(
        Expression = mean(log2(Expression + 1), na.rm = TRUE),
        SEM = sd(log2(Expression + 1), na.rm = TRUE) / sqrt(n()),
        .groups = "drop"
      ) %>%
      group_by(Tissue, Gene) %>%
      complete(
        Stage_Label = stages,
        fill = list(Expression = NA_real_, SEM = NA_real_)
      ) %>%
      ungroup() %>%
      mutate(SEM = ifelse(is.na(SEM), 0, SEM)) %>%
      rename(Stage = Stage_Label)
  })

  if (nrow(all_markers) == 0) {
    stop("No marker expression data available for Panel C.")
  }

  all_markers$Stage <- factor(all_markers$Stage, levels = stages)

  # Create multi-panel plot
  p <- ggplot(all_markers, aes(x = Stage, y = Expression, group = Gene)) +
    geom_line(color = "#809BCE", size = 1) +
    geom_point(size = 3, color = "#809BCE") +
    geom_errorbar(aes(ymin = Expression - SEM, ymax = Expression + SEM),
      width = 0.2, alpha = 0.5
    ) +
    facet_wrap(~ paste0(Gene, " (", Tissue, ")"), ncol = 4, scales = "free_y") +
    labs(
      title = "Validation with Known Developmental Markers",
      subtitle = "Expression levels (log2 TPM) across developmental stages",
      x = "Developmental Stage",
      y = "Expression (log2 TPM)"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
      plot.title = element_text(size = 12, face = "bold"),
      plot.subtitle = element_text(size = 10),
      strip.text = element_text(size = 8, face = "bold")
    )

  p
}

# Combine all panels
panel_a <- create_panel_a()
panel_b <- create_panel_b()
panel_c <- create_panel_c()

# Create final figure
final_figure <- (panel_a | panel_b) / panel_c +
  plot_layout(heights = c(1.2, 1)) +
  plot_annotation(
    tag_levels = "A",
    theme = theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 12)
    )
  )

# Save figure
ggsave(file.path(output_dir, "figure4_biological_validation.pdf"),
  final_figure,
  width = 16, height = 12, dpi = 300
)
ggsave(file.path(output_dir, "figure4_biological_validation.png"),
  final_figure,
  width = 16, height = 12, dpi = 300
)

cat("Figure 4 saved to:", output_dir, "\n")
