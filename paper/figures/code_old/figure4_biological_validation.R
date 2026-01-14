#!/usr/bin/env Rscript
# ==============================================================================
# Figure 4: Biological Validation
# Standalone script - no external source() dependencies
# ==============================================================================

# ==============================================================================
# 1. PACKAGE LOADING
# ==============================================================================
required_packages <- c(
  "tidyverse", "patchwork", "viridis", "ggraph", "igraph",
  "scales", "data.table", "readxl", "grid", "jsonlite", "gprofiler2"
)

load_required_packages <- function(packages) {
  missing <- packages[!sapply(packages, requireNamespace, quietly = TRUE)]
  if (length(missing) > 0) {
    stop(
      "Missing required packages: ", paste(missing, collapse = ", "),
      "\nInstall with: install.packages(c('", paste(missing, collapse = "', '"), "'))"
    )
  }
  invisible(lapply(packages, library, character.only = TRUE, warn.conflicts = FALSE))
}

load_required_packages(required_packages)

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
BASE_DIR <- "/Users/tianyuan/Desktop/github_dev/pig-dev-stage"
OUTPUT_DIR <- file.path(BASE_DIR, "paper/figures/out")
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Tissues with model results
TISSUES_WITH_MODELS <- c("Muscle", "Brain", "Liver", "Blood", "Lung")

# All tissues with expression data (for marker validation)
ALL_TISSUES <- c(
  "Muscle", "Brain", "Liver", "Blood", "Lung",
  "Adipose", "Small intestine", "Testis"
)

# Tissue file mapping
TISSUE_FILE_MAP <- c(
  "Muscle" = "Muscle.expr_tpm.txt.gz",
  "Brain" = "Brain.expr_tpm.txt.gz",
  "Liver" = "Liver.expr_tpm.txt.gz",
  "Blood" = "Blood.expr_tpm.txt.gz",
  "Lung" = "Lung.expr_tpm.txt.gz",
  "Adipose" = "Adipose.expr_tpm.txt.gz",
  "Small intestine" = "Small_intestine.expr_tpm.txt.gz",
  "Testis" = "Testis.expr_tpm.txt.gz"
)

# Stage definitions
STAGE_ORDER <- c(
  "Infant_0_20d", "Early childhood_21_59d", "Pre_pubertal_60_149d",
  "Post_pubertal_150_365d", "Adult_>365d"
)

STAGE_MAPPING <- c(
  "Infant_0_20d" = "Infant",
  "Early childhood_21_59d" = "Early",
  "Pre_pubertal_60_149d" = "Pre-pub",
  "Post_pubertal_150_365d" = "Post-pub",
  "Adult_>365d" = "Post-pub"
)

STAGE_LABELS <- c("Infant", "Early", "Pre-pub", "Post-pub")

# Known developmental markers
MARKER_INFO <- tibble(
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

# Tissue colors
TISSUE_COLORS <- c(
  "Adipose" = "#FFD700", "Blood" = "#DC143C",
  "Brain" = "#4169E1", "Liver" = "#8B4513",
  "Lung" = "#32CD32", "Muscle" = "#FF6347",
  "Small intestine" = "#FFA500", "Testis" = "#9370DB"
)

set.seed(42)

# ==============================================================================
# 3. INLINED UTILITY FUNCTIONS
# ==============================================================================

# --- Nature theme ---
nature_theme <- function(base_size = 8, base_family = "sans") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      plot.title = element_text(size = base_size + 2, face = "bold", color = "black"),
      plot.subtitle = element_text(size = base_size, color = "black"),
      axis.text = element_text(size = base_size, color = "black"),
      axis.title = element_text(size = base_size, color = "black"),
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.ticks = element_line(color = "black", linewidth = 0.5),
      legend.text = element_text(size = base_size - 1),
      legend.title = element_text(size = base_size, face = "bold"),
      legend.key.size = unit(0.8, "lines"),
      legend.background = element_blank(),
      legend.key = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(size = base_size, face = "bold"),
      plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")
    )
}

# --- Metadata loader ---
load_pig_metadata <- function() {
  metadata_path <- file.path(BASE_DIR, "data", "PigGTEx_v0.MetaTable.xlsx")

  if (!file.exists(metadata_path)) {
    stop("Metadata file not found: ", metadata_path)
  }

  raw <- readxl::read_excel(metadata_path)

  parse_age_days <- function(age_value) {
    if (is.na(age_value)) {
      return(NA_real_)
    }
    if (is.numeric(age_value)) {
      return(as.numeric(age_value))
    }

    age_str <- tolower(trimws(as.character(age_value)))
    if (age_str == "" || grepl("unknown", age_str)) {
      return(NA_real_)
    }

    value <- suppressWarnings(as.numeric(gsub("[^0-9.]+", "", age_str)))
    if (is.na(value)) {
      return(NA_real_)
    }

    if (grepl("day", age_str)) {
      return(value)
    }
    if (grepl("week", age_str)) {
      return(value * 7)
    }
    if (grepl("month", age_str)) {
      return(value * 30)
    }
    if (grepl("year", age_str)) {
      return(value * 365)
    }

    NA_real_
  }

  assign_stage <- function(age_days) {
    if (is.na(age_days)) {
      return(NA_character_)
    }
    if (age_days <= 20) {
      return("Infant_0_20d")
    }
    if (age_days <= 59) {
      return("Early childhood_21_59d")
    }
    if (age_days <= 149) {
      return("Pre_pubertal_60_149d")
    }
    if (age_days <= 365) {
      return("Post_pubertal_150_365d")
    }
    "Adult_>365d"
  }

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

  metadata
}

# ==============================================================================
# 4. DATA LOADING WITH ERROR HANDLING
# ==============================================================================
load_data <- function() {
  cat("Loading data...\n")

  # Load metadata
  metadata <- tryCatch(
    load_pig_metadata() %>%
      filter(!is.na(Stage)),
    error = function(e) {
      warning("Failed to load metadata: ", e$message)
      NULL
    }
  )

  if (is.null(metadata)) {
    stop("Cannot proceed without metadata")
  }

  cat(sprintf("  Loaded %d samples from metadata\n", nrow(metadata)))

  # Load GO enrichment data
  go_file <- file.path(BASE_DIR, "results", "enrichment_analysis", "all_tissues_go_enrichment.csv")
  go_data <- NULL
  if (file.exists(go_file)) {
    go_data <- tryCatch(
      read_csv(go_file, show_col_types = FALSE),
      error = function(e) {
        warning("Failed to load GO enrichment: ", e$message)
        NULL
      }
    )
    if (!is.null(go_data)) {
      cat(sprintf("  Loaded %d GO enrichment terms\n", nrow(go_data)))
    }
  } else {
    warning("GO enrichment file not found: ", go_file)
  }

  # Load model results (for top genes enrichment)
  results_list <- list()
  for (tissue in TISSUES_WITH_MODELS) {
    path <- file.path(BASE_DIR, "machine_learning/model_outputs", paste0(tissue, "_results.json"))
    if (file.exists(path)) {
      results_list[[tissue]] <- fromJSON(path)
      cat(sprintf("  Loaded ML results for %s\n", tissue))
    }
  }

  list(metadata = metadata, go_data = go_data, results = results_list)
}

# ==============================================================================
# 5. PANEL CREATION FUNCTIONS
# ==============================================================================

# Panel A: Comparative Matrix Overview
create_panel_a <- function(data) {
  cat("Creating Panel A: Comparative Matrix Overview...\n")

  if (is.null(data$go_enrichment)) {
    return(ggplot() +
      theme_void() +
      labs(title = "No GO Enrichment Data"))
  }

  # We will use the existing "all_tissues_go_enrichment.csv" data loaded in data$go_enrichment
  # Extract top terms to display in matrix
  top_terms <- data$go_enrichment %>%
    group_by(Tissue) %>%
    arrange(p_value) %>%
    slice_head(n = 5) %>% # Top 5 per tissue for the matrix
    ungroup() %>%
    arrange(Tissue, p_value) %>%
    # Fix factor order for plotting
    mutate(term_name = factor(term_name, levels = unique(rev(term_name))))

  # Create Matrix Plot
  p <- ggplot(top_terms, aes(x = Tissue, y = term_name)) +
    geom_point(aes(size = -log10(p_value), fill = Tissue), shape = 21, color = "black") +
    scale_size_continuous(range = c(3, 8), name = "-log10(p)") +
    scale_fill_brewer(palette = "Set1") +
    labs(
      title = "Top Functional Themes",
      subtitle = "Top 5 Terms per Tissue (FDR < 0.05)",
      x = NULL, y = NULL
    ) +
    nature_theme() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
      axis.text.y = element_text(size = 8),
      panel.grid.major = element_line(color = "grey90", linewidth = 0.2),
      panel.border = element_rect(color = "black", fill = NA)
    )

  attr(p, "matrix_info") <- list(
    n_terms = nrow(top_terms),
    tissues = unique(top_terms$Tissue)
  )
  p
}

# Panel B: Enrichment Map Network Analysis
create_panel_b <- function(results_list) {
  cat("Creating Panel B: Enrichment Map Network Analysis...\n")

  if (length(results_list) == 0) {
    return(ggplot() +
      theme_void() +
      labs(title = "No ML Results"))
  }

  # 1. Run Relaxed Enrichment for All Tissues
  all_enrichment <- list()

  for (tissue in names(results_list)) {
    res <- results_list[[tissue]]
    gene_ids <- if (!is.null(res$top_genes)) res$top_genes else res$top_features

    if (is.null(gene_ids) || length(gene_ids) == 0) next

    # Use enough genes to get signal
    query_genes <- gene_ids[1:min(200, length(gene_ids))]

    tryCatch(
      {
        # Relaxed threshold to ensure we get terms for the network
        gost_result <- gost(
          query = query_genes,
          organism = "sscrofa",
          sources = c("GO:BP", "KEGG", "REAC"),
          significant = FALSE, # Get all results, filter later
          user_threshold = 1.0,
          correction_method = "fdr",
          evcodes = TRUE
        )

        if (!is.null(gost_result) && !is.null(gost_result$result) && nrow(gost_result$result) > 0) {
          enrich_df <- gost_result$result %>%
            mutate(Tissue = tissue) %>%
            filter(term_size < 3000) %>% # Exclude top-level terms
            arrange(p_value) %>%
            slice_head(n = 6) %>% # REDUCED: Take top 6 per tissue to reduce clutter (30 max nodes)
            select(Tissue, source, term_id, term_name, p_value, intersection, intersection_size)

          all_enrichment[[tissue]] <- enrich_df
          cat(sprintf("  %s: %d terms selected for network\n", tissue, nrow(enrich_df)))
        }
      },
      error = function(e) warning(sprintf("Enrichment failed for %s: %s", tissue, e$message))
    )
  }

  if (length(all_enrichment) == 0) {
    return(ggplot() +
      theme_void() +
      labs(title = "No Enrichment Found"))
  }

  # Combine and Identify Nodes
  network_nodes <- bind_rows(all_enrichment) %>%
    group_by(term_id) %>%
    # Assign term to the tissue with best p-value
    arrange(p_value) %>%
    summarise(
      term_name = first(term_name),
      Best_Tissue = first(Tissue),
      min_p = first(p_value),
      intersection = first(intersection),
      node_size = first(intersection_size),
      source = first(source),
      .groups = "drop"
    )

  cat(sprintf("  Building network with %d nodes...\n", nrow(network_nodes)))

  # 2. Build Edges (Shared Genes)
  get_genes <- function(x) unlist(strsplit(x, ","))
  term_genes <- lapply(network_nodes$intersection, get_genes)
  names(term_genes) <- network_nodes$term_id

  edges <- data.frame(from = character(), to = character(), weight = numeric())

  n_nodes <- nrow(network_nodes)
  if (n_nodes > 1) {
    edge_list <- list()
    k <- 1
    for (i in 1:(n_nodes - 1)) {
      genes_i <- term_genes[[i]]
      for (j in (i + 1):n_nodes) {
        genes_j <- term_genes[[j]]
        intersect_len <- length(intersect(genes_i, genes_j))
        union_len <- length(union(genes_i, genes_j))

        if (union_len > 0) {
          ji <- intersect_len / union_len
          # Link if significant overlap (Jaccard > 0.25) - slightly stricter
          if (ji > 0.25) {
            edge_list[[k]] <- data.frame(
              from = network_nodes$term_id[i],
              to = network_nodes$term_id[j],
              weight = ji
            )
            k <- k + 1
          }
        }
      }
    }
    if (length(edge_list) > 0) edges <- bind_rows(edge_list)
  }

  cat(sprintf("  Created %d edges.\n", nrow(edges)))

  # 3. Create Graph & Filter Singletons
  graph_full <- graph_from_data_frame(d = edges, vertices = network_nodes, directed = FALSE)

  # Filter to connected components only (Degree > 0)
  # visualize only the network, not the islands
  V(graph_full)$degree <- degree(graph_full)
  graph <- induced_subgraph(graph_full, V(graph_full)$degree > 0)

  if (gorder(graph) == 0) {
    # Fallback if everything is disconnected (unlikely but possible)
    graph <- graph_full
  }

  # Add node attributes
  # Recalculate degree for subgraph
  V(graph)$degree <- degree(graph)

  # Smarter Labeling:
  # 1. Identify connected components (clusters)
  cl <- components(graph)
  V(graph)$cluster <- cl$membership

  # 2. For each cluster, pick the node with the lowest p-value (Best Representative)
  #    PLUS any very high degree nodes (Hubs) that are different

  # We can't easily iterate igraph object in dplyr style, so create a label vector
  node_data <- data.frame(
    name = V(graph)$name,
    term_name = V(graph)$term_name,
    p_value = V(graph)$min_p,
    degree = V(graph)$degree,
    cluster = V(graph)$cluster,
    stringsAsFactors = FALSE
  )

  # Select labels
  labels_to_show <- node_data %>%
    group_by(cluster) %>%
    arrange(p_value) %>%
    slice(1) %>% # Always label the best term in the cluster
    pull(name)

  # Also add major hubs if not already included
  hubs <- node_data %>%
    filter(degree > 3) %>%
    pull(name)
  labels_to_show <- unique(c(labels_to_show, hubs))

  V(graph)$label <- ifelse(V(graph)$name %in% labels_to_show,
    str_trunc(V(graph)$term_name, 25), NA
  )

  # 4. Plot with ggraph
  # Factor Tissue for color consistency (need to update node data in graph or aesthetics)
  # ggraph uses the graph object. Attributes are in V(graph).

  p <- ggraph(graph, layout = "fr") + # Use Fruchterman-Reingold for connected implementation
    geom_edge_link(aes(alpha = weight), color = "grey60", width = 0.5, show.legend = FALSE) +
    geom_node_point(aes(fill = Best_Tissue, size = node_size), shape = 21, color = "white", stroke = 1.5) +
    geom_node_text(aes(label = label), repel = TRUE, size = 3, fontface = "bold", bg.color = "white", bg.r = 0.15) +
    scale_fill_brewer(palette = "Set1", name = "Tissue", breaks = TISSUES_WITH_MODELS) +
    scale_size_continuous(range = c(3, 10), name = "# Genes") +
    scale_edge_width(range = c(0.2, 1)) +
    labs(
      title = "Enrichment Network Map",
      subtitle = "Connected Functional Modules (Jaccard > 0.25)",
      x = NULL, y = NULL
    ) +
    theme_void() +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(size = 9, color = "grey40"),
      legend.position = "right",
      plot.margin = margin(10, 10, 10, 10)
    )

  attr(p, "network_info") <- list(
    n_nodes = nrow(network_nodes),
    n_edges = nrow(edges),
    categories = unique(network_nodes$source)
  )

  p
}


# ==============================================================================
# 6. FIGURE ASSEMBLY AND SAVE
# ==============================================================================
create_figure <- function(data) {
  cat("Assembling figure...\n")

  panel_a <- create_panel_a(data)
  panel_b <- create_panel_b(data$results)

  # Combine panels side by side
  final_figure <- panel_a | panel_b +
    plot_layout(widths = c(1, 1)) +
    plot_annotation(
      tag_levels = "A",
      theme = theme(
        plot.background = element_rect(fill = "white", color = NA)
      )
    )

  # Save figure
  output_path <- file.path(OUTPUT_DIR, "figure4_biological_validation")

  ggsave(paste0(output_path, ".pdf"), final_figure,
    width = 280, height = 150, units = "mm", dpi = 300
  )
  cat(sprintf("Saved: %s.pdf\n", output_path))

  ggsave(paste0(output_path, ".png"), final_figure,
    width = 280, height = 150, units = "mm", dpi = 300
  )
  cat(sprintf("Saved: %s.png\n", output_path))

  list(
    figure = final_figure,
    panel_a_info = attr(panel_a, "go_info"),
    panel_b_info = attr(panel_b, "network_info")
  )
}

# ==============================================================================
# 7. SUMMARY STATISTICS OUTPUT
# ==============================================================================
write_summary <- function(data, figure_info) {
  cat("Writing summary statistics...\n")

  go_data <- data$go_data

  # GO term statistics
  go_stats <- ""
  if (!is.null(go_data)) {
    go_summary <- go_data %>%
      mutate(q_value = ifelse(is.na(q_value), p_value, q_value)) %>%
      filter(!is.na(q_value), q_value < 0.05) %>%
      group_by(Tissue, source) %>%
      summarise(n_terms = n(), .groups = "drop")

    go_stats <- paste(
      sprintf(
        "  %s (%s): %d top terms",
        go_summary$Tissue, go_summary$source, go_summary$n_terms
      ),
      collapse = "\n"
    )
  }

  # Panel B info (Pathways)
  pathway_stats <- ""
  if (!is.null(figure_info$panel_b_info)) {
    pathway_stats <- sprintf(
      "Network Nodes: %d\nNetwork Edges: %d\nCategories: %s",
      figure_info$panel_b_info$n_nodes,
      figure_info$panel_b_info$n_edges,
      paste(figure_info$panel_b_info$categories, collapse = ", ")
    )
  }

  summary_text <- sprintf(
    "FIGURE 4: BIOLOGICAL VALIDATION
================================
Generated: %s
Script: figure4_biological_validation.R

================================================================================
INPUT DATA SOURCES
================================================================================
- Metadata: data/PigGTEx_v0.MetaTable.xlsx (%d samples)
- GO enrichment: results/enrichment_analysis/all_tissues_go_enrichment.csv
- ML model results: machine_learning/model_outputs/*_results.json

================================================================================
KEY STATISTICS
================================================================================

PANEL A: GO ENRICHMENT
----------------------
%s

PANEL B: ENRICHMENT MAP NETWORK
-----------------------------------
%s

================================================================================
FIGURE SPECIFICATIONS
================================================================================
- Output file: figure4_biological_validation.pdf
- Dimensions: 280mm x 150mm
- DPI: 300
- Panels: A (GO enrichment), B (Enrichment Map Network)
",
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    nrow(data$metadata),
    if (go_stats != "") go_stats else "GO data not available",
    if (pathway_stats != "") pathway_stats else "Pathway data not available"
  )

  output_path <- file.path(OUTPUT_DIR, "figure4_summary.txt")
  writeLines(summary_text, output_path)
  cat(sprintf("Saved: %s\n", output_path))
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
main <- function() {
  cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
  cat("Figure 4: Biological Validation\n")
  cat("=" %>% rep(70) %>% paste(collapse = ""), "\n\n")

  # Load data
  data <- load_data()

  # Create figure
  figure_info <- create_figure(data)

  # Write summary
  write_summary(data, figure_info)

  cat("\nFigure 4 completed successfully!\n")
}

# Run
main()
