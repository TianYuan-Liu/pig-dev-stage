#!/usr/bin/env Rscript
# ==============================================================================
# Figure 3: Functional Enrichment & Biological Validation
# ==============================================================================
# Panels:
#   A - GO enrichment dot matrix (top terms per tissue)
#   B - Enrichment network map (connected functional modules)
# ==============================================================================

# ==============================================================================
# 1. SETUP
# ==============================================================================
suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(ggraph)
  library(igraph)
  library(gprofiler2)
})

# Load shared modules
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("--file=", "", file_arg))))
  }
  "/Users/tianyuan/Desktop/github_dev/pig-dev-stage/paper/figures/R"
}
script_dir <- get_script_dir()
source(file.path(script_dir, "_theme.R"))
source(file.path(script_dir, "_utils.R"))

cat("==============================================================================\n")
cat("Figure 3: Functional Enrichment & Biological Validation\n")
cat("==============================================================================\n\n")

set.seed(42)

# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
cat("Loading data...\n")

metadata <- load_metadata()
results_list <- load_all_ml_results()
go_data <- load_go_enrichment()

cat(sprintf("  Metadata: %d samples\n", nrow(metadata)))
cat(sprintf("  ML results: %d tissues\n", length(results_list)))
if (!is.null(go_data)) {
  cat(sprintf("  GO enrichment: %d terms\n", nrow(go_data)))
}

# ==============================================================================
# 3. PANEL A: GO ENRICHMENT DOT MATRIX
# ==============================================================================
# ==============================================================================
# 3. PANEL A: GO ENRICHMENT DOT MATRIX (REMOVED)
# ==============================================================================
# Panel A has been removed as per user request.


# ==============================================================================
# 4. PANEL B: ENRICHMENT NETWORK MAP
# ==============================================================================
create_panel_b <- function(results_list) {
  cat("Creating Panel B: Enrichment network...\n")

  if (length(results_list) == 0) {
    return(ggplot() +
      theme_void() +
      labs(title = "No ML results"))
  }

  # Run enrichment for all tissues
  all_enrichment <- list()

  for (tissue in names(results_list)) {
    res <- results_list[[tissue]]
    gene_ids <- if (!is.null(res$top_genes)) res$top_genes else res$top_features

    if (is.null(gene_ids) || length(gene_ids) == 0) next

    query_genes <- gene_ids[1:min(200, length(gene_ids))]

    tryCatch(
      {
        gost_result <- gost(
          query = query_genes,
          organism = "sscrofa",
          sources = c("GO:BP", "KEGG", "REAC"),
          significant = FALSE,
          user_threshold = 1.0,
          correction_method = "fdr",
          evcodes = TRUE
        )

        if (!is.null(gost_result) && !is.null(gost_result$result) && nrow(gost_result$result) > 0) {
          enrich_df <- gost_result$result %>%
            mutate(Tissue = tissue) %>%
            filter(term_size < 3000) %>%
            arrange(p_value) %>%
            slice_head(n = 6) %>%
            select(Tissue, source, term_id, term_name, p_value, intersection, intersection_size)

          all_enrichment[[tissue]] <- enrich_df
          cat(sprintf("  %s: %d terms\n", tissue, nrow(enrich_df)))
        }
      },
      error = function(e) {
        warning(sprintf("Enrichment failed for %s: %s", tissue, e$message))
      }
    )
  }

  if (length(all_enrichment) == 0) {
    return(ggplot() +
      theme_void() +
      labs(title = "No enrichment found"))
  }

  # Build network nodes
  network_nodes <- bind_rows(all_enrichment) %>%
    group_by(term_id) %>%
    arrange(p_value) %>%
    summarise(
      term_name = first(term_name),
      Best_Tissue = first(Tissue),
      min_p = first(p_value),
      intersection = first(intersection),
      node_size = first(intersection_size),
      source = first(source),
      .groups = "drop"
    ) %>%
    mutate(
      source_label = case_when(
        source == "GO:BP" ~ "GO BP",
        source == "REAC" ~ "Reactome",
        TRUE ~ source
      ),
      source_label = factor(source_label, levels = c("GO BP", "KEGG", "Reactome"))
    )

  cat(sprintf("  Building network with %d nodes...\n", nrow(network_nodes)))

  # Build edges (shared genes)
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

  cat(sprintf("  Created %d edges\n", nrow(edges)))

  # Create graph
  graph_full <- graph_from_data_frame(d = edges, vertices = network_nodes, directed = FALSE)
  V(graph_full)$degree <- degree(graph_full)

  # Filter to connected nodes
  graph <- induced_subgraph(graph_full, V(graph_full)$degree > 0)

  if (gorder(graph) == 0) {
    graph <- graph_full
  }

  V(graph)$degree <- degree(graph)

  # Smart labeling: label cluster representatives
  cl <- components(graph)
  V(graph)$cluster <- cl$membership

  node_data <- data.frame(
    name = V(graph)$name,
    term_name = V(graph)$term_name,
    p_value = V(graph)$min_p,
    degree = V(graph)$degree,
    cluster = V(graph)$cluster,
    stringsAsFactors = FALSE
  )

  labels_to_show <- node_data %>%
    group_by(cluster) %>%
    arrange(p_value) %>%
    slice(1) %>%
    pull(name)

  hubs <- node_data %>%
    filter(degree > 3) %>%
    pull(name)
  labels_to_show <- unique(c(labels_to_show, hubs))

  V(graph)$label <- ifelse(
    V(graph)$name %in% labels_to_show,
    str_wrap(V(graph)$term_name, width = 28),
    ""
  )

  # Plot
  legend_source_data <- tibble(
    source_label = factor(
      c("GO BP", "KEGG", "Reactome"),
      levels = c("GO BP", "KEGG", "Reactome")
    )
  )

  ggraph(graph, layout = "fr") +
    geom_edge_link(aes(alpha = weight), color = "gray70", width = 0.3, show.legend = FALSE) +
    geom_node_point(
      aes(fill = Best_Tissue, size = node_size, shape = source_label),
      color = "white", stroke = 0.6,
      show.legend = c(fill = TRUE, shape = TRUE, size = TRUE)
    ) +
    geom_point(
      data = legend_source_data,
      aes(x = 0, y = 0, shape = source_label),
      inherit.aes = FALSE,
      size = 3,
      alpha = 0
    ) +
    geom_node_text(
      aes(label = label),
      repel = TRUE,
      size = 2.4,
      fontface = "bold",
      bg.color = "white",
      bg.r = 0.1,
      point.padding = unit(0.3, "lines"),
      box.padding = unit(0.5, "lines"),
      force = 10,
      force_pull = 0.5,
      max.overlaps = 100
    ) +
    scale_fill_tissue(breaks = names(results_list)) +
    scale_shape_manual(
      values = c("GO BP" = 21, "KEGG" = 22, "Reactome" = 24),
      name = "Source",
      drop = FALSE
    ) +
    scale_size_continuous(
      range = c(2, 8),
      name = "Genes",
      breaks = scales::pretty_breaks(n = 3),
      guide = "legend"
    ) +
    guides(
      fill = guide_legend(
        order = 1,
        override.aes = list(shape = 21, size = 3.5),
        ncol = 1
      ),
      shape = guide_legend(
        title = "Source",
        order = 2,
        override.aes = list(size = 3, color = "black", fill = "gray80"),
        ncol = 1
      ),
      size = guide_legend(
        order = 3,
        override.aes = list(shape = 21, fill = "gray70", color = "white", stroke = 0.4),
        ncol = 1
      )
    ) +
    labs(
      title = "Functional Module Network",
      subtitle = sprintf(
        "Connected terms (Jaccard > 0.25); %d tissues, %d terms",
        length(results_list),
        nrow(network_nodes)
      ),
      x = NULL, y = NULL
    ) +
    nature_theme() +
    theme(
      axis.text = element_blank(),
      axis.title = element_blank(),
      axis.ticks = element_blank(),
      axis.line = element_blank(),
      legend.position = "right",
      legend.box = "vertical",
      legend.box.just = "left",
      legend.spacing.y = unit(1.2, "mm"),
      legend.key.height = unit(3, "mm"),
      legend.key.width = unit(3, "mm"),
      legend.title = element_text(size = 6, face = "bold"),
      legend.text = element_text(size = 5.5),
      plot.margin = margin(4, 4, 4, 4, "mm")
    )
}

# ==============================================================================
# 5. FIGURE ASSEMBLY
# ==============================================================================
cat("\nAssembling figure...\n")

# panel_a <- create_panel_a(go_data) # Removed
panel_b <- create_panel_b(results_list)

# Layout: Only Panel B
fig3 <- panel_b +
  plot_annotation(
    # tag_levels = list(c("a", "b")), # No tags needed for single panel
    theme = theme(
      plot.background = element_rect(fill = "white", color = NA),
      plot.tag = element_text(size = 8, face = "bold")
    )
  )

# ==============================================================================
# 6. SAVE FIGURE
# ==============================================================================
cat("\nSaving figure...\n")
# Increased height to accommodate both panels comfortably
# Adjusted height for single panel
save_figure(fig3, "fig3_functional_enrichment", width = 183, height = 120)

# ==============================================================================
# 7. SUMMARY STATISTICS
# ==============================================================================
cat("\nWriting summary...\n")

go_stats <- if (!is.null(go_data)) {
  go_data %>%
    filter(!is.na(p_value), p_value < 0.05) %>%
    count(Tissue, source, name = "n_terms") %>%
    arrange(Tissue, source)
} else {
  tibble(Tissue = "N/A", source = "N/A", n_terms = 0)
}

summary_text <- sprintf(
  "
FIGURE 3: FUNCTIONAL ENRICHMENT
===============================
Generated: %s

PANEL (Only): ENRICHMENT NETWORK
---------------------------
Tissues analyzed: %s
Network construction: Jaccard similarity > 0.25

Figure dimensions: 183mm × 120mm
",
  format(Sys.time(), "%%Y-%%m-%%d %%H:%%M"),
  # paste(capture.output(print(go_stats, n = Inf)), collapse = "\n"), # Removed stats for Panel A
  paste(names(results_list), collapse = ", ")
)

write_summary(summary_text, "fig3_summary.txt")

cat("\n✓ Figure 3 completed successfully!\n")
