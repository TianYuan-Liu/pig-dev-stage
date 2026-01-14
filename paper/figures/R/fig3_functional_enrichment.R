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
create_panel_a <- function(go_data) {
  cat("Creating Panel A: GO enrichment matrix...\n")
  
  if (is.null(go_data)) {
    return(ggplot() + theme_void() + 
             labs(title = "GO Enrichment Data Not Available"))
  }
  
  # Get top terms per tissue
  top_terms <- go_data %>%
    filter(!is.na(p_value), p_value < 0.05) %>%
    group_by(Tissue) %>%
    arrange(p_value) %>%
    slice_head(n = 5) %>%
    ungroup() %>%
    mutate(
      term_name = str_trunc(term_name, 40),
      term_name = factor(term_name, levels = unique(rev(term_name)))
    )
  
  if (nrow(top_terms) == 0) {
    return(ggplot() + theme_void() + 
             labs(title = "No significant GO terms found"))
  }
  
  ggplot(top_terms, aes(x = Tissue, y = term_name)) +
    geom_point(
      aes(size = -log10(p_value), fill = Tissue),
      shape = 21, color = "black", stroke = 0.3
    ) +
    scale_size_continuous(range = c(2, 6), name = "-log10(p)") +
    scale_fill_tissue(guide = "none") +
    labs(
      title = "Top Enriched GO Terms",
      subtitle = "Top 5 per tissue (p < 0.05)",
      x = NULL, y = NULL
    ) +
    nature_theme() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
      axis.text.y = element_text(size = 5),
      panel.grid.major = element_line(color = "gray95", linewidth = 0.2),
      legend.position = "right"
    )
}

# ==============================================================================
# 4. PANEL B: ENRICHMENT NETWORK MAP
# ==============================================================================
create_panel_b <- function(results_list) {
  cat("Creating Panel B: Enrichment network...\n")
  
  if (length(results_list) == 0) {
    return(ggplot() + theme_void() + labs(title = "No ML results"))
  }
  
  # Run enrichment for all tissues
  all_enrichment <- list()
  
  for (tissue in names(results_list)) {
    res <- results_list[[tissue]]
    gene_ids <- if (!is.null(res$top_genes)) res$top_genes else res$top_features
    
    if (is.null(gene_ids) || length(gene_ids) == 0) next
    
    query_genes <- gene_ids[1:min(200, length(gene_ids))]
    
    tryCatch({
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
    }, error = function(e) {
      warning(sprintf("Enrichment failed for %s: %s", tissue, e$message))
    })
  }
  
  if (length(all_enrichment) == 0) {
    return(ggplot() + theme_void() + labs(title = "No enrichment found"))
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
  
  hubs <- node_data %>% filter(degree > 3) %>% pull(name)
  labels_to_show <- unique(c(labels_to_show, hubs))
  
  V(graph)$label <- ifelse(
    V(graph)$name %in% labels_to_show,
    str_trunc(V(graph)$term_name, 25),
    NA
  )
  
  # Plot
  ggraph(graph, layout = "fr") +
    geom_edge_link(aes(alpha = weight), color = "gray60", width = 0.4, show.legend = FALSE) +
    geom_node_point(aes(fill = Best_Tissue, size = node_size), shape = 21, color = "white", stroke = 1) +
    geom_node_text(aes(label = label), repel = TRUE, size = 2, fontface = "bold", bg.color = "white", bg.r = 0.1) +
    scale_fill_tissue(breaks = names(results_list)) +
    scale_size_continuous(range = c(2, 8), name = "# Genes") +
    labs(
      title = "Functional Module Network",
      subtitle = "Connected terms (Jaccard > 0.25)",
      x = NULL, y = NULL
    ) +
    theme_void() +
    theme(
      plot.title = element_text(face = "bold", size = 8),
      plot.subtitle = element_text(size = 6, color = "gray40"),
      legend.position = "right",
      legend.text = element_text(size = 5),
      legend.title = element_text(size = 6, face = "bold"),
      plot.margin = margin(4, 4, 4, 4, "mm")
    )
}

# ==============================================================================
# 5. FIGURE ASSEMBLY
# ==============================================================================
cat("\nAssembling figure...\n")

panel_a <- create_panel_a(go_data)
panel_b <- create_panel_b(results_list)

fig3 <- panel_a | panel_b +
  plot_layout(widths = c(1, 1.2)) +
  plot_annotation(
    tag_levels = "A",
    theme = theme(
      plot.background = element_rect(fill = "white", color = NA),
      plot.tag = element_text(size = 8, face = "bold")
    )
  )

# ==============================================================================
# 6. SAVE FIGURE
# ==============================================================================
cat("\nSaving figure...\n")
save_figure(fig3, "fig3_functional_enrichment", width = 183, height = 140)

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

summary_text <- sprintf("
FIGURE 3: FUNCTIONAL ENRICHMENT
===============================
Generated: %s

PANEL A: GO ENRICHMENT
----------------------
Terms per tissue (p < 0.05):
%s

PANEL B: ENRICHMENT NETWORK
---------------------------
Tissues analyzed: %s
Network construction: Jaccard similarity > 0.25

Figure dimensions: 183mm × 140mm
",
  format(Sys.time(), "%%Y-%%m-%%d %%H:%%M"),
  paste(capture.output(print(go_stats, n = Inf)), collapse = "\n"),
  paste(names(results_list), collapse = ", ")
)

write_summary(summary_text, "fig3_summary.txt")

cat("\n✓ Figure 3 completed successfully!\n")
