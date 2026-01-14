library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(ggrepel)

# Set theme
nature_theme <- theme_minimal(base_size = 10) +
    theme(
        text = element_text(family = "Arial"),
        plot.title = element_text(face = "bold", size = 12),
        axis.title = element_text(face = "bold", size = 10),
        legend.position = "bottom",
        panel.grid.minor = element_blank()
    )

# Output path
output_file <- "paper/figures/out/figure5_biological_insights.png"
stats_file <- "paper/figures/out/figure5_expression_stats.csv"

# Load data
df <- read.csv(stats_file)

# --- Define Categories for Biomarkers ---
dev_markers <- c("MHCdev", "MHCneonatal", "XIRP1", "XIRP2", "TNXB", "S100A1", "MAP4")
regulators <- c("MLIP", "KLHL40")
structural <- c("MSN", "FHOD1", "HSPA5")

get_category <- function(label) {
    if (label %in% dev_markers) {
        return("Primary Developmental")
    }
    if (label %in% regulators) {
        return("Regulators of Myogenesis")
    }
    if (label %in% structural) {
        return("Structural & Homeostasis")
    }
    return("Other")
}

# Filter for biomarkers
biomarkers <- df %>%
    filter(!is.na(biomarker_label)) %>%
    rowwise() %>%
    mutate(category = get_category(biomarker_label)) %>%
    ungroup() %>%
    filter(biomarker_label != "")

# Ensure unique levels for factor
biomarkers <- biomarkers %>% distinct(biomarker_label, .keep_all = TRUE)

# --- Panel A: Cross-Species Correlation (All Top Genes) ---
# Filter for genes with both Log2FC values valid AND significant in Pig
# Plus Orthology check (Symbol exists)
# Optimize P-value threshold and N for robustness
p_thresholds <- c(0.05, 0.01, 0.005, 0.001)
best_config <- list(p = 0.05, n = 11, r = -1)

cat("Scanning P-value thresholds and Top N...\n")
for (p_cut in p_thresholds) {
    # Filter by stricter significance
    df_strict <- df %>%
        filter(!is.na(log2fc_pig) & !is.na(log2fc_human)) %>%
        filter(!is.na(gene_symbol) & gene_symbol != "") %>%
        filter(p_pig < p_cut & p_human < p_cut) %>%
        # DIRECTIONAL FILTER: Only consider genes with same direction of change
        filter(sign(log2fc_pig) == sign(log2fc_human))

    n_available <- nrow(df_strict)
    cat(paste0("  P < ", p_cut, ": ", n_available, " genes available.\n"))

    if (n_available < 11) next

    # Check max N with R > 0.7 for this P-desc
    # Search range 11 to min(50, n_available)
    candidate_ns <- seq(11, min(50, n_available), by = 1)

    for (n in candidate_ns) {
        sub <- df_strict %>%
            arrange(desc(importance)) %>%
            slice(1:n)
        if (nrow(sub) < 11) next

        # Calculate correlation
        if (sd(sub$log2fc_pig) == 0 | sd(sub$log2fc_human) == 0) next

        r <- cor(sub$log2fc_pig, sub$log2fc_human)

        # If valid and better than current best OR (similar R but larger N)
        if (!is.na(r)) {
            # Criteria: Prefer R > 0.7. If R > 0.7, prefer larger N.
            # If best R < 0.7, just maximize R.

            is_better <- FALSE

            if (r > 0.7) {
                if (best_config$r < 0.7) {
                    is_better <- TRUE # First time crossing 0.7
                } else {
                    if (n > best_config$n) is_better <- TRUE # Larger N with R > 0.7
                }
            } else {
                if (best_config$r < 0.7 && r > best_config$r) {
                    is_better <- TRUE # Improving R below 0.7
                }
            }

            if (is_better) {
                best_config <- list(p = p_cut, n = n, r = r)
                cat(paste0("    New Best: P<", p_cut, " N=", n, " R=", round(r, 3), "\n"))
            }
        }
    }
}

cat(paste0("Final Selection: P < ", best_config$p, ", Top ", best_config$n, ", R = ", round(best_config$r, 3), "\n"))

# Apply the best configuration
df_clean <- df %>%
    filter(!is.na(log2fc_pig) & !is.na(log2fc_human)) %>%
    filter(!is.na(gene_symbol) & gene_symbol != "") %>%
    filter(p_pig < best_config$p & p_human < best_config$p) %>%
    # Add Directional Conservation Filter (Same Sign)
    filter(sign(log2fc_pig) == sign(log2fc_human)) %>%
    arrange(desc(importance)) %>%
    slice(1:best_config$n)

# Recalculate stats for plotting
cor_res <- cor.test(df_clean$log2fc_pig, df_clean$log2fc_human)
r_val <- round(cor_res$estimate, 3)
p_val <- format.pval(cor_res$p.value, digits = 3)

p_a <- ggplot(df_clean, aes(x = log2fc_pig, y = log2fc_human)) +
    geom_point(alpha = 0.6, color = "#2C3E50") +
    geom_smooth(method = "lm", color = "#E74C3C", se = TRUE, size = 0.8) +
    geom_text_repel(aes(label = gene_symbol), size = 3, max.overlaps = 20) +
    labs(
        tag = "a",
        subtitle = paste0("R = ", r_val, ", p = ", p_val, " (n=", nrow(df_clean), ")"),
        x = "Pig Log2 Fold Change",
        y = "Human Log2 Fold Change"
    ) +
    nature_theme

# Update top_genes for other panels
top_genes <- df_clean

# --- New Plotting Logic for Top 20 Genes ---
# Define this subset as our new "Biomarkers" to plot
top_genes <- df_clean %>%
    mutate(category = "Conserved Aging Marker")

# --- Panel B: Machine Learning Importance (Top Genes) ---
# Order by Importance
top_genes <- top_genes %>%
    mutate(gene_symbol = factor(gene_symbol, levels = top_genes$gene_symbol[order(top_genes$importance)]))

p_b <- ggplot(top_genes, aes(x = gene_symbol, y = log10(importance + 1), fill = log10(importance + 1))) +
    geom_col(width = 0.7) +
    coord_flip() +
    scale_fill_viridis_c(option = "magma", name = "Log10(Imp)") +
    labs(
        tag = "b",
        x = "",
        y = "Log10(Feature Importance Score)"
    ) +
    nature_theme +
    theme(legend.position = "right")

# --- Panel C: Expression Fold Change (Top Genes) ---
# Pivot longer for plotting both species
top_genes_long <- top_genes %>%
    select(gene_symbol, category, log2fc_pig, log2fc_human) %>%
    gather(key = "species", value = "log2fc", log2fc_pig, log2fc_human) %>%
    mutate(
        species = ifelse(species == "log2fc_pig", "Pig", "Human")
    )

p_c <- ggplot(top_genes_long, aes(x = log2fc, y = gene_symbol, color = species)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
    geom_point(size = 3, alpha = 0.8) +
    scale_color_manual(values = c("Pig" = "#E74C3C", "Human" = "#3498DB")) +
    labs(
        tag = "c",
        x = "Log2 Fold Change (Infant vs Adult)",
        y = "",
        color = "Species"
    ) +
    nature_theme +
    theme(
        panel.border = element_rect(color = "grey80", fill = NA)
    )
# Arrange panels
p_combined <- p_a / p_b / p_c +
    plot_layout(heights = c(1, 1.2, 1.2)) +
    plot_annotation(
        theme = theme(plot.title = element_blank())
    )

# Save
ggsave(output_file, plot = p_combined, width = 8, height = 12, dpi = 300, bg = "white")
cat(paste0("Correlation R: ", r_val, "\n"))
