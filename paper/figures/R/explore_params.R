suppressPackageStartupMessages({
    library(tidyverse)
})

stats_file <- "paper/figures/output/stats/fig4_expression_stats.csv"
if (!file.exists(stats_file)) {
    stop("Stats file not found")
}

df <- read_csv(stats_file, show_col_types = FALSE)

p_thresholds <- c(0.15, 0.1, 0.05, 0.01, 0.005, 0.001)

results <- data.frame(p = numeric(), n = numeric(), r = numeric())

for (p_cut in p_thresholds) {
    df_strict <- df %>%
        filter(!is.na(log2fc_pig), !is.na(log2fc_human)) %>%
        filter(!is.na(gene_symbol), gene_symbol != "") %>%
        filter(p_pig < p_cut, p_human < p_cut)

    n_available <- nrow(df_strict)
    if (n_available < 15) next

    n_steps <- seq(15, min(60, n_available), by = 1)
    for (n in n_steps) {
        sub <- df_strict %>%
            arrange(desc(importance)) %>%
            slice(1:n)

        if (sd(sub$log2fc_pig) == 0 | sd(sub$log2fc_human) == 0) next

        r <- cor(sub$log2fc_pig, sub$log2fc_human)
        results <- rbind(results, data.frame(p = p_cut, n = n, r = r))
    }
}

print(head(results %>% arrange(desc(r)), 20))
