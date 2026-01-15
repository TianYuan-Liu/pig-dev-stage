# Figure Panel Documentation

This document provides detailed descriptions of each panel in the manuscript figures. This documentation is designed to help AI agents and other automated systems understand the content and structure of each visualization.

## Figure 1: Study Design and Methodology

**Overall Purpose**: Overview of the experimental design, sample distribution, and classification schemes used in the developmental stage prediction models.

### Panel 1a: Developmental Stage Timeline
- **Type**: Timeline visualization (horizontal bar chart)
- **Content**: Displays the 5 developmental stages used in the study:
  - Infant: 0-20 days
  - Early childhood: 21-59 days
  - Pre-pubertal: 60-149 days
  - Post-pubertal: 150-365 days
  - Adult: >365 days
- **Visual Elements**: Colored rectangles for each stage, stage labels above, age ranges below
- **Data Source**: Defined in R script (`STAGE_ORDER`, `STAGE_COLORS`)
- **Purpose**: Provides temporal context for the developmental stages

### Panel 1b: Machine Learning Workflow Diagram
- **Type**: SVG diagram (placeholder in R script)
- **Content**: Visual representation of the machine learning pipeline/workflow
- **Visual Elements**: Empty placeholder in R code; actual diagram loaded as SVG
- **Data Source**: External SVG file (`figure1b.svg`)
- **Purpose**: Illustrates the methodology and computational workflow

### Panel 1c: Sample Distribution Heatmap
- **Type**: Heatmap (tissues × developmental stages)
- **Content**: Sample counts for each tissue-stage combination
- **Visual Elements**: 
  - Rows: Tissues (ordered by total sample count)
  - Columns: Developmental stages
  - Color intensity: Sample count (viridis color scale)
  - Text labels: Sample numbers in each cell
- **Data Source**: Metadata filtered by tissues with ML models
- **Purpose**: Shows data availability and distribution across tissues and stages

### Panel 1d: Classification Schemes Bar Chart
- **Type**: Stacked bar chart
- **Content**: Sample counts per tissue, colored by classification scheme (2-class, 3-class, or 4-class)
- **Visual Elements**:
  - X-axis: Tissues (ordered by sample count)
  - Y-axis: Sample count
  - Colors: Different classification schemes
  - Text labels: Total sample count above each bar
- **Data Source**: ML results list (scheme and n_samples per tissue)
- **Purpose**: Shows which classification scheme was used for each tissue based on data availability

---

## Figure 2: Model Performance

**Overall Purpose**: Comprehensive evaluation of machine learning model performance across tissues, including accuracy metrics, confusion patterns, and age-stage relationships.

### Panel 2a: Performance Metrics Heatmap
- **Type**: Heatmap with text annotations
- **Content**: Four performance metrics for each tissue:
  - Balanced Accuracy (with 95% confidence intervals)
  - F1-macro (with 95% confidence intervals)
  - Mean Absolute Error (MAE)
  - Spearman correlation coefficient (ρ)
- **Visual Elements**:
  - Rows: Tissues (ordered by Balanced Accuracy)
  - Columns: Performance metrics
  - Color intensity: Metric values (performance color scale)
  - Text labels: Metric values with confidence intervals in parentheses
- **Data Source**: Extracted from ML results (`extract_performance_metrics()`)
- **Purpose**: Provides quantitative comparison of model performance across tissues

### Panel 2b: Confusion Matrices Grid
- **Type**: Grid of confusion matrices (one per tissue)
- **Content**: Normalized confusion matrices showing prediction accuracy patterns
- **Visual Elements**:
  - Each subplot: One tissue
  - X-axis: Predicted stage
  - Y-axis: True stage
  - Color intensity: Percentage of predictions (white to green gradient, midpoint at 50%)
  - Text labels: Percentage values in each cell
  - Subtitle: Adjacent error rate (predictions off by one stage)
- **Data Source**: Confusion matrices from ML results
- **Purpose**: Reveals systematic prediction errors and stage confusion patterns

### Panel 2c: Age vs Stage Correlation Scatter Plots
- **Type**: Faceted scatter plots with LOESS smoothing
- **Content**: Relationship between chronological age (days) and developmental stage
- **Visual Elements**:
  - X-axis: Age in days (log10 scale)
  - Y-axis: Developmental stage (1-5, labeled as Infant, Early, Pre-pub, Post-pub, Adult)
  - Points: Individual samples (jittered, subsampled if >100)
  - Smooth line: LOESS fit with confidence band
  - Text annotation: Spearman correlation coefficient (ρ) per tissue
  - Facets: One panel per tissue
- **Data Source**: Metadata (Age_days, Stage) filtered by tissue
- **Purpose**: Validates that chronological age correlates with assigned developmental stage

---

## Figure 3: Functional Enrichment & Biological Validation

**Overall Purpose**: Biological interpretation of model features through gene ontology enrichment and functional module analysis.

### Panel 3a: GO Enrichment Dot Matrix
- **Type**: Dot plot matrix
- **Content**: Top 5 enriched GO terms per tissue (p < 0.05)
- **Visual Elements**:
  - X-axis: Tissues
  - Y-axis: GO term names (truncated to 40 characters)
  - Point size: -log10(p-value) (larger = more significant)
  - Point color: Tissue-specific colors
  - Only significant terms (p < 0.05) shown
- **Data Source**: GO enrichment data (`load_go_enrichment()`)
- **Purpose**: Identifies biological processes enriched in top predictive features per tissue

### Panel 3b: Enrichment Network Map
- **Type**: Network graph (force-directed layout)
- **Content**: Functional modules connected by shared genes
- **Visual Elements**:
  - Nodes: Enriched terms (GO:BP, KEGG, Reactome)
    - Size: Number of genes in intersection
    - Color: Tissue where term was most significant
    - Shape: Source database (GO BP = circle, KEGG = square, Reactome = triangle)
  - Edges: Connections between terms (Jaccard similarity > 0.25)
    - Edge width/alpha: Jaccard index strength
  - Labels: Selected term names (cluster representatives and hubs)
- **Data Source**: 
  - Top features from ML models (top 200 genes per tissue)
  - g:Profiler enrichment results
  - Network built from gene intersections
- **Purpose**: Reveals functional modules and cross-tissue biological themes

---

## Figure 4: Cross-Species Comparison (Pig vs Human)

**Overall Purpose**: Validation of developmental markers through cross-species comparison between pig and human muscle tissue.

### Panel 4a: Cross-Species Fold Change Correlation
- **Type**: Scatter plot with regression line
- **Content**: Correlation between pig and human log2 fold changes (Infant vs Adult)
- **Visual Elements**:
  - X-axis: Pig log2FC (Infant vs Adult)
  - Y-axis: Human log2FC (Infant vs Adult)
  - Points: Individual genes (labeled with gene symbols)
  - Regression line: Linear fit with confidence band
  - Reference lines: Dashed lines at x=0 and y=0
  - Text annotation: Pearson correlation (R), p-value, and sample size (n)
- **Data Source**: 
  - `fig4_expression_stats.csv` (fold changes and p-values)
  - Genes selected by: p < threshold in both species, same direction, top by importance
- **Purpose**: Tests conservation of developmental expression patterns between species

### Panel 4b: Feature Importance Bar Chart
- **Type**: Horizontal bar chart
- **Content**: Feature importance scores for conserved markers
- **Visual Elements**:
  - Y-axis: Gene symbols (ordered by importance)
  - X-axis: Log10(Feature Importance + 1)
  - Bar color: Viridis magma scale (intensity = importance)
- **Data Source**: Same genes as Panel 4a, ordered by importance from ML model
- **Purpose**: Shows which conserved genes were most important for pig developmental stage prediction

### Panel 4c: Paired Fold Change Comparison
- **Type**: Lollipop/dot plot (paired comparison)
- **Content**: Side-by-side comparison of pig and human fold changes for each gene
- **Visual Elements**:
  - Y-axis: Gene symbols (ordered by importance, same as Panel 4b)
  - X-axis: Log2FC (Infant vs Adult)
  - Points: Pig (blue) and Human (red) values
  - Lines: Connect paired values for each gene
  - Reference line: Dashed vertical line at x=0
- **Data Source**: Same genes as Panels 4a and 4b
- **Purpose**: Visualizes direction and magnitude of fold changes in both species simultaneously

---

## Data Files in This Directory

- `fig1_summary.txt`: Summary statistics for Figure 1 (sample counts, classification schemes)
- `fig2_summary.txt`: Summary statistics for Figure 2 (performance metrics, error rates, correlations)
- `fig3_summary.txt`: Summary statistics for Figure 3 (GO enrichment counts)
- `fig4_summary.txt`: Summary statistics for Figure 4 (correlation analysis, conserved markers)
- `fig4_expression_stats.csv`: Expression statistics for cross-species comparison
- `fig4_orthology_mapping.csv`: Gene orthology mapping between pig and human

## Related Files

- **R Scripts**: `paper/figures/R/fig*.R` - Source code for generating each figure
- **Output Images**: `paper/figures/output/png/fig*.png` and `paper/figures/output/pdf/fig*.pdf`
- **SVG Workflow**: `paper/figures/output/svg/figure1b.svg` - Machine learning workflow diagram

## Notes for AI Agents

1. **Panel Labels**: All panels are labeled with lowercase letters (a, b, c, d) in the top-left corner
2. **Figure Dimensions**: All figures are 183mm wide (standard journal width)
3. **Color Schemes**: Consistent color palettes are defined in `_theme.R` (tissue colors, stage colors, scheme colors)
4. **Data Filtering**: Each figure applies tissue-specific filtering based on which tissues have ML models
5. **Statistical Tests**: Confidence intervals are 95% where applicable; p-values are FDR-corrected for enrichment analyses
