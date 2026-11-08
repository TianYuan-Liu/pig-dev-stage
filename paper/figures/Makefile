# ==============================================================================
# Makefile for Paper Figures
# Pig Developmental Stage Classification
# ==============================================================================
# Usage:
#   make all       - Generate all figures (main + supplementary)
#   make main      - Generate main figures only
#   make supp      - Generate supplementary figures only
#   make supp-pdf  - Compile supplementary PDF
#   make fig1      - Generate Figure 1 only
#   make fig4-data - Prepare Figure 4 data (Python)
#   make clean     - Remove all outputs
#   make help      - Show this help
# ==============================================================================

# Configuration
R := Rscript
PYTHON := python3
R_DIR := R
PY_DIR := python
PROV_DIR := provenance
OUT_DIR := output
PAPER_DIR := ..

# Mermaid
MERMAID_SRC := mermaid/fig1b_pipeline.mmd
MERMAID_CFG := mermaid/config.json
MERMAID_PNG := $(OUT_DIR)/png/figure1b.png

# Figure scripts
FIG1_SCRIPT := $(R_DIR)/fig1_study_design.R
FIG2_SCRIPT := $(R_DIR)/fig2_honest_eval.R
FIG3_SCRIPT := $(R_DIR)/fig3_functional_enrichment.R
FIG4_DATA_SCRIPT := $(PY_DIR)/fig4_data_prep.py
FIG4_SCRIPT := $(R_DIR)/fig4_cross_species.R
FIG5_SCRIPT := $(R_DIR)/fig5_nonlinearity_sweep.R   # linear-conservation sweep

# Supplementary figure scripts (all R unless noted; rendered as Supplementary
# Fig. S1-S4 by document order in supplementary.tex).
# Consolidation (2026-05): the former stability figure, per-class-recall figure,
# and VIP-feature heatmap were merged out of the supplement -- per-class recall +
# stability + stage-correlation now live in the single composite
# figS1_model_evaluation; the VIP-feature trajectory heatmap moved into main
# Fig. 3.
SUPP_MODELEVAL_SCRIPT := $(R_DIR)/figS1_model_evaluation.R  # composite: per-class recall + stability
SUPP_BATCH_SCRIPT := $(R_DIR)/figS4_batch_effects.R         # batch effects (UMAP)
SUPP_MARKER_SCRIPT := ../../review/analyses/marker_gene_validation_v3.py  # marker validation (Python, 6-stage)
SUPP_HET_SCRIPT   := $(R_DIR)/figS8_heterogeneity.R         # subgroup heterogeneity
SUPP_TXSCATTER_SCRIPT := $(R_DIR)/figS_transfer_scatter.R   # per-cell transfer correlation scatter
SUPP_TXSCATTER_DUMP   := provenance/dump_transfer_scatter.py  # per-sample (score,level) dump

# Output files
FIG1_PDF := $(OUT_DIR)/pdf/fig1_study_design.pdf
FIG2_PDF := $(OUT_DIR)/pdf/fig2_honest_eval.pdf
FIG3_PDF := $(OUT_DIR)/pdf/fig3_functional_enrichment.pdf
FIG4_PDF := $(OUT_DIR)/pdf/fig4_cross_species.pdf
FIG5_PDF := $(OUT_DIR)/pdf/fig5_nonlinearity_sweep.pdf
FIG5_DATA_SRC := data/transfer_6stage/expB_sweep.csv
FIG4_DATA := $(OUT_DIR)/stats/fig4_expression_stats.csv

# Supplementary outputs (filenames are internal; rendered S-numbers come from order)
FIGS_MODELEVAL_PDF := $(OUT_DIR)/pdf/figS1_model_evaluation.pdf
FIGS_BATCH_PDF  := $(OUT_DIR)/pdf/figS4_batch_effects.pdf
FIGS_MARKER_PDF := $(OUT_DIR)/pdf/figS6_marker_genes_v3.pdf
FIGS_HET_PDF    := $(OUT_DIR)/pdf/figS8_heterogeneity.pdf
FIGS_TXSCATTER_PDF := $(OUT_DIR)/pdf/figS_transfer_scatter.pdf
TXSCATTER_DATA  := data/transfer_scatter_6stage.csv
SUPP_DOC := $(PAPER_DIR)/supplementary.pdf

# Shared dependencies
SHARED_DEPS := $(R_DIR)/_theme.R $(R_DIR)/_utils.R config/nature_style.yaml

# ==============================================================================
# TARGETS
# ==============================================================================

.PHONY: all main supp supp-pdf fig1 fig2 fig3 fig4 fig5 fig4-data mermaid clean help

# Default target - generates everything
all: main supp
	@echo ""
	@echo "✓ All figures generated successfully!"
	@echo "  Outputs in: $(OUT_DIR)/pdf/ and $(OUT_DIR)/png/"

# Main figures only
main: fig1 fig2 fig3 fig4 fig5
	@echo ""
	@echo "✓ Main figures complete"

# Supplementary figures (S1-S4 by document order)
supp: $(FIGS_MODELEVAL_PDF) $(FIGS_BATCH_PDF) $(FIGS_MARKER_PDF) $(FIGS_HET_PDF) $(FIGS_TXSCATTER_PDF)
	@echo ""
	@echo "✓ Supplementary figures complete"

# Compile supplementary PDF (Nature-compliant)
supp-pdf: supp
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Compiling Supplementary PDF (Nature format)"
	@echo "═══════════════════════════════════════════════════════════════"
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode supplementary.tex
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode supplementary.tex
	@echo ""
	@echo "✓ Supplementary PDF compiled: $(SUPP_DOC)"
	@ls -lh $(SUPP_DOC)

# Individual figures
fig1: $(FIG1_PDF)
	@echo "✓ Figure 1 complete"

fig2: $(FIG2_PDF)
	@echo "✓ Figure 2 complete"

fig3: $(FIG3_PDF)
	@echo "✓ Figure 3 complete"

fig4: $(FIG4_PDF)
	@echo "✓ Figure 4 complete"

fig5: $(FIG5_PDF)
	@echo "✓ Figure 5 complete"

fig4-data: $(FIG4_DATA)
	@echo "✓ Figure 4 data prepared"

mermaid: $(MERMAID_PNG)
	@echo "✓ Mermaid diagram rendered"

# ==============================================================================
# RULES
# ==============================================================================

# Mermaid diagram for Panel B (PNG via Chromium for foreignObject support)
$(MERMAID_PNG): $(MERMAID_SRC) $(MERMAID_CFG)
	@echo "Rendering mermaid diagram..."
	mmdc -i $(MERMAID_SRC) -o $(MERMAID_PNG) -c $(MERMAID_CFG) -t neutral -w 1800 -H 680 -b white -s 4

$(FIG1_PDF): $(FIG1_SCRIPT) $(SHARED_DEPS) $(MERMAID_PNG)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Generating Figure 1: Study Design"
	@echo "═══════════════════════════════════════════════════════════════"
	$(R) $(FIG1_SCRIPT)

$(FIG2_PDF): $(FIG2_SCRIPT) $(SHARED_DEPS)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Generating Figure 2: Model Performance"
	@echo "═══════════════════════════════════════════════════════════════"
	$(R) $(FIG2_SCRIPT)

$(FIG3_PDF): $(FIG3_SCRIPT) $(SHARED_DEPS)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Generating Figure 3: Functional Enrichment"
	@echo "═══════════════════════════════════════════════════════════════"
	$(R) $(FIG3_SCRIPT)

$(FIG4_DATA): $(FIG4_DATA_SCRIPT)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Preparing Figure 4 Data (Python)"
	@echo "═══════════════════════════════════════════════════════════════"
	$(PYTHON) $(FIG4_DATA_SCRIPT)

# Figure 4 is built directly from the authoritative 6-stage transfer artifact.
FIG4_DATA_SRC := data/determinants_6stage.json

# Rebuild the determinants artifact reproducibly: transfer_6stage.py --exp matrix
# writes the per-cell matrix + tests (brain forebrain-aggregated by construction),
# then build_determinants_6stage.py assembles determinants_6stage.json. No circular
# step, no hardcoded brain constants. (Heavy: loads the 7-species Cardoso atlas.)
DET_DEPS := $(PROV_DIR)/transfer_6stage.py $(PROV_DIR)/build_determinants_6stage.py \
            $(PROV_DIR)/load_dgtex.py $(PROV_DIR)/stage_mapping_v2.py
$(FIG4_DATA_SRC): $(DET_DEPS)
	$(PYTHON) $(PROV_DIR)/transfer_6stage.py --exp matrix
	$(PYTHON) $(PROV_DIR)/build_determinants_6stage.py

$(FIG4_PDF): $(FIG4_SCRIPT) $(FIG4_DATA_SRC) $(SHARED_DEPS)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Generating Figure 4: Cross-Species Transfer"
	@echo "═══════════════════════════════════════════════════════════════"
	$(R) $(FIG4_SCRIPT)

$(FIG5_PDF): $(FIG5_SCRIPT) $(FIG5_DATA_SRC) $(SHARED_DEPS)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Generating Figure 5: Linear-conservation nonlinearity sweep"
	@echo "═══════════════════════════════════════════════════════════════"
	$(R) $(FIG5_SCRIPT)

# Supplementary figures - one R script per figure (house style)
$(FIGS_MODELEVAL_PDF): $(SUPP_MODELEVAL_SCRIPT) $(SHARED_DEPS)
	@echo "Generating Supplementary Fig S1: model evaluation (per-class recall + stability)"
	$(R) $(SUPP_MODELEVAL_SCRIPT)

$(FIGS_BATCH_PDF): $(SUPP_BATCH_SCRIPT) $(SHARED_DEPS)
	@echo "Generating Supplementary Fig: batch effects (UMAP)"
	$(R) $(SUPP_BATCH_SCRIPT)

$(FIGS_MARKER_PDF): $(SUPP_MARKER_SCRIPT)
	@echo "Generating Supplementary Fig S6: marker-gene validation (v3, Python, 6-stage)"
	$(PYTHON) $(SUPP_MARKER_SCRIPT)

$(FIGS_HET_PDF): $(SUPP_HET_SCRIPT) $(SHARED_DEPS)
	@echo "Generating Supplementary Fig: subgroup heterogeneity"
	$(R) $(SUPP_HET_SCRIPT)

# Per-cell cross-species transfer correlation scatter (Supplementary Fig. S5).
# The per-sample (projected score, target level) table is dumped by the canonical
# transfer pipeline (reused verbatim; each cell's rho is checked against the
# committed determinants_6stage.json before the CSV is written).
$(TXSCATTER_DATA): $(SUPP_TXSCATTER_DUMP) provenance/transfer_6stage.py
	@echo "Dumping per-sample transfer scores (fidelity-checked vs determinants_6stage.json)"
	python3 $(SUPP_TXSCATTER_DUMP)

$(FIGS_TXSCATTER_PDF): $(SUPP_TXSCATTER_SCRIPT) $(TXSCATTER_DATA) $(SHARED_DEPS)
	@echo "Generating Supplementary Fig: per-cell transfer correlation scatter"
	$(R) $(SUPP_TXSCATTER_SCRIPT)

# ==============================================================================
# UTILITIES
# ==============================================================================

clean:
	@echo "Cleaning output directories..."
	rm -rf $(OUT_DIR)/pdf/*.pdf
	rm -rf $(OUT_DIR)/png/*.png
	rm -rf $(OUT_DIR)/stats/*.txt
	@echo "✓ Clean complete"

clean-all: clean
	rm -rf $(OUT_DIR)/stats/*.csv
	@echo "✓ Full clean complete (including data files)"

help:
	@echo ""
	@echo "Paper Figures Makefile"
	@echo "======================"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all        Generate all figures - main + supplementary (default)"
	@echo "  main       Generate main figures only (Fig 1-4)"
	@echo "  supp       Generate supplementary figures only (Fig S1-S3)"
	@echo "  supp-pdf   Compile supplementary PDF (Nature format)"
	@echo "  fig1       Generate Figure 1: Study Design"
	@echo "  fig2       Generate Figure 2: Model Performance"
	@echo "  fig3       Generate Figure 3: Functional Enrichment"
	@echo "  fig4       Generate Figure 4: Cross-Species Comparison"
	@echo "  fig4-data  Prepare data for Figure 4 (Python)"
	@echo "  clean      Remove PDF/PNG outputs"
	@echo "  clean-all  Remove all outputs including data"
	@echo "  help       Show this help message"
	@echo ""
	@echo "Outputs:"
	@echo "  PDF:   output/pdf/"
	@echo "  PNG:   output/png/"
	@echo "  Stats: output/stats/"
	@echo ""
