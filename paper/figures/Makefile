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
OUT_DIR := output
PAPER_DIR := ..

# Mermaid
MERMAID_SRC := mermaid/fig1b_pipeline.mmd
MERMAID_CFG := mermaid/config.json
MERMAID_PNG := $(OUT_DIR)/png/figure1b.png

# Figure scripts
FIG1_SCRIPT := $(R_DIR)/fig1_study_design.R
FIG2_SCRIPT := $(R_DIR)/fig2_model_performance.R
FIG3_SCRIPT := $(R_DIR)/fig3_functional_enrichment.R
FIG4_DATA_SCRIPT := $(PY_DIR)/fig4_data_prep.py
FIG4_SCRIPT := $(R_DIR)/fig4_cross_species.R
SUPP_SCRIPT := $(R_DIR)/supplementary_figures.R

# Output files
FIG1_PDF := $(OUT_DIR)/pdf/fig1_study_design.pdf
FIG2_PDF := $(OUT_DIR)/pdf/fig2_model_performance.pdf
FIG3_PDF := $(OUT_DIR)/pdf/fig3_functional_enrichment.pdf
FIG4_PDF := $(OUT_DIR)/pdf/fig4_cross_species.pdf
FIG4_DATA := $(OUT_DIR)/stats/fig4_expression_stats.csv

# Supplementary outputs
FIGS1_PDF := $(OUT_DIR)/pdf/figS1_cross_species_sensitivity.pdf
FIGS2_PDF := $(OUT_DIR)/pdf/figS2_tissue_specificity_stability.pdf
FIGS3_PDF := $(OUT_DIR)/pdf/figS3_extended_performance.pdf
SUPP_DOC := $(PAPER_DIR)/supplementary_standalone.pdf

# Shared dependencies
SHARED_DEPS := $(R_DIR)/_theme.R $(R_DIR)/_utils.R config/nature_style.yaml

# ==============================================================================
# TARGETS
# ==============================================================================

.PHONY: all main supp supp-pdf fig1 fig2 fig3 fig4 fig4-data mermaid clean help

# Default target - generates everything
all: main supp
	@echo ""
	@echo "✓ All figures generated successfully!"
	@echo "  Outputs in: $(OUT_DIR)/pdf/ and $(OUT_DIR)/png/"

# Main figures only
main: fig1 fig2 fig3 fig4
	@echo ""
	@echo "✓ Main figures complete"

# Supplementary figures
supp: $(FIGS1_PDF) $(FIGS2_PDF) $(FIGS3_PDF)
	@echo ""
	@echo "✓ Supplementary figures complete"

# Compile supplementary PDF (Nature-compliant)
supp-pdf: supp
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Compiling Supplementary PDF (Nature format)"
	@echo "═══════════════════════════════════════════════════════════════"
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode supplementary_standalone.tex
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode supplementary_standalone.tex
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

$(FIG4_PDF): $(FIG4_SCRIPT) $(FIG4_DATA) $(SHARED_DEPS)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Generating Figure 4: Cross-Species Comparison"
	@echo "═══════════════════════════════════════════════════════════════"
	$(R) $(FIG4_SCRIPT)

# Supplementary figures - all generated by single script
$(FIGS1_PDF) $(FIGS2_PDF) $(FIGS3_PDF): $(SUPP_SCRIPT) $(SHARED_DEPS)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Generating Supplementary Figures S1-S3"
	@echo "═══════════════════════════════════════════════════════════════"
	$(R) $(SUPP_SCRIPT)

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
