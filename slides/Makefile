# Makefile for Cardiff University Slides

# Configuration
PROJECT = presentation
SOURCE_DIR = .
THEME_DIR = theme
BUILD_DIR = build
OUTPUT_DIR = .

# Main File to Compile (change this to your specific presentation)
MAIN_FILE = presentation
SOURCE = $(SOURCE_DIR)/$(MAIN_FILE).tex

# Tools
LATEX = pdflatex
LATEX_FLAGS = -output-directory=$(BUILD_DIR) -interaction=nonstopmode

# Environment to find the theme
export TEXINPUTS := .:$(THEME_DIR):$(TEXINPUTS)

# Targets
.PHONY: all clean help

all: $(BUILD_DIR)
	@echo "Building $(MAIN_FILE)..."
	$(LATEX) $(LATEX_FLAGS) $(SOURCE)
	$(LATEX) $(LATEX_FLAGS) $(SOURCE)
	@mv $(BUILD_DIR)/$(MAIN_FILE).pdf $(OUTPUT_DIR)/$(PROJECT).pdf
	@echo "Done! Output is $(PROJECT).pdf"

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(PROJECT).pdf

help:
	@echo "Cardiff Slide Builder"
	@echo "---------------------"
	@echo "make        : Build the presentation"
	@echo "make clean  : Remove build artifacts"
