# Nature/Cell-Style Paper Reorganization Plan

## Executive Summary
This document outlines a comprehensive reorganization strategy to align your manuscript with Nature/Cell publication standards, ensuring logical flow, proper figure citation order, and modular LaTeX structure.

## Current Analysis

### Figure Citation Order (First Appearance in Text)
1. **Abstract**: Table 1, Fig 1, Fig 2, Fig 3, Fig 4, Fig S1-S6
2. **Results Section 1**: Fig 1, Fig S2, Fig S3-S6
3. **Results Section 2**: Fig 2, Table 1
4. **Results Section 3**: Fig 3
5. **Results Section 4**: Fig 4
6. **Discussion**: Fig 3, Fig 4 (re-referenced)

### Issues Identified
- ✓ Figures are cited in correct order (1→2→3→4)
- ✓ Table 1 appears with Figure 2 (performance metrics)
- ⚠️ Supplementary figures cited in abstract (unusual for Nature/Cell)
- ⚠️ LaTeX file is monolithic (not modular)
- ⚠️ PNG and PDF versions are mixed in supplementary

## Proposed Reorganization

### 1. Modular LaTeX Structure
```
paper/
├── main.tex                    # Master document
├── sections/
│   ├── 00_abstract.tex        # Abstract
│   ├── 01_introduction.tex    # Introduction
│   ├── 02_results.tex         # Results (can be split further)
│   ├── 03_discussion.tex      # Discussion
│   ├── 04_methods.tex         # Methods
│   └── 05_references.tex      # References
├── figures/
│   ├── main/                  # Main figures (Fig 1-4)
│   └── supplementary/         # Supplementary figures
├── tables/
│   ├── main/                  # Main tables
│   └── supplementary/         # Supplementary tables
└── supplementary/
    ├── supplementary.tex      # Supplementary information document
    └── supplementary_methods.tex
```

### 2. Figure Organization (Nature/Cell Format)

#### Main Figures (Limited to 4-6 for Nature/Cell)
- **Figure 1**: Study Overview and Methodology
  - Panel A: Age harmonization scheme
  - Panel B: Sample distribution heatmap
  - Panel C: ML pipeline workflow
  - Panel D: Tissue sample counts & granularity

- **Figure 2**: Classifier Performance
  - Panel A: Performance metrics table/plot
  - Panel B: Confusion matrices
  - Panel C: Age correlation plots

- **Figure 3**: Molecular Signatures
  - Panel A: UMAP embeddings by tissue/stage
  - Panel B: Feature overlap Venn/UpSet
  - Panel C: Marker expression trajectories

- **Figure 4**: Biological Validation
  - Panel A: GO enrichment heatmap
  - Panel B: Network visualization
  - Panel C: Canonical marker tracking

#### Supplementary Figures
- **Figure S1**: Extended sample distribution details
- **Figure S2-S5**: PNG versions for presentations
- **Figure S6**: Multi-panel layout example
- **Figure S7**: Cross-tissue validation results
- **Figure S8**: Additional performance metrics

### 3. Text Reorganization

#### Abstract (150-200 words for Nature)
- Remove supplementary figure citations
- Focus on main findings and impact
- End with broad significance statement

#### Introduction (3-4 paragraphs)
1. Problem statement & significance
2. Current limitations
3. Our approach & innovations
4. Overview of findings (no detailed results)

#### Results (4-6 subsections)
1. Study design and cohort characteristics (Fig 1)
2. Classifier achieves high tissue-specific accuracy (Fig 2, Table 1)
3. Molecular signatures reflect developmental programs (Fig 3)
4. Functional validation confirms biological relevance (Fig 4)
5. Cross-tissue generalization reveals domain specificity
6. [Optional] Clinical/agricultural applications

#### Discussion (4-5 paragraphs)
1. Summary of key findings
2. Biological insights & interpretation
3. Technical innovations & advantages
4. Limitations & future directions
5. Conclusions & broader impact

#### Methods
- Move to end (Nature style) or supplementary (Cell style)
- Keep concise in main text (~1000 words)
- Detailed protocols in supplementary

### 4. Citation Standardization

#### Figures
- Main text: "Fig. 1a", "Fig. 2", "Figs. 3 and 4"
- Supplementary: "Supplementary Fig. 1", "Extended Data Fig. 1"

#### Tables
- Main text: "Table 1"
- Supplementary: "Supplementary Table 1"

#### Equations
- Number only referenced equations
- Use \eqref{} for citations

### 5. LaTeX Best Practices

#### Preamble Organization
```latex
% Document class
\documentclass[11pt]{article}

% Essential packages
\usepackage{natbib}      % Nature-style citations
\usepackage{graphicx}
\usepackage{subcaption}  % For multi-panel figures

% Journal-specific formatting
\usepackage{nature}      % If available
```

#### Figure Management
```latex
% Main figure with subpanels
\begin{figure}[htbp]
  \centering
  \begin{subfigure}[b]{0.48\textwidth}
    \includegraphics[width=\textwidth]{figures/main/fig1a.pdf}
    \caption{}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.48\textwidth}
    \includegraphics[width=\textwidth]{figures/main/fig1b.pdf}
    \caption{}
  \end{subfigure}
  \caption{\textbf{Study overview.} (a) Age harmonization. (b) Sample distribution.}
  \label{fig:overview}
\end{figure}
```

### 6. Implementation Checklist

#### Phase 1: Structure (Immediate)
- [ ] Create directory structure
- [ ] Split LaTeX into modular files
- [ ] Set up main.tex with \input{} commands
- [ ] Move figures to organized folders

#### Phase 2: Content (1-2 days)
- [ ] Revise abstract (remove supp citations)
- [ ] Tighten introduction
- [ ] Reorganize results for narrative flow
- [ ] Ensure figure citations are sequential

#### Phase 3: Formatting (1 day)
- [ ] Standardize all citations (Fig., Table, Eq.)
- [ ] Add bold to figure titles
- [ ] Format references (Nature style)
- [ ] Create proper supplementary document

#### Phase 4: Validation (Few hours)
- [ ] Verify all cross-references work
- [ ] Check figure quality (300 dpi minimum)
- [ ] Ensure reproducible compilation
- [ ] Test with journal template if available

### 7. Quality Checks

#### Figure Citation Order
✓ All figures cited in numerical order
✓ Each figure cited before it appears
✓ No forward references to later figures

#### Content Flow
✓ Methods support all results claims
✓ Discussion addresses all key findings
✓ Limitations are acknowledged
✓ Future directions are specific

#### Technical Requirements
✓ Line numbers for review (if required)
✓ Double-spaced text (if required)
✓ Continuous line numbering
✓ PDF/A compliance for archival

## Next Steps

1. **Backup current version** as `pig_age_stage_original.tex`
2. **Create modular structure** with section files
3. **Reorganize figures** into main/supplementary
4. **Update citations** throughout manuscript
5. **Generate clean PDF** for internal review
6. **Create submission package** with all components

## Journal-Specific Adjustments

### For Nature
- Limit to 3000 words (excluding methods/references)
- Maximum 4 main figures/tables combined
- Extended Data figures allowed (up to 10)
- Methods at end of main text

### For Cell
- Limit to 5000 words (excluding methods)
- Maximum 7 main figures/tables
- Supplementary figures unlimited
- STAR Methods in supplementary

### For Nature Communications
- No strict word limit
- Flexible figure count
- Methods in main text
- Extensive supplementary allowed

## Version Control Recommendations

```bash
# Create reorganization branch
git checkout -b reorganize-nature-style

# Track changes systematically
git add sections/*.tex
git commit -m "Modularize LaTeX structure"

git add figures/
git commit -m "Reorganize figures for journal format"

# Create release version
git tag -a v1.0-nature -m "Nature-style submission version"
```

## Success Metrics

- [ ] Manuscript compiles without errors
- [ ] All figures/tables cited in order
- [ ] Word count within journal limits
- [ ] Supplementary materials complete
- [ ] Co-authors approve reorganization
- [ ] Journal template requirements met

---
*Document prepared: $(date)*
*Target journal: Nature/Cell family*
*Status: Ready for implementation*