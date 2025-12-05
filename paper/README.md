# Paper Organization

## 📁 Folder Structure

```
paper/
├── 📄 pig_age_stage.tex      - Original manuscript (full version)
├── 📄 pig_age_stage.pdf      - Original compiled PDF (18 pages)
├── 📄 main_modular.tex       - Modular Nature/Cell version
├── 📄 main_modular.pdf       - Nature/Cell compiled PDF (9 pages)
├── 📄 pig_age_stage.bib      - Bibliography file
│
├── sections/                 - Modular LaTeX sections
│   ├── 00_abstract.tex      - Concise abstract (150 words)
│   ├── 01_introduction.tex  - Introduction section
│   ├── 02_results.tex       - Results sections
│   ├── 03_discussion.tex    - Discussion section
│   ├── 04_methods.tex       - Methods (concise version)
│   ├── 05_figures_tables.tex - All figures and tables
│   └── 06_references.tex    - References
│
├── supplementary/           - Supplementary materials
│   └── supplementary_information.tex
│
├── figures/                 - Figure organization
│   ├── main/               - Main manuscript figures
│   └── supplementary/      - Supplementary figures
│
├── tables/                  - Table organization
│   ├── main/               - Main manuscript tables
│   └── supplementary/      - Supplementary tables
│
├── archive/                 - Archived files
│   ├── *.txt               - Original text drafts
│   ├── *.rtf               - RTF versions
│   └── *_backup.tex        - Backup files
│
└── docs/                    - Documentation
    ├── REORGANIZATION_PLAN.md
    ├── REORGANIZATION_COMPLETE.md
    └── COMPILATION_SUCCESS.md
```

## 🚀 Quick Commands

### Compile Original Version:
```bash
pdflatex pig_age_stage.tex
pdflatex pig_age_stage.tex  # Run twice for references
```

### Compile Modular Version:
```bash
pdflatex main_modular.tex
pdflatex main_modular.tex  # Run twice for references
```

### Compile Supplementary:
```bash
cd supplementary
pdflatex supplementary_information.tex
cd ..
```

### Clean Build Files:
```bash
rm -f *.aux *.log *.out *.fls *.fdb_latexmk *.synctex.gz
```

## 📝 Key Files

- **For submission**: Use `main_modular.tex` (Nature/Cell format)
- **For review**: Use `pig_age_stage.tex` (comprehensive version)
- **Supplementary**: Located in `supplementary/` folder

## 🎯 Journal Formats

### Nature (3000 words)
- Use `main_modular.tex`
- 4 figures/tables max
- Methods at end

### Cell (5000 words)
- Use `main_modular.tex`
- 7 figures/tables max
- STAR Methods in supplementary

### Nature Communications
- Use `pig_age_stage.tex`
- No strict limits
- Flexible format

---
*Organized and ready for submission*