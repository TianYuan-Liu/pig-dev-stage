# Project Rules

## Paper and Thesis Synchronization
- The thesis (`thesis/`) is built based on the paper (`paper/`). Any content change made to one MUST be reflected in the other to keep them in sync.

## Bibliography
- **NEVER** modify `paper/pig-age-human/pig-age-human.bib`. AI models can hallucinate fake citations — this file is maintained manually.
- Both `paper/` and `thesis/` use `paper/pig-age-human/pig-age-human.bib` as their shared citation source. Do not create separate or duplicate bib files.
- Only cite references that already exist in the bib file. If a new citation is needed, print the full BibTeX entry so the user can manually add it to the bib file.

## Compilation
- Always compile the paper and/or thesis after making changes to `.tex` files. Use the compilation commands from MEMORY.md.

## Data Integrity
- **NEVER** simulate fake data or use mock/placeholder data. All data must come from actual experimental results or existing pipeline outputs.
- Do not fabricate numbers, statistics, p-values, or any quantitative results.
