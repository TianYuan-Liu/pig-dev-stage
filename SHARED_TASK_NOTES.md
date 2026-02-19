# Shared Task Notes

## Current Status
- Missing citations analysis COMPLETE and VALIDATED. See `MISSING_CITATIONS.txt` for 14 missing bib entries.
- All 8 PubMed-indexed references validated via PMID lookup — all correct.
- Non-PubMed refs (baker_reproducibility_2016 = Nature News; hastie_elements_2009 & barlow_statistical_1972 = books; jaccard_distribution_1912 = classic paper; efron_bootstrap_1979 & benjamini_controlling_1995 = stats papers) verified by description.
- Fixed a file location error: `baker_reproducibility_2016` second use is in `6_Conclusions.tex` line 70, not `5_Discussion.tex`.

## Action Items for Human Developer
1. **Validate each suggested reference** in `MISSING_CITATIONS.txt` is the correct paper
2. **Add validated BibTeX entries** to `paper/pig-age-human/pig-age-human.bib`
3. **Fix the `scoris_klk6_2002` key** — appears to be a misspelling. Decide whether to rename it in the thesis or keep the key as-is in the bib file
4. **Choose UMAP citation** — arXiv preprint vs Nat Biotechnol paper for `mcinnes_umap_2018`
5. After adding entries, recompile thesis to verify all citations resolve
6. Check that `paper/paper.tex` also compiles cleanly (it may share some of these keys)

## Notes
- No content changes were made to any thesis or paper files
- The thesis uses 103 unique citation keys; the bib file has 180 entries
- MISSING_CITATIONS.txt includes ready-to-paste BibTeX entries for all 14 missing refs
