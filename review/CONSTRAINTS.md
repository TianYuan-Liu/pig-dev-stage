# Autonomous Revision Loop — Hard Constraints

Every agent invoked in this revision cycle MUST read this file before acting, and
the orchestrator enforces these as gates around every commit.

## 1. No data fabrication
Every quantitative claim added to the paper, supplementary, thesis, or response
document MUST cite a JSON/CSV file in `machine_learning/model_outputs/`,
`machine_learning/analysis/results/`, or `review/analyses/results/`. If a number
doesn't exist, the analysis runs first; the text edit comes second. The
`critical-code-reviewer` final pass flags any unsourced statistic for revert.

## 2. Bibliography is read-only
`paper/pig-age-human/pig-age-human.bib` is **never** modified by an agent.

- Baseline SHA-256: `2f55092b47ba2a066b99b15edf1a8429a7c8498bfe9ed0b7d3309886ee0a94d9`
- Orchestrator gate: `shasum -a 256 paper/pig-age-human/pig-age-human.bib` must
  return that exact hash before every commit.
- New citations: print BibTeX entries to `review/new_bib_entries.bib`
  (append-only). The user pastes them into the bib manually after the loop.
- Only `\cite{key}` for keys that already exist in `pig-age-human.bib`. Verify
  with `grep -c "@.*{<key>," paper/pig-age-human/pig-age-human.bib`.

## 3. Paper ↔ thesis synchronisation
Every commit that touches `paper/paper.tex` or `paper/supplementary.tex` MUST
also touch the corresponding section of the thesis
(`thesis/MPhilThesis-Latex-Template/`). The orchestrator checks
`git diff --name-only HEAD~1` and reverts the commit if the sync rule is broken.

| Paper section | Thesis mirror |
|---|---|
| `paper/paper.tex` Results | `thesis/.../4_Results/4_Results.tex` |
| `paper/paper.tex` Discussion | `thesis/.../5_Discussion/5_Discussion.tex` |
| `paper/paper.tex` Conclusions | `thesis/.../6_Conclusions/6_Conclusions.tex` |
| `paper/supplementary.tex` figs/tables | `thesis/.../7_Supplementary/` |

## 4. LaTeX must compile
After every text-change commit:
1. `cd paper && pdflatex -interaction=nonstopmode paper.tex && bibtex paper && pdflatex -interaction=nonstopmode paper.tex && pdflatex -interaction=nonstopmode paper.tex`
2. `cd paper && pdflatex -interaction=nonstopmode supplementary.tex && pdflatex -interaction=nonstopmode supplementary.tex`
3. Exit code from each step must be 0. Any "undefined reference" or "citation undefined" output triggers a revert.

## 5. No overstating
The editor explicitly flagged "overstated conclusions". Each new claim must be
qualified: "in our dataset", "limited to N samples", "subject to platform
variation", "in the postnatal lifespan window", etc. The `academic-grammar-clarity`
final pass audits this.

## 6. Real data only
- Cross-species expansion (R1.2) uses `data/cardoso_moreira_2019/Human.RPKM.tsv`
  (the 136 MB file, gitignored) with ortholog mapping JSONs already in that
  folder. No synthetic concordance.
- Heterogeneity sensitivity (R2.1) uses real BioProject/Breed/Sex columns from
  `data/PigGTEx_v0.MetaTable.csv`.
- Batch-effect UMAP (R3.1) uses real per-tissue TPM matrices in
  `data/pigGTEx/*.expr_tpm.txt.gz`.
- Marker-gene panel (R3.3) genes must be literature-curated (cite the source in
  the response document); expression values must come from the actual TPM
  matrices, not from random/synthetic data.

## 7. No new dependencies
Stick to packages already in `pyproject.toml` / `requirements.txt`:
`numpy, pandas, scikit-learn, lightgbm, optuna, matplotlib, seaborn, umap-learn,
mygene, requests, scipy, statsmodels`. If a new package is needed, the agent
must request user approval (log a `crash` row with `reason=new_dependency_needed`).

## 8. Project-specific conventions (from MEMORY.md)
- Stage names: `Infant`, `Early childhood`, `Pre-pubertal`, `Post-pubertal`, `Adult` (NEW format; do not use `Infant_0_20d` etc.).
- `DataLoader` sets `Sample_ID` as index, not column — access via `.index`, never `["Sample_ID"]`.
- Run the pipeline from project root (`/Users/tianyuan/Desktop/github_dev/pig-dev-stage`), not from `machine_learning/`.
- `paper/figures/R/_utils.R::load_metadata` reads CSV not XLSX.
- Metadata path: `data/PigGTEx_v0.MetaTable.csv` (not `.xlsx`).

## 9. Branch hygiene
- Working branch: `revision/bmc-r1-may11`.
- Each reviewer point lands as ONE commit (or a small commit cluster) with a
  message starting `R<n>.<m>: <one-line summary>`.
- Never `git push --force`, never reset past the branch base.
- Every commit's message must include the reviewer point identifier
  (`R3.1`, `R1.2`, etc.) for traceability.

## 10. Logging
`review/revision_log.tsv` is the autoresearch-style log. Columns:
`commit\treviewer_point\tgates_passed\tstatus\tdescription`
- One row per attempt (whether kept, discarded, or crashed).
- `gates_passed`: comma-separated codes (`latex`, `thesis_sync`, `bib_unchanged`, `data_real`, `no_overstate`).
- `status`: `keep` / `discard` / `crash`.
- Append-only; never rewrite history.
- This file is **untracked by git** (autoresearch convention — it is a local log,
  not part of the deliverable).
