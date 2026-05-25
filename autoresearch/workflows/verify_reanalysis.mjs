export const meta = {
  name: 'autoresearch-verify-reanalysis',
  description: 'Independently audit every pipeline step for bugs + multi-experiment re-analysis of ML-feature vs DE-gene cross-species r',
  phases: [
    { title: 'Step audit', detail: 'independent re-derivation of each pipeline step from raw data' },
    { title: 'Experiments', detail: 'multi-method investigation of ML-feature r vs DE-gene r' },
    { title: 'Verify', detail: 'adversarial check of the central verdict' },
    { title: 'Synthesize', detail: 'reconcile findings, corrected numbers, verdict' },
  ],
}

const ROOT = '/Users/tianyuan/Desktop/github_dev/.claude-worktree-dgtex-v2'

const COMMON = `Working dir: ${ROOT}. Use python3 (base conda env has pandas/numpy/scipy/scikit-learn/lightgbm). ` +
  `Write your OWN scratch script to /tmp (unique name) and run it — do NOT just trust the repo's code; ` +
  `the point is to catch bugs by INDEPENDENT re-derivation from the raw data. Report numbers verbatim.\n\n` +
  `Key facts & paths:\n` +
  `- Ortholog table: review/analyses/results/pig_human_one_to_one_orthologs.csv (cols: pig_gene_id [ENSSSCG...], human_gene_id [ENSG...], symbol). Strict 1:1, immutable.\n` +
  `- Pig expression: data/pigGTEx/<Tissue>.expr_tpm.txt.gz (tab, index=pig gene id, cols=BioSample ids).\n` +
  `- Pig metadata: data/PigGTEx_v0.MetaTable.csv (BioSample, Tissue class, Sub categories, Age). Stages from age via days_to_stage.\n` +
  `- dGTEx human expression: data/dgtex/expression/gene_tpm_dgtex_v1_<slug>.gct.gz; metadata xlsx in data/dgtex/metadata/. AGECOHORT 1=Infant..4=Post-pubertal.\n` +
  `- Driver (parameterized, wraps frozen evaluator): autoresearch/autoresearch_driver.py exposes build_merged(name,cfg) -> (merged_df, info) and select_genes(merged,cfg). ` +
  `merged_df cols include pig_gene_id, human_gene_id, log2fc_pig, log2fc_human, p_pig, fdr_pig.\n` +
  `- ML model outputs (developmental-STAGE classifier, pig data): machine_learning/model_outputs/<T>_loopC_ridge25_results.json (ridge winner; top_genes ranked by |coef|, scale-CONFOUNDED on unstandardized log2(TPM+1)). ` +
  `<T>_ridgestd_results.json (SAME ridge model/BA; top_genes ranked by |coef|*std = correct standardized importance).\n` +
  `- Baseline 'confirmatory' driver cfg (paste this dict; ortho_table loaded separately):\n` +
  `    cfg=dict(label='x',panel=[T],ortholog='strict_1to1',ortho_table=pd.read_csv('review/analyses/results/pig_human_one_to_one_orthologs.csv'),\n` +
  `      filter='pig_anchored',statistic='spearman',pig_young=['Infant','Early childhood'],pig_old=['Post-pubertal','Adult'],\n` +
  `      human_young_cohorts=[1],human_old_cohorts=[4],pig_method='weighted',drop_pct=30.0,pig_min_tpm=1.0,human_min_tpm=10.0,\n` +
  `      top_n=300,bootstrap_B=100,bootstrap_frac=0.8,restrict_ids=None,restrict_col='human',per_tissue={},pig_pool={})\n` +
  `  (sys.path.insert for '.' and 'review/analyses' before importing autoresearch.autoresearch_driver / cross_species_all_tissues.)\n` +
  `- Reference results to check against: DE-gene pig_anchored cross-species Spearman r = Muscle 0.535 (n=293), Lung 0.362 (n=233). ` +
  `Panel confirmatory mean_r=0.356. ML winner ridge BA: Muscle 0.915, Lung 0.958. Muscle & Lung are in BOTH the ML panel and the cross-species panel.`

const AUDIT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    step: { type: 'string' },
    independently_recomputed: { type: 'boolean' },
    matches_repo: { type: 'boolean' },
    bug_found: { type: 'boolean' },
    severity: { type: 'string', enum: ['none', 'minor', 'major'] },
    finding: { type: 'string' },
    numbers: { type: 'string' },
  },
  required: ['step', 'independently_recomputed', 'matches_repo', 'bug_found', 'severity', 'finding', 'numbers'],
}

const EXP_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    experiment: { type: 'string' },
    method: { type: 'string' },
    results_table: { type: 'string' },
    conclusion: { type: 'string' },
    bug_suspected: { type: 'boolean' },
    caveats: { type: 'string' },
  },
  required: ['experiment', 'method', 'results_table', 'conclusion', 'bug_suspected', 'caveats'],
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    claim: { type: 'string' },
    verdict: { type: 'string', enum: ['supported', 'refuted', 'uncertain'] },
    reasoning: { type: 'string' },
    strongest_counterevidence: { type: 'string' },
  },
  required: ['claim', 'verdict', 'reasoning', 'strongest_counterevidence'],
}

// ── Phase 1: independent step audits ──────────────────────────────────────
const AUDITS = [
  { id: 'gene_id_matching', task:
    `Audit gene-ID matching — the most likely source of a deflated overlap. Load the pig Muscle expression index (gene IDs), ` +
    `the ortholog table pig_gene_id column, and the ridge Muscle top_genes (from <T>_loopC_ridge25_results.json AND <T>_ridgestd_results.json). ` +
    `Check: (a) are ID formats identical (versioned vs unversioned, prefix)? (b) how many pig expression IDs intersect ortholog pig_gene_id? ` +
    `(c) of the ridge top-1000, how many are valid pig gene IDs, how many are in the ortholog table at all, how many in the merged-table universe (orthologs expressed in both)? ` +
    `Decompose WHY only ~102/1000 land in the universe — is it legitimate (most top features simply aren't 1:1 orthologs / not expressed in human) or an ID-format BUG? Report exact counts.` },
  { id: 'pig_fc', task:
    `Audit the pig log2FC. For Muscle, INDEPENDENTLY recompute log2((mean_old+pseudo)/(mean_young+pseudo)) for ~5 specific genes from the raw pig TPM matrix ` +
    `(young=Infant+Early childhood, old=Post-pubertal+Adult, per metadata) and compare to merged_df['log2fc_pig'] from build_merged. ` +
    `Note: the frozen pig FC uses purity-z WEIGHTING + 30% drop, so exact equality is not expected — assess whether the sign and rough magnitude are sane and the direction (old vs young) is correct (not flipped).` },
  { id: 'human_fc', task:
    `Audit the human (dGTEx) log2FC. For Muscle, independently load the dGTEx muscle gct + metadata, bin AGECOHORT 1 (young) vs 4 (old), recompute median-based log2FC for ~5 genes, ` +
    `and compare sign/magnitude to merged_df['log2fc_human']. Confirm young/old are not swapped and cohort mapping is correct.` },
  { id: 'gene_selection_filter', task:
    `Audit the pig_anchored gene selection. Independently re-implement the filter (fdr_pig<0.10 AND |log2fc_pig|>0.5 AND |log2fc_human|>0.5) on merged_df and compare the selected gene count to select_genes(merged,cfg) for Muscle and Lung. ` +
    `Also verify the BH-FDR is applied to PIG p-values (not human) and that selection is pig-anchored (significance only on pig side). Report counts (expect Muscle 293, Lung 233).` },
  { id: 'crossspecies_stat', task:
    `Audit the cross-species statistic. For Muscle and Lung, independently compute scipy spearmanr(log2fc_pig, log2fc_human) on the pig_anchored gene set and compare to the reported r (Muscle 0.535, Lung 0.362). ` +
    `Independently run an ortholog-scramble permutation (shuffle human FC, 1000x, seed 42) and report empirical p. Flag any discrepancy.` },
  { id: 'ml_importance', task:
    `Audit the ML feature importance. Confirm ridge_continuous get_feature_importance = |coef| on UNSTANDARDIZED log2(TPM+1) (scale-confounded), and ridge_std = |coef|*std (correct). ` +
    `Independently load the Muscle ridge model output, take its top genes by BOTH rankings, and report how different the two top-100 lists are (Jaccard). ` +
    `Verify ridge_std BA == ridge_continuous BA (same model). State which ranking is the statistically correct 'importance'.` },
]

phase('Step audit')
const auditAgents = AUDITS.map(a => agent(
  `INDEPENDENT BUG AUDIT — step "${a.id}".\n\n${COMMON}\n\nYour task: ${a.task}\n\n` +
  `Return: step="${a.id}", independently_recomputed (did you recompute from raw data, not just rerun repo code), ` +
  `matches_repo (do your numbers match the repo's), bug_found, severity, finding (what you concluded), numbers (the key figures).`,
  { label: `audit:${a.id}`, phase: 'Step audit', model: 'sonnet', schema: AUDIT_SCHEMA }))

// ── Phase 2: experiments on the central question ──────────────────────────
const EXPERIMENTS = [
  { id: 'id_decomposition', task:
    `Decompose the ML-feature universe. For Muscle and Lung: of the ridge top-1000 features, count how many are (a) in the ortholog table, (b) expressed in human (in merged universe). ` +
    `Then compute the ML-feature cross-species r WITHOUT the orthology constraint being a hidden filter — i.e., among the ridge top features that ARE orthologs-expressed-in-both, take top-50/100/200 and compute spearman(log2fc_pig,log2fc_human). ` +
    `Use BOTH the |coef| and |coef|*std rankings. Report a table.` },
  { id: 'importance_methods', task:
    `Compute the ML-feature cross-species r under MULTIPLE importance measures for Muscle and Lung at top-50/100/200: ` +
    `(1) ridge |coef| (from _loopC_ridge25), (2) ridge |coef|*std (from _ridgestd), (3) sklearn permutation_importance of a ridge fit on log2(TPM+1) with stage labels (balanced_accuracy scorer, n_repeats=5, seed=42) — rank genes by mean importance, restrict to orthologs-expressed-in-both. ` +
    `Report r for each method and compare to DE-gene r (0.535 Muscle, 0.362 Lung). Which method gives the highest ML-feature r? Does any reach/exceed the DE-gene r?` },
  { id: 'expected_ordering', task:
    `Test whether DE-gene r >= ML-feature r is EXPECTED (not a bug). On the Muscle merged table (orthologs expressed in both), compute the cross-species spearman r for gene sets of equal size (n=200) ranked by 4 different criteria: ` +
    `(a) |log2fc_pig| descending (pure DE magnitude), (b) pig FDR ascending (significance), (c) ridge |coef|*std importance, (d) 200 RANDOM orthologs (mean of 50 random draws, seed 42). ` +
    `Rank the resulting r values. The hypothesis to test: selecting by FC magnitude/significance naturally yields the HIGHEST cross-species r, so DE>ML is expected. Report the r for each criterion.` },
  { id: 'perm_null_ml', task:
    `Establish significance of the ML-feature r. For Muscle and Lung, take the ridge |coef|*std top-100 orthologs-expressed-in-both, compute the observed spearman r, then an ortholog-scramble permutation null (shuffle human FC, 1000x, seed 42) and report empirical p. ` +
    `Is the ML-feature r (≈0.27-0.34) significantly above chance? Report observed r, null mean, empirical p.` },
]

phase('Experiments')
const expAgents = EXPERIMENTS.map(e => agent(
  `EXPERIMENT — "${e.id}".\n\n${COMMON}\n\nYour task: ${e.task}\n\n` +
  `Return: experiment="${e.id}", method (what you ran), results_table (numbers as text), conclusion, bug_suspected, caveats.`,
  { label: `exp:${e.id}`, phase: 'Experiments', model: 'sonnet', schema: EXP_SCHEMA }))

const [audits, exps] = await parallel([
  () => Promise.all(auditAgents),
  () => Promise.all(expAgents),
])

// ── Phase 3: adversarial verification of the central verdict ──────────────
const expSummary = JSON.stringify((exps || []).filter(Boolean).map(e => ({
  experiment: e.experiment, conclusion: e.conclusion, bug_suspected: e.bug_suspected, results: e.results_table })), null, 1)

phase('Verify')
const CLAIM = 'The finding "cross-species r on the ML model\'s top features (~0.27-0.34, standardized importance) is LOWER than r on the DE-selected genes (0.535 Muscle / 0.362 Lung)" is CORRECT and EXPECTED (DE selection targets large concordant fold-changes which maximize the correlation), NOT a bug.'
const verifyAgents = [0, 1, 2].map(i => agent(
  `ADVERSARIAL VERIFICATION (skeptic #${i + 1}). Try to REFUTE this claim using the experiment evidence. ` +
  `Default to 'refuted' or 'uncertain' if the evidence is weak or a bug is plausible.\n\n` +
  `CLAIM: ${CLAIM}\n\nExperiment findings:\n${expSummary}\n\n` +
  `${COMMON}\n\nYou MAY run your own quick check if needed. Consider: could the low ML-feature r be a residual bug (ID mismatch, wrong importance, wrong FC sign, label leakage)? ` +
  `Or is DE>ML genuinely expected? Return: claim (restate briefly), verdict (supported/refuted/uncertain), reasoning, strongest_counterevidence.`,
  { label: `verify:skeptic${i + 1}`, phase: 'Verify', model: 'sonnet', schema: VERDICT_SCHEMA }))
const verdicts = await Promise.all(verifyAgents)

// ── Phase 4: synthesis ────────────────────────────────────────────────────
phase('Synthesize')
const synth = await agent(
  `SYNTHESIZE the verification re-analysis. Inputs below.\n\n` +
  `STEP AUDITS:\n${JSON.stringify((audits || []).filter(Boolean), null, 1)}\n\n` +
  `EXPERIMENTS:\n${JSON.stringify((exps || []).filter(Boolean), null, 1)}\n\n` +
  `ADVERSARIAL VERDICTS:\n${JSON.stringify((verdicts || []).filter(Boolean), null, 1)}\n\n` +
  `Produce a clear written report: (1) any BUGS found in any step (severity + fix), (2) the corrected ML-feature cross-species r per tissue/method, ` +
  `(3) the definitive verdict on whether 'ML-feature r < DE-gene r' is a bug or expected, with the strongest evidence, ` +
  `(4) what (if anything) should change in the committed results, (5) confidence level. Be concrete and quantitative.`,
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' })

return {
  audits: (audits || []).filter(Boolean),
  experiments: (exps || []).filter(Boolean),
  verdicts: (verdicts || []).filter(Boolean),
  synthesis: synth,
  bugs_found: (audits || []).filter(Boolean).filter(a => a.bug_found),
}
