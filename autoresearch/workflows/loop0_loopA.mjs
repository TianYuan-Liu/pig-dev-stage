export const meta = {
  name: 'autoresearch-dgtex-loop0-loopA',
  description: 'dGTEx cross-species autoresearch: Loop 0 tissue rescues + Loop A gene-selection variants',
  phases: [
    { title: 'Loop 0 rescues', detail: 'Small Intestine (pool), Heart (permissive bins), Liver (GO:0032502)' },
    { title: 'Loop A variants', detail: '8 single-knob gene-selection variants vs baseline' },
  ],
}

const ROOT = '/Users/tianyuan/Desktop/github_dev/.claude-worktree-dgtex-v2'
const OUT = 'autoresearch/uniform_results'
const BASELINE = { mean_r: 0.35589176, min_r: 0.16189436, config: 'strict_1to1|pig_anchored|spearman', panel_size: 5 }

// ── schemas ──────────────────────────────────────────────────────────────
const VARIANT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    variant_id: { type: 'string' },
    command: { type: 'string' },
    mean_r: { type: ['number', 'null'] },
    min_r: { type: ['number', 'null'] },
    max_r: { type: ['number', 'null'] },
    all_tissues_valid: { type: 'boolean' },
    per_tissue_r: { type: 'object', additionalProperties: { type: ['number', 'null'] } },
    status: { type: 'string', enum: ['ok', 'crash'] },
    notes: { type: 'string' },
  },
  required: ['variant_id', 'command', 'mean_r', 'min_r', 'max_r', 'all_tissues_valid', 'status', 'notes'],
}

const RESCUE_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    rescue_code: { type: 'string' },
    tissue: { type: 'string' },
    runnable: { type: 'boolean' },
    command: { type: 'string' },
    r: { type: ['number', 'null'] },
    n_genes: { type: ['integer', 'null'] },
    n_pig_young: { type: ['integer', 'null'] },
    n_pig_old: { type: ['integer', 'null'] },
    ci_low: { type: ['number', 'null'] },
    ci_high: { type: ['number', 'null'] },
    passes_criterion: { type: 'boolean' },
    criterion: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['rescue_code', 'tissue', 'runnable', 'command', 'r', 'passes_criterion', 'criterion', 'notes'],
}

const runHint = `Run from ${ROOT}. Use python3 (miniforge base env has pandas/numpy/scipy/openpyxl). ` +
  `The driver prints a JSON object to stdout AND writes it to --out; read the --out file with Read. ` +
  `Extract: top-level mean_r, min_r, max_r, all_tissues_valid, and per_tissue[<tissue>].r for each panel tissue. ` +
  `Do NOT modify any source file. Report numbers verbatim from the JSON.`

// ── Loop A variants: each is ONE knob flip vs the locked baseline ──────────
const VARIANTS = [
  { id: 'v1_human_permissive_bins', flags: '--human-young-cohorts 1,2 --human-old-cohorts 3,4', note: 'permissive human bins (cohort 1+2 vs 3+4); MORE complex (custom bins)' },
  { id: 'v2_pig_permissive_bins', flags: '--pig-young "Infant,Early childhood,Pre-pubertal"', note: 'permissive pig young bin (+Pre-pubertal); MORE complex (custom bins)' },
  { id: 'v3_pig_median_fc', flags: '--pig-method median', note: 'median pig FC instead of weighted; SIMPLER (no purity weighting)' },
  { id: 'v4_drop0', flags: '--drop-pct 0', note: 'no marker-purity sample drop; SIMPLER (removes a step)' },
  { id: 'v5_low_expr', flags: '--pig-min-tpm 0.5 --human-min-tpm 5', note: 'lower expression filters; neutral complexity (threshold values)' },
  { id: 'v6_high_expr', flags: '--pig-min-tpm 3 --human-min-tpm 30', note: 'higher expression filters; neutral complexity (threshold values)' },
  { id: 'v7_top300_pigfc', flags: '--filter top_n_pigfc --top-n 300', note: 'NEW filter: top-300 by |pig log2FC|, no FDR; MORE complex (new filter)' },
  { id: 'v8_bootstrap_stable', flags: '--filter bootstrap_stable --bootstrap-B 100', note: 'NEW filter: pig_anchored genes stable in >=80% of pig bootstraps; MUCH MORE complex' },
]

phase('Loop 0 rescues')

// Rescue A — Small Intestine via pig sub-region pooling
const rescueA = agent(
  `Loop 0 RESCUE A — Small Intestine, via pig sub-region pooling.\n\n` +
  `${runHint}\n\n` +
  `Background: dGTEx tissue "Small Intestine" exists; on the pig side PigGTEx uses sub-region ` +
  `Tissue-class labels. Pooling the TRUE small-intestine sub-regions only — "Small intestine", ` +
  `"Ileum", "Jejunum", "Duodenum" (NOT Colon/Large_intestine, which is large intestine) — recovers samples.\n\n` +
  `Run EXACTLY:\n` +
  `cd ${ROOT} && python3 -m autoresearch.autoresearch_driver --label rescueA_small_intestine ` +
  `--tissues "Small Intestine" ` +
  `--pig-pool '{"Small Intestine":["Small_intestine","Ileum","Jejunum","Duodenum"]}' ` +
  `--out ${OUT}/rescueA_small_intestine.json 1>/dev/null 2>${OUT}/rescueA_small_intestine.log\n\n` +
  `Then Read ${OUT}/rescueA_small_intestine.json. Report tissue="Small Intestine", rescue_code="A", runnable=true. ` +
  `Pull r, n_genes (= per_tissue["Small Intestine"].n), n_pig_young/n_pig_old (= sample_info["Small Intestine"].pig_young/pig_old), ci_low, ci_high.\n\n` +
  `ACCEPTANCE CRITERION: n_pig_young>=3 AND n_pig_old>=3 AND r>0 with a valid p-value. ` +
  `Set passes_criterion accordingly. If the driver errors or the JSON is missing, set runnable=false and explain in notes. ` +
  `Do not retry with different pools — this is a one-shot rescue.`,
  { label: 'rescueA:SmallIntestine', phase: 'Loop 0 rescues', model: 'sonnet', schema: RESCUE_SCHEMA }
)

// Rescue B — Heart via permissive dGTEx bins
const rescueB = agent(
  `Loop 0 RESCUE B — Heart, via permissive dGTEx age bins.\n\n` +
  `${runHint}\n\n` +
  `Background: under default strict bins (human cohort 1 vs 4) Heart gives r≈-0.02 (noise, dGTEx 3 vs 3). ` +
  `The intervention: permissive human bins — cohort 1+2 (young) vs 3+4 (old) — applied ONLY to Heart.\n\n` +
  `Run EXACTLY:\n` +
  `cd ${ROOT} && python3 -m autoresearch.autoresearch_driver --label rescueB_heart ` +
  `--tissues "Heart" ` +
  `--per-tissue-bins '{"Heart":{"human_young_cohorts":[1,2],"human_old_cohorts":[3,4]}}' ` +
  `--out ${OUT}/rescueB_heart.json 1>/dev/null 2>${OUT}/rescueB_heart.log\n\n` +
  `Then Read ${OUT}/rescueB_heart.json. Report tissue="Heart", rescue_code="B", runnable=true, with r, n_genes, ` +
  `n_pig_young, n_pig_old, ci_low, ci_high.\n\n` +
  `ACCEPTANCE CRITERION: r>0.10 AND the bootstrap 95% CI [ci_low, ci_high] excludes 0 (both bounds same sign as r, i.e. ci_low>0). ` +
  `Set passes_criterion accordingly. One-shot; do not vary the bins.`,
  { label: 'rescueB:Heart', phase: 'Loop 0 rescues', model: 'sonnet', schema: RESCUE_SCHEMA }
)

// Rescue E — Liver via GO:0032502 (developmental process) gene restriction
const rescueE = agent(
  `Loop 0 RESCUE E — Liver, via GO:0032502 developmental-process gene restriction.\n\n` +
  `${runHint}\n\n` +
  `Background: Liver gives r≈-0.32 (adult-hepatic genes flip direction in dGTEx adolescents). ` +
  `Hypothesis: restricting to developmental-process genes (GO:0032502 and its descendants) reverses this.\n\n` +
  `STEP 1 — build the human gene list. Write & run a python script (requests is installed) that queries mygene.info ` +
  `for human genes annotated to GO:0032502 and its descendant terms, collecting unversioned Ensembl gene IDs (ENSG...). ` +
  `Use the biological-process annotation. A workable query (paginate with fetch_all):\n` +
  `  import requests\n` +
  `  ids=set(); \n` +
  `  r=requests.get('https://mygene.info/v3/query', params={'q':'go.BP.id:"GO:0032502"','species':'human','fields':'ensembl.gene','size':1000,'fetch_all':True}).json()\n` +
  `  # follow _scroll_id with scroll_id param until no hits; collect hit['ensembl']['gene'] (may be str or list)\n` +
  `  Write the unique ENSG IDs (one per line) to ${OUT}/go_dev_human_ensg.txt\n` +
  `Report how many IDs you collected in notes. If mygene is unreachable, set runnable=false and explain.\n\n` +
  `STEP 2 — evaluate Liver restricted to that gene set:\n` +
  `cd ${ROOT} && python3 -m autoresearch.autoresearch_driver --label rescueE_liver_go ` +
  `--tissues "Liver" --restrict-genes ${OUT}/go_dev_human_ensg.txt --restrict-col human ` +
  `--out ${OUT}/rescueE_liver_go.json 1>/dev/null 2>${OUT}/rescueE_liver_go.log\n\n` +
  `Then Read ${OUT}/rescueE_liver_go.json. Report tissue="Liver", rescue_code="E", runnable=true, with r, n_genes, ` +
  `n_pig_young, n_pig_old, ci_low, ci_high. Note in notes: the gene-list size and whether restriction reduced n_genes.\n\n` +
  `ACCEPTANCE CRITERION: r>0 AND bootstrap 95% CI excludes 0 (ci_low>0) — i.e. it must REVERSE the negative baseline. ` +
  `Set passes_criterion accordingly. One-shot.`,
  { label: 'rescueE:Liver', phase: 'Loop 0 rescues', model: 'sonnet', schema: RESCUE_SCHEMA }
)

phase('Loop A variants')

const variantAgents = VARIANTS.map(v => agent(
  `Loop A variant "${v.id}" — ONE knob flip vs the locked baseline ` +
  `(strict_1to1 | pig_anchored | spearman, 5-tissue panel: Muscle, Lung, Testis, Spleen, Adipose Tissue).\n` +
  `Complexity note (for the record): ${v.note}\n\n` +
  `${runHint}\n\n` +
  `Run EXACTLY:\n` +
  `cd ${ROOT} && python3 -m autoresearch.autoresearch_driver --label ${v.id} ${v.flags} ` +
  `--out ${OUT}/${v.id}.json 1>/dev/null 2>${OUT}/${v.id}.log\n\n` +
  `Then Read ${OUT}/${v.id}.json and return: variant_id="${v.id}", the exact command, mean_r, min_r, max_r, ` +
  `all_tissues_valid, per_tissue_r (a map tissue->r for all 5 panel tissues), status="ok" (or "crash" if the run failed), ` +
  `and notes (one line; mention the complexity note and anything unusual). Report numbers verbatim.`,
  { label: `loopA:${v.id}`, phase: 'Loop A variants', model: 'haiku', schema: VARIANT_SCHEMA }
))

const [loop0, loopA] = await parallel([
  () => Promise.all([rescueA, rescueB, rescueE]),
  () => Promise.all(variantAgents),
])

const blocked = [
  { rescue_code: 'C', tissue: 'Colon', reason: 'dGTEx colon expression file not present locally (only 8 dGTEx tissues downloaded)' },
  { rescue_code: 'D', tissue: 'Lymph Node', reason: 'dGTEx lymph_node expression file not present locally' },
  { rescue_code: 'F', tissue: 'Blood Vessel', reason: 'dGTEx blood_vessel/artery expression file not present locally' },
  { rescue_code: 'G', tissue: 'Pituitary', reason: 'dGTEx pituitary expression file not present locally' },
]

return {
  baseline: BASELINE,
  loop0_rescues: (loop0 || []).filter(Boolean),
  loop0_blocked: blocked,
  loopA_variants: (loopA || []).filter(Boolean),
}
