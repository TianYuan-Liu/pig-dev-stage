#!/usr/bin/env python3
"""Loop C — validate the Loop A winner (ext3) per tissue.

For each panel tissue, under the winning config (strict_1to1 | pig_anchored |
spearman, pig>=3 / human>=30 TPM, no purity-drop), reports:
  - r and bootstrap 95% CI (from the frozen evaluator)
  - ortholog-scramble permutation empirical p (shuffle pig<->human pairing)
  - leave-one-gene-out r range
  - tissue-marker sanity (how many curated markers are in the gene set + their
    directional concordance)

Reuses the parameterized driver's build_merged + select_genes (which call the
FROZEN evaluator). No new science in the correlation itself.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "review" / "analyses"))

import cross_species_all_tissues as xs  # noqa: E402
from autoresearch.autoresearch_driver import build_merged, select_genes, TISSUE_REGISTRY  # noqa: E402
from autoresearch.run_purity_generic import TISSUE_MARKERS, load_pig_symbol_map  # noqa: E402

PANEL = ["Muscle", "Lung", "Testis", "Spleen", "Adipose Tissue"]
N_PERM = 1000
SEED = 42


def winner_cfg():
    ortho = pd.read_csv(ROOT / "review/analyses/results/pig_human_one_to_one_orthologs.csv")
    return {
        "label": "loopC_validate", "panel": PANEL,
        "ortholog": "strict_1to1", "ortho_table": ortho,
        "filter": "pig_anchored", "statistic": "spearman",
        "pig_young": ["Infant", "Early childhood"], "pig_old": ["Post-pubertal", "Adult"],
        "human_young_cohorts": [1], "human_old_cohorts": [4],
        "pig_method": "weighted", "drop_pct": 0.0,
        "pig_min_tpm": 3.0, "human_min_tpm": 30.0,
        "top_n": 300, "bootstrap_B": 100, "bootstrap_frac": 0.8,
        "restrict_ids": None, "restrict_col": "human",
        "per_tissue": {}, "pig_pool": {},
    }


def validate_tissue(name: str, cfg: dict) -> dict:
    merged, info = build_merged(name, cfg)
    if merged is None:
        return {"tissue": name, "status": "skipped", "reason": info.get("skipped")}
    sub = select_genes(merged, cfg)
    x = sub["log2fc_pig"].values.astype(float)
    y = sub["log2fc_human"].values.astype(float)
    n = len(x)
    r_obs, p_obs = stats.spearmanr(x, y)

    # bootstrap CI (consistent with the pipeline's bootstrap_pearson_ci)
    _, ci_low, ci_high = xs.bootstrap_pearson_ci(x, y)

    # ortholog-scramble permutation: shuffle human FC vs pig FC (random pairing)
    rng = np.random.default_rng(SEED)
    null = np.empty(N_PERM)
    yc = y.copy()
    for i in range(N_PERM):
        rng.shuffle(yc)
        null[i] = stats.spearmanr(x, yc)[0]
    emp_p = (1.0 + np.sum(np.abs(null) >= abs(r_obs))) / (N_PERM + 1.0)

    # leave-one-gene-out r range
    if n <= 200:
        loo = [stats.spearmanr(np.delete(x, i), np.delete(y, i))[0] for i in range(n)]
    else:  # subsample LOO for large sets (deterministic stride)
        idx = np.linspace(0, n - 1, 200).astype(int)
        loo = [stats.spearmanr(np.delete(x, i), np.delete(y, i))[0] for i in idx]
    loo_lo, loo_hi = float(min(loo)), float(max(loo))

    # tissue-marker sanity
    pig_tissue = TISSUE_REGISTRY[name][1]
    sym2id = load_pig_symbol_map(xs.PIG_SYM_CACHE)
    panel_syms = TISSUE_MARKERS.get(pig_tissue, [])
    marker_ids = {sym2id[s.upper()] for s in panel_syms if s.upper() in sym2id}
    in_set = sub[sub["pig_gene_id"].isin(marker_ids)]
    n_markers_in = int(len(in_set))
    concordant = int((np.sign(in_set["log2fc_pig"]) == np.sign(in_set["log2fc_human"])).sum()) if n_markers_in else 0

    return {
        "tissue": name, "status": "ok", "n_genes": int(n),
        "r": float(r_obs), "p": float(p_obs),
        "ci_low": float(ci_low) if not np.isnan(ci_low) else None,
        "ci_high": float(ci_high) if not np.isnan(ci_high) else None,
        "ci_excludes_zero": bool((not np.isnan(ci_low)) and ci_low > 0),
        "perm_empirical_p": float(emp_p), "n_perm": N_PERM,
        "loo_r_min": loo_lo, "loo_r_max": loo_hi,
        "marker_panel_size": len(panel_syms),
        "markers_in_set": n_markers_in, "markers_concordant": concordant,
    }


def main() -> int:
    cfg = winner_cfg()
    out = {"config": "strict_1to1 | pig_anchored | spearman | pig>=3 human>=30 TPM | drop0",
           "n_perm": N_PERM, "seed": SEED, "per_tissue": {}}
    print(f"Loop C validation (N_perm={N_PERM}, seed={SEED})")
    print("=" * 78)
    for t in PANEL:
        res = validate_tissue(t, cfg)
        out["per_tissue"][t] = res
        if res["status"] != "ok":
            print(f"{t:16s} SKIPPED ({res.get('reason')})")
            continue
        ci = f"[{res['ci_low']:+.3f},{res['ci_high']:+.3f}]" if res["ci_low"] is not None else "NA"
        print(f"{t:16s} r={res['r']:+.3f} n={res['n_genes']:>4} CI={ci} "
              f"perm_p={res['perm_empirical_p']:.4f} "
              f"LOO=[{res['loo_r_min']:+.3f},{res['loo_r_max']:+.3f}] "
              f"markers={res['markers_concordant']}/{res['markers_in_set']}")
    rs = [r["r"] for r in out["per_tissue"].values() if r.get("status") == "ok"]
    out["mean_r"] = float(np.mean(rs)) if rs else None
    out["min_r"] = float(min(rs)) if rs else None
    out_path = ROOT / "review/analyses/results/v2_loopC_validation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nmean_r={out['mean_r']:.4f} min_r={out['min_r']:.4f}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
