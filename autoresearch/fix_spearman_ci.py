#!/usr/bin/env python3
"""Bug-1 fix (from the verification re-analysis): the frozen compute_stat reports
a PEARSON bootstrap CI even when the winning statistic is Spearman. The point
estimate, p, perm-p and LOO are all correct — only the CI is mislabelled.

This does NOT modify the frozen evaluator (compute_stat / bootstrap_pearson_ci).
It recomputes the correct SPEARMAN percentile-bootstrap CI (same n=1000, seed=42
methodology) for the confirmatory baseline winner and patches only the reported
CI fields in v2_uniform_winner.json + the fig4 summary, tagging ci_method.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "review" / "analyses"))
import cross_species_all_tissues as xs  # noqa: E402
from autoresearch.autoresearch_driver import build_merged, select_genes  # noqa: E402

PANEL = ["Muscle", "Lung", "Testis", "Spleen", "Adipose Tissue"]
N_BOOT, SEED = xs.N_BOOTSTRAP, 42


def bootstrap_spearman_ci(x, y, n=N_BOOT, seed=SEED):
    """Percentile bootstrap CI for Spearman r — mirrors bootstrap_pearson_ci
    exactly but uses stats.spearmanr."""
    if len(x) < 5:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    rs, ns = [], len(x)
    for _ in range(n):
        idx = rng.integers(0, ns, ns)
        xs_, ys_ = x[idx], y[idx]
        if np.std(xs_) > 0 and np.std(ys_) > 0:
            r, _ = stats.spearmanr(xs_, ys_)
            if not np.isnan(r):
                rs.append(r)
    rs = np.array(rs)
    return float(np.mean(rs)), float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def baseline_cfg(panel):
    return dict(label="x", panel=panel, ortholog="strict_1to1",
        ortho_table=pd.read_csv(ROOT / "review/analyses/results/pig_human_one_to_one_orthologs.csv"),
        filter="pig_anchored", statistic="spearman",
        pig_young=["Infant", "Early childhood"], pig_old=["Post-pubertal", "Adult"],
        human_young_cohorts=[1], human_old_cohorts=[4], pig_method="weighted",
        drop_pct=30.0, pig_min_tpm=1.0, human_min_tpm=10.0, top_n=300,
        bootstrap_B=100, bootstrap_frac=0.8, restrict_ids=None, restrict_col="human",
        per_tissue={}, pig_pool={})


def main():
    wj = ROOT / "review/analyses/results/v2_uniform_winner.json"
    win = json.load(open(wj))
    print("Recomputing SPEARMAN bootstrap CIs (n=1000, seed=42):")
    for t in PANEL:
        cfg = baseline_cfg([t])
        merged, _ = build_merged(t, cfg)
        sub = select_genes(merged, cfg)
        x = sub["log2fc_pig"].values.astype(float)
        y = sub["log2fc_human"].values.astype(float)
        _, lo, hi = bootstrap_spearman_ci(x, y)
        old = win["per_tissue"][t]
        print(f"  {t:16s} r={old['r']:+.3f}  Pearson-CI [{old['ci_low']:+.3f},{old['ci_high']:+.3f}]"
              f"  ->  Spearman-CI [{lo:+.3f},{hi:+.3f}]")
        old["ci_low"], old["ci_high"] = lo, hi
        old["ci_method"] = "spearman_percentile_bootstrap_n1000_seed42"
    win["note"] = win.get("note", "") + (" | CI corrected to Spearman bootstrap (Bug-1 fix from "
                  "verification re-analysis; point estimates/p/perm-p/LOO unchanged).")
    wj.write_text(json.dumps(win, indent=2))
    print(f"Patched {wj}")

    # regenerate fig4 summary CIs from the patched winner + existing Loop C perm/LOO
    val = json.load(open(ROOT / "review/analyses/results/v2_loopC_validation.json"))
    stats_path = ROOT / "paper/figures/output/stats/fig4_v2_dgtex_summary.tsv"
    cols = ["tissue", "n_genes", "spearman_r", "ci_low", "ci_high", "p_value",
            "directional_concordance_pct", "perm_empirical_p", "loo_r_min", "loo_r_max", "tier"]
    lines = ["\t".join(cols)]
    for t in PANEL:
        p = win["per_tissue"][t]; v = val["per_tissue"].get(t, {})
        perm = v.get("perm_empirical_p")
        tier = "1" if (p["r"] >= 0.40 and perm is not None and perm < 0.05) else "2"
        lines.append("\t".join([t, str(p["n_genes"]), f"{p['r']:.4f}",
            f"{p['ci_low']:.4f}", f"{p['ci_high']:.4f}", f"{p['p']:.3e}", f"{p['dc']:.1f}",
            f"{perm:.4f}" if perm is not None else "",
            f"{v.get('loo_r_min'):.4f}" if v.get("loo_r_min") is not None else "",
            f"{v.get('loo_r_max'):.4f}" if v.get("loo_r_max") is not None else "", tier]))
    lines.append("\t".join(["PANEL_MEAN", "", f"{win['aggregate']['mean_r']:.4f}", "", "", "", "", "", "", "", ""]))
    stats_path.write_text("\n".join(lines) + "\n")
    print(f"Regenerated {stats_path}")


if __name__ == "__main__":
    main()
