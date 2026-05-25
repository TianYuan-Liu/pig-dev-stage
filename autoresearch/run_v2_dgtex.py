#!/usr/bin/env python3
"""Per-cell cross-species runner under dGTEx-only v2 protocol.

For one tissue (selected via --cell-index N), apply the uniform pig-anchored
protocol against dGTEx as the sole human reference:

  1. Load pig TPM, apply v2 binning (Infant+EarlyChildhood vs Post-pubertal+Adult)
  2. Load dGTEx TPM for the same tissue, apply v2 binning (cohort 1 vs cohort 4)
  3. Restrict to validated 1:1 pig↔human orthologs
  4. Weighted purity-z log2FC computation (uniform method from previous winner)
  5. Stratified marker-purity drop 30% per stage
  6. Expression filter: pig median TPM ≥ 1, human median TPM ≥ 10
  7. Pig-anchored gene filter: fdr_pig < 0.10 AND |log2fc| > 0.5 (both species)
  8. Pearson r + 95% bootstrap CI + directional concordance
  9. 1000-permutation ortholog-pair null distribution
 10. Marker-gene sanity check using existing TISSUE_MARKERS panel
 11. Save per-cell JSON to autoresearch/v2_dgtex_results/{tissue_slug}.json

Usage:
  python -m autoresearch.run_v2_dgtex --cell-index 0   # Muscle
  python -m autoresearch.run_v2_dgtex --cell-index 7   # Heart
  python -m autoresearch.run_v2_dgtex --all            # Loop over all 8
"""

from __future__ import annotations

import argparse
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

from autoresearch.run_uniform_cross_species import (  # noqa: E402
    apply_uniform_drop,
    compute_pig_fc,
)
from autoresearch.run_weighted_fc import median_fc  # noqa: E402
from autoresearch.run_purity_generic import TISSUE_MARKERS, load_pig_symbol_map  # noqa: E402
from autoresearch.stage_mapping_v2 import (  # noqa: E402
    INCLUDED_TISSUES,
    N_MIN_SAMPLES_PER_BIN,
    PIG_OLD_STAGES,
    PIG_YOUNG_STAGES,
)
from autoresearch.load_dgtex import load_dgtex_tissue  # noqa: E402

OUT_DIR = ROOT / "autoresearch" / "v2_dgtex_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIFORM_METHOD = "weighted"
MIN_PIG_TPM = 1.0
MIN_HUMAN_TPM = 10.0
DROP_PCT = 30.0
N_PERMUTATIONS = 1000
SEED = 42


# Override the cross_species_all_tissues module's stage globals to match v2
xs.PIG_YOUNG = set(PIG_YOUNG_STAGES)
xs.PIG_OLD = set(PIG_OLD_STAGES)


def compute_dgtex_fc(
    h_expr: pd.DataFrame, h_young: list[str], h_old: list[str]
) -> pd.DataFrame:
    """Human FC: median across samples (small dGTEx n; weighted not justified)."""
    return median_fc(h_expr, h_young, h_old)


def permutation_test(
    pig_fc: np.ndarray,
    human_fc: np.ndarray,
    pig_fdr: np.ndarray,
    n_perm: int = N_PERMUTATIONS,
    seed: int = SEED,
) -> dict:
    """Ortholog-pair permutation: randomly re-pair pig and human log2FC vectors,
    re-apply pig-anchored filter, recompute Pearson r. Returns empirical p."""
    rng = np.random.default_rng(seed)
    mask = (
        (pig_fdr < xs.FDR_THRESHOLD)
        & (np.abs(pig_fc) > xs.FC_THRESHOLD)
        & (np.abs(human_fc) > xs.FC_THRESHOLD)
    )
    n_obs = int(mask.sum())
    if n_obs < 5:
        return {"empirical_p": None, "n_perm": n_perm, "n_obs": n_obs}
    r_obs = stats.pearsonr(pig_fc[mask], human_fc[mask])[0]

    null_rs = []
    for _ in range(n_perm):
        perm = rng.permutation(len(human_fc))
        hum_shuf = human_fc[perm]
        m = (
            (pig_fdr < xs.FDR_THRESHOLD)
            & (np.abs(pig_fc) > xs.FC_THRESHOLD)
            & (np.abs(hum_shuf) > xs.FC_THRESHOLD)
        )
        if m.sum() < 5:
            null_rs.append(0.0)
            continue
        null_rs.append(stats.pearsonr(pig_fc[m], hum_shuf[m])[0])
    null_arr = np.asarray(null_rs)
    n_above = int((null_arr >= r_obs).sum())
    return {
        "observed_r": float(r_obs),
        "n_obs": n_obs,
        "n_perm": n_perm,
        "null_mean": float(null_arr.mean()),
        "null_std": float(null_arr.std()),
        "null_p99": float(np.percentile(null_arr, 99)),
        "n_above": n_above,
        "empirical_p": (n_above + 1) / (n_perm + 1),
    }


def marker_sanity(
    merged: pd.DataFrame, pa_set: set[str], pig_tissue: str
) -> dict:
    """Check canonical markers from TISSUE_MARKERS panel: how many appear in
    the pig-anchored set, and how many are directionally concordant."""
    panel = TISSUE_MARKERS.get(pig_tissue, [])
    if not panel:
        return {
            "markers_in_panel": 0,
            "markers_in_set": 0,
            "markers_concordant": 0,
            "marker_hits": [],
        }
    pa_genes = merged[merged["pig_gene_id"].isin(pa_set)]
    panel_upper = {p.upper() for p in panel}
    sym2id = load_pig_symbol_map(xs.PIG_SYM_CACHE)
    panel_pig_ids = {sym2id[s] for s in panel_upper if s in sym2id}
    hits_df = pa_genes[pa_genes["pig_gene_id"].isin(panel_pig_ids)]
    hits = []
    for _, r in hits_df.iterrows():
        hits.append({
            "gene_symbol": r["gene_symbol"],
            "log2fc_pig": float(r["log2fc_pig"]),
            "log2fc_human": float(r["log2fc_human"]),
            "concordant": bool(np.sign(r["log2fc_pig"]) == np.sign(r["log2fc_human"])),
        })
    return {
        "markers_in_panel": len(panel),
        "markers_in_set": int(len(hits)),
        "markers_concordant": int(sum(h["concordant"] for h in hits)),
        "marker_hits": hits,
    }


def analyse_tissue_dgtex(
    dgtex_smts: str, pig_tissue: str, tissue_file_name: str, ortho: pd.DataFrame
) -> dict:
    """Run uniform pig-anchored protocol on one (pig, dGTEx) tissue pair."""
    print(f"\n--- {dgtex_smts} (pig={pig_tissue}, dgtex_file={tissue_file_name}) ---")

    # 1. Pig side
    p_expr, p_young, p_old = xs.load_pig_tissue(pig_tissue)
    print(f"  pig initial: young={len(p_young)} old={len(p_old)}  (matrix {p_expr.shape})")
    if len(p_young) < N_MIN_SAMPLES_PER_BIN or len(p_old) < N_MIN_SAMPLES_PER_BIN:
        return {
            "dgtex_tissue": dgtex_smts,
            "skipped": "pig_low_n_initial",
            "n_pig_young": len(p_young),
            "n_pig_old": len(p_old),
        }
    keep = p_expr.median(axis=1) >= MIN_PIG_TPM
    p_expr = p_expr.loc[keep]
    p_young, p_old, qc = apply_uniform_drop(p_expr, p_young, p_old, pig_tissue, DROP_PCT)
    if len(p_young) < N_MIN_SAMPLES_PER_BIN or len(p_old) < N_MIN_SAMPLES_PER_BIN:
        return {"dgtex_tissue": dgtex_smts, "skipped": "pig_low_n_after_drop", "qc": qc}
    p_fc = compute_pig_fc(p_expr, p_young, p_old, UNIFORM_METHOD, pig_tissue)
    print(f"  pig after QC: young={len(p_young)} old={len(p_old)}  genes={len(p_fc)}")

    # 2. dGTEx side
    h_expr, h_young, h_old = load_dgtex_tissue(dgtex_smts, tissue_file_name)
    print(f"  dGTEx initial: young={len(h_young)} old={len(h_old)}  (matrix {h_expr.shape})")
    if len(h_young) < N_MIN_SAMPLES_PER_BIN or len(h_old) < N_MIN_SAMPLES_PER_BIN:
        return {
            "dgtex_tissue": dgtex_smts,
            "skipped": "human_low_n",
            "n_human_young": len(h_young),
            "n_human_old": len(h_old),
        }
    if MIN_HUMAN_TPM > 0:
        h_expr = h_expr.loc[h_expr.median(axis=1) >= MIN_HUMAN_TPM]
    h_fc = compute_dgtex_fc(h_expr, h_young, h_old)
    print(f"  dGTEx genes after expression filter: {len(h_fc)}")

    # 3. Cross-species merge on orthologs
    o_ = ortho[
        ortho["pig_gene_id"].isin(p_fc.index)
        & ortho["human_gene_id"].isin(h_fc.index)
    ]
    rows = []
    for _, r in o_.iterrows():
        rows.append({
            "gene_symbol":   r["symbol"],
            "pig_gene_id":   r["pig_gene_id"],
            "human_gene_id": r["human_gene_id"],
            "log2fc_pig":    p_fc.loc[r["pig_gene_id"], "log2fc"],
            "log2fc_human":  h_fc.loc[r["human_gene_id"], "log2fc"],
            "p_pig":         p_fc.loc[r["pig_gene_id"], "pvalue"],
            "p_human":       h_fc.loc[r["human_gene_id"], "pvalue"],
        })
    merged = pd.DataFrame(rows)
    if len(merged) < 50:
        return {
            "dgtex_tissue": dgtex_smts,
            "skipped": "too_few_orthologs",
            "n_orthologs_tested": len(merged),
        }
    merged = xs.add_fdr(merged, "p_pig", "fdr_pig")
    merged = xs.add_fdr(merged, "p_human", "fdr_human")

    # 4. Pig-anchored filter
    pa = merged[
        (merged["fdr_pig"] < xs.FDR_THRESHOLD)
        & (merged["log2fc_pig"].abs() > xs.FC_THRESHOLD)
        & (merged["log2fc_human"].abs() > xs.FC_THRESHOLD)
    ].copy()

    n_pa = int(len(pa))
    if n_pa < 5:
        pa_stats = {"n_genes": n_pa, "pearson_r": None, "directional_concordance": None}
        perm = {"empirical_p": None, "n_obs": n_pa}
        marker = {
            "markers_in_panel": 0,
            "markers_in_set": 0,
            "markers_concordant": 0,
            "marker_hits": [],
        }
    else:
        x = pa["log2fc_pig"].values
        y = pa["log2fc_human"].values
        r, p_pearson = stats.pearsonr(x, y)
        rs, p_sp = stats.spearmanr(x, y)
        dc = 100.0 * (np.sign(x) == np.sign(y)).sum() / n_pa
        _, ci_low, ci_high = xs.bootstrap_pearson_ci(x, y)
        pa_stats = {
            "n_genes": n_pa,
            "pearson_r": float(r),
            "pearson_p": float(p_pearson),
            "spearman_r": float(rs),
            "spearman_p": float(p_sp),
            "ci_low": float(ci_low) if not np.isnan(ci_low) else None,
            "ci_high": float(ci_high) if not np.isnan(ci_high) else None,
            "directional_concordance": float(dc),
        }
        perm = permutation_test(
            merged["log2fc_pig"].values,
            merged["log2fc_human"].values,
            merged["fdr_pig"].values,
        )
        marker = marker_sanity(merged, set(pa["pig_gene_id"]), pig_tissue)

    print(
        f"  RESULT: n={pa_stats.get('n_genes')}  "
        f"r={pa_stats.get('pearson_r')}  dc={pa_stats.get('directional_concordance')}%  "
        f"perm p={perm.get('empirical_p')}"
    )

    return {
        "dgtex_tissue": dgtex_smts,
        "pig_tissue": pig_tissue,
        "config": {
            "method": UNIFORM_METHOD,
            "min_pig_tpm": MIN_PIG_TPM,
            "min_human_tpm": MIN_HUMAN_TPM,
            "drop_pct": DROP_PCT,
            "human_reference": "dGTEx_v1_2026-01-30",
            "human_young_cohort": "AGECOHORT=1 (Infant 0-2y)",
            "human_old_cohort": "AGECOHORT=4 (Post-pubertal 13-18y)",
        },
        "qc": qc,
        "sample_sizes": {
            "pig_young": len(p_young),
            "pig_old": len(p_old),
            "human_young": len(h_young),
            "human_old": len(h_old),
        },
        "n_orthologs_tested": int(len(merged)),
        "pig_anchored": pa_stats,
        "permutation": perm,
        "marker_sanity": marker,
        "_gene_table": pa[
            [
                "gene_symbol", "pig_gene_id", "human_gene_id",
                "log2fc_pig", "log2fc_human",
                "p_pig", "p_human", "fdr_pig", "fdr_human",
            ]
        ].to_dict("records") if n_pa < 5000 else None,
    }


def tissue_slug(dgtex_smts: str) -> str:
    return dgtex_smts.lower().replace(" ", "_")


def run_one(cell_index: int) -> dict:
    dgtex_smts, pig_tissue, tissue_file_name = INCLUDED_TISSUES[cell_index]
    ortho = xs.build_one_to_one_orthologs()
    result = analyse_tissue_dgtex(dgtex_smts, pig_tissue, tissue_file_name, ortho)
    out_path = OUT_DIR / f"{tissue_slug(dgtex_smts)}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"  wrote {out_path}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--cell-index", type=int, help="Tissue index 0..7")
    group.add_argument("--all", action="store_true", help="Loop over all 8 tissues")
    args = ap.parse_args()
    if args.all:
        for i in range(len(INCLUDED_TISSUES)):
            run_one(i)
    else:
        if not 0 <= args.cell_index < len(INCLUDED_TISSUES):
            ap.error(f"--cell-index must be in 0..{len(INCLUDED_TISSUES)-1}")
        run_one(args.cell_index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
