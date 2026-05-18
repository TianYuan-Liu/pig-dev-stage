#!/usr/bin/env python3
"""Brain QC: re-run the Brain cross-species comparison after dropping
PigGTEx brain samples that show evidence of skeletal-muscle contamination.

Methods supported:
  marker          - Drop top --drop-pct of samples ranked by mean log2(TPM+1)
                    of canonical skeletal-muscle markers (5-gene panel from
                    the response letter: TNNC2, SMPX, MYBPHL, CHRNA1, LRRC2).
  extended_marker - As above but with a ~17-gene myogenic marker panel.
  nmf             - Fit NMF (k=8) on the pig brain log2(TPM+1) matrix,
                    identify the NMF component most correlated with the
                    5-gene marker score (blind to outcome), drop top
                    --drop-pct of samples by that component's loading.

Writes a JSON file in the same schema as
review/analyses/results/cross_species_all_tissues.json so the autoresearch
metric extractor can pick it up via --alltis-json.

Usage:
  python autoresearch/run_brain_qc.py --method marker --drop-pct 20 \
         --out review/analyses/results/cross_species_brain_qc_marker20.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import NMF
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "review" / "analyses"))

# Import helpers from the published cross-species script
import cross_species_all_tissues as xs  # noqa: E402

# ---------------------------------------------------------------------------
# Marker panels
# ---------------------------------------------------------------------------
MARKER_5 = ["TNNC2", "SMPX", "MYBPHL", "CHRNA1", "LRRC2"]
# Extended skeletal-muscle marker panel (sarcomeric/myogenic):
MARKER_EXT = [
    "TNNC2", "SMPX", "MYBPHL", "CHRNA1", "LRRC2",
    "MYH1", "MYH2", "MYH7", "ACTN3", "MB", "DES", "MYOG", "MYOD1",
    "CKM", "MYL1", "TNNT3", "TNNI2", "MYBPC1",
]
# Brain-identity markers (drop samples with LOW score = poor brain purity):
# pan-neuronal + glial canonical genes
BRAIN_MARKERS = [
    "NEFL", "NEFM", "NEFH", "MAP2", "SYN1", "SYP", "SNAP25", "STMN2",
    "GFAP", "OLIG2", "MBP", "PLP1", "VGAT", "VGLUT1", "GAD1",
    "SLC17A7", "RBFOX3", "TUBB3", "DLG4", "GRIN1",
]


def load_pig_symbol_map(cache: Path) -> dict[str, str]:
    """Invert pig_id_to_symbol.json to a dict symbol(upper) -> pig_id."""
    d = json.loads(cache.read_text())
    sym2id: dict[str, str] = {}
    for pid, sym in d.items():
        if not sym:
            continue
        s = sym.upper()
        # Take first occurrence; published ortholog table uses uniquely-mapped
        # symbols anyway, so collisions don't matter for the signature.
        sym2id.setdefault(s, pid)
    return sym2id


def compute_marker_score(expr: pd.DataFrame, marker_pig_ids: list[str]) -> pd.Series:
    """Per-sample mean log2(TPM+1) across the provided marker pig Ensembl IDs."""
    present = [g for g in marker_pig_ids if g in expr.index]
    if not present:
        raise SystemExit("ERROR: none of the marker gene IDs are in the expression matrix")
    sub = np.log2(expr.loc[present].astype(float).values + 1.0)
    return pd.Series(sub.mean(axis=0), index=expr.columns, name="marker_score")


def select_keep_by_marker(score: pd.Series, candidate: list[str], drop_pct: float,
                          per_group: dict[str, list[str]] | None = None
                          ) -> tuple[list[str], list[str]]:
    """Within `candidate` samples, keep those whose score is below the (100-drop_pct)
    percentile. If `per_group` is supplied (e.g. {'young': [...], 'old': [...]}),
    the percentile is computed within each group separately so that stage and
    contamination are not conflated.
    Returns (kept, dropped)."""
    if not candidate:
        return [], []
    if per_group is None:
        s = score.reindex(candidate).dropna()
        if s.empty:
            return list(candidate), []
        thresh = float(np.percentile(s.values, 100.0 - drop_pct))
        kept = s.index[s.values <= thresh].tolist()
        dropped = s.index[s.values > thresh].tolist()
        return kept, dropped

    kept_all, dropped_all = [], []
    for group_name, samples in per_group.items():
        sg = score.reindex(samples).dropna()
        if sg.empty:
            kept_all.extend(samples)
            continue
        thresh = float(np.percentile(sg.values, 100.0 - drop_pct))
        kept_all.extend(sg.index[sg.values <= thresh].tolist())
        dropped_all.extend(sg.index[sg.values > thresh].tolist())
    return kept_all, dropped_all


def select_keep_by_nmf(expr_log: pd.DataFrame, candidate: list[str], drop_pct: float,
                       n_components: int, marker_score: pd.Series, rng_seed: int = 42
                       ) -> tuple[list[str], list[str], int, float]:
    """Fit NMF on the candidate-subset of the (log-transformed) expression matrix,
    pick the component most correlated with the marker score, drop top drop_pct of
    samples by that component's loading."""
    sub = expr_log[candidate].astype(float).values
    sub = np.clip(sub, 0, None)  # NMF requires non-negative
    n_components = max(2, min(n_components, sub.shape[1] - 1, sub.shape[0] - 1))
    model = NMF(n_components=n_components, init="nndsvda", random_state=rng_seed,
                max_iter=500, tol=1e-3)
    W = model.fit_transform(sub)        # genes x k
    H = model.components_                # k x samples
    # Identify the component most correlated with the marker score
    ms = marker_score.reindex(candidate).fillna(0.0).values
    rs = []
    for k in range(H.shape[0]):
        r, _ = stats.pearsonr(H[k], ms)
        rs.append(r if np.isfinite(r) else 0.0)
    rs = np.array(rs)
    best_k = int(np.argmax(np.abs(rs)))
    best_r = float(rs[best_k])
    loadings = pd.Series(H[best_k], index=candidate)
    if best_r < 0:
        # Flip so high loading <=> high contamination
        loadings = -loadings
    thresh = float(np.percentile(loadings.values, 100.0 - drop_pct))
    kept = loadings.index[loadings.values <= thresh].tolist()
    dropped = loadings.index[loadings.values > thresh].tolist()
    return kept, dropped, best_k, best_r


def analyse_brain_filtered(ortho: pd.DataFrame, drop_pct: float, method: str,
                           n_components: int) -> dict:
    """Replicate xs.analyse_tissue('Brain', 'Brain', ...) but with sample filtering."""
    print(f"\n=== Brain QC ({method}, drop_pct={drop_pct}%) ===")

    # ---- HUMAN (unchanged) ----
    h_expr, h_young, h_old = xs.load_human_tissue(xs.HUMAN_RPKM, "Brain")
    print(f"  Human brain: young n={len(h_young)}, old n={len(h_old)}")
    h_fc = xs.compute_fc_pvals(h_expr, h_young, h_old, use_ttest=True)

    # ---- PIG: load and filter ----
    p_expr, p_young, p_old = xs.load_pig_tissue("Brain")
    candidate = sorted(set(p_young) | set(p_old))
    print(f"  Pig brain raw: young n={len(p_young)}, old n={len(p_old)} "
          f"(candidate union n={len(candidate)})")

    # Build marker score for *all* candidate samples
    pig_sym2id = load_pig_symbol_map(xs.PIG_SYM_CACHE)
    panel = (MARKER_EXT if method in ("extended_marker", "stratified_extended_marker")
             else MARKER_5)
    marker_ids = [pig_sym2id[s.upper()] for s in panel if s.upper() in pig_sym2id]
    missing = [s for s in panel if s.upper() not in pig_sym2id]
    if missing:
        print(f"  WARNING: marker genes not in pig symbol cache: {missing}")
    print(f"  Using {len(marker_ids)}/{len(panel)} marker genes for signature")
    marker_score = compute_marker_score(p_expr[candidate], marker_ids)

    nmf_info: dict = {}
    if method in ("marker", "extended_marker", "stratified_marker", "stratified_extended_marker"):
        per_group = ({"young": p_young, "old": p_old}
                     if method.startswith("stratified") else None)
        kept, dropped = select_keep_by_marker(marker_score, candidate, drop_pct,
                                              per_group=per_group)
    elif method == "random":
        # Permutation control: randomly drop drop_pct of samples within each stage.
        # If purity-based filtering r is meaningfully higher than the random
        # distribution, it implies real signal cleanup; if not, the gain is
        # just from subset reduction.
        rng = np.random.default_rng(int(drop_pct * 100))
        kept, dropped = [], []
        for samples in (p_young, p_old):
            n = len(samples)
            n_drop = int(round(n * drop_pct / 100.0))
            idx_drop = set(rng.choice(n, size=n_drop, replace=False))
            for i, s in enumerate(samples):
                if i in idx_drop:
                    dropped.append(s)
                else:
                    kept.append(s)
    elif method in ("stratified_purity", "stratified_purity_ratio"):
        # Compute brain-purity score (mean log2(TPM+1) over canonical brain markers)
        brain_ids = [pig_sym2id[s.upper()] for s in BRAIN_MARKERS
                     if s.upper() in pig_sym2id]
        if not brain_ids:
            raise SystemExit("No brain marker genes found in pig symbol cache")
        purity = compute_marker_score(p_expr[candidate], brain_ids)
        if method == "stratified_purity_ratio":
            # Drop by muscle_marker / brain_marker ratio (high = contamination)
            score = marker_score / purity.replace(0, np.nan)
            score = score.fillna(score.median())
            print(f"  Using muscle/brain ratio score for filtering")
        else:
            # Drop bottom drop_pct by brain purity (low purity = bad sample)
            # We achieve this by negating purity then using the same top-drop logic
            score = -purity
            print(f"  Using brain-purity score for filtering (drop LOW-purity)")
        per_group = {"young": p_young, "old": p_old}
        kept, dropped = select_keep_by_marker(score, candidate, drop_pct,
                                              per_group=per_group)
    elif method == "nmf":
        log_expr = np.log2(p_expr[candidate].astype(float) + 1.0)
        # Reduce to a manageable gene set for NMF (top 5,000 by variance)
        var = log_expr.var(axis=1)
        top_genes = var.sort_values(ascending=False).head(5000).index
        kept, dropped, best_k, best_r = select_keep_by_nmf(
            log_expr.loc[top_genes], candidate, drop_pct,
            n_components=n_components, marker_score=marker_score)
        nmf_info = {"nmf_components": n_components, "best_component": best_k,
                    "best_pearson_to_marker_score": best_r, "n_genes_used": int(len(top_genes))}
    else:
        raise SystemExit(f"unknown method: {method}")

    p_young_f = [s for s in p_young if s in kept]
    p_old_f = [s for s in p_old if s in kept]
    print(f"  After QC: young n={len(p_young_f)}, old n={len(p_old_f)} "
          f"(dropped {len(dropped)} of {len(candidate)})")

    # Dropped per-class breakdown for transparency
    dropped_young = [s for s in dropped if s in p_young]
    dropped_old = [s for s in dropped if s in p_old]
    print(f"    of which young dropped: {len(dropped_young)}, old dropped: {len(dropped_old)}")

    if len(p_young_f) < 3 or len(p_old_f) < 3:
        return {"tissue": "Brain", "skipped": "insufficient_after_filter",
                "drop_pct": drop_pct, "method": method,
                "n_dropped": len(dropped)}

    p_fc = xs.compute_fc_pvals(p_expr, p_young_f, p_old_f, use_ttest=False)

    # ---- Match orthologs and compute strict / pig-anchored ----
    o = ortho[ortho["pig_gene_id"].isin(p_fc.index) &
              ortho["human_gene_id"].isin(h_fc.index)].copy()
    print(f"  One-to-one orthologs in both matrices: {len(o)}")

    rows = []
    for _, r in o.iterrows():
        rows.append({
            "gene_symbol": r["symbol"],
            "pig_gene_id": r["pig_gene_id"],
            "human_gene_id": r["human_gene_id"],
            "log2fc_pig": p_fc.loc[r["pig_gene_id"], "log2fc"],
            "log2fc_human": h_fc.loc[r["human_gene_id"], "log2fc"],
            "p_pig": p_fc.loc[r["pig_gene_id"], "pvalue"],
            "p_human": h_fc.loc[r["human_gene_id"], "pvalue"],
        })
    merged = pd.DataFrame(rows)
    merged = xs.add_fdr(merged, "p_pig", "fdr_pig")
    merged = xs.add_fdr(merged, "p_human", "fdr_human")

    strict = merged[
        (merged["fdr_pig"] < xs.FDR_THRESHOLD) &
        (merged["fdr_human"] < xs.FDR_THRESHOLD) &
        (merged["log2fc_pig"].abs() > xs.FC_THRESHOLD) &
        (merged["log2fc_human"].abs() > xs.FC_THRESHOLD)
    ].copy()
    pig_anchored = merged[
        (merged["fdr_pig"] < xs.FDR_THRESHOLD) &
        (merged["log2fc_pig"].abs() > xs.FC_THRESHOLD) &
        (merged["log2fc_human"].abs() > xs.FC_THRESHOLD)
    ].copy()

    def corr_stats(df_: pd.DataFrame, label: str) -> dict:
        if len(df_) < 5:
            return {"label": label, "n_genes": int(len(df_)),
                    "pearson_r": None, "pearson_p": None,
                    "ci_low": None, "ci_high": None,
                    "directional_concordance": None,
                    "n_concordant": int((np.sign(df_["log2fc_pig"]) ==
                                         np.sign(df_["log2fc_human"])).sum()) if len(df_) > 0 else 0}
        x = df_["log2fc_pig"].values
        y = df_["log2fc_human"].values
        r, p = stats.pearsonr(x, y)
        nc = int((np.sign(x) == np.sign(y)).sum())
        pct = 100.0 * nc / len(df_)
        _, ci_low, ci_high = xs.bootstrap_pearson_ci(x, y)
        return {"label": label, "n_genes": int(len(df_)),
                "pearson_r": float(r), "pearson_p": float(p),
                "ci_low": float(ci_low) if not np.isnan(ci_low) else None,
                "ci_high": float(ci_high) if not np.isnan(ci_high) else None,
                "directional_concordance": float(pct),
                "n_concordant": nc}

    s_stats = corr_stats(strict, "strict_FDR_both")
    p_stats = corr_stats(pig_anchored, "pig_anchored")
    print(f"    strict        : n={s_stats['n_genes']}, r={s_stats['pearson_r']}")
    print(f"    pig-anchored  : n={p_stats['n_genes']}, r={p_stats['pearson_r']}")
    return {
        "tissue": "Brain",
        "pig_tissue": "Brain",
        "qc_method": method,
        "qc_drop_pct": drop_pct,
        "qc_dropped_total": len(dropped),
        "qc_dropped_young": len(dropped_young),
        "qc_dropped_old": len(dropped_old),
        "qc_nmf": nmf_info,
        "n_orthologs_tested": int(len(merged)),
        "sample_sizes": {
            "pig_young": len(p_young_f), "pig_old": len(p_old_f),
            "human_young": len(h_young), "human_old": len(h_old),
        },
        "human_young_stages": sorted(xs.HUMAN_YOUNG),
        "human_old_stages": sorted(xs.HUMAN_OLD),
        "pig_young_stages": sorted(xs.PIG_YOUNG),
        "pig_old_stages": sorted(xs.PIG_OLD),
        "strict": s_stats,
        "pig_anchored": p_stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method",
                    choices=["marker", "extended_marker", "nmf",
                             "stratified_marker", "stratified_extended_marker",
                             "stratified_purity", "stratified_purity_ratio",
                             "random"],
                    default="marker")
    ap.add_argument("--drop-pct", type=float, default=20.0)
    ap.add_argument("--nmf-components", type=int, default=8)
    ap.add_argument("--out", type=str, required=True,
                    help="Output JSON path (drop-in replacement for cross_species_all_tissues.json)")
    args = ap.parse_args()

    ortho = xs.build_one_to_one_orthologs()
    brain = analyse_brain_filtered(ortho, args.drop_pct, args.method, args.nmf_components)

    # Read the published all-tissues JSON to copy over the Liver entry unchanged.
    base = json.loads(xs.JSON_OUT.read_text())
    base["per_tissue"]["Brain"] = brain
    base["analysis"] = f"cross_species_brain_qc_{args.method}_drop{int(args.drop_pct)}"
    base["brain_qc"] = {
        "method": args.method,
        "drop_pct": args.drop_pct,
        "marker_panel": MARKER_5 if args.method != "extended_marker" else MARKER_EXT,
        "n_dropped": brain.get("qc_dropped_total"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(base, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
