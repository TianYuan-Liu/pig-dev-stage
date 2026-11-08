#!/usr/bin/env python3
"""K-sensitivity + rank-product analysis for joint cross-species features.

Reads per-seed full-importance parquet files (seed_importance_{pig,human}_{tissue}.parquet)
written by pipeline.py and computes:
  - Joint stable sets at K = 50, 100, 200, 500, 1000 with threshold 8/10
  - Pearson r + 1000-bootstrap 95% CI of log2 FC on each joint set
  - Rank-product joint score (alternative criterion that doesn't require
    membership in BOTH top-K stable sets)

Writes:
  sensitivity_K.json                  per-tissue, per-K joint metrics
  fig_sensitivity_K.pdf               joint size vs K curve, r vs K curve
  joint_features_{tissue}_K{K}.csv    joint set per K per tissue (sensible K values)
  joint_rank_product_{tissue}.csv     alternative joint set via rank product
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

OUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUT_DIR.parent.parent.parent
ORTHO_CSV = (PROJECT_ROOT / "review" / "analyses" / "results"
             / "pig_human_one_to_one_orthologs.csv")

TISSUES = ["Muscle", "Brain", "Liver", "Lung", "Blood"]
K_VALUES = [50, 100, 200, 500, 1000]
STABILITY_THRESHOLD = 8  # >= 8/10 seeds
N_BOOTSTRAP = 1000
PSEUDO = 0.01

TISSUE_COLOR = {
    "Muscle": "#C97A9A", "Brain":  "#5B8FF9", "Liver":  "#E0A458",
    "Lung":   "#82D173", "Blood":  "#B45355",
}
PIG_YOUNG = {"Infant", "Early childhood"}
PIG_OLD = {"Post-pubertal", "Adult"}


def load_ortho() -> pd.DataFrame:
    df = pd.read_csv(ORTHO_CSV)
    df = df[df["is_one_to_one"]] if "is_one_to_one" in df.columns else df
    df["symbol"] = df["symbol"].str.upper()
    return df[["symbol", "pig_gene_id", "human_gene_id"]]


def stable_at_k(seed_imp: pd.DataFrame, k: int,
                threshold: int) -> Tuple[List[str], pd.Series, pd.Series]:
    """seed_imp: rows = genes, columns = seed_xx. Returns stable genes (in top-K
    of >= threshold seeds), per-gene seed count, mean rank across seeds where
    top-K.

    Important: features tied at gain=0 are NOT counted as top-K. This is
    because LightGBM assigns gain=0 to features it never used; many
    features share gain=0 (sometimes thousands), and treating those ties
    as top-ranked would massively inflate the stable set. We define top-K
    via nlargest, which gives a strict ordering within ties and never
    expands past K members.
    """
    n_seeds = pd.Series(0, index=seed_imp.index, dtype=int)
    rank_records: Dict[str, List[int]] = defaultdict(list)
    for col in seed_imp.columns:
        s = seed_imp[col]
        # Exclude exact zeros from top-K eligibility (never-used features)
        nonzero = s[s > 0]
        top = nonzero.nlargest(k)
        for rank, gid in enumerate(top.index, start=1):
            n_seeds[gid] += 1
            rank_records[gid].append(rank)
    mean_rank = pd.Series(
        {g: float(np.mean(r)) for g, r in rank_records.items()},
        dtype=float,
    )
    stable = n_seeds[n_seeds >= threshold].index.tolist()
    return stable, n_seeds, mean_rank


def fold_change_from_cache(tissue: str, species: str,
                           ortho: pd.DataFrame) -> Optional[pd.Series]:
    """Compute log2FC (Adult/Infant) on cached raw matrix."""
    cache_dir = OUT_DIR / "cache"
    if species == "pig":
        raw_p = cache_dir / f"pig_{tissue}_rawTPM.parquet"
        y_csv = cache_dir / f"pig_{tissue}_y.csv"
        if not raw_p.exists():
            return None
        raw = pd.read_parquet(raw_p)
        y = pd.read_csv(y_csv, index_col=0).iloc[:, 0]
        young = y[y.isin(PIG_YOUNG)].index
        old = y[y.isin(PIG_OLD)].index
    else:
        # Try Cardoso path then GSE muscle
        if tissue == "Muscle":
            raw_p = cache_dir / "human_muscle_raw.parquet"
            y_csv = cache_dir / "human_muscle_y.csv"
        else:
            raw_p = cache_dir / f"human_cardoso_{tissue}_raw.parquet"
            y_csv = cache_dir / f"human_cardoso_{tissue}_y.csv"
        if not raw_p.exists():
            return None
        raw = pd.read_parquet(raw_p)
        y = pd.read_csv(y_csv, index_col=0).iloc[:, 0]
        young = y[y == 0].index
        old = y[y == 1].index
    if len(young) == 0 or len(old) == 0:
        return None
    yvals = raw.loc[young].mean(axis=0)
    ovals = raw.loc[old].mean(axis=0)
    fc = np.log2((ovals + PSEUDO) / (yvals + PSEUDO))
    return fc


def bootstrap_pearson(x: np.ndarray, y: np.ndarray,
                      n: int = N_BOOTSTRAP, seed: int = 42
                      ) -> Tuple[float, float, float]:
    if len(x) < 5:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        xs, ys = x[idx], y[idx]
        if np.std(xs) > 0 and np.std(ys) > 0:
            r, _ = stats.pearsonr(xs, ys)
            if not np.isnan(r):
                rs.append(r)
    rs = np.array(rs)
    return (float(np.mean(rs)),
            float(np.percentile(rs, 2.5)),
            float(np.percentile(rs, 97.5)))


def analyse_tissue(tissue: str, ortho: pd.DataFrame) -> dict:
    pig_p = OUT_DIR / f"seed_importance_pig_{tissue}.parquet"
    human_p = OUT_DIR / f"seed_importance_human_{tissue}.parquet"
    if not pig_p.exists():
        return {"tissue": tissue, "error": "pig importance file missing"}
    pig_seed = pd.read_parquet(pig_p)
    human_avail = human_p.exists()
    human_seed = pd.read_parquet(human_p) if human_avail else None

    pig_fc = fold_change_from_cache(tissue, "pig", ortho)
    human_fc = fold_change_from_cache(tissue, "human", ortho) if human_avail else None

    result = {
        "tissue": tissue,
        "human_available": human_avail,
        "k_sensitivity": [],
        "rank_product": None,
    }

    pig_to_sym = dict(zip(ortho["pig_gene_id"], ortho["symbol"]))
    human_to_sym = dict(zip(ortho["human_gene_id"], ortho["symbol"]))
    pid_to_hid = dict(zip(ortho["pig_gene_id"], ortho["human_gene_id"]))
    hid_to_pid = dict(zip(ortho["human_gene_id"], ortho["pig_gene_id"]))

    for k in K_VALUES:
        pig_stable, pig_nseed, pig_mr = stable_at_k(
            pig_seed, k, STABILITY_THRESHOLD)
        entry = {
            "k": k,
            "pig_stable_n": len(pig_stable),
            "human_stable_n": None,
            "joint_n": 0,
            "pearson_r": None,
            "ci_low": None,
            "ci_high": None,
            "directional_concordance_pct": None,
        }
        if human_avail:
            human_stable, human_nseed, human_mr = stable_at_k(
                human_seed, k, STABILITY_THRESHOLD)
            entry["human_stable_n"] = len(human_stable)
            # Intersection via ortholog mapping
            pig_set = set(pig_stable)
            human_set = set(human_stable)
            joint_pairs = [
                (pid, pid_to_hid[pid]) for pid in pig_set
                if pid_to_hid.get(pid) in human_set
            ]
            entry["joint_n"] = len(joint_pairs)
            if joint_pairs and pig_fc is not None and human_fc is not None:
                pids = [p for p, _ in joint_pairs]
                hids = [h for _, h in joint_pairs]
                x = pig_fc.reindex(pids).values
                y_v = human_fc.reindex(hids).values
                ok = np.isfinite(x) & np.isfinite(y_v)
                x, y_v = x[ok], y_v[ok]
                if len(x) >= 5 and np.std(x) > 0 and np.std(y_v) > 0:
                    r, _ = stats.pearsonr(x, y_v)
                    r_mean, ci_lo, ci_hi = bootstrap_pearson(x, y_v)
                    pct = 100.0 * np.sum(np.sign(x) == np.sign(y_v)) / len(x)
                    entry["pearson_r"] = float(r)
                    entry["ci_low"] = float(ci_lo)
                    entry["ci_high"] = float(ci_hi)
                    entry["directional_concordance_pct"] = float(pct)
        result["k_sensitivity"].append(entry)

        # Save joint CSV at selected K values
        if k in (100, 200, 500, 1000) and human_avail and entry["joint_n"] > 0:
            rows = []
            pig_stable, _, pig_mr = stable_at_k(pig_seed, k, STABILITY_THRESHOLD)
            human_stable, _, human_mr = stable_at_k(human_seed, k, STABILITY_THRESHOLD)
            for pid in pig_stable:
                hid = pid_to_hid.get(pid)
                if hid in set(human_stable):
                    sym = pig_to_sym.get(pid) or human_to_sym.get(hid) or ""
                    rows.append({
                        "symbol": sym,
                        "pig_gene_id": pid,
                        "human_gene_id": hid,
                        "pig_mean_rank": float(pig_mr.get(pid, np.nan)),
                        "human_mean_rank": float(human_mr.get(hid, np.nan)),
                        "log2fc_pig": float(pig_fc.get(pid, np.nan))
                                      if pig_fc is not None else None,
                        "log2fc_human": float(human_fc.get(hid, np.nan))
                                        if human_fc is not None else None,
                    })
            df = pd.DataFrame(rows).sort_values(
                ["pig_mean_rank", "human_mean_rank"]
            ).reset_index(drop=True)
            df.to_csv(OUT_DIR / f"joint_features_{tissue}_K{k}.csv", index=False)

    # ---- Rank-product joint score
    if human_avail:
        # Per-seed rank, but features with gain=0 (never used) get a sentinel
        # rank of n_features + 1 instead of being tied at top via method='min'.
        def _per_seed_rank(df: pd.DataFrame) -> pd.DataFrame:
            n_feat = df.shape[0]
            out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
            for col in df.columns:
                s = df[col]
                nz = s[s > 0]
                # assign rank by gain descending, ties broken by first-seen
                ranked_nz = nz.sort_values(ascending=False).rank(
                    method="first").astype(int)
                # invert: largest gain → rank 1
                ranks = pd.Series(
                    {g: int(len(nz) - r + 1) for g, r in ranked_nz.items()},
                    dtype=float,
                )
                # zero-gain features get sentinel rank
                ranks = ranks.reindex(df.index).fillna(n_feat + 1)
                out[col] = ranks
            return out

        pig_rank = _per_seed_rank(pig_seed)
        human_rank = _per_seed_rank(human_seed)
        pig_mean_rank = pig_rank.mean(axis=1)
        human_mean_rank = human_rank.mean(axis=1)

        # Map to ortholog-paired DataFrame
        rows = []
        for pid, hid in zip(ortho["pig_gene_id"], ortho["human_gene_id"]):
            if pid in pig_mean_rank.index and hid in human_mean_rank.index:
                pr = pig_mean_rank[pid]
                hr = human_mean_rank[hid]
                # Rank-product (lower is better)
                rp = np.sqrt(pr * hr)
                rows.append({
                    "pig_gene_id": pid, "human_gene_id": hid,
                    "symbol": pig_to_sym.get(pid) or human_to_sym.get(hid) or "",
                    "pig_mean_rank": float(pr),
                    "human_mean_rank": float(hr),
                    "rank_product": float(rp),
                })
        rp_df = pd.DataFrame(rows).sort_values("rank_product").reset_index(drop=True)
        # log2FC on top of rank-product list
        if pig_fc is not None and human_fc is not None:
            rp_df["log2fc_pig"] = rp_df["pig_gene_id"].map(pig_fc)
            rp_df["log2fc_human"] = rp_df["human_gene_id"].map(human_fc)
        rp_df.to_csv(OUT_DIR / f"joint_rank_product_{tissue}.csv", index=False)

        # Compute r on top 100 by rank-product
        top100 = rp_df.head(100)
        if "log2fc_pig" in top100.columns:
            x = top100["log2fc_pig"].values
            y_v = top100["log2fc_human"].values
            ok = np.isfinite(x) & np.isfinite(y_v)
            x, y_v = x[ok], y_v[ok]
            if len(x) >= 5 and np.std(x) > 0 and np.std(y_v) > 0:
                r, _ = stats.pearsonr(x, y_v)
                r_mean, ci_lo, ci_hi = bootstrap_pearson(x, y_v)
                pct = 100.0 * np.sum(np.sign(x) == np.sign(y_v)) / len(x)
                result["rank_product"] = {
                    "top_n_used": int(len(x)),
                    "pearson_r": float(r),
                    "ci_low": float(ci_lo),
                    "ci_high": float(ci_hi),
                    "directional_concordance_pct": float(pct),
                }
    return result


def fig_sensitivity(all_results: List[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    for r in all_results:
        if "k_sensitivity" not in r:
            continue
        t = r["tissue"]
        col = TISSUE_COLOR.get(t, "#888")
        ks = [e["k"] for e in r["k_sensitivity"]]
        ns = [e["joint_n"] for e in r["k_sensitivity"]]
        if r["human_available"]:
            ax.plot(ks, ns, marker="o", color=col, label=t, lw=2)
        else:
            ax.scatter([], [], color=col, label=f"{t} (no human)")
    ax.set_xscale("log")
    ax.set_xlabel("Top-K stability set size")
    ax.set_ylabel("Joint conserved gene set size")
    ax.set_title("Joint set size vs K", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)

    ax = axes[1]
    for r in all_results:
        if "k_sensitivity" not in r or not r["human_available"]:
            continue
        t = r["tissue"]
        col = TISSUE_COLOR.get(t, "#888")
        ks = [e["k"] for e in r["k_sensitivity"] if e["pearson_r"] is not None]
        rs = [e["pearson_r"] for e in r["k_sensitivity"] if e["pearson_r"] is not None]
        if ks:
            ax.plot(ks, rs, marker="o", color=col, label=t, lw=2)
        # rank-product reference
        rp = r.get("rank_product")
        if rp and rp.get("pearson_r") is not None:
            ax.axhline(rp["pearson_r"], color=col, ls=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Top-K stability set size")
    ax.set_ylabel(r"Pearson $r$ on joint set (log$_2$FC)")
    ax.set_title("Concordance vs K", fontweight="bold")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_ylim(-0.5, 1.0)
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    out = OUT_DIR / "fig_sensitivity_K.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def main() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "figure.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    ortho = load_ortho()
    all_results = []
    for t in TISSUES:
        try:
            r = analyse_tissue(t, ortho)
        except Exception as exc:
            r = {"tissue": t, "error": str(exc)}
        all_results.append(r)

    out_json = OUT_DIR / "sensitivity_K.json"
    out_json.write_text(json.dumps({"tissues": all_results}, indent=2,
                                   default=str))
    print(f"  {out_json.name}")
    # Pretty summary
    print("\nSummary (joint_n / r) by K:")
    print(f"{'tissue':10s} " + "  ".join(f"K={k:<5d}" for k in K_VALUES) +
          "    rank-prod r")
    for r in all_results:
        if "k_sensitivity" not in r:
            continue
        cells = []
        for e in r["k_sensitivity"]:
            rstr = f"r={e['pearson_r']:.2f}" if e["pearson_r"] is not None else "r=--"
            cells.append(f"{e['joint_n']:>3d}/{rstr}")
        rp = r.get("rank_product") or {}
        rp_str = (f"r={rp.get('pearson_r'):.2f}" if rp.get("pearson_r") is not None
                  else "--")
        print(f"{r['tissue']:10s} " + "  ".join(cells) + f"   {rp_str}")

    fig_sensitivity(all_results)


if __name__ == "__main__":
    main()
