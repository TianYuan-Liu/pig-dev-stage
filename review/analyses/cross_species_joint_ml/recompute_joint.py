#!/usr/bin/env python3
"""Recompute joint sets and concordance from saved per-seed importance,
applying the bug-fixed stable_set (non-zero-gain only). No retraining.

Updates per-tissue results_{tissue}.json with corrected joint set, recomputes
concordance, regenerates features_*.csv and joint_features_*.csv at K=100
(primary, per spec) and at K=50/200/500/1000 as sensitivity-tagged columns
in the JSON.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUT_DIR.parent.parent.parent
ORTHO_CSV = (PROJECT_ROOT / "review" / "analyses" / "results"
             / "pig_human_one_to_one_orthologs.csv")
ANNOT_CACHE = OUT_DIR / "cache" / "mygene_annotations.json"

TISSUES = ["Muscle", "Brain", "Liver", "Lung", "Blood"]
STABILITY_THRESHOLD = 8
PRIMARY_K = 100
SENSITIVITY_K = [50, 100, 200, 500, 1000]
N_BOOTSTRAP = 1000
PSEUDO = 0.01
PIG_YOUNG = {"Infant", "Early childhood"}
PIG_OLD = {"Post-pubertal", "Adult"}


def load_ortho() -> pd.DataFrame:
    df = pd.read_csv(ORTHO_CSV)
    df = df[df["is_one_to_one"]] if "is_one_to_one" in df.columns else df
    df["symbol"] = df["symbol"].str.upper()
    return df[["symbol", "pig_gene_id", "human_gene_id"]]


def stable_at_k(seed_imp: pd.DataFrame, k: int, threshold: int
                ) -> Tuple[List[str], pd.DataFrame]:
    """Stable features = in top-K (by gain, EXCLUDING zero-gain) of >= threshold seeds."""
    counter: Dict[str, int] = defaultdict(int)
    rank_records: Dict[str, List[int]] = defaultdict(list)
    gain_records: Dict[str, List[float]] = defaultdict(list)
    for col in seed_imp.columns:
        s = seed_imp[col]
        nonzero = s[s > 0]
        top = nonzero.nlargest(k)
        for rank, (g, v) in enumerate(top.items(), start=1):
            counter[g] += 1
            rank_records[g].append(rank)
            gain_records[g].append(float(v))
    rows = []
    for g, c in counter.items():
        rows.append({
            "gene_id": g,
            "n_seeds_top": int(c),
            "mean_rank": float(np.mean(rank_records[g])),
            "median_rank": float(np.median(rank_records[g])),
            "mean_gain": float(np.mean(gain_records[g])),
        })
    tab = pd.DataFrame(rows).sort_values(
        ["n_seeds_top", "mean_gain"], ascending=[False, False]
    ).reset_index(drop=True)
    stable = tab[tab["n_seeds_top"] >= threshold]["gene_id"].tolist()
    return stable, tab


def fold_change_from_cache(tissue: str, species: str) -> Optional[pd.Series]:
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
    fc = np.log2((raw.loc[old].mean(axis=0) + PSEUDO)
                 / (raw.loc[young].mean(axis=0) + PSEUDO))
    return fc


def bootstrap_pearson(x, y, n=N_BOOTSTRAP, seed=42):
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


def seed_jaccard(seed_imp: pd.DataFrame, k: int) -> float:
    seed_sets = []
    for col in seed_imp.columns:
        s = seed_imp[col]
        nonzero = s[s > 0]
        seed_sets.append(set(nonzero.nlargest(k).index))
    vals = []
    for i in range(len(seed_sets)):
        for j in range(i + 1, len(seed_sets)):
            a = seed_sets[i]
            b = seed_sets[j]
            if a or b:
                vals.append(len(a & b) / len(a | b))
    return float(np.mean(vals)) if vals else float("nan")


def annotate_category(symbol: str, cache: dict) -> str:
    return cache.get(symbol, {}).get("category", "other")


def recompute_tissue(tissue: str, ortho: pd.DataFrame, annot: dict) -> dict:
    print(f"\n>> {tissue}")
    pig_p = OUT_DIR / f"seed_importance_pig_{tissue}.parquet"
    human_p = OUT_DIR / f"seed_importance_human_{tissue}.parquet"
    if not pig_p.exists():
        return {"tissue": tissue, "error": "missing pig importance parquet"}
    pig_seed = pd.read_parquet(pig_p)
    pig_seed.index.name = "pig_gene_id"
    human_avail = human_p.exists()
    human_seed = pd.read_parquet(human_p) if human_avail else None
    if human_avail:
        human_seed.index.name = "human_gene_id"

    pig_to_sym = dict(zip(ortho["pig_gene_id"], ortho["symbol"]))
    human_to_sym = dict(zip(ortho["human_gene_id"], ortho["symbol"]))
    pid_to_hid = dict(zip(ortho["pig_gene_id"], ortho["human_gene_id"]))

    pig_fc = fold_change_from_cache(tissue, "pig")
    human_fc = fold_change_from_cache(tissue, "human") if human_avail else None

    # Primary K=100 stable + table
    pig_stable, pig_tab = stable_at_k(pig_seed, PRIMARY_K, STABILITY_THRESHOLD)
    pig_tab.to_csv(OUT_DIR / f"features_pig_{tissue}.csv", index=False)
    pig_jacc = seed_jaccard(pig_seed, PRIMARY_K)
    print(f"  pig stable@{PRIMARY_K}={len(pig_stable)}, jaccard={pig_jacc:.2f}")

    if human_avail:
        human_stable, human_tab = stable_at_k(human_seed, PRIMARY_K,
                                              STABILITY_THRESHOLD)
        human_tab.to_csv(OUT_DIR / f"features_human_{tissue}.csv", index=False)
        human_jacc = seed_jaccard(human_seed, PRIMARY_K)
        print(f"  human stable@{PRIMARY_K}={len(human_stable)}, "
              f"jaccard={human_jacc:.2f}")
    else:
        human_stable, human_tab, human_jacc = [], pd.DataFrame(), None

    # Joint at primary K
    joint_pairs: List[Tuple[str, str]] = []
    if human_avail:
        pig_set = set(pig_stable)
        human_set = set(human_stable)
        for pid in pig_set:
            hid = pid_to_hid.get(pid)
            if hid and hid in human_set:
                joint_pairs.append((pid, hid))

    # Sensitivity per K (without rebuilding tables)
    sens = []
    for k in SENSITIVITY_K:
        ps, _ = stable_at_k(pig_seed, k, STABILITY_THRESHOLD)
        entry = {"k": k, "pig_stable_n": len(ps)}
        if human_avail:
            hs, _ = stable_at_k(human_seed, k, STABILITY_THRESHOLD)
            entry["human_stable_n"] = len(hs)
            jp = [(pid, pid_to_hid[pid]) for pid in set(ps)
                  if pid_to_hid.get(pid) in set(hs)]
            entry["joint_n"] = len(jp)
            entry["expected_by_chance"] = (k * k / max(pig_seed.shape[0], 1))
            if jp and pig_fc is not None and human_fc is not None:
                x = pig_fc.reindex([p for p, _ in jp]).values
                y_v = human_fc.reindex([h for _, h in jp]).values
                ok = np.isfinite(x) & np.isfinite(y_v)
                x, y_v = x[ok], y_v[ok]
                if len(x) >= 5 and np.std(x) > 0 and np.std(y_v) > 0:
                    r, _ = stats.pearsonr(x, y_v)
                    r_m, ci_lo, ci_hi = bootstrap_pearson(x, y_v)
                    entry.update({
                        "pearson_r": float(r),
                        "ci_low": float(ci_lo),
                        "ci_high": float(ci_hi),
                        "directional_concordance_pct":
                            100.0 * np.sum(np.sign(x) == np.sign(y_v)) / len(x),
                    })
                else:
                    entry.update({"pearson_r": None})
            else:
                entry.update({"pearson_r": None})
        sens.append(entry)

    # Concordance at primary K
    concordance: dict = {
        "n_joint_genes_for_fc": 0, "pearson_r": None, "pearson_p": None,
        "bootstrap_mean_r": None, "ci_low": None, "ci_high": None,
        "n_concordant_direction": 0, "directional_concordance_pct": None,
    }
    if joint_pairs and pig_fc is not None and human_fc is not None:
        pids = [p for p, _ in joint_pairs]
        hids = [h for _, h in joint_pairs]
        x = pig_fc.reindex(pids).values
        y_v = human_fc.reindex(hids).values
        ok = np.isfinite(x) & np.isfinite(y_v)
        x, y_v = x[ok], y_v[ok]
        if len(x) >= 5 and np.std(x) > 0 and np.std(y_v) > 0:
            r, p = stats.pearsonr(x, y_v)
            r_m, ci_lo, ci_hi = bootstrap_pearson(x, y_v)
            concordance.update({
                "n_joint_genes_for_fc": int(len(x)),
                "pearson_r": float(r), "pearson_p": float(p),
                "bootstrap_mean_r": float(r_m),
                "ci_low": float(ci_lo), "ci_high": float(ci_hi),
                "n_concordant_direction": int(np.sum(np.sign(x) == np.sign(y_v))),
                "directional_concordance_pct":
                    100.0 * np.sum(np.sign(x) == np.sign(y_v)) / len(x),
            })

    # Joint CSV
    pig_rank_map = {r["gene_id"]: r["mean_rank"] for _, r in pig_tab.iterrows()}
    pig_nseed_map = {r["gene_id"]: r["n_seeds_top"]
                     for _, r in pig_tab.iterrows()}
    human_rank_map = ({r["gene_id"]: r["mean_rank"]
                       for _, r in human_tab.iterrows()}
                      if human_avail else {})
    human_nseed_map = ({r["gene_id"]: r["n_seeds_top"]
                        for _, r in human_tab.iterrows()}
                       if human_avail else {})
    rows = []
    for (pid, hid) in joint_pairs:
        sym = pig_to_sym.get(pid) or human_to_sym.get(hid) or ""
        rows.append({
            "symbol": sym,
            "pig_gene_id": pid,
            "human_gene_id": hid,
            "pig_n_seeds_top100": int(pig_nseed_map.get(pid, 0)),
            "pig_mean_rank": float(pig_rank_map.get(pid, np.nan)),
            "human_n_seeds_top100": int(human_nseed_map.get(hid, 0)),
            "human_mean_rank": float(human_rank_map.get(hid, np.nan)),
            "log2fc_pig": float(pig_fc.get(pid, np.nan))
                          if pig_fc is not None else None,
            "log2fc_human": float(human_fc.get(hid, np.nan))
                            if human_fc is not None else None,
            "category": annotate_category(sym, annot),
        })
    joint_df = pd.DataFrame(rows)
    if len(joint_df) > 0:
        joint_df = joint_df.sort_values(
            ["pig_n_seeds_top100", "human_n_seeds_top100", "pig_mean_rank"],
            ascending=[False, False, True]
        ).reset_index(drop=True)
    else:
        # Empty joint set - write empty CSV with proper column schema
        joint_df = pd.DataFrame(columns=[
            "symbol", "pig_gene_id", "human_gene_id",
            "pig_n_seeds_top100", "pig_mean_rank",
            "human_n_seeds_top100", "human_mean_rank",
            "log2fc_pig", "log2fc_human", "category",
        ])
    joint_df.to_csv(OUT_DIR / f"joint_features_{tissue}.csv", index=False)

    # Annotation summary
    cats: Dict[str, int] = defaultdict(int)
    for _, r in joint_df.iterrows():
        cats[r["category"] or "other"] += 1

    # Read existing result for fields like pig sample counts
    rj = OUT_DIR / f"results_{tissue}.json"
    base = json.loads(rj.read_text()) if rj.exists() else {}

    result = {
        **base,
        "tissue": tissue,
        "human_available": human_avail,
        "n_orthologs_used": int(pig_seed.shape[0]),
        "primary_K": PRIMARY_K,
        "stability_threshold": STABILITY_THRESHOLD,
        "pig": {
            **(base.get("pig") or {}),
            "stable_top100_n": len(pig_stable),
            "jaccard_top100_mean": pig_jacc,
            "n_unique_features_ever_used": int((pig_seed > 0).any(axis=1).sum()),
        },
        "human": {
            **(base.get("human") or {}),
            "stable_top100_n": len(human_stable),
            "jaccard_top100_mean": human_jacc,
            "n_unique_features_ever_used":
                int((human_seed > 0).any(axis=1).sum()) if human_avail else 0,
        } if human_avail else {"available": False},
        "joint_set_n": int(len(joint_pairs)),
        "joint_features": [
            {"symbol": r["symbol"], "pig_gene_id": r["pig_gene_id"],
             "human_gene_id": r["human_gene_id"],
             "pig_n_seeds": int(r["pig_n_seeds_top100"]),
             "human_n_seeds": int(r["human_n_seeds_top100"]),
             "log2fc_pig": (float(r["log2fc_pig"])
                            if pd.notna(r["log2fc_pig"]) else None),
             "log2fc_human": (float(r["log2fc_human"])
                              if pd.notna(r["log2fc_human"]) else None),
             "category": r["category"]}
            for _, r in joint_df.iterrows()
        ],
        "concordance": concordance,
        "annotation_summary": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
        "k_sensitivity": sens,
    }
    rj.write_text(json.dumps(result, indent=2, default=str))
    print(f"  joint@{PRIMARY_K} = {len(joint_pairs)} genes; "
          f"r = {concordance['pearson_r']}")
    return result


def main():
    ortho = load_ortho()
    annot = (json.loads(ANNOT_CACHE.read_text())
             if ANNOT_CACHE.exists() else {})
    summary = {}
    for t in TISSUES:
        r = recompute_tissue(t, ortho, annot)
        summary[t] = {
            "joint_n": r.get("joint_set_n", 0),
            "r": (r.get("concordance") or {}).get("pearson_r"),
        }
    (OUT_DIR / "summary_all_tissues.json").write_text(
        json.dumps(summary, indent=2))
    print("\nSummary:")
    for t, v in summary.items():
        print(f"  {t}: joint_n={v['joint_n']}, r={v['r']}")


if __name__ == "__main__":
    main()
