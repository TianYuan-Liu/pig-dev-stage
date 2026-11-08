"""Variant E — Intersection then rank-product.

Pre-filter to genes with non-zero gain in >=1 seed in BOTH species (the
"ever used in both" pool). Within that intersection, compute the same
rank-product score as Variant C and take top N ∈ {30, 100, 200}.
No retraining; only runs for tissues with human data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _utils as U  # noqa: E402

VARIANT = "E"
OUT_DIR = HERE
TISSUES_WITH_HUMAN = U.HUMAN_TISSUES
TOP_N_LIST = [30, 100, 200]
PRIMARY_N = 100


def run_tissue(tissue: str, ortho: pd.DataFrame, annot: dict) -> dict:
    pig_seed = U.load_pig_seed_imp(tissue)
    human_seed = U.load_human_seed_imp(tissue)
    if pig_seed is None or human_seed is None:
        return {"variant": VARIANT, "tissue": tissue,
                "error": "missing seed importance parquet"}

    pig_ever = (pig_seed > 0).any(axis=1)
    human_ever = (human_seed > 0).any(axis=1)
    pig_ever_set = set(pig_ever[pig_ever].index)
    human_ever_set = set(human_ever[human_ever].index)

    # Build the intersection via ortholog mapping
    pid_to_hid = dict(zip(ortho["pig_gene_id"], ortho["human_gene_id"]))
    pid_to_sym = dict(zip(ortho["pig_gene_id"], ortho["symbol"]))
    hid_to_sym = dict(zip(ortho["human_gene_id"], ortho["symbol"]))

    intersect_pairs: List = []
    for pid in pig_ever_set:
        hid = pid_to_hid.get(pid)
        if hid and hid in human_ever_set:
            intersect_pairs.append((pid, hid))
    U.log_row(VARIANT, tissue, "filter", "n_intersect", len(intersect_pairs))

    # Rank-product within the intersection
    pig_rank = U.per_seed_rank(pig_seed)
    human_rank = U.per_seed_rank(human_seed)
    pig_mean_rank = pig_rank.mean(axis=1)
    human_mean_rank = human_rank.mean(axis=1)

    rows = []
    for pid, hid in intersect_pairs:
        pr = float(pig_mean_rank[pid])
        hr = float(human_mean_rank[hid])
        rows.append({
            "symbol": pid_to_sym.get(pid) or hid_to_sym.get(hid) or "",
            "pig_gene_id": pid,
            "human_gene_id": hid,
            "pig_mean_rank": pr,
            "human_mean_rank": hr,
            "rank_product": float(np.sqrt(pr * hr)),
        })
    rp_df = pd.DataFrame(rows).sort_values("rank_product").reset_index(drop=True)

    pig_fc = U.pig_fold_change(tissue)
    human_fc = U.human_fold_change(tissue)
    if pig_fc is not None:
        rp_df["log2fc_pig"] = rp_df["pig_gene_id"].map(pig_fc)
    if human_fc is not None:
        rp_df["log2fc_human"] = rp_df["human_gene_id"].map(human_fc)

    sweep: Dict[int, dict] = {}
    primary_rows: List[dict] = []
    for N in TOP_N_LIST:
        top = rp_df.head(N).copy()
        top["category"] = top["symbol"].apply(
            lambda s: U.classify_symbol(s, annot))
        pairs = list(zip(top["pig_gene_id"], top["human_gene_id"]))
        conc = U.compute_concordance(pairs, pig_fc, human_fc)
        sweep[N] = {
            "n": N,
            "joint_n": len(pairs),
            "concordance": conc,
            "annotation_summary": U.annotation_summary(
                top["symbol"].tolist(), annot),
        }
        if N == PRIMARY_N:
            primary_rows = top.to_dict(orient="records")
            top.to_csv(OUT_DIR / f"joint_features_{tissue}.csv", index=False)

    # save intersection rank-product table
    rp_df.to_csv(OUT_DIR / f"rank_product_intersection_{tissue}.csv",
                 index=False)

    primary = sweep[PRIMARY_N]
    # Expected by chance for "ever used in both" pool size
    n_orthologs = pig_seed.shape[0]
    expected_overlap_n = U.expected_n_by_chance(
        len(pig_ever_set), len(human_ever_set), n_orthologs=n_orthologs)
    intersection_n = len(intersect_pairs)
    # Reporting ratio: how much bigger is intersection than chance
    inter_ratio = (intersection_n / expected_overlap_n
                   if expected_overlap_n else None)

    result = {
        "variant": VARIANT,
        "tissue": tissue,
        "human_available": True,
        "n_orthologs_used": n_orthologs,
        "pig_ever_used_n": int(pig_ever.sum()),
        "human_ever_used_n": int(human_ever.sum()),
        "intersection_n": intersection_n,
        "expected_by_chance_intersection": expected_overlap_n,
        "intersection_vs_chance_ratio": inter_ratio,
        "primary_N": PRIMARY_N,
        "joint_set_n": min(PRIMARY_N, intersection_n),
        # for variant comparison: the "joint" denominator we care about
        "expected_by_chance": expected_overlap_n,
        "ratio_observed_to_chance":
            (min(PRIMARY_N, intersection_n) / expected_overlap_n
             if expected_overlap_n else None),
        "concordance": primary["concordance"],
        "annotation_summary": primary["annotation_summary"],
        "joint_features": primary_rows[:200],
        "N_sweep": sweep,
    }
    (OUT_DIR / f"results_{tissue}.json").write_text(
        json.dumps(result, indent=2, default=str))
    U.log_row(VARIANT, tissue, "joint", "joint_n",
              min(PRIMARY_N, intersection_n))
    U.log_row(VARIANT, tissue, "joint", "pearson_r",
              primary["concordance"].get("pearson_r"))
    print(f"[E] {tissue}: intersection={intersection_n}, "
          f"top-{PRIMARY_N} r="
          f"{primary['concordance'].get('pearson_r')}", flush=True)
    return result


def main():
    ortho = U.load_ortho()
    annot = U.load_annot_cache()
    summary = {}
    for tissue in TISSUES_WITH_HUMAN:
        try:
            r = run_tissue(tissue, ortho, annot)
            summary[tissue] = r
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            U.log_row(VARIANT, tissue, "crash", "exception",
                      str(exc).replace("\t", " ")[:200])
            summary[tissue] = {"variant": VARIANT, "tissue": tissue,
                               "error": str(exc)}
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
