"""Variant C — Rank product.

joint_score(g) = sqrt(pig_mean_rank * human_mean_rank). Zero-gain features
get sentinel rank = n_features + 1. Top N ∈ {30, 100, 200} by joint_score.
No retraining. Only runs for tissues with human data.
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

VARIANT = "C"
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

    pig_rank = U.per_seed_rank(pig_seed)
    human_rank = U.per_seed_rank(human_seed)
    pig_mean_rank = pig_rank.mean(axis=1)
    human_mean_rank = human_rank.mean(axis=1)

    pid_to_sym = dict(zip(ortho["pig_gene_id"], ortho["symbol"]))
    hid_to_sym = dict(zip(ortho["human_gene_id"], ortho["symbol"]))

    rows = []
    for pid, hid in zip(ortho["pig_gene_id"], ortho["human_gene_id"]):
        if pid in pig_mean_rank.index and hid in human_mean_rank.index:
            pr = pig_mean_rank[pid]
            hr = human_mean_rank[hid]
            rp = float(np.sqrt(pr * hr))
            rows.append({
                "symbol": pid_to_sym.get(pid) or hid_to_sym.get(hid) or "",
                "pig_gene_id": pid,
                "human_gene_id": hid,
                "pig_mean_rank": float(pr),
                "human_mean_rank": float(hr),
                "rank_product": rp,
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
        joint_pairs = list(zip(top["pig_gene_id"], top["human_gene_id"]))
        conc = U.compute_concordance(joint_pairs, pig_fc, human_fc)
        sweep[N] = {
            "n": N,
            "joint_n": len(joint_pairs),
            "concordance": conc,
            "annotation_summary": U.annotation_summary(
                top["symbol"].tolist(), annot),
        }
        if N == PRIMARY_N:
            primary_rows = top.to_dict(orient="records")
            top.to_csv(OUT_DIR / f"joint_features_{tissue}.csv", index=False)

    # save full rank-product table
    rp_df.to_csv(OUT_DIR / f"rank_product_{tissue}.csv", index=False)

    primary = sweep[PRIMARY_N]
    # expected by chance: any random gene has expected rank product
    # ~ sqrt(N_orthologs/2 * N_orthologs/2) = N/2 for null hypothesis;
    # the top-N by rank product carves out the top fraction. We report
    # joint_n / expected_n where expected_n = N (since top-N is fixed),
    # which is 1 by construction; instead use a more meaningful measure:
    # how many of the top-N have non-zero gain in BOTH species — that's
    # the "informative" fraction.
    n_orthologs = rp_df.shape[0]
    informative_count = 0
    for r in primary_rows:
        pid = r.get("pig_gene_id")
        hid = r.get("human_gene_id")
        if pid in pig_seed.index and hid in human_seed.index:
            pig_used = bool((pig_seed.loc[pid] > 0).any())
            human_used = bool((human_seed.loc[hid] > 0).any())
            if pig_used and human_used:
                informative_count += 1

    # Use Pig "ever used" set and human "ever used" as baseline pools
    pig_ever = int((pig_seed > 0).any(axis=1).sum())
    human_ever = int((human_seed > 0).any(axis=1).sum())
    # Expected overlap = (pig_ever * human_ever) / n_orthologs under random
    expected_overlap_n = U.expected_n_by_chance(pig_ever, human_ever,
                                                n_orthologs=n_orthologs)

    result = {
        "variant": VARIANT,
        "tissue": tissue,
        "human_available": True,
        "n_orthologs_used": n_orthologs,
        "primary_N": PRIMARY_N,
        "joint_set_n": PRIMARY_N,
        "informative_in_topN_count": informative_count,
        "expected_by_chance": expected_overlap_n,
        "ratio_observed_to_chance":
            (informative_count / expected_overlap_n)
            if expected_overlap_n else None,
        "concordance": primary["concordance"],
        "annotation_summary": primary["annotation_summary"],
        "joint_features": primary_rows[:200],  # truncate to keep file small
        "N_sweep": sweep,
        "pig_ever_used_n": pig_ever,
        "human_ever_used_n": human_ever,
    }
    (OUT_DIR / f"results_{tissue}.json").write_text(
        json.dumps(result, indent=2, default=str))
    U.log_row(VARIANT, tissue, "joint", "joint_n", PRIMARY_N)
    U.log_row(VARIANT, tissue, "joint", "pearson_r",
              primary["concordance"].get("pearson_r"))
    print(f"[C] {tissue}: top-{PRIMARY_N} r="
          f"{primary['concordance'].get('pearson_r')}, "
          f"informative_in_top={informative_count}", flush=True)
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
