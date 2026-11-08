"""Variant A — Pig-anchored.

For each pig-stable@100 (>=8/10 seeds) gene, retain in the joint set if:
  (i) sign(log2FC_pig) == sign(log2FC_human)
  (ii) |log2FC_human| > threshold
Sweep threshold ∈ {0.2, 0.3, 0.5}; primary = 0.3.
No retraining — reuses cached importance parquets and raw matrices.
Reports for tissues with human data (Muscle, Brain, Liver). Pig-only
results saved for Lung and Blood.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _utils as U  # noqa: E402

VARIANT = "A"
OUT_DIR = HERE
TISSUES = U.TISSUES
TOP_K = 100
HUMAN_FC_THRESHOLDS = [0.2, 0.3, 0.5]
PRIMARY_THRESHOLD = 0.3


def run_tissue(tissue: str, ortho: pd.DataFrame, annot: dict) -> dict:
    pig_seed = U.load_pig_seed_imp(tissue)
    if pig_seed is None:
        return {"variant": VARIANT, "tissue": tissue,
                "error": "missing pig importance parquet"}
    pig_stable, pig_tab = U.stable_set_from_seeds(pig_seed, TOP_K)
    pig_tab.to_csv(OUT_DIR / f"features_pig_{tissue}.csv", index=False)
    U.log_row(VARIANT, tissue, "load", "pig_stable_n", len(pig_stable))

    pid_to_hid = dict(zip(ortho["pig_gene_id"], ortho["human_gene_id"]))
    pid_to_sym = dict(zip(ortho["pig_gene_id"], ortho["symbol"]))
    pig_rank = dict(zip(pig_tab["gene_id"], pig_tab["mean_rank"]))
    pig_nseed = dict(zip(pig_tab["gene_id"], pig_tab["n_seeds_top"]))

    pig_fc = U.pig_fold_change(tissue)
    human_fc = U.human_fold_change(tissue)
    human_avail = human_fc is not None

    out_per_threshold: Dict[float, dict] = {}
    primary_rows: List[dict] = []

    if not human_avail:
        # pig-only: just record pig stable set, no joint
        result = {
            "variant": VARIANT,
            "tissue": tissue,
            "human_available": False,
            "pig_stable_n": len(pig_stable),
            "joint_set_n": 0,
            "expected_by_chance": None,
            "ratio_observed_to_chance": None,
            "concordance": {"n_joint_for_fc": 0, "pearson_r": None,
                            "ci_low": None, "ci_high": None,
                            "directional_concordance_pct": None,
                            "n_concordant_direction": 0},
            "joint_features": [],
            "annotation_summary": {},
            "threshold_sweep": {},
        }
        (OUT_DIR / f"results_{tissue}.json").write_text(
            json.dumps(result, indent=2, default=str))
        pd.DataFrame(columns=[
            "symbol", "pig_gene_id", "human_gene_id",
            "log2fc_pig", "log2fc_human", "category",
        ]).to_csv(OUT_DIR / f"joint_features_{tissue}.csv", index=False)
        return result

    for thr in HUMAN_FC_THRESHOLDS:
        rows = []
        for pid in pig_stable:
            hid = pid_to_hid.get(pid)
            if not hid:
                continue
            sym = pid_to_sym.get(pid, "")
            pfc = (float(pig_fc.get(pid, np.nan))
                   if pig_fc is not None and pid in pig_fc.index else np.nan)
            hfc = (float(human_fc.get(hid, np.nan))
                   if hid in human_fc.index else np.nan)
            if not (np.isfinite(pfc) and np.isfinite(hfc)):
                continue
            if np.sign(pfc) != np.sign(hfc):
                continue
            if abs(hfc) <= thr:
                continue
            rows.append({
                "symbol": sym,
                "pig_gene_id": pid,
                "human_gene_id": hid,
                "log2fc_pig": pfc,
                "log2fc_human": hfc,
                "pig_n_seeds_top100": int(pig_nseed.get(pid, 0)),
                "pig_mean_rank": float(pig_rank.get(pid, np.nan)),
                "category": U.classify_symbol(sym, annot),
            })
        joint_pairs = [(r["pig_gene_id"], r["human_gene_id"]) for r in rows]
        conc = U.compute_concordance(joint_pairs, pig_fc, human_fc)
        out_per_threshold[thr] = {
            "threshold": thr, "joint_n": len(rows), "concordance": conc,
            "annotation_summary": U.annotation_summary(
                [r["symbol"] for r in rows if r["symbol"]], annot),
        }
        if abs(thr - PRIMARY_THRESHOLD) < 1e-9:
            primary_rows = rows

    joint_df = pd.DataFrame(primary_rows)
    joint_df.to_csv(OUT_DIR / f"joint_features_{tissue}.csv", index=False)

    expected_n = U.expected_n_by_chance(len(pig_stable), len(pig_stable),
                                        n_orthologs=pig_seed.shape[0])
    # Better expected-by-chance for pig-anchored: at threshold 0.3 we'd retain
    # genes whose human |log2FC| > 0.3 AND concordant sign. Estimate as
    # |pig_stable| * P(|hfc|>thr & sign match | random gene).
    if human_fc is not None and pig_fc is not None:
        # Use empirical distribution of |hfc| over orthologs as a baseline
        all_pids = ortho["pig_gene_id"].tolist()
        all_hids = ortho["human_gene_id"].tolist()
        common = [(p, h) for p, h in zip(all_pids, all_hids)
                  if p in pig_fc.index and h in human_fc.index]
        pfcs = np.array([pig_fc[p] for p, _ in common])
        hfcs = np.array([human_fc[h] for _, h in common])
        ok = np.isfinite(pfcs) & np.isfinite(hfcs)
        pfcs, hfcs = pfcs[ok], hfcs[ok]
        if len(pfcs):
            empir_pass = (np.abs(hfcs) > PRIMARY_THRESHOLD) & (np.sign(pfcs) == np.sign(hfcs))
            p_pass = float(np.mean(empir_pass))
            expected_n = float(len(pig_stable) * p_pass)
    primary = out_per_threshold[PRIMARY_THRESHOLD]
    primary["expected_by_chance"] = expected_n
    primary["ratio_observed_to_chance"] = (
        primary["joint_n"] / expected_n if expected_n else None)

    result = {
        "variant": VARIANT,
        "tissue": tissue,
        "human_available": True,
        "pig_stable_n": len(pig_stable),
        "primary_threshold": PRIMARY_THRESHOLD,
        "joint_set_n": primary["joint_n"],
        "expected_by_chance": expected_n,
        "ratio_observed_to_chance": primary["ratio_observed_to_chance"],
        "concordance": primary["concordance"],
        "annotation_summary": primary["annotation_summary"],
        "joint_features": primary_rows,
        "threshold_sweep": out_per_threshold,
    }
    (OUT_DIR / f"results_{tissue}.json").write_text(
        json.dumps(result, indent=2, default=str))
    U.log_row(VARIANT, tissue, "joint", "joint_n", primary["joint_n"])
    U.log_row(VARIANT, tissue, "joint", "pearson_r",
              primary["concordance"].get("pearson_r"))
    print(f"[A] {tissue}: pig_stable={len(pig_stable)}, joint@thr0.3="
          f"{primary['joint_n']}, r={primary['concordance'].get('pearson_r')}",
          flush=True)
    return result


def main():
    ortho = U.load_ortho()
    annot = U.load_annot_cache()
    summary = {}
    for tissue in TISSUES:
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
