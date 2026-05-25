#!/usr/bin/env python3
"""Finalize Loop 0 + Loop A: pick the winner by the lexicographic objective
(max mean_r; tie-break max min_r; then simplicity), regenerate the canonical
v2_uniform_winner.json, write the full variant-sweep audit CSV, and append
every attempt (kept / discarded / blocked) to autoresearch_results.tsv.

Reads the per-config driver outputs already written under
autoresearch/uniform_results/.  Pure bookkeeping — no new science.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "autoresearch" / "uniform_results"
PANEL = ["Muscle", "Lung", "Testis", "Spleen", "Adipose Tissue"]

# (cell_id, label-file, loop, status, complexity-note)
LOOPA = [
    ("strict_1to1|pig_anchored|spearman|baseline", None,          "A",     "keep_baseline", "locked baseline"),
    ("v1_human_permissive_bins",        "v1_human_permissive_bins",        "A_ext", "discard", "better than baseline but custom bins; below ext3"),
    ("v2_pig_permissive_bins",          "v2_pig_permissive_bins",          "A_ext", "discard", "Testis collapses to ~0; worse mean"),
    ("v3_pig_median_fc",                "v3_pig_median_fc",                "A_ext", "discard", "simpler FC but worse mean"),
    ("v4_drop0",                        "v4_drop0",                        "A_ext", "discard", "ties baseline (simpler) but below ext3"),
    ("v5_low_expr",                     "v5_low_expr",                     "A_ext", "discard", "better than baseline but below ext3"),
    ("v6_high_expr",                    "v6_high_expr",                    "A_ext", "discard", "high TPM filters +0.030; below ext3"),
    ("v7_top300_pigfc",                 "v7_top300_pigfc",                 "A_ext", "discard", "new top-N filter; much worse"),
    ("v8_bootstrap_stable",             "v8_bootstrap_stable",             "A_ext", "discard", "new bootstrap filter; better mean worse min; complex"),
    ("ext1_high_expr_permissive",       "ext1_high_expr_permissive",       "A_ext", "discard", "high expr + permissive bins; below ext3"),
    ("ext2_pig5_human50",               "ext2_pig5_human50",               "A_ext", "discard", "pig5/human50; Spleen n=20 collapse"),
    ("ext3_high_expr_drop0",            "ext3_high_expr_drop0",            "A_ext", "keep",    "WINNER pig3 human30 drop0; +0.085 mean_r; removes purity-drop step"),
    ("ext4_high_expr_drop0_permissive", "ext4_high_expr_drop0_permissive", "A_ext", "discard", "ext3 + permissive bins; lower mean than ext3"),
]

LOOP0 = [
    ("rescueA_small_intestine_pool",  "rescueA_small_intestine", "0", "discard", "Small Intestine pooled pig n=107/31 but r=-0.095 (<0): fails r>0"),
    ("rescueB_heart_permissive_bins", "rescueB_heart",           "0", "discard", "Heart permissive bins add no human samples; r=-0.095 CI spans 0"),
    ("rescueE_liver_go0032502",       "rescueE_liver_go",        "0", "discard", "Liver restricted to 6362 GO:0032502 genes; r=-0.069 still negative CI spans 0"),
    ("rescueC_colon_blocked",   None, "0", "discard", "dGTEx colon expression file not present locally (data-blocked)"),
    ("rescueD_lymphnode_blocked", None, "0", "discard", "dGTEx lymph_node expression file not present locally (data-blocked)"),
    ("rescueF_bloodvessel_blocked", None, "0", "discard", "dGTEx blood_vessel/artery expression file not present locally (data-blocked)"),
    ("rescueG_pituitary_blocked", None, "0", "discard", "dGTEx pituitary expression file not present locally (data-blocked)"),
]


def load(label_file):
    if label_file is None:
        return None
    p = RES / f"{label_file}.json"
    return json.load(open(p)) if p.exists() else None


def main():
    import subprocess
    sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).decode().strip()

    # ---- winner = ext3 ----
    w = load("ext3_high_expr_drop0")
    assert w is not None and w["all_tissues_valid"], "winner ext3 missing/invalid"
    cfg = w["config"]
    winner = {
        "config": {"ortholog": cfg["ortholog"], "filter": cfg["filter"], "statistic": cfg["statistic"]},
        "protocol": {
            "pig_method": cfg["pig_method"],
            "pig_min_tpm": cfg["pig_min_tpm"],
            "pig_drop_pct": cfg["drop_pct"],
            "human_min_tpm": cfg["human_min_tpm"],
            "human_method": "median",
            "human_young_cohort": cfg["human_young_cohorts"],
            "human_old_cohort": cfg["human_old_cohorts"],
        },
        "objective": "max mean_r; tie-break max min_r within 0.005 of max; then simplicity",
        "provenance": "Loop A autoresearch winner ext3 (high-expression filter pig>=3/human>=30 TPM, "
                      "no purity-drop). Supersedes baseline (mean_r 0.3559 -> 0.4404). "
                      "Caveat: higher TPM thresholds reduce gene counts (Lung n=49); validated in Loop C.",
        "aggregate": {
            "mean_r": w["mean_r"], "min_r": w["min_r"], "max_r": w["max_r"],
            "n_tissues": w["n_tissues_with_r"],
        },
        "per_tissue": {
            t: {
                "n_genes": w["per_tissue"][t]["n"],
                "r": w["per_tissue"][t]["r"],
                "p": w["per_tissue"][t]["p"],
                "dc": w["per_tissue"][t]["dc"],
                "ci_low": w["per_tissue"][t]["ci_low"],
                "ci_high": w["per_tissue"][t]["ci_high"],
            } for t in PANEL
        },
    }
    out = ROOT / "review/analyses/results/v2_uniform_winner.json"
    out.write_text(json.dumps(winner, indent=2))
    print(f"Wrote {out}: ext3 mean_r={w['mean_r']:.4f} min_r={w['min_r']:.4f}")

    # ---- variant-sweep audit CSV ----
    rows = []
    for cell_id, lf, loop, status, note in LOOPA + LOOP0:
        d = load(lf)
        if d is not None and "mean_r" in d:
            mean_r, min_r, max_r = d["mean_r"], d["min_r"], d["max_r"]
            psize = len([t for t in d["panel"] if d["per_tissue"][t]["r"] is not None])
        elif d is not None and "per_tissue" in d:  # single-tissue rescue
            t = d["panel"][0]; pt = d["per_tissue"][t]
            mean_r = min_r = max_r = pt["r"]; psize = 1
        else:
            mean_r = min_r = max_r = None; psize = 5 if loop.startswith("A") else 1
        rows.append({"cell_id": cell_id, "loop": loop, "mean_r": mean_r,
                     "min_r": min_r, "max_r": max_r, "panel_size": psize,
                     "status": status, "description": note})
    csv_out = ROOT / "review/analyses/results/v2_loopA_variant_sweep.csv"
    with open(csv_out, "w", newline="") as fh:
        wri = csv.DictWriter(fh, fieldnames=["loop", "cell_id", "mean_r", "min_r", "max_r", "panel_size", "status", "description"])
        wri.writeheader()
        for r in rows:
            wri.writerow({k: r[k] for k in wri.fieldnames})
    print(f"Wrote {csv_out}: {len(rows)} rows")

    # ---- append to autoresearch_results.tsv ----
    tsv = ROOT / "autoresearch" / "autoresearch_results.tsv"
    def fmt(x):
        return "" if x is None else f"{x:.4f}"
    with open(tsv, "a") as fh:
        for r in rows:
            if r["cell_id"].endswith("baseline"):
                continue  # baseline row already present
            fh.write("\t".join([sha, r["loop"], r["cell_id"], fmt(r["mean_r"]),
                                 fmt(r["min_r"]), fmt(r["max_r"]), str(r["panel_size"]),
                                 r["status"], r["description"]]) + "\n")
    print(f"Appended {len(rows)-1} rows to {tsv} (commit {sha})")


if __name__ == "__main__":
    main()
