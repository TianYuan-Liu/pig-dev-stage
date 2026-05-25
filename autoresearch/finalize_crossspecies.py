#!/usr/bin/env python3
"""Finalize the cross-species deliverables with the CONFIRMATORY vs EXPLORATORY
framing.

Decision (per user): a cross-species validation must NOT be optimized. So the
pre-specified BASELINE protocol is the confirmatory validation result; the Loop A
search winner (ext3) is reported only as an exploratory "maximum conserved
signal", with the explicit caveat that the protocol was selected to maximize r.

Writes:
  review/analyses/results/v2_uniform_winner.json          (BASELINE, confirmatory)
  review/analyses/results/v2_uniform_exploratory_max.json (ext3, exploratory)
  paper/figures/output/stats/fig4_v2_dgtex_summary.tsv     (BASELINE headline)
  paper/figures/output/stats/fig4_v2_dgtex_exploratory.tsv (ext3 companion)
and relabels the Loop A rows in autoresearch_results.tsv.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "autoresearch" / "uniform_results"
PANEL = ["Muscle", "Lung", "Testis", "Spleen", "Adipose Tissue"]

SPECS = {
    "baseline": {
        "driver": "baseline_confirmatory.json",
        "loopc": "v2_loopC_validation.json",
        "role": "confirmatory",
        "protocol": {"pig_method": "weighted", "pig_min_tpm": 1.0, "pig_drop_pct": 30.0,
                     "human_min_tpm": 10.0, "human_method": "median",
                     "human_young_cohort": [1], "human_old_cohort": [4]},
        "note": "Pre-specified uniform protocol. This is the confirmatory cross-species "
                "validation: the config was NOT selected to maximize r.",
    },
    "ext3": {
        "driver": "ext3_high_expr_drop0.json",
        "loopc": "v2_loopC_validation_ext3.json",
        "role": "exploratory_max",
        "protocol": {"pig_method": "weighted", "pig_min_tpm": 3.0, "pig_drop_pct": 0.0,
                     "human_min_tpm": 30.0, "human_method": "median",
                     "human_young_cohort": [1], "human_old_cohort": [4]},
        "note": "EXPLORATORY maximum: the Loop A search selected this config because it "
                "maximizes mean_r. Reported as an upper bound on the conserved signal under "
                "defensible high-expression filtering, NOT as an unbiased validation "
                "(selection-optimism bias). The ortholog-scramble permutation p still "
                "controls against chance pairing for this fixed config.",
    },
}


def build_json(name: str) -> dict:
    spec = SPECS[name]
    d = json.load(open(RES / spec["driver"]))
    val = json.load(open(ROOT / "review/analyses/results" / spec["loopc"]))
    per = {}
    for t in PANEL:
        w = d["per_tissue"][t]
        v = val["per_tissue"].get(t, {})
        per[t] = {
            "n_genes": w["n"], "r": w["r"], "p": w["p"], "dc": w["dc"],
            "ci_low": w["ci_low"], "ci_high": w["ci_high"],
            "perm_empirical_p": v.get("perm_empirical_p"),
            "loo_r_min": v.get("loo_r_min"), "loo_r_max": v.get("loo_r_max"),
        }
    return {
        "config": {"ortholog": "strict_1to1", "filter": "pig_anchored", "statistic": "spearman"},
        "protocol": spec["protocol"],
        "role": spec["role"],
        "objective": "max mean_r; tie-break max min_r (Loop A search objective)",
        "note": spec["note"],
        "aggregate": {"mean_r": d["mean_r"], "min_r": d["min_r"], "max_r": d["max_r"],
                      "n_tissues": d["n_tissues_with_r"]},
        "per_tissue": per,
    }


def fig4_tsv(payload: dict, path: Path):
    cols = ["tissue", "n_genes", "spearman_r", "ci_low", "ci_high", "p_value",
            "directional_concordance_pct", "perm_empirical_p", "loo_r_min", "loo_r_max", "tier"]
    lines = ["\t".join(cols)]
    for t in PANEL:
        p = payload["per_tissue"][t]
        perm = p["perm_empirical_p"]
        tier = "1" if (p["r"] >= 0.40 and perm is not None and perm < 0.05) else "2"
        lines.append("\t".join([
            t, str(p["n_genes"]), f"{p['r']:.4f}",
            f"{p['ci_low']:.4f}" if p["ci_low"] is not None else "",
            f"{p['ci_high']:.4f}" if p["ci_high"] is not None else "",
            f"{p['p']:.3e}", f"{p['dc']:.1f}",
            f"{perm:.4f}" if perm is not None else "",
            f"{p['loo_r_min']:.4f}" if p["loo_r_min"] is not None else "",
            f"{p['loo_r_max']:.4f}" if p["loo_r_max"] is not None else "",
            tier]))
    lines.append("\t".join(["PANEL_MEAN", "", f"{payload['aggregate']['mean_r']:.4f}",
                            "", "", "", "", "", "", "", ""]))
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    conf = build_json("baseline")
    expl = build_json("ext3")
    (ROOT / "review/analyses/results/v2_uniform_winner.json").write_text(json.dumps(conf, indent=2))
    (ROOT / "review/analyses/results/v2_uniform_exploratory_max.json").write_text(json.dumps(expl, indent=2))
    stats = ROOT / "paper/figures/output/stats"
    stats.mkdir(parents=True, exist_ok=True)
    fig4_tsv(conf, stats / "fig4_v2_dgtex_summary.tsv")
    fig4_tsv(expl, stats / "fig4_v2_dgtex_exploratory.tsv")

    print("CONFIRMATORY (baseline, headline cross-species validation):")
    print(f"  mean_r={conf['aggregate']['mean_r']:.4f}  min_r={conf['aggregate']['min_r']:.4f}")
    print("EXPLORATORY-MAX (ext3, selection-optimism caveat):")
    print(f"  mean_r={expl['aggregate']['mean_r']:.4f}  min_r={expl['aggregate']['min_r']:.4f}")

    # relabel Loop A rows in results.tsv: baseline=confirmatory, ext3=exploratory_max
    tsv = ROOT / "autoresearch" / "autoresearch_results.tsv"
    out = []
    for ln in tsv.read_text().splitlines():
        f = ln.split("\t")
        if len(f) >= 9 and f[2] == "ext3_high_expr_drop0":
            f[7] = "exploratory_max"
            f[8] = "Loop A search max (NOT the validation); reported as exploratory upper bound w/ permutation control"
            ln = "\t".join(f)
        elif len(f) >= 9 and f[2] == "strict_1to1|pig_anchored|spearman":
            f[7] = "keep_confirmatory"
            f[8] = "pre-specified baseline = CONFIRMATORY cross-species validation (mean_r 0.3559)"
            ln = "\t".join(f)
        out.append(ln)
    tsv.write_text("\n".join(out) + "\n")
    print(f"Relabeled Loop A rows in {tsv}")
    return 0


if __name__ == "__main__":
    main()
