#!/usr/bin/env python3
"""Loop D (delivery) — aggregate the Loop A winner + Loop C validation into a
single per-tissue table for the paper's Fig 4 (dGTEx cross-species panel).

Reads:
  review/analyses/results/v2_uniform_winner.json   (Loop A winner: ext3)
  review/analyses/results/v2_loopC_validation.json  (Loop C per-tissue validation)
Writes:
  paper/figures/output/stats/fig4_v2_dgtex_summary.tsv  (one row per panel tissue)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = ["Muscle", "Lung", "Testis", "Spleen", "Adipose Tissue"]


def main() -> int:
    win = json.load(open(ROOT / "review/analyses/results/v2_uniform_winner.json"))
    val = json.load(open(ROOT / "review/analyses/results/v2_loopC_validation.json"))
    out_dir = ROOT / "paper" / "figures" / "output" / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "fig4_v2_dgtex_summary.tsv"

    cols = ["tissue", "n_genes", "spearman_r", "ci_low", "ci_high", "p_value",
            "directional_concordance_pct", "perm_empirical_p", "loo_r_min",
            "loo_r_max", "tier"]
    lines = ["\t".join(cols)]
    for t in PANEL:
        w = win["per_tissue"][t]
        v = val["per_tissue"].get(t, {})
        perm_p = v.get("perm_empirical_p")
        ci_low = w["ci_low"]
        # Tier 1 = strong & validated; Tier 2 = weak/marginal
        tier = "1" if (w["r"] >= 0.40 and perm_p is not None and perm_p < 0.05) else "2"
        row = [
            t, str(w["n_genes"]), f"{w['r']:.4f}",
            f"{ci_low:.4f}" if ci_low is not None else "",
            f"{w['ci_high']:.4f}" if w["ci_high"] is not None else "",
            f"{w['p']:.3e}", f"{w['dc']:.1f}",
            f"{perm_p:.4f}" if perm_p is not None else "",
            f"{v.get('loo_r_min'):.4f}" if v.get("loo_r_min") is not None else "",
            f"{v.get('loo_r_max'):.4f}" if v.get("loo_r_max") is not None else "",
            tier,
        ]
        lines.append("\t".join(row))

    agg = win["aggregate"]
    lines.append("\t".join(["PANEL_MEAN", "", f"{agg['mean_r']:.4f}", "", "", "", "",
                            "", "", "", ""]))
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    main()
