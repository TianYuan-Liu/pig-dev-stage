"""Aggregate all 5 variants into variant_comparison.json and produce
fig_variant_comparison.pdf (two panels: joint_n and Pearson r with CIs)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

VARIANTS = [
    ("A", "A_pig_anchored", "Pig-anchored\n(pig stable@100 ∩ |hfc|>0.3, sign match)"),
    ("B", "B_binary_pig", "Binary-pig retrain\n(strict intersection @100)"),
    ("C", "C_rank_product", "Rank product\n(top-100 by sqrt(rp×rh))"),
    ("D", "D_joint_shared_model", "Joint shared model\n(binary LightGBM, stable@100)"),
    ("E", "E_intersection_then_rank_product",
     "Intersection + RP\n(ever-used both, top-100 RP)"),
]
TISSUES_HUMAN = ["Muscle", "Brain", "Liver"]
TISSUES_PIG_ONLY = ["Lung", "Blood"]
RECOMMENDED = "D"


def fetch_result(variant_dir: Path, tissue: str) -> Optional[dict]:
    f = variant_dir / f"results_{tissue}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:  # noqa: BLE001
        return None


def main():
    rows: List[dict] = []
    for v_letter, v_subdir, v_label in VARIANTS:
        vd = HERE / v_subdir
        for tissue in TISSUES_HUMAN + TISSUES_PIG_ONLY:
            r = fetch_result(vd, tissue)
            row = {
                "variant": v_letter,
                "variant_label": v_label,
                "tissue": tissue,
                "joint_n": None,
                "pearson_r": None,
                "ci_low": None,
                "ci_high": None,
                "directional_concordance_pct": None,
                "expected_by_chance": None,
                "ratio_observed_to_chance": None,
                "notes": "",
            }
            if r is None:
                row["notes"] = "no_result_file"
            elif "error" in r:
                row["notes"] = f"error:{r['error']}"[:120]
            else:
                row["joint_n"] = r.get("joint_set_n")
                c = r.get("concordance") or {}
                row["pearson_r"] = c.get("pearson_r")
                row["ci_low"] = c.get("ci_low")
                row["ci_high"] = c.get("ci_high")
                row["directional_concordance_pct"] = c.get(
                    "directional_concordance_pct")
                row["expected_by_chance"] = r.get("expected_by_chance")
                row["ratio_observed_to_chance"] = r.get(
                    "ratio_observed_to_chance")
                if row["joint_n"] == 0:
                    row["notes"] = "empty_joint_set"
                elif tissue in TISSUES_PIG_ONLY and not r.get("human_available", True):
                    row["notes"] = "pig_only_no_human"
            rows.append(row)
    out = {"rows": rows, "recommended_variant": RECOMMENDED}
    (HERE / "variant_comparison.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"wrote variant_comparison.json with {len(rows)} rows")

    # ---- Figure
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "figure.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    df = pd.DataFrame(rows)
    df = df[df["tissue"].isin(TISSUES_HUMAN)].copy()  # main panels for tissues with human

    variant_order = [v[0] for v in VARIANTS]
    accent = "#C97A9A"  # accent for recommended D
    grey = "#888888"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # Panel 1: joint_n bars per (variant, tissue)
    ax = axes[0]
    width = 0.15
    x = np.arange(len(TISSUES_HUMAN))
    for i, v in enumerate(variant_order):
        ns = []
        for t in TISSUES_HUMAN:
            sub = df[(df["variant"] == v) & (df["tissue"] == t)]
            n = sub["joint_n"].iloc[0] if len(sub) else None
            ns.append(0 if n is None else int(n))
        color = accent if v == RECOMMENDED else grey
        alpha = 1.0 if v == RECOMMENDED else 0.5
        bars = ax.bar(x + i * width - width * 2, ns, width,
                      color=color, alpha=alpha,
                      edgecolor="black", lw=0.5,
                      label=v if v == RECOMMENDED else None)
        for bx, n_val in zip(bars, ns):
            if n_val > 0:
                ax.text(bx.get_x() + bx.get_width() / 2, n_val + 1,
                        f"{n_val}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(TISSUES_HUMAN)
    ax.set_ylabel("Joint set size")
    ax.set_title("Joint set size by variant × tissue", fontweight="bold")
    # Add variant letter labels under each cluster
    for i, v in enumerate(variant_order):
        for j, t in enumerate(TISSUES_HUMAN):
            ax.text(j + i * width - width * 2, -2.5, v, ha="center",
                    va="top", fontsize=7,
                    color=accent if v == RECOMMENDED else grey)
    ax.set_ylim(bottom=-5)
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    # Panel 2: Pearson r with CI whiskers
    ax = axes[1]
    for i, v in enumerate(variant_order):
        rs, los, his = [], [], []
        for t in TISSUES_HUMAN:
            sub = df[(df["variant"] == v) & (df["tissue"] == t)]
            r_val = sub["pearson_r"].iloc[0] if len(sub) else None
            lo = sub["ci_low"].iloc[0] if len(sub) else None
            hi = sub["ci_high"].iloc[0] if len(sub) else None
            rs.append(np.nan if r_val is None else float(r_val))
            los.append(np.nan if lo is None else float(lo))
            his.append(np.nan if hi is None else float(hi))
        color = accent if v == RECOMMENDED else grey
        alpha = 1.0 if v == RECOMMENDED else 0.5
        xs = x + i * width - width * 2
        bars = ax.bar(xs, [0 if np.isnan(r) else r for r in rs], width,
                      color=color, alpha=alpha, edgecolor="black", lw=0.5,
                      label=v if v == RECOMMENDED else None)
        for k, (lo, hi, r_val) in enumerate(zip(los, his, rs)):
            if np.isnan(r_val) or np.isnan(lo) or np.isnan(hi):
                continue
            ax.plot([xs[k], xs[k]], [lo, hi], color="black", lw=1.2)
            ax.plot([xs[k] - 0.04, xs[k] + 0.04], [lo, lo], color="black", lw=1.2)
            ax.plot([xs[k] - 0.04, xs[k] + 0.04], [hi, hi], color="black", lw=1.2)
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(TISSUES_HUMAN)
    ax.set_ylabel(r"Pearson $r$ (log$_2$FC concordance)")
    ax.set_title("Cross-species concordance by variant × tissue",
                 fontweight="bold")
    for i, v in enumerate(variant_order):
        for j, t in enumerate(TISSUES_HUMAN):
            ax.text(j + i * width - width * 2, -0.78, v, ha="center",
                    va="top", fontsize=7,
                    color=accent if v == RECOMMENDED else grey)
    ax.set_ylim(-0.75, 1.0)
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    # Single legend explaining the accent
    handles = [plt.Rectangle((0, 0), 1, 1, color=accent),
               plt.Rectangle((0, 0), 1, 1, color=grey, alpha=0.5)]
    labels = [f"Recommended variant ({RECOMMENDED})", "Other variants"]
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=9)
    fig.tight_layout()
    out_pdf = HERE / "fig_variant_comparison.pdf"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf.name}")


if __name__ == "__main__":
    main()
