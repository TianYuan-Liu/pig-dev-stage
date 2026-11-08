#!/usr/bin/env python3
"""Figure generation for joint cross-species ML pipeline.

Inputs (per tissue, produced by pipeline.py):
  features_pig_{tissue}.csv, features_human_{tissue}.csv,
  joint_features_{tissue}.csv, results_{tissue}.json

Outputs:
  fig_top_features_{tissue}.pdf  - pig top-30 | human top-30 | joint
  fig_concordance_panel.pdf      - forest plot of joint-ML r per tissue
  fig_method_comparison.pdf      - FDR-baseline gene-set size vs joint-ML
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR = PROJECT_ROOT / "review" / "analyses" / "cross_species_joint_ml"
RESULTS_DIR = PROJECT_ROOT / "review" / "analyses" / "results"
ORTHO_CSV = RESULTS_DIR / "pig_human_one_to_one_orthologs.csv"
BASELINE_JSON = RESULTS_DIR / "cross_species_all_tissues.json"
BASELINE_GENES = RESULTS_DIR / "cross_species_per_tissue_genes.csv"

TISSUES = ["Muscle", "Brain", "Liver", "Lung", "Blood"]
TISSUE_COLOR = {
    "Muscle": "#C97A9A",
    "Brain":  "#5B8FF9",
    "Liver":  "#E0A458",
    "Lung":   "#82D173",
    "Blood":  "#B45355",
}

sns.set_theme(style="white", context="paper")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_ortho() -> pd.DataFrame:
    df = pd.read_csv(ORTHO_CSV)
    df = df[df["is_one_to_one"]] if "is_one_to_one" in df.columns else df
    df["symbol"] = df["symbol"].str.upper()
    return df[["symbol", "pig_gene_id", "human_gene_id"]]


def _id_to_symbol(ortho: pd.DataFrame) -> Dict[str, str]:
    m = {}
    for _, r in ortho.iterrows():
        m[r["pig_gene_id"]] = r["symbol"]
        m[r["human_gene_id"]] = r["symbol"]
    return m


def fig_top_features(tissue: str, ortho: pd.DataFrame) -> None:
    feat_pig = OUT_DIR / f"features_pig_{tissue}.csv"
    feat_h = OUT_DIR / f"features_human_{tissue}.csv"
    joint_csv = OUT_DIR / f"joint_features_{tissue}.csv"
    res_json = OUT_DIR / f"results_{tissue}.json"
    if not feat_pig.exists():
        print(f"  SKIP fig_top_features {tissue}: missing pig features")
        return
    res = json.loads(res_json.read_text())
    id2sym = _id_to_symbol(ortho)

    pig_tab = pd.read_csv(feat_pig).head(30)
    pig_tab["symbol"] = pig_tab["gene_id"].map(id2sym).fillna(pig_tab["gene_id"])

    has_human = feat_h.exists() and res.get("human_available", False)
    if has_human:
        h_tab = pd.read_csv(feat_h).head(30)
        h_tab["symbol"] = h_tab["gene_id"].map(id2sym).fillna(h_tab["gene_id"])
        joint_df = pd.read_csv(joint_csv) if joint_csv.exists() else pd.DataFrame()
    else:
        h_tab = pd.DataFrame()
        joint_df = pd.DataFrame()

    fig = plt.figure(figsize=(11, 8))
    if has_human:
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9], wspace=0.55)
    else:
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.6, 0.4], wspace=0.55)
    color = TISSUE_COLOR.get(tissue, "#888")

    # ---- Pig top-30
    axp = fig.add_subplot(gs[0, 0])
    ypos = np.arange(len(pig_tab))[::-1]
    axp.barh(ypos, pig_tab["mean_gain"], color=color, edgecolor="black",
             linewidth=0.3)
    for y, (_, r) in zip(ypos, pig_tab.iterrows()):
        axp.text(r["mean_gain"], y, f" {int(r['n_seeds_top'])}/10",
                 va="center", fontsize=7, color="#444")
    axp.set_yticks(ypos)
    axp.set_yticklabels(pig_tab["symbol"], fontsize=7)
    axp.set_xlabel("Mean gain (LightGBM)")
    axp.set_title(f"Pig top-30 ({tissue})", fontweight="bold", color=color)
    axp.invert_xaxis() if False else None

    # ---- Human top-30
    axh = fig.add_subplot(gs[0, 1])
    if has_human and len(h_tab) > 0:
        ypos = np.arange(len(h_tab))[::-1]
        axh.barh(ypos, h_tab["mean_gain"], color=color, alpha=0.7,
                 edgecolor="black", linewidth=0.3)
        for y, (_, r) in zip(ypos, h_tab.iterrows()):
            axh.text(r["mean_gain"], y, f" {int(r['n_seeds_top'])}/10",
                     va="center", fontsize=7, color="#444")
        axh.set_yticks(ypos)
        axh.set_yticklabels(h_tab["symbol"], fontsize=7)
        axh.set_xlabel("Mean gain (LightGBM)")
        axh.set_title(f"Human top-30 ({tissue})", fontweight="bold", color=color)
    else:
        axh.text(0.5, 0.5,
                 ("Human postnatal RNA-seq matrix\n"
                  "unavailable for this tissue\n"
                  "(Cardoso-Moreira 2019 lacks\n"
                  f"{tissue.lower()} samples)."),
                 ha="center", va="center", transform=axh.transAxes,
                 fontsize=9, color="#666")
        axh.set_xticks([]); axh.set_yticks([])
        for sp in ("left", "bottom"):
            axh.spines[sp].set_visible(False)

    # ---- Joint set
    axj = fig.add_subplot(gs[0, 2])
    if has_human and len(joint_df) > 0:
        # Show top 30 joint genes by combined seed-stability
        joint_df = joint_df.copy()
        joint_df["combined_n"] = (
            joint_df["pig_n_seeds_top100"] + joint_df["human_n_seeds_top100"])
        jt = joint_df.sort_values(
            ["combined_n", "pig_mean_rank"],
            ascending=[False, True]
        ).head(30)
        ypos = np.arange(len(jt))[::-1]
        axj.scatter(jt["log2fc_pig"], ypos, s=22, color=color,
                    edgecolor="black", linewidth=0.4,
                    label="pig log2FC")
        axj.scatter(jt["log2fc_human"], ypos, s=22, color="white",
                    edgecolor=color, linewidth=1.2,
                    label="human log2FC")
        axj.axvline(0, color="grey", lw=0.4, ls="--")
        axj.set_yticks(ypos)
        axj.set_yticklabels(jt["symbol"], fontsize=7)
        axj.set_xlabel("log$_2$ FC (Adult / Infant)")
        axj.set_title(f"Joint set (n={len(joint_df)})", fontweight="bold",
                      color=color)
        axj.legend(loc="lower right", fontsize=7)
    else:
        n_j = res.get("joint_set_n", 0) if res else 0
        msg = (f"Joint set: 0 genes\n(no human data)" if not has_human
               else f"Joint set: {n_j} genes")
        axj.text(0.5, 0.5, msg, ha="center", va="center",
                 transform=axj.transAxes, fontsize=10, color="#666")
        axj.set_xticks([]); axj.set_yticks([])
        for sp in ("left", "bottom"):
            axj.spines[sp].set_visible(False)

    fig.suptitle(
        f"{tissue}: joint cross-species feature selection",
        fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT_DIR / f"fig_top_features_{tissue}.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def fig_concordance_panel() -> None:
    rows = []
    for t in TISSUES:
        rj = OUT_DIR / f"results_{t}.json"
        if not rj.exists():
            continue
        d = json.loads(rj.read_text())
        c = d.get("concordance") or {}
        rows.append({
            "tissue": t,
            "joint_n": d.get("joint_set_n", 0),
            "r": c.get("pearson_r"),
            "ci_low": c.get("ci_low"),
            "ci_high": c.get("ci_high"),
            "pct_concord": c.get("directional_concordance_pct"),
            "human_avail": d.get("human_available", False),
        })
    df = pd.DataFrame(rows)
    df["order"] = df["tissue"].apply(TISSUES.index)
    df = df.sort_values("order").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ypos = np.arange(len(df))[::-1]
    for y, (_, r) in zip(ypos, df.iterrows()):
        col = TISSUE_COLOR.get(r["tissue"], "#888")
        if r["r"] is None or (isinstance(r["r"], float) and np.isnan(r["r"])):
            txt = ("no joint set (no human matrix)" if not r["human_avail"]
                   else f"joint n={r['joint_n']}; r unavailable")
            ax.text(0.03, y, f"{r['tissue']}: {txt}",
                    va="center", ha="left",
                    transform=ax.get_yaxis_transform(),
                    fontsize=9, color="#666")
            continue
        if r["ci_low"] is not None and r["ci_high"] is not None:
            ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=col, lw=2.5,
                    solid_capstyle="round")
        ax.scatter([r["r"]], [y], s=90, color=col, edgecolor="black",
                   linewidth=0.5, zorder=4)
        ci = (f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
              if r["ci_low"] is not None else "")
        pct = (f"; {r['pct_concord']:.0f}% same-sign"
               if r["pct_concord"] is not None else "")
        ax.text(1.04, y,
                f"{r['tissue']}: r={r['r']:.2f} {ci}  "
                f"(n={r['joint_n']}{pct})",
                va="center", ha="left",
                transform=ax.get_yaxis_transform(),
                fontsize=9)
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(df["tissue"])
    ax.set_ylim(-0.5, len(df) - 0.5)
    ax.set_xlim(-0.2, 1.1)
    ax.set_xlabel(r"Pearson $r$ on joint conserved gene set "
                  r"(pig vs human log$_2$ FC, Infant$\to$Adult)")
    ax.set_title(
        "Cross-species concordance on joint ML-selected gene sets",
        fontweight="bold")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    out = OUT_DIR / "fig_concordance_panel.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def fig_method_comparison() -> None:
    """Side-by-side baseline FDR vs joint ML gene-set sizes."""
    # Baseline counts come from cross_species_all_tissues.json + the published
    # muscle 66-gene set
    baseline = {"Muscle": 66, "Brain": None, "Liver": None,
                "Lung": None, "Blood": None}
    if BASELINE_JSON.exists():
        bj = json.loads(BASELINE_JSON.read_text())
        for human_t, info in bj.get("per_tissue", {}).items():
            if "strict" in info and info["strict"] is not None:
                baseline[human_t] = info["strict"].get("n_genes")

    joint_sizes = {}
    for t in TISSUES:
        rj = OUT_DIR / f"results_{t}.json"
        if rj.exists():
            joint_sizes[t] = json.loads(rj.read_text()).get("joint_set_n", 0)
        else:
            joint_sizes[t] = 0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(TISSUES))
    w = 0.4
    base_vals = [baseline.get(t) if baseline.get(t) is not None else 0
                 for t in TISSUES]
    joint_vals = [joint_sizes.get(t, 0) for t in TISSUES]
    ax.bar(x - w / 2, base_vals, w, label="Baseline FDR + |log2FC|",
           color="#999999", edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, joint_vals, w, label="Joint ML stability",
           color=[TISSUE_COLOR[t] for t in TISSUES],
           edgecolor="black", linewidth=0.4)
    # annotate counts (NA for tissues without baseline)
    for i, t in enumerate(TISSUES):
        bv = baseline.get(t)
        label = "NA" if bv is None else str(bv)
        ax.text(i - w / 2, max(bv if bv else 0, 1) + 1, label, ha="center",
                fontsize=8, color="#444")
        ax.text(i + w / 2, max(joint_vals[i], 1) + 1, str(joint_vals[i]),
                ha="center", fontsize=8, color="#000")
    ax.set_xticks(x)
    ax.set_xticklabels(TISSUES)
    ax.set_ylabel("Number of cross-species genes")
    ax.set_title("Method comparison: baseline filtering vs joint ML "
                 "feature selection", fontweight="bold")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(0.01, 0.01,
             "Baseline: prior exploratory FDR<0.10 & |log2FC|>0.5 in both "
             "species (Cardoso-Moreira; muscle from prior figure). "
             "Joint ML: stable top-100 in >=8/10 seeds in both species.",
             fontsize=7, color="#555")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = OUT_DIR / "fig_method_comparison.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def main() -> None:
    ortho = load_ortho()
    for t in TISSUES:
        try:
            fig_top_features(t, ortho)
        except Exception as exc:
            print(f"  FAIL {t}: {exc}")
    fig_concordance_panel()
    fig_method_comparison()


if __name__ == "__main__":
    main()
