#!/usr/bin/env python3
"""Generate figS8_heterogeneity.pdf/png from the precomputed CSVs.

Three panels:
  (a) Sample-count heatmap: tissue (rows) x stage (cols), one heatmap per
      subgroup dimension (Breed, Sex, BioProject) shown as small multiples.
      Cells are coloured by within-stage proportion of the dominant level
      (a proxy for stratification imbalance: 1.0 = single subgroup
      dominates; 0.2 = perfectly even mix of 5 levels).
  (b) Per-stratum BA strip plot per tissue (Breed and Sex pooled), with the
      tissue marginal BA drawn as a dashed line.
  (c) Permutation null distribution vs. observed std(BA) for muscle (the
      tissue with the most subgroup levels and the most significant
      permutation result).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "review" / "analyses" / "results"
FIG_PDF = PROJECT_ROOT / "paper" / "figures" / "output" / "pdf" / "figS8_heterogeneity.pdf"
FIG_PNG = PROJECT_ROOT / "paper" / "figures" / "output" / "png" / "figS8_heterogeneity.png"
FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
FIG_PNG.parent.mkdir(parents=True, exist_ok=True)

STAGE_ORDER = ["Infant", "Early childhood", "Pre-pubertal", "Post-pubertal", "Adult"]
TISSUES = ["Muscle", "Brain", "Liver", "Blood", "Lung"]

audit = pd.read_csv(RESULTS_DIR / "heterogeneity_stratification.csv")
perf = pd.read_csv(RESULTS_DIR / "heterogeneity_subgroup_performance.csv")
perm = pd.read_csv(RESULTS_DIR / "heterogeneity_permutation_pvalues.csv")
preds = pd.read_csv(RESULTS_DIR / "heterogeneity_predictions.csv")


# ---------------------------------------------------------------------------
# Panel (a): dominance heatmap
# ---------------------------------------------------------------------------

def dominance_matrix(dimension: str) -> pd.DataFrame:
    """For each (tissue, stage), compute fraction of samples in the dominant
    subgroup level under `dimension`. 1.0 = monolithic; 1/k = perfectly even.
    Returns matrix with index=tissue, columns=stage, values in [0,1] or NaN.
    """
    rows = []
    for t in TISSUES:
        sub = audit[(audit.tissue == t) & (audit.dimension == dimension)]
        for s in STAGE_ORDER:
            ss = sub[sub.stage == s]
            if ss.empty:
                rows.append({"tissue": t, "stage": s, "dominance": np.nan, "total": 0})
                continue
            total = ss["n_samples"].sum()
            dominance = ss["n_samples"].max() / total if total > 0 else np.nan
            rows.append({"tissue": t, "stage": s, "dominance": dominance, "total": int(total)})
    df = pd.DataFrame(rows)
    mat = df.pivot(index="tissue", columns="stage", values="dominance").reindex(index=TISSUES, columns=STAGE_ORDER)
    tot = df.pivot(index="tissue", columns="stage", values="total").reindex(index=TISSUES, columns=STAGE_ORDER)
    return mat, tot


# ---------------------------------------------------------------------------
# Panel (c): permutation null for muscle/breed (largest n_levels)
# Re-run a small permutation to plot the null distribution (saves doing
# 1000-vector storage in CSV; this re-uses the cached CV predictions).
# ---------------------------------------------------------------------------

def permutation_null_for_panel(tissue: str = "Muscle", dimension: str = "Sex",
                                n_permutations: int = 1000, seed: int = 42,
                                min_group_n: int = 10) -> tuple:
    """Recompute null distribution of std(BA) for plotting."""
    from machine_learning.data_processing.data_loader import DataLoader

    # Get tissue metadata
    dl = DataLoader(PROJECT_ROOT / "data/pigGTEx", PROJECT_ROOT / "data/PigGTEx_v0.MetaTable.csv")
    dl.load_metadata()
    meta_full = dl.metadata
    tissue_meta = meta_full[
        (meta_full["Tissue"] == tissue)
        & (meta_full["Tissue_Main"] == tissue)
        & (meta_full["Sub_categories"] == tissue)
    ].copy()

    tp = preds[preds.tissue == tissue].copy()
    tp.set_index("sample_id", inplace=True)
    md = tissue_meta.loc[tp.index].copy()

    # Normalize subgroup column
    if dimension == "Sex":
        norm = md["Sex"].astype(str).str.strip().str.lower().map(lambda v: {
            "male": "Male", "female": "Female", "pooled": "Pooled",
            "neuter": "Neuter", "unknown": "Unknown", "nan": "Unknown", "": "Unknown",
            "hermaphrodite": "Hermaphrodite",
        }.get(v, "Unknown"))
        md["dim"] = norm
    else:
        md["dim"] = md[dimension].astype(str).fillna("Unknown")

    # Identify usable levels
    counts = md["dim"].value_counts()
    usable = []
    for lvl, n in counts.items():
        if n < min_group_n:
            continue
        sub = md[md["dim"] == lvl]
        if tp.loc[sub.index, "y_true"].nunique() < 2:
            continue
        usable.append(lvl)

    md_use = md[md["dim"].isin(usable)].copy()
    p_use = tp.loc[md_use.index].copy()
    obs_labels = md_use["dim"].values
    stage_array = md_use["Stage"].astype(str).values

    def spread(label_arr):
        bas = []
        for lvl in usable:
            mask = (label_arr == lvl)
            if mask.sum() < min_group_n:
                continue
            sp = p_use.iloc[mask.nonzero()[0]]
            if sp["y_true"].nunique() < 2:
                continue
            bas.append(balanced_accuracy_score(sp["y_true"], sp["y_pred"]))
        return float(np.std(bas)) if len(bas) >= 2 else np.nan

    observed = spread(obs_labels)
    rng = np.random.default_rng(seed)
    stage_to_idx = {s: np.where(stage_array == s)[0] for s in np.unique(stage_array)}
    null = []
    for _ in range(n_permutations):
        permuted = obs_labels.copy()
        for idx in stage_to_idx.values():
            sub = permuted[idx]
            rng.shuffle(sub)
            permuted[idx] = sub
        s = spread(permuted)
        if not np.isnan(s):
            null.append(s)
    return observed, np.asarray(null), usable


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
})

fig = plt.figure(figsize=(13, 9), constrained_layout=False)
gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.5, 1.2], hspace=0.55, wspace=0.45,
                       left=0.07, right=0.97, top=0.94, bottom=0.07)

# ----- Panel (a): three heatmaps -----
for j, dim in enumerate(["Breed", "Sex", "BioProject"]):
    ax = fig.add_subplot(gs[0, j])
    mat, tot = dominance_matrix(dim)
    im = ax.imshow(mat.values, cmap="YlOrRd", vmin=0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_ORDER, rotation=30, ha="right")
    ax.set_yticks(range(len(TISSUES)))
    ax.set_yticklabels(TISSUES)
    for i, t in enumerate(TISSUES):
        for k, s in enumerate(STAGE_ORDER):
            val = mat.values[i, k]
            n = tot.values[i, k]
            if np.isnan(val):
                ax.text(k, i, "-", ha="center", va="center", fontsize=8, color="grey")
            else:
                txtcolor = "white" if val > 0.65 else "black"
                ax.text(k, i, f"{val:.2f}\nn={int(n)}", ha="center", va="center",
                        fontsize=7, color=txtcolor)
    ax.set_title(f"({chr(97)}{j+1}) Dominance: {dim}", loc="left")
    if j == 2:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Top-level\nfraction", fontsize=8)

# ----- Panel (b): per-stratum BA strip+box plot -----
ax_b = fig.add_subplot(gs[1, :])
# Only Breed + Sex, exclude Overall rows and tissues with <2 levels
plot_df = perf[(perf.dimension.isin(["Breed", "Sex"]))].copy()
# Skip tissues that have only one subgroup level (no spread)
keep = plot_df.groupby(["tissue", "dimension"]).filter(lambda g: len(g) >= 2)
keep["tissue_dim"] = keep["tissue"] + "\n" + keep["dimension"]
order = sorted(keep["tissue_dim"].unique())
# Strip plot with size scaled by n_samples
size_min, size_max = 30, 280
keep["size_pt"] = size_min + (keep["n_samples"] - keep["n_samples"].min()) / (
    keep["n_samples"].max() - keep["n_samples"].min() + 1e-9
) * (size_max - size_min)

# Marginal BA per tissue
marginal = perf[perf.dimension == "Overall"].set_index("tissue")["balanced_accuracy"]

palette = {"Breed": "#1f77b4", "Sex": "#ff7f0e"}
for i, td in enumerate(order):
    sub = keep[keep["tissue_dim"] == td]
    color = palette[sub["dimension"].iloc[0]]
    jitter = (np.random.RandomState(i).rand(len(sub)) - 0.5) * 0.35
    ax_b.scatter(np.full(len(sub), i) + jitter, sub["balanced_accuracy"],
                  s=sub["size_pt"], alpha=0.7, edgecolor="white", linewidth=0.8,
                  color=color, zorder=3)
    # Label each point with the level
    for x_off, (_, row) in zip(jitter, sub.iterrows()):
        ax_b.text(i + x_off, row["balanced_accuracy"] + 0.012,
                   str(row["level"])[:18], ha="center", va="bottom",
                   fontsize=6.5, color="#333333", rotation=0)
    # Draw marginal BA as dashed line
    t = sub["tissue"].iloc[0]
    if t in marginal.index:
        ax_b.hlines(marginal[t], i - 0.4, i + 0.4, colors="black", linestyles="--",
                     linewidth=1.3, zorder=2)

ax_b.set_xticks(range(len(order)))
ax_b.set_xticklabels(order)
ax_b.set_ylim(0.4, 1.05)
ax_b.set_ylabel("Balanced accuracy (CV predictions, subgroup-stratified)")
ax_b.set_title("(b) Subgroup performance vs. marginal BA (dashed black). Point size scales with n.", loc="left")

# Legend (colours and a size-reference)
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker="o", linestyle="None", color="w", markerfacecolor=palette["Breed"],
            markersize=8, label="Breed level"),
    Line2D([0], [0], marker="o", linestyle="None", color="w", markerfacecolor=palette["Sex"],
            markersize=8, label="Sex level"),
    Line2D([0], [0], color="black", linestyle="--", label="Tissue marginal BA"),
]
ax_b.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

# ----- Panel (c): permutation null distribution for Muscle / Sex (most significant) -----
ax_c = fig.add_subplot(gs[2, 0])
obs_ms, null_ms, levels_ms = permutation_null_for_panel("Muscle", "Sex", n_permutations=1000, seed=42)
ax_c.hist(null_ms, bins=30, color="#cccccc", edgecolor="white", label="Null (1000 perms)")
ax_c.axvline(obs_ms, color="#d62728", linewidth=2.2, label=f"Observed = {obs_ms:.3f}")
pval = (null_ms >= obs_ms).sum() / len(null_ms)
ax_c.set_xlabel("std(BA) across Sex subgroups")
ax_c.set_ylabel("Permutation count")
ax_c.set_title(f"(c) Muscle / Sex: permutation p = {pval:.3f}", loc="left")
ax_c.legend(fontsize=7, loc="upper right")

# Muscle / BioProject
ax_d = fig.add_subplot(gs[2, 1])
obs_mp, null_mp, levels_mp = permutation_null_for_panel("Muscle", "BioProject", n_permutations=1000, seed=42)
ax_d.hist(null_mp, bins=30, color="#cccccc", edgecolor="white", label="Null (1000 perms)")
ax_d.axvline(obs_mp, color="#d62728", linewidth=2.2, label=f"Observed = {obs_mp:.3f}")
pval2 = (null_mp >= obs_mp).sum() / len(null_mp)
ax_d.set_xlabel("std(BA) across BioProject subgroups")
ax_d.set_ylabel("Permutation count")
ax_d.set_title(f"(d) Muscle / BioProject: p = {pval2:.3f}", loc="left")
ax_d.legend(fontsize=7, loc="upper right")

# Liver / Breed (also significant)
ax_e = fig.add_subplot(gs[2, 2])
obs_lb, null_lb, levels_lb = permutation_null_for_panel("Liver", "Breed", n_permutations=1000, seed=42)
ax_e.hist(null_lb, bins=30, color="#cccccc", edgecolor="white", label="Null (1000 perms)")
ax_e.axvline(obs_lb, color="#d62728", linewidth=2.2, label=f"Observed = {obs_lb:.3f}")
pval3 = (null_lb >= obs_lb).sum() / len(null_lb)
ax_e.set_xlabel("std(BA) across Breed subgroups")
ax_e.set_ylabel("Permutation count")
ax_e.set_title(f"(e) Liver / Breed: p = {pval3:.3f}", loc="left")
ax_e.legend(fontsize=7, loc="upper right")

fig.suptitle("Figure S8 - Within-stage subgroup heterogeneity and its effect on per-tissue classifier performance",
              fontsize=11, y=0.985)

fig.savefig(FIG_PDF, bbox_inches="tight")
fig.savefig(FIG_PNG, bbox_inches="tight", dpi=300)
print(f"Wrote {FIG_PDF}")
print(f"Wrote {FIG_PNG}")
print(f"Panel (c) Muscle/Sex: observed std={obs_ms:.3f}, p={pval:.3f}")
print(f"Panel (d) Muscle/BioProject: observed std={obs_mp:.3f}, p={pval2:.3f}")
print(f"Panel (e) Liver/Breed: observed std={obs_lb:.3f}, p={pval3:.3f}")
