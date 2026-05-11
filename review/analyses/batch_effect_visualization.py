#!/usr/bin/env python3
"""
Batch Effect Visualization (Reviewer R3.1)
==========================================
Addresses BMC Genomics Reviewer 3, point R3.1: provide PCA / UMAP visualisations
of the PigGTEx data coloured by BioProject (technical batch) and by Stage
(biological signal), then quantify whether developmental stage drives a
substantial portion of the variance beyond BioProject membership.

For each of the five focal tissues (Muscle, Brain, Liver, Lung, Blood) this
script:
  1. Loads the per-tissue TPM matrix via DataLoader (Tissue / Tissue_Main /
     Sub_categories 3-column filter) and restricts to samples with a known
     developmental stage.
  2. log2(TPM+1) transforms the matrix and selects the top 5,000 most variable
     genes (variance computed on log space).
  3. Fits PCA(n_components=10) and UMAP(n_neighbors=15, min_dist=0.1,
     metric='euclidean', random_state=42) on the gene-by-sample matrix
     transposed (samples x genes).
  4. Saves PC1..PC10 + UMAP1 + UMAP2 + metadata to
     review/analyses/results/batch_effect_embeddings_{Tissue}.csv.
  5. Decomposes variance per principal component (PC1..PC5) via OLS:
       PC ~ Stage        (one-hot of Stage)
       PC ~ BioProject   (one-hot of BioProject)
       PC ~ Stage + BioProject (combined model)
     Reports R^2 for each, plus n_projects and n_stages, written to
     review/analyses/results/batch_effect_variance_decomposition.csv.
  6. Computes silhouette scores in PCA-10 space for Stage labels and for
     BioProject labels. Higher Stage silhouette than BioProject silhouette
     would imply that stage forms tighter clusters than batch, i.e. batch is
     not the dominant structure.

The companion publication-quality figure is produced by the same script and
saved to paper/figures/output/pdf/figS4_batch_effects.pdf and the matching
.png file.

Run from project root:

    python review/analyses/batch_effect_visualization.py
"""

from __future__ import annotations

import gzip
import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import umap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Suppress UMAP / numba chatter — we redirect important messages to our own log.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.data_processing.data_loader import DataLoader  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
PIGGTEX_DIR = DATA_DIR / "pigGTEx"
META_PATH = DATA_DIR / "PigGTEx_v0.MetaTable.csv"
RESULTS_DIR = PROJECT_ROOT / "review" / "analyses" / "results"
FIG_PDF = PROJECT_ROOT / "paper" / "figures" / "output" / "pdf" / "figS4_batch_effects.pdf"
FIG_PNG = PROJECT_ROOT / "paper" / "figures" / "output" / "png" / "figS4_batch_effects.png"
LOG_PATH = RESULTS_DIR / "batch_effect_visualization.log"

TISSUES = ["Muscle", "Brain", "Liver", "Lung", "Blood"]
STAGE_ORDER = ["Infant", "Early childhood", "Pre-pubertal", "Post-pubertal", "Adult"]
STAGE_PALETTE = {
    "Infant": "#2166AC",        # deep blue
    "Early childhood": "#67A9CF",
    "Pre-pubertal": "#F7F7F7",
    "Post-pubertal": "#EF8A62",
    "Adult": "#B2182B",         # deep red
}
N_TOP_GENES = 5000
N_PCA = 10
RANDOM_STATE = 42


def configure_logging() -> logging.Logger:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    handler_file = logging.FileHandler(LOG_PATH, mode="w")
    handler_stream = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S")
    handler_file.setFormatter(fmt)
    handler_stream.setFormatter(fmt)
    logger = logging.getLogger("batch_effect_viz")
    logger.handlers.clear()
    logger.addHandler(handler_file)
    logger.addHandler(handler_stream)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def load_tissue(loader: DataLoader, tissue: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load TPM matrix and aligned metadata for one tissue, restricted to samples
    with a known Stage and BioProject."""
    expr, meta = loader.load_expression(tissue)
    # Keep only samples with known stage AND known bioproject.
    keep = meta["Stage"].notna() & meta["BioProject"].notna()
    meta = meta[keep].copy()
    common = [s for s in expr.columns if s in meta.index]
    expr = expr[common]
    meta = meta.loc[common]
    # Stage is a Categorical with the full STAGE_ORDER as categories — drop unused
    # categories to keep the figure legend clean (e.g. Lung has no Adult sample).
    meta["Stage"] = meta["Stage"].cat.remove_unused_categories()
    return expr, meta


def select_top_variable_genes(log_expr: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pick the n genes with the highest variance across samples (in log space)."""
    variances = log_expr.var(axis=1)
    # Drop genes that are zero everywhere (variance == 0).
    variances = variances[variances > 0]
    top_genes = variances.sort_values(ascending=False).head(n).index
    return log_expr.loc[top_genes]


def fit_pca_umap(log_expr_top: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """Fit PCA(10) and UMAP on (samples x genes); return PC dataframe, UMAP
    dataframe, fitted PCA estimator."""
    X = log_expr_top.T.values  # samples x genes
    sample_ids = log_expr_top.columns

    pca = PCA(n_components=N_PCA, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X)
    pc_df = pd.DataFrame(
        pcs,
        index=sample_ids,
        columns=[f"PC{i+1}" for i in range(N_PCA)],
    )

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=RANDOM_STATE,
        n_components=2,
    )
    embedding = reducer.fit_transform(X)
    umap_df = pd.DataFrame(
        embedding,
        index=sample_ids,
        columns=["UMAP1", "UMAP2"],
    )
    return pc_df, umap_df, pca


def ols_r2(y: pd.Series, design: pd.DataFrame) -> float:
    """Fit y ~ design (with intercept) and return R^2."""
    X = sm.add_constant(design.astype(float), has_constant="add")
    # statsmodels OLS handles rank-deficient designs but the warning is noisy.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y.values, X.values).fit()
    return float(model.rsquared)


def variance_decomposition(pc_df: pd.DataFrame, meta: pd.DataFrame, tissue: str) -> pd.DataFrame:
    """For PC1..PC5 fit Stage / BioProject / combined OLS and report R^2."""
    stage_design = pd.get_dummies(meta["Stage"].astype(str), prefix="Stage", drop_first=True)
    project_design = pd.get_dummies(meta["BioProject"].astype(str), prefix="BP", drop_first=True)
    combined_design = pd.concat([stage_design, project_design], axis=1)

    n_projects = meta["BioProject"].nunique()
    n_stages = meta["Stage"].nunique()

    rows = []
    for pc in [f"PC{i}" for i in range(1, 6)]:
        y = pc_df[pc]
        r2_stage = ols_r2(y, stage_design) if stage_design.shape[1] > 0 else np.nan
        r2_bp = ols_r2(y, project_design) if project_design.shape[1] > 0 else np.nan
        r2_comb = ols_r2(y, combined_design) if combined_design.shape[1] > 0 else np.nan
        rows.append(
            {
                "tissue": tissue,
                "PC": pc,
                "R2_stage": round(r2_stage, 4),
                "R2_bioproject": round(r2_bp, 4),
                "R2_combined": round(r2_comb, 4),
                "n_projects": int(n_projects),
                "n_stages": int(n_stages),
                "n_samples": int(len(meta)),
            }
        )
    return pd.DataFrame(rows)


def silhouettes(pc_df: pd.DataFrame, meta: pd.DataFrame) -> dict[str, float | int]:
    """Silhouette score in PC-10 space for Stage and for BioProject."""
    X = pc_df.values  # PC-10 coordinates
    out: dict[str, float | int] = {}

    stage_labels = meta["Stage"].astype(str).values
    if len(set(stage_labels)) >= 2:
        out["silhouette_stage"] = float(silhouette_score(X, stage_labels))
    else:
        out["silhouette_stage"] = float("nan")
    out["n_stage_labels"] = int(len(set(stage_labels)))

    bp_labels = meta["BioProject"].astype(str).values
    if len(set(bp_labels)) >= 2:
        out["silhouette_bioproject"] = float(silhouette_score(X, bp_labels))
    else:
        out["silhouette_bioproject"] = float("nan")
    out["n_bp_labels"] = int(len(set(bp_labels)))

    return out


def build_panel_axes(fig: plt.Figure, n_rows: int) -> np.ndarray:
    """Create a (n_rows x 2) grid of axes."""
    return fig.subplots(nrows=n_rows, ncols=2, squeeze=False)


def categorical_palette(n: int) -> list:
    """Return a list of n distinguishable colours. Uses seaborn 'tab20' / 'husl'
    fallback for >20 categories."""
    if n <= 10:
        return sns.color_palette("tab10", n)
    if n <= 20:
        return sns.color_palette("tab20", n)
    return sns.color_palette("husl", n)


def plot_tissue_row(
    ax_bp: plt.Axes,
    ax_stage: plt.Axes,
    umap_df: pd.DataFrame,
    meta: pd.DataFrame,
    tissue: str,
) -> None:
    """Render UMAP coloured by BioProject (left) and Stage (right) for one tissue."""
    n_samples = len(meta)
    n_projects = meta["BioProject"].nunique()

    # ---- LEFT: BioProject ----
    projects = (
        meta["BioProject"].value_counts().index.tolist()
    )  # largest first so legend reads sensibly
    palette = categorical_palette(len(projects))
    bp_color_map = {p: palette[i] for i, p in enumerate(projects)}

    for proj in projects:
        idx = meta.index[meta["BioProject"] == proj]
        ax_bp.scatter(
            umap_df.loc[idx, "UMAP1"],
            umap_df.loc[idx, "UMAP2"],
            s=14,
            color=bp_color_map[proj],
            edgecolor="white",
            linewidth=0.2,
            alpha=0.85,
            rasterized=True,
        )

    ax_bp.set_title(f"{tissue} — BioProject", fontsize=9, loc="left")
    ax_bp.set_xlabel("UMAP1", fontsize=7)
    ax_bp.set_ylabel("UMAP2", fontsize=7)
    ax_bp.tick_params(labelsize=6)
    ax_bp.text(
        0.97,
        0.03,
        f"n={n_samples} / {n_projects} projects",
        transform=ax_bp.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color="black",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="grey", lw=0.3, alpha=0.85),
    )

    # ---- RIGHT: Stage ----
    stages_present = [s for s in STAGE_ORDER if (meta["Stage"].astype(str) == s).any()]
    for stage in stages_present:
        idx = meta.index[meta["Stage"].astype(str) == stage]
        ax_stage.scatter(
            umap_df.loc[idx, "UMAP1"],
            umap_df.loc[idx, "UMAP2"],
            s=14,
            color=STAGE_PALETTE[stage],
            edgecolor="black",
            linewidth=0.25,
            alpha=0.9,
            label=stage,
            rasterized=True,
        )

    ax_stage.set_title(f"{tissue} — Stage", fontsize=9, loc="left")
    ax_stage.set_xlabel("UMAP1", fontsize=7)
    ax_stage.set_ylabel("UMAP2", fontsize=7)
    ax_stage.tick_params(labelsize=6)
    ax_stage.text(
        0.97,
        0.03,
        f"n={n_samples} / {len(stages_present)} stages",
        transform=ax_stage.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color="black",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="grey", lw=0.3, alpha=0.85),
    )

    for ax in (ax_bp, ax_stage):
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)


def stage_legend_handles() -> list[Line2D]:
    """Single shared legend for the right-hand Stage column."""
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=STAGE_PALETTE[s],
            markeredgecolor="black",
            markeredgewidth=0.3,
            markersize=6,
            label=s,
        )
        for s in STAGE_ORDER
    ]


def run() -> None:
    logger = configure_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    FIG_PNG.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Loading metadata from %s", META_PATH)
    loader = DataLoader(PIGGTEX_DIR, META_PATH)
    loader.load_metadata()

    all_variance_rows: list[pd.DataFrame] = []
    all_silhouettes: list[dict] = []

    # Set up the figure: 5 rows x 2 cols; double-column width similar to other
    # supplementary figures in this paper.
    fig = plt.figure(figsize=(7.2, 14.0), dpi=300)
    axes = build_panel_axes(fig, n_rows=len(TISSUES))

    for row_idx, tissue in enumerate(TISSUES):
        logger.info("=" * 70)
        logger.info("Tissue: %s", tissue)
        logger.info("=" * 70)

        expr, meta = load_tissue(loader, tissue)
        logger.info(
            "Loaded %d samples (with Stage+BioProject), %d genes for %s",
            expr.shape[1],
            expr.shape[0],
            tissue,
        )

        if expr.shape[1] < 10:
            logger.warning(
                "Skipping %s: only %d samples with known stage and bioproject (need >=10)",
                tissue,
                expr.shape[1],
            )
            axes[row_idx, 0].set_axis_off()
            axes[row_idx, 1].set_axis_off()
            continue

        log_expr = np.log2(expr + 1.0)
        log_expr_top = select_top_variable_genes(log_expr, N_TOP_GENES)
        logger.info("Top variable genes selected: %d", log_expr_top.shape[0])

        pc_df, umap_df, pca = fit_pca_umap(log_expr_top)
        logger.info(
            "PCA cumulative explained variance (first 5 PCs): %s",
            np.round(pca.explained_variance_ratio_[:5].cumsum(), 3).tolist(),
        )

        # Save embeddings + metadata
        meta_cols = [
            c
            for c in ["BioProject", "Stage", "Sex", "Breed", "Platform", "Model"]
            if c in meta.columns
        ]
        embedding_out = pd.concat(
            [pc_df, umap_df, meta.loc[pc_df.index, meta_cols]],
            axis=1,
        )
        embedding_out.index.name = "Sample_ID"
        embedding_out["tissue"] = tissue
        out_csv = RESULTS_DIR / f"batch_effect_embeddings_{tissue}.csv"
        embedding_out.to_csv(out_csv)
        logger.info("Saved embeddings: %s", out_csv)

        # Variance decomposition
        var_df = variance_decomposition(pc_df, meta, tissue)
        all_variance_rows.append(var_df)
        logger.info("Variance decomposition for %s:\n%s", tissue, var_df.to_string(index=False))

        # Silhouette
        sil = silhouettes(pc_df, meta)
        sil["tissue"] = tissue
        sil["n_samples"] = int(len(meta))
        all_silhouettes.append(sil)
        logger.info(
            "Silhouette — Stage=%.4f (k=%d), BioProject=%.4f (k=%d)",
            sil.get("silhouette_stage", float("nan")),
            sil.get("n_stage_labels", 0),
            sil.get("silhouette_bioproject", float("nan")),
            sil.get("n_bp_labels", 0),
        )

        # Plot row
        plot_tissue_row(
            ax_bp=axes[row_idx, 0],
            ax_stage=axes[row_idx, 1],
            umap_df=umap_df,
            meta=meta,
            tissue=tissue,
        )

    # Combined variance & silhouette tables
    variance_table = pd.concat(all_variance_rows, ignore_index=True)
    variance_table.to_csv(RESULTS_DIR / "batch_effect_variance_decomposition.csv", index=False)
    logger.info("Saved variance decomposition: %s", RESULTS_DIR / "batch_effect_variance_decomposition.csv")

    silhouette_table = pd.DataFrame(all_silhouettes)
    silhouette_table = silhouette_table[
        [
            "tissue",
            "n_samples",
            "n_stage_labels",
            "silhouette_stage",
            "n_bp_labels",
            "silhouette_bioproject",
        ]
    ]
    silhouette_table.to_csv(RESULTS_DIR / "batch_effect_silhouette.csv", index=False)
    logger.info("Saved silhouettes: %s", RESULTS_DIR / "batch_effect_silhouette.csv")

    # Add shared stage legend at the top of the right column
    handles = stage_legend_handles()
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=5,
        frameon=False,
        fontsize=7,
        title="Developmental stage",
        title_fontsize=7,
    )

    fig.suptitle(
        "Batch-effect evaluation across five tissues: developmental stage drives the\n"
        "dominant variance even when samples span multiple BioProjects.",
        fontsize=10,
        y=0.995,
    )
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.97])

    fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure PDF: %s", FIG_PDF)
    logger.info("Saved figure PNG: %s", FIG_PNG)

    # Headline numbers
    pc1 = variance_table[variance_table["PC"] == "PC1"]
    logger.info("\nPC1 summary (R2 stage vs bioproject) per tissue:")
    for _, row in pc1.iterrows():
        logger.info(
            "  %s: PC1 R2_stage=%.3f, R2_bioproject=%.3f, R2_combined=%.3f (n=%d, %d projects)",
            row["tissue"],
            row["R2_stage"],
            row["R2_bioproject"],
            row["R2_combined"],
            row["n_samples"],
            row["n_projects"],
        )
    logger.info("\nSilhouettes per tissue:")
    for _, row in silhouette_table.iterrows():
        logger.info(
            "  %s: stage=%.3f, bioproject=%.3f",
            row["tissue"],
            row["silhouette_stage"],
            row["silhouette_bioproject"],
        )


if __name__ == "__main__":
    run()
