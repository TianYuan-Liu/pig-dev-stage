#!/usr/bin/env python3
"""
Batch Effect Diagnostic Analysis for Developmental Stage Classification.

Quantifies whether batch variables (BioProject, Breed_group, LibraryLayout,
Platform) confound the developmental stage classifier. Four analyses:

  1. PCA Visualization — PC1 vs PC2 colored by Stage and batch variables
  2. PERMANOVA Variance Partitioning — R² for each factor
  3. Cramer's V Confounding — association between Stage and batch variables
  4. Silhouette Analysis — clustering coherence by Stage vs BioProject
"""

import sys
import json
import time
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.data_processing.data_loader import DataLoader
from machine_learning.data_processing.preprocessing import ExpressionPreprocessor
from machine_learning.utils.helpers import load_and_prepare_tissue, to_samples_x_genes_df

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DEFAULT_TISSUES = ["Muscle", "Liver", "Brain", "Blood", "Lung"]
BATCH_COLUMNS = ["TechBatch", "BioProject", "Breed_group", "LibraryLayout", "Platform"]
N_PCA_COMPONENTS = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_batch_metadata() -> pd.DataFrame:
    """Load full metadata table and return it indexed by Sample_ID."""
    loader = DataLoader(
        data_dir=PROJECT_ROOT / "data/pigGTEx",
        metadata_path=PROJECT_ROOT / "data/PigGTEx_v0.MetaTable.csv",
    )
    meta = loader.load_metadata()
    return meta


def _prepare_tissue_data(
    tissue_name: str,
) -> Optional[Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, str, pd.DataFrame]]:
    """
    Load expression + stage labels and merge batch metadata.

    Returns
    -------
    (X_log_scaled, y, sample_ids, gene_names, scheme_name, batch_df) or None.
    X_log_scaled is samples x genes after log2(TPM+1) + StandardScaler.
    batch_df has columns from BATCH_COLUMNS, indexed by sample_ids.
    """
    prepared = load_and_prepare_tissue(tissue_name)
    if prepared is None:
        return None

    X_raw, y, sample_ids, gene_names, scheme_name = prepared

    # log2(TPM+1) and transpose to samples x genes
    X_log = np.log2(X_raw.values.T + 1)  # (n_samples, n_genes)

    # StandardScaler for PCA / distance-based analyses
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    X_scaled_df = pd.DataFrame(X_scaled, index=sample_ids, columns=gene_names)

    # Load batch metadata and align
    full_meta = _load_batch_metadata()
    available_cols = [c for c in BATCH_COLUMNS if c in full_meta.columns]
    batch_df = full_meta.loc[full_meta.index.intersection(sample_ids), available_cols].copy()
    # Reindex to match sample order
    batch_df = batch_df.reindex(sample_ids)

    return X_scaled_df, y, sample_ids, gene_names, scheme_name, batch_df


def _pca_reduce(X: np.ndarray, n_components: int = N_PCA_COMPONENTS) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X_pca, explained_variance_ratio) using up to n_components PCs."""
    n_components = min(n_components, min(X.shape))
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca, pca.explained_variance_ratio_


# ===================================================================
# Analysis 1: PCA Visualization
# ===================================================================

def run_pca_visualization(
    X_scaled: pd.DataFrame,
    y: np.ndarray,
    batch_df: pd.DataFrame,
    tissue_name: str,
    output_dir: Path,
) -> Dict:
    """PCA colored by Stage and each batch variable. Saves a 4-panel figure."""

    X_pca, var_ratio = _pca_reduce(X_scaled.values, n_components=min(10, min(X_scaled.shape)))

    # Determine label columns
    label_map = {"Stage": pd.Series(y, index=X_scaled.index)}
    for col in BATCH_COLUMNS:
        if col in batch_df.columns and batch_df[col].notna().sum() > 0:
            label_map[col] = batch_df[col]

    n_panels = len(label_map)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, (label_name, labels) in zip(axes, label_map.items()):
        series = labels.copy()
        # For BioProject: keep top 8 and group rest as "Other"
        if label_name == "BioProject":
            top8 = series.value_counts().head(8).index
            series = series.apply(lambda x: x if x in top8 else "Other")

        unique_labels = sorted(series.dropna().unique(), key=str)
        n_colors = len(unique_labels)
        cmap = cm.get_cmap("tab20" if n_colors > 10 else "tab10", max(n_colors, 1))
        color_dict = {lbl: cmap(i / max(n_colors - 1, 1)) for i, lbl in enumerate(unique_labels)}

        for lbl in unique_labels:
            mask = series == lbl
            ax.scatter(
                X_pca[mask.values, 0],
                X_pca[mask.values, 1],
                c=[color_dict[lbl]],
                label=str(lbl),
                s=15,
                alpha=0.7,
            )

        ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%})")
        ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%})")
        ax.set_title(f"{tissue_name} — {label_name}")
        # Legend: at most 12 entries to avoid overflow
        if n_colors <= 12:
            ax.legend(fontsize=6, markerscale=1.5, loc="best")

    plt.tight_layout()
    fig_path = output_dir / f"{tissue_name}_pca_batch.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    result = {
        "variance_explained_top10": [float(v) for v in var_ratio[:10]],
        "cumulative_var_top10": float(var_ratio[:10].sum()),
        "figure": str(fig_path),
    }
    print(f"  PCA figure saved: {fig_path}")
    return result


# ===================================================================
# Analysis 1b: UMAP Visualization
# ===================================================================

def run_umap_visualization(
    X_scaled: pd.DataFrame,
    y: np.ndarray,
    batch_df: pd.DataFrame,
    tissue_name: str,
    output_dir: Path,
) -> Optional[Dict]:
    """UMAP colored by Stage and BioProject. Saves a 2-panel figure.

    Requires umap-learn; returns None if not installed.
    """
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed, skipping UMAP visualization")
        return None

    # PCA pre-reduction (standard UMAP best practice)
    X_pca, _ = _pca_reduce(X_scaled.values, n_components=N_PCA_COMPONENTS)
    n_pca_used = X_pca.shape[1]

    # UMAP embedding
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42,
        metric="euclidean",
    )
    X_umap = reducer.fit_transform(X_pca)

    # Build label panels: Stage + BioProject
    label_map = {"Stage": pd.Series(y, index=X_scaled.index)}
    if "BioProject" in batch_df.columns and batch_df["BioProject"].notna().sum() > 0:
        label_map["BioProject"] = batch_df["BioProject"]

    n_panels = len(label_map)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, (label_name, labels) in zip(axes, label_map.items()):
        series = labels.copy()
        # For BioProject: keep top 8 and group rest as "Other"
        if label_name == "BioProject":
            top8 = series.value_counts().head(8).index
            series = series.apply(lambda x: x if x in top8 else "Other")

        unique_labels = sorted(series.dropna().unique(), key=str)
        n_colors = len(unique_labels)
        cmap = cm.get_cmap("tab20" if n_colors > 10 else "tab10", max(n_colors, 1))
        color_dict = {lbl: cmap(i / max(n_colors - 1, 1)) for i, lbl in enumerate(unique_labels)}

        for lbl in unique_labels:
            mask = series == lbl
            ax.scatter(
                X_umap[mask.values, 0],
                X_umap[mask.values, 1],
                c=[color_dict[lbl]],
                label=str(lbl),
                s=15,
                alpha=0.7,
            )

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"{tissue_name} — {label_name}")
        if n_colors <= 12:
            ax.legend(fontsize=6, markerscale=1.5, loc="best")

    plt.tight_layout()
    fig_path = output_dir / f"{tissue_name}_umap_batch.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    result = {
        "n_neighbors": 15,
        "min_dist": 0.1,
        "pca_components_input": int(n_pca_used),
        "figure": str(fig_path),
    }
    print(f"  UMAP figure saved: {fig_path}")
    return result


# ===================================================================
# Analysis 2: PERMANOVA Variance Partitioning
# ===================================================================

def _permanova(distance_matrix: np.ndarray, grouping: np.ndarray, n_permutations: int = 999) -> Dict:
    """
    Pure-numpy PERMANOVA (Anderson 2001).

    Parameters
    ----------
    distance_matrix : (n, n) square distance matrix
    grouping : (n,) group labels (integers or strings)
    n_permutations : number of permutations for p-value

    Returns
    -------
    dict with R2, F_statistic, p_value
    """
    n = len(grouping)
    groups = pd.Categorical(grouping)
    codes = groups.codes
    unique_groups = np.unique(codes)
    n_groups = len(unique_groups)

    if n_groups < 2 or n_groups >= n:
        return {"R2": np.nan, "F_statistic": np.nan, "p_value": np.nan}

    # Squared distances (Gower's centered matrix approach)
    D2 = distance_matrix ** 2

    # Total sum of squares
    SS_total = D2.sum() / (2 * n)

    def _compute_ss_within(codes_arr):
        ss_w = 0.0
        for g in unique_groups:
            mask = codes_arr == g
            n_g = mask.sum()
            if n_g > 1:
                D2_g = D2[np.ix_(mask, mask)]
                ss_w += D2_g.sum() / (2 * n_g)
        return ss_w

    SS_within = _compute_ss_within(codes)
    SS_between = SS_total - SS_within

    R2 = SS_between / SS_total if SS_total > 0 else 0.0

    df_between = n_groups - 1
    df_within = n - n_groups
    F_stat = (SS_between / df_between) / (SS_within / df_within) if df_within > 0 and SS_within > 0 else 0.0

    # Permutation test
    rng = np.random.RandomState(42)
    n_ge = 1  # include observed
    for _ in range(n_permutations):
        perm_codes = rng.permutation(codes)
        ss_w_perm = _compute_ss_within(perm_codes)
        ss_b_perm = SS_total - ss_w_perm
        f_perm = (ss_b_perm / df_between) / (ss_w_perm / df_within) if df_within > 0 and ss_w_perm > 0 else 0.0
        if f_perm >= F_stat:
            n_ge += 1

    p_value = n_ge / (n_permutations + 1)

    return {"R2": float(R2), "F_statistic": float(F_stat), "p_value": float(p_value)}


def run_permanova(
    X_scaled: pd.DataFrame,
    y: np.ndarray,
    batch_df: pd.DataFrame,
    tissue_name: str,
    n_permutations: int = 999,
) -> Dict:
    """Run PERMANOVA for Stage and each batch variable."""

    print(f"  PERMANOVA (PCA-reduced, {n_permutations} permutations)...")

    X_pca, _ = _pca_reduce(X_scaled.values, n_components=N_PCA_COMPONENTS)
    dist_matrix = squareform(pdist(X_pca, metric="euclidean"))

    results = {}

    # Stage
    res = _permanova(dist_matrix, y, n_permutations)
    results["Stage"] = res
    print(f"    Stage:          R²={res['R2']:.4f}  p={res['p_value']:.4f}")

    # Batch variables
    for col in BATCH_COLUMNS:
        if col not in batch_df.columns:
            continue
        vals = batch_df[col].values
        valid = pd.notna(vals)
        if valid.sum() < 10:
            continue

        # Use all samples, filling NaN with "Unknown"
        grouping = np.where(pd.isna(vals), "Unknown", vals.astype(str))
        n_unique = len(np.unique(grouping))
        if n_unique < 2:
            continue

        res = _permanova(dist_matrix, grouping, n_permutations)
        results[col] = res
        print(f"    {col:16s}: R²={res['R2']:.4f}  p={res['p_value']:.4f}")

    # Flag if any batch R² > Stage R²
    stage_r2 = results.get("Stage", {}).get("R2", 0)
    batch_vars = {k: v for k, v in results.items() if k != "Stage"}
    max_batch_r2 = max((v["R2"] for v in batch_vars.values()), default=0)
    if max_batch_r2 > stage_r2 and not np.isnan(max_batch_r2):
        dominant = max(batch_vars, key=lambda k: batch_vars[k]["R2"])
        results["warning"] = (
            f"Batch variable '{dominant}' explains more variance (R²={max_batch_r2:.4f}) "
            f"than Stage (R²={stage_r2:.4f})"
        )
        print(f"    WARNING: {results['warning']}")
    else:
        results["warning"] = None

    return results


# ===================================================================
# Analysis 3: Cramer's V Confounding Test
# ===================================================================

def _cramers_v(x: np.ndarray, y_vals: np.ndarray) -> Dict:
    """Compute Cramer's V between two categorical arrays."""
    ct = pd.crosstab(pd.Series(x, name="x"), pd.Series(y_vals, name="y"))
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"cramers_v": np.nan, "chi2": np.nan, "p_value": np.nan, "dof": 0}

    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.values.sum()
    k = min(ct.shape) - 1
    v = np.sqrt(chi2 / (n * k)) if k > 0 and n > 0 else 0.0

    return {"cramers_v": float(v), "chi2": float(chi2), "p_value": float(p), "dof": int(dof)}


def run_cramers_v(
    y: np.ndarray,
    batch_df: pd.DataFrame,
    tissue_name: str,
) -> Dict:
    """Cramer's V between Stage and each batch variable."""

    print("  Cramer's V confounding test...")
    results = {}

    for col in BATCH_COLUMNS:
        if col not in batch_df.columns:
            continue
        vals = batch_df[col].values
        valid = pd.notna(vals)
        if valid.sum() < 10:
            continue

        batch_vals = np.where(pd.isna(vals), "Unknown", vals.astype(str))
        res = _cramers_v(y.astype(str), batch_vals)
        results[col] = res

        strength = "STRONG" if res["cramers_v"] > 0.5 else "moderate" if res["cramers_v"] > 0.3 else "weak"
        print(f"    Stage x {col:16s}: V={res['cramers_v']:.3f} ({strength})  p={res['p_value']:.2e}")

    return results


# ===================================================================
# Analysis 4: Silhouette Analysis
# ===================================================================

def run_silhouette_analysis(
    X_scaled: pd.DataFrame,
    y: np.ndarray,
    batch_df: pd.DataFrame,
    tissue_name: str,
    output_dir: Path,
) -> Dict:
    """Compare silhouette scores: Stage vs BioProject clustering."""

    print("  Silhouette analysis...")
    X_pca, _ = _pca_reduce(X_scaled.values, n_components=N_PCA_COMPONENTS)

    results = {}

    # Stage silhouette
    y_arr = np.asarray(y)
    n_stage_classes = len(np.unique(y_arr))
    if n_stage_classes >= 2:
        sil_stage = silhouette_score(X_pca, y_arr, metric="euclidean")
        results["Stage"] = float(sil_stage)
        print(f"    Silhouette (Stage):      {sil_stage:.4f}")
    else:
        results["Stage"] = np.nan

    # Batch variable silhouettes
    for col in BATCH_COLUMNS:
        if col not in batch_df.columns:
            continue
        vals = batch_df[col].values
        labels = np.where(pd.isna(vals), "Unknown", vals.astype(str))
        n_unique = len(np.unique(labels))
        if n_unique < 2:
            continue

        sil = silhouette_score(X_pca, labels, metric="euclidean")
        results[col] = float(sil)
        print(f"    Silhouette ({col:16s}): {sil:.4f}")

    # Flag if batch silhouette > stage silhouette
    stage_sil = results.get("Stage", np.nan)
    batch_sils = {k: v for k, v in results.items() if k != "Stage" and not np.isnan(v)}
    if batch_sils:
        max_batch_key = max(batch_sils, key=batch_sils.get)
        max_batch_sil = batch_sils[max_batch_key]
        if max_batch_sil > stage_sil and not np.isnan(stage_sil):
            results["warning"] = (
                f"Samples cluster more by '{max_batch_key}' (sil={max_batch_sil:.4f}) "
                f"than by Stage (sil={stage_sil:.4f})"
            )
            print(f"    WARNING: {results['warning']}")
        else:
            results["warning"] = None
    else:
        results["warning"] = None

    # Bar chart
    plot_labels = [k for k in results if k != "warning" and not np.isnan(results[k])]
    plot_values = [results[k] for k in plot_labels]
    if plot_labels:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#2196F3" if k == "Stage" else "#FF9800" for k in plot_labels]
        ax.barh(plot_labels, plot_values, color=colors)
        ax.set_xlabel("Silhouette Score")
        ax.set_title(f"{tissue_name} — Silhouette: Stage vs Batch")
        ax.axvline(0, color="grey", linewidth=0.5)
        plt.tight_layout()
        fig_path = output_dir / f"{tissue_name}_silhouette_comparison.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        results["figure"] = str(fig_path)
        print(f"  Silhouette figure saved: {fig_path}")

    return results


# ===================================================================
# Tissue-level orchestrator
# ===================================================================

def run_tissue_batch_diagnostics(
    tissue_name: str,
    output_dir: Path,
    n_permutations: int = 999,
) -> Optional[Dict]:
    """Run all batch diagnostic analyses for a single tissue."""

    print(f"\n{'='*60}")
    print(f"  {tissue_name}")
    print(f"{'='*60}")

    data = _prepare_tissue_data(tissue_name)
    if data is None:
        print(f"  {tissue_name} not eligible for classification")
        return None

    X_scaled, y, sample_ids, gene_names, scheme_name, batch_df = data
    print(f"  Samples: {len(y)}, Genes: {X_scaled.shape[1]}, Scheme: {scheme_name}")

    # Report batch variable availability
    for col in BATCH_COLUMNS:
        if col in batch_df.columns:
            n_unique = batch_df[col].nunique(dropna=True)
            n_valid = batch_df[col].notna().sum()
            print(f"  {col}: {n_unique} unique values, {n_valid}/{len(y)} samples")

    result = {
        "tissue": tissue_name,
        "scheme": scheme_name,
        "n_samples": len(y),
        "n_genes": int(X_scaled.shape[1]),
    }

    # 1. PCA
    result["pca"] = run_pca_visualization(X_scaled, y, batch_df, tissue_name, output_dir)

    # 1b. UMAP
    umap_result = run_umap_visualization(X_scaled, y, batch_df, tissue_name, output_dir)
    result["umap"] = umap_result  # None if umap-learn not installed

    # 2. PERMANOVA
    result["permanova"] = run_permanova(X_scaled, y, batch_df, tissue_name, n_permutations)

    # 3. Cramer's V
    result["cramers_v"] = run_cramers_v(y, batch_df, tissue_name)

    # 4. Silhouette
    result["silhouette"] = run_silhouette_analysis(X_scaled, y, batch_df, tissue_name, output_dir)

    return result


# ===================================================================
# Cross-tissue runner
# ===================================================================

def run_batch_diagnostics(
    tissues: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    n_permutations: int = 999,
) -> Dict:
    """Run batch diagnostics across multiple tissues."""

    if tissues is None:
        tissues = DEFAULT_TISSUES
    if output_dir is None:
        output_dir = PROJECT_ROOT / "machine_learning/analysis/results/batch_diagnostics"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BATCH EFFECT DIAGNOSTIC ANALYSIS")
    print("=" * 80)
    print(f"Tissues:        {tissues}")
    print(f"Permutations:   {n_permutations}")
    print(f"Output:         {output_dir}")

    t_start = time.time()
    json_path = output_dir / "batch_diagnostics_results.json"

    def _json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    # Load existing results so re-runs merge rather than overwrite
    if json_path.exists():
        try:
            with open(json_path) as f:
                all_results = json.load(f)
        except (json.JSONDecodeError, IOError):
            all_results = {}
    else:
        all_results = {}

    for tissue in tissues:
        result = run_tissue_batch_diagnostics(
            tissue, output_dir,
            n_permutations=n_permutations,
        )
        if result:
            all_results[tissue] = result
            # Incremental save — crash-safe
            with open(json_path, "w") as f:
                json.dump(all_results, f, indent=2, default=_json_default)

    elapsed = time.time() - t_start

    # ---- Summary table ----
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    rows = []
    for tissue, r in all_results.items():
        row = {"Tissue": tissue, "Scheme": r["scheme"], "N": r["n_samples"]}

        # PERMANOVA R² for Stage
        perm = r.get("permanova", {})
        row["R²(Stage)"] = f"{perm.get('Stage', {}).get('R2', np.nan):.4f}"

        # Highest batch R²
        batch_r2s = {k: v["R2"] for k, v in perm.items() if k not in ("Stage", "warning") and isinstance(v, dict)}
        if batch_r2s:
            max_k = max(batch_r2s, key=batch_r2s.get)
            row["R²(max batch)"] = f"{batch_r2s[max_k]:.4f}"
            row["Max batch var"] = max_k
        else:
            row["R²(max batch)"] = "N/A"
            row["Max batch var"] = "N/A"

        # Cramer's V (max)
        cv = r.get("cramers_v", {})
        cv_vals = {k: v["cramers_v"] for k, v in cv.items() if isinstance(v, dict)}
        if cv_vals:
            max_cv_k = max(cv_vals, key=cv_vals.get)
            row["V(max)"] = f"{cv_vals[max_cv_k]:.3f}"
            row["V variable"] = max_cv_k
        else:
            row["V(max)"] = "N/A"
            row["V variable"] = "N/A"

        # Silhouette
        sil = r.get("silhouette", {})
        row["Sil(Stage)"] = f"{sil.get('Stage', np.nan):.4f}"

        rows.append(row)

    if rows:
        summary_df = pd.DataFrame(rows)
        print(summary_df.to_string(index=False))

        # Save summary CSV
        csv_path = output_dir / "batch_diagnostics_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"\nSummary CSV: {csv_path}")

    # Overall assessment
    print(f"\n--- Overall Assessment ---")
    for tissue, r in all_results.items():
        warnings_found = []
        perm = r.get("permanova", {})
        if perm.get("warning"):
            warnings_found.append(f"PERMANOVA: {perm['warning']}")
        sil = r.get("silhouette", {})
        if sil.get("warning"):
            warnings_found.append(f"Silhouette: {sil['warning']}")

        if warnings_found:
            print(f"  {tissue}: POTENTIAL BATCH EFFECTS")
            for w in warnings_found:
                print(f"    - {w}")
        else:
            print(f"  {tissue}: No major batch effect warnings")

    print(f"\nTotal time: {elapsed:.1f}s")

    # Save full results JSON (final write with all tissues)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    print(f"Results JSON: {json_path}")

    return all_results


# ===================================================================
# CLI entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch effect diagnostic analysis for developmental stage classification"
    )
    parser.add_argument(
        "--tissues", nargs="+", default=DEFAULT_TISSUES,
        help="Tissues to evaluate",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=999,
        help="Number of permutations for PERMANOVA (default: 999)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: analysis/results/batch_diagnostics/)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else None

    run_batch_diagnostics(
        tissues=args.tissues,
        output_dir=out,
        n_permutations=args.n_permutations,
    )


if __name__ == "__main__":
    main()
