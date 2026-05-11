#!/usr/bin/env python3
"""
Heterogeneity Sensitivity Analysis (Reviewer R2.1)
====================================================
Address BMC Genomics R2.1: quantify within-stage heterogeneity and test
whether the model's per-tissue performance varies by subgroup (Breed, Sex,
BioProject) or whether the marginal balanced accuracy is misleading.

Methodology choice (see R2.1 prompt option):
---------------------------------------------
The trained-model JSONs do not store per-sample CV predictions, only per-fold
aggregate metrics. We therefore re-run the SAME outer CV split that the
pipeline used (StratifiedKFold n_splits=5, shuffle=True, random_state=42 --
see `machine_learning/run_pipeline.py` lines 306-311) and re-fit each fold
with the SAME hyperparameters that the pipeline already tuned for that fold
(stored as `cross_validation.per_fold_details[i].best_params` in the JSONs).
This is mathematically equivalent to using the original CV predictions --
no inner-tuning loop is re-run, so the analysis takes <10 minutes total.

Outputs (under review/analyses/results/):
  * heterogeneity_stratification.csv   -- long-format sample counts per
                                          (tissue, stage, dimension, level)
  * heterogeneity_subgroup_performance.csv -- per-subgroup balanced accuracy
                                          per tissue, with n>=10 filter
  * heterogeneity_permutation_pvalues.csv -- permutation p-values testing
                                          whether observed BA disparities
                                          across breed/sex within a stage
                                          exceed chance (1000 shuffles)
  * heterogeneity_predictions.csv      -- per-sample CV predictions cached
                                          for figure generation
"""

from __future__ import annotations

import gzip
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.data_processing.data_loader import DataLoader
from machine_learning.data_processing.preprocessing import ExpressionPreprocessor
from machine_learning.data_processing.stage_selection import StageGranularitySelector
from machine_learning.model_training.hyperparameter_tuning import FIXED_PARAMS
from machine_learning.model_training.models import OrdinalLightGBM
from machine_learning.utils.helpers import to_samples_x_genes_df

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("R2.1")

TISSUES = ["Muscle", "Brain", "Liver", "Blood", "Lung"]
RESULTS_DIR = PROJECT_ROOT / "review" / "analyses" / "results"
MODEL_OUTPUTS_DIR = PROJECT_ROOT / "machine_learning" / "model_outputs"
DATA_DIR = PROJECT_ROOT / "data"
RANDOM_STATE = 42  # Same outer-CV seed as run_pipeline.py
MIN_GROUP_N = 10   # subgroup threshold for stratified BA
N_PERMUTATIONS = 1000


# ---------------------------------------------------------------------------
# Metadata audit (sample counts)
# ---------------------------------------------------------------------------

def normalize_sex(series: pd.Series) -> pd.Series:
    """Collapse case-variants (e.g. 'male' vs 'Male')."""
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "male": "Male",
        "female": "Female",
        "pooled": "Pooled",
        "neuter": "Neuter",
        "hermaphrodite": "Hermaphrodite",
        "unknown": "Unknown",
        "nan": "Unknown",
        "": "Unknown",
    }
    return s.map(lambda v: mapping.get(v, "Unknown"))


def normalize_breed(series: pd.Series) -> pd.Series:
    """Collapse missing/unknown to 'Unknown'; keep cross labels as-is."""
    s = series.astype(str).str.strip()
    s = s.replace({"nan": "Unknown", "": "Unknown"})
    return s


def normalize_stage(series: pd.Series) -> pd.Series:
    """Stage column already in proper format; just convert to string."""
    return series.astype(str)


def stratification_audit(tissue_metadata: pd.DataFrame, tissue: str) -> pd.DataFrame:
    """For each (Stage, dimension) compute sample counts and write long-form rows."""
    rows = []
    md = tissue_metadata.copy()
    md["Stage"] = normalize_stage(md["Stage"])
    md["Sex"] = normalize_sex(md["Sex"])
    md["Breed"] = normalize_breed(md["Breed"])
    md["BioProject"] = md["BioProject"].astype(str).fillna("Unknown")

    for dim in ["Breed", "Sex", "BioProject"]:
        counts = md.groupby(["Stage", dim], observed=True).size().reset_index(name="n_samples")
        for _, r in counts.iterrows():
            rows.append({
                "tissue": tissue,
                "stage": r["Stage"],
                "dimension": dim,
                "level": r[dim],
                "n_samples": int(r["n_samples"]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CV prediction reproduction
# ---------------------------------------------------------------------------

def load_tissue_data(tissue: str) -> Optional[Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, str, dict, pd.DataFrame]]:
    """Load expression + labels for a tissue exactly as run_pipeline.py does.

    Returns (X_genes_x_samples, y_int, sample_ids, gene_names, scheme_name,
             scheme_dict, sample_metadata).
    """
    data_loader = DataLoader(
        data_dir=DATA_DIR / "pigGTEx",
        metadata_path=DATA_DIR / "PigGTEx_v0.MetaTable.csv",
    )
    data_loader.load_metadata()
    expr_data, metadata = data_loader.load_expression(tissue)

    X = expr_data
    y = metadata["Stage"].values
    sample_ids = metadata.index.values

    selector = StageGranularitySelector()
    stage_counts = pd.Series(y).value_counts()
    scheme_name, scheme = selector.select_scheme(stage_counts, tissue)
    if scheme is None:
        return None

    # Remap stages and filter -1 (samples whose stage is not in the scheme labels)
    if "mapping" in scheme:
        labels = scheme.get("labels", [])
        stage_mapping = scheme["mapping"]
        y_mapped = [stage_mapping.get(s, s) for s in y]
        label_to_int = {label: i for i, label in enumerate(labels)}
        y_int = np.array([label_to_int.get(label, -1) for label in y_mapped])
        valid_mask = y_int != -1
        X = X.loc[:, valid_mask]
        y_int = y_int[valid_mask]
        sample_ids = sample_ids[valid_mask]
        metadata = metadata.loc[sample_ids].copy()
    else:
        y_int = y

    gene_names = X.index.values
    # Re-attach the merged-stage label for reporting (e.g. Post-pubertal/Adult)
    metadata = metadata.copy()
    metadata["StageLabel"] = [scheme["labels"][i] for i in y_int]
    return X, y_int, sample_ids, gene_names, scheme_name, scheme, metadata


def reproduce_cv_predictions(
    tissue: str,
    fold_best_params: List[dict],
    X: pd.DataFrame,
    y: np.ndarray,
    sample_ids: np.ndarray,
    gene_names: np.ndarray,
    seed: int = RANDOM_STATE,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Replay run_pipeline.py's outer CV using stored per-fold hyperparameters.

    Returns a DataFrame indexed by sample_id with columns
    {fold, y_true, y_pred} for every sample.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    X_T = X.T  # samples x genes -- pipeline does the same

    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_T, y)):
        X_train_raw = X.iloc[:, train_idx]
        X_test_raw = X.iloc[:, test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        train_ids = sample_ids[train_idx]
        test_ids = sample_ids[test_idx]

        # Preprocess (fit on train only -- no data leakage)
        pp = ExpressionPreprocessor()
        X_train_proc = pp.fit_transform(X_train_raw)
        X_test_proc = pp.transform(X_test_raw)

        X_train_df = to_samples_x_genes_df(X_train_proc, train_ids, pp, gene_names)
        X_test_df = to_samples_x_genes_df(X_test_proc, test_ids, pp, gene_names)

        # Re-fit with the stored per-fold best_params (no inner tuning)
        best_params = fold_best_params[fold_i].copy()
        # Some params (n_estimators) saved as float in JSON; cast to int where appropriate
        for k in ("num_leaves", "max_depth", "min_data_in_leaf", "n_estimators"):
            if k in best_params:
                best_params[k] = int(best_params[k])

        model = OrdinalLightGBM(**best_params, **FIXED_PARAMS, seed=seed + fold_i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_df, y_train)
        y_pred = model.predict(X_test_df)

        for sid, yt, yp in zip(test_ids, y_test, y_pred):
            rows.append({"sample_id": sid, "fold": fold_i + 1, "y_true": int(yt), "y_pred": int(yp)})

        ba = balanced_accuracy_score(y_test, y_pred)
        logger.info(f"{tissue} fold {fold_i+1}: n_test={len(y_test)}, BA={ba:.3f}")

    return pd.DataFrame(rows).set_index("sample_id")


# ---------------------------------------------------------------------------
# Subgroup performance
# ---------------------------------------------------------------------------

def compute_subgroup_ba(preds: pd.DataFrame, metadata: pd.DataFrame, tissue: str,
                        dimensions: List[str] = ("Breed", "Sex", "BioProject")) -> pd.DataFrame:
    """Compute balanced accuracy stratified by each dimension level (n>=MIN_GROUP_N)."""
    rows = []
    # Marginal BA on all samples
    marginal_ba = balanced_accuracy_score(preds["y_true"], preds["y_pred"])
    rows.append({
        "tissue": tissue,
        "dimension": "Overall",
        "level": "All",
        "n_samples": int(len(preds)),
        "n_stages_represented": int(preds["y_true"].nunique()),
        "balanced_accuracy": float(marginal_ba),
    })

    # Subgroup BAs
    md = metadata.loc[preds.index].copy()
    md["Sex"] = normalize_sex(md["Sex"])
    md["Breed"] = normalize_breed(md["Breed"])
    md["BioProject"] = md["BioProject"].astype(str).fillna("Unknown")

    for dim in dimensions:
        for level, sub in md.groupby(dim, observed=True):
            if len(sub) < MIN_GROUP_N:
                continue
            sub_preds = preds.loc[sub.index]
            # Need at least 2 distinct true classes for balanced_accuracy to be meaningful
            n_classes_present = sub_preds["y_true"].nunique()
            if n_classes_present < 2:
                continue
            ba = balanced_accuracy_score(sub_preds["y_true"], sub_preds["y_pred"])
            rows.append({
                "tissue": tissue,
                "dimension": dim,
                "level": str(level),
                "n_samples": int(len(sub_preds)),
                "n_stages_represented": int(n_classes_present),
                "balanced_accuracy": float(ba),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test_within_stage(
    preds: pd.DataFrame,
    metadata: pd.DataFrame,
    tissue: str,
    dimension: str,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = RANDOM_STATE,
) -> Optional[dict]:
    """Test whether the spread of subgroup BAs along `dimension` exceeds chance.

    Null: subgroup labels are exchangeable WITHIN each stage. We shuffle the
    chosen label column within stage strata, recompute per-level BAs (n>=MIN_GROUP_N),
    and compare the observed std of subgroup BAs to the permutation distribution.
    A small p-value means subgroup performance differs more than chance would
    predict given the stage composition of each subgroup.

    Returns dict with keys: observed_std, perm_mean, perm_std, pvalue,
    n_levels_used, n_permutations.
    """
    md = metadata.loc[preds.index].copy()
    if dimension == "Sex":
        md[dimension] = normalize_sex(md[dimension])
    elif dimension == "Breed":
        md[dimension] = normalize_breed(md[dimension])
    else:
        md[dimension] = md[dimension].astype(str).fillna("Unknown")

    # Identify analysable subgroups (>=MIN_GROUP_N AND >=2 stages represented).
    level_sizes = md[dimension].value_counts()
    usable_levels = []
    for level, n in level_sizes.items():
        if n < MIN_GROUP_N:
            continue
        sub = md[md[dimension] == level]
        sub_pred = preds.loc[sub.index]
        if sub_pred["y_true"].nunique() < 2:
            continue
        usable_levels.append(level)

    if len(usable_levels) < 2:
        logger.info(f"{tissue}/{dimension}: <2 usable levels, skipping permutation")
        return None

    md_use = md[md[dimension].isin(usable_levels)].copy()
    preds_use = preds.loc[md_use.index].copy()

    def compute_spread(label_array: np.ndarray) -> float:
        bas = []
        for lvl in usable_levels:
            mask = (label_array == lvl)
            if mask.sum() < MIN_GROUP_N:
                continue
            sub_p = preds_use.iloc[mask.nonzero()[0]]
            if sub_p["y_true"].nunique() < 2:
                continue
            bas.append(balanced_accuracy_score(sub_p["y_true"], sub_p["y_pred"]))
        return float(np.std(bas)) if len(bas) >= 2 else np.nan

    obs_labels = md_use[dimension].values
    observed_std = compute_spread(obs_labels)
    if np.isnan(observed_std):
        return None

    rng = np.random.default_rng(seed)
    stage_array = md_use["Stage"].astype(str).values
    # Pre-compute per-stage index arrays once
    stage_to_idx = {s: np.where(stage_array == s)[0] for s in np.unique(stage_array)}
    perm_stds = []
    for _ in range(n_permutations):
        permuted = obs_labels.copy()
        # Shuffle within each stage independently. Fancy indexing returns a
        # copy, so we must shuffle a sliced array and write it back.
        for idx in stage_to_idx.values():
            sub = permuted[idx]
            rng.shuffle(sub)
            permuted[idx] = sub
        s = compute_spread(permuted)
        if not np.isnan(s):
            perm_stds.append(s)

    perm_stds = np.asarray(perm_stds)
    if len(perm_stds) == 0:
        return None
    # One-sided p: probability that random label shuffles produce >= observed spread
    pvalue = float((perm_stds >= observed_std).sum() + 1) / (len(perm_stds) + 1)

    return {
        "tissue": tissue,
        "dimension": dimension,
        "n_levels_used": len(usable_levels),
        "usable_levels": ";".join(map(str, usable_levels)),
        "observed_std_ba": float(observed_std),
        "perm_null_mean_std": float(perm_stds.mean()),
        "perm_null_q95_std": float(np.quantile(perm_stds, 0.95)),
        "pvalue": pvalue,
        "n_permutations": int(len(perm_stds)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Stratification audit + 2. CV reproduction + 3. Subgroup BA + 4. Permutation ----------
    audit_rows: List[pd.DataFrame] = []
    perf_rows: List[pd.DataFrame] = []
    perm_rows: List[dict] = []
    pred_frames: List[pd.DataFrame] = []

    for tissue in TISSUES:
        logger.info(f"=== {tissue} ===")
        tissue_pack = load_tissue_data(tissue)
        if tissue_pack is None:
            logger.warning(f"{tissue}: stage scheme returned None, skipping")
            continue
        X, y_int, sample_ids, gene_names, scheme_name, scheme, metadata = tissue_pack
        logger.info(f"{tissue}: X={X.shape}, scheme={scheme_name}, n={len(y_int)}")

        # 1) Stratification audit
        audit = stratification_audit(metadata, tissue)
        audit_rows.append(audit)

        # 2) Load fold hyperparameters
        with open(MODEL_OUTPUTS_DIR / f"{tissue}_results.json") as f:
            model_out = json.load(f)
        per_fold = model_out["cross_validation"]["per_fold_details"]
        fold_params = [pf["best_params"] for pf in per_fold]
        fold_seed = model_out["cross_validation"].get("seed", RANDOM_STATE)
        n_folds = model_out["cross_validation"].get("n_folds", 5)
        logger.info(f"{tissue}: replaying {n_folds}-fold CV (seed={fold_seed})")

        # 3) Reproduce CV predictions
        preds = reproduce_cv_predictions(
            tissue=tissue,
            fold_best_params=fold_params,
            X=X, y=y_int, sample_ids=sample_ids, gene_names=gene_names,
            seed=fold_seed, n_folds=n_folds,
        )
        # Sanity: compare reproduced fold BAs to stored ones
        for fold_i in range(n_folds):
            sub = preds[preds["fold"] == fold_i + 1]
            ba_repro = balanced_accuracy_score(sub["y_true"], sub["y_pred"])
            ba_stored = per_fold[fold_i]["test_balanced_accuracy"]
            logger.info(f"{tissue} fold {fold_i+1}: BA reproduced={ba_repro:.4f}, stored={ba_stored:.4f}")

        # Add stage label and metadata for the figure stage
        preds["stage_int"] = preds["y_true"]
        preds["stage_label"] = [scheme["labels"][i] for i in preds["y_true"]]
        preds["pred_label"] = [scheme["labels"][i] for i in preds["y_pred"]]
        preds["tissue"] = tissue
        pred_frames.append(preds.reset_index())

        # 4) Subgroup performance
        perf = compute_subgroup_ba(preds, metadata, tissue)
        perf_rows.append(perf)

        # 5) Permutation tests (Breed, Sex, BioProject for Muscle)
        for dim in ["Breed", "Sex"]:
            res = permutation_test_within_stage(preds, metadata, tissue, dim,
                                                 n_permutations=N_PERMUTATIONS, seed=RANDOM_STATE)
            if res is not None:
                perm_rows.append(res)
        if tissue == "Muscle":
            res_bp = permutation_test_within_stage(preds, metadata, tissue, "BioProject",
                                                    n_permutations=N_PERMUTATIONS, seed=RANDOM_STATE)
            if res_bp is not None:
                perm_rows.append(res_bp)

    # ---------- Save outputs ----------
    audit_df = pd.concat(audit_rows, ignore_index=True)
    audit_df.to_csv(RESULTS_DIR / "heterogeneity_stratification.csv", index=False)
    logger.info(f"Wrote {RESULTS_DIR / 'heterogeneity_stratification.csv'} (n={len(audit_df)})")

    perf_df = pd.concat(perf_rows, ignore_index=True)
    perf_df.to_csv(RESULTS_DIR / "heterogeneity_subgroup_performance.csv", index=False)
    logger.info(f"Wrote {RESULTS_DIR / 'heterogeneity_subgroup_performance.csv'} (n={len(perf_df)})")

    perm_df = pd.DataFrame(perm_rows)
    perm_df.to_csv(RESULTS_DIR / "heterogeneity_permutation_pvalues.csv", index=False)
    logger.info(f"Wrote {RESULTS_DIR / 'heterogeneity_permutation_pvalues.csv'} (n={len(perm_df)})")

    preds_all = pd.concat(pred_frames, ignore_index=True)
    preds_all.to_csv(RESULTS_DIR / "heterogeneity_predictions.csv", index=False)
    logger.info(f"Wrote {RESULTS_DIR / 'heterogeneity_predictions.csv'} (n={len(preds_all)})")

    # Quick summary
    logger.info("---- summary ----")
    logger.info("Overall BAs (reproduced):")
    for tissue in TISSUES:
        row = perf_df[(perf_df["tissue"] == tissue) & (perf_df["dimension"] == "Overall")]
        if not row.empty:
            logger.info(f"  {tissue}: {row.iloc[0]['balanced_accuracy']:.3f}")
    logger.info("Subgroup BA spread per tissue/dimension:")
    for tissue in TISSUES:
        for dim in ["Breed", "Sex", "BioProject"]:
            sub = perf_df[(perf_df["tissue"] == tissue) & (perf_df["dimension"] == dim)]
            if not sub.empty:
                logger.info(f"  {tissue}/{dim}: n_levels={len(sub)}, BA range [{sub['balanced_accuracy'].min():.3f}, {sub['balanced_accuracy'].max():.3f}], std={sub['balanced_accuracy'].std():.3f}")


if __name__ == "__main__":
    main()
