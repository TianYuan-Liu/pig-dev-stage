#!/usr/bin/env python3
"""
Joint cross-species ML pipeline for tissue-aware developmental feature selection
=================================================================================

Trains an ordinal LightGBM model on pig and a binary LightGBM model on human
over the SAME 7,485-ortholog feature space. A gene becomes a "joint conserved
feature" only if it appears in the top-K importance ranking of both species'
models across most random seeds. This procedure replaces post-hoc FDR + |log2FC|
filtering as the cross-species feature-selection mechanism for the manuscript.

Outputs (all in review/analyses/cross_species_joint_ml/):
  features_pig_{tissue}.csv      per-seed top-100 lists + stability summary
  features_human_{tissue}.csv    per-seed top-100 lists + stability summary
  joint_features_{tissue}.csv    pig stable @100 INTERSECT human stable @100
  results_{tissue}.json          stability counts, concordance, annotation summary
  fig_top_features_{tissue}.pdf  pig top-30 | human top-30 | joint
  fig_concordance_panel.pdf      forest plot of joint-ML r per tissue
  fig_method_comparison.pdf      gene-set size baseline FDR vs joint ML
  log.tsv                        autoresearch attempt log

Design notes:
- Pig model uses OrdinalLightGBM (5-stage classifier) with 5-fold StratifiedKFold
  per seed; importance averaged across folds, then top-100 per seed.
- Human model uses LightGBM binary (Infant vs Adult) with leave-pair-out CV
  for tissues with n<=20 samples per species; importance averaged across LPO
  folds, then top-100 per seed.
- Joint stability set: gene appears in top-100 in >= 8 of 10 seeds in BOTH species.
- Lung and Blood: no Cardoso-Moreira human matrix exists for either tissue.
  Pipeline produces pig-only features and records joint_set=NA for these.

Author: AUTORESEARCH joint cross-species ML / R1.2 follow-up
"""

from __future__ import annotations

import gzip
import json
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# Make repository's machine_learning package importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from machine_learning.model_training.models import OrdinalLightGBM  # noqa: E402

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = PROJECT_ROOT / "data"
HUMAN_RPKM = DATA_DIR / "cardoso_moreira_2019" / "Human.RPKM.tsv"
HUMAN_MUSCLE_CSV = DATA_DIR / "human_muscle" / "GSE257558_read_counts_healty.csv"
PIG_META_CSV = DATA_DIR / "PigGTEx_v0.MetaTable.csv"
PIG_TPM_DIR = DATA_DIR / "pigGTEx"

ORTHO_CSV = (PROJECT_ROOT / "review" / "analyses" / "results"
             / "pig_human_one_to_one_orthologs.csv")
HUMAN_SYM_CACHE = DATA_DIR / "cardoso_moreira_2019" / "_human_id_to_symbol.json"
BASELINE_GENES_CSV = (PROJECT_ROOT / "review" / "analyses" / "results"
                      / "cross_species_per_tissue_genes.csv")

OUT_DIR = PROJECT_ROOT / "review" / "analyses" / "cross_species_joint_ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
ANNOT_CACHE = CACHE_DIR / "mygene_annotations.json"
LOG_FILE = OUT_DIR / "log.tsv"

# ============================================================================
# CONFIGURATION
# ============================================================================
TISSUES = ["Muscle", "Brain", "Liver", "Lung", "Blood"]

# Human tissue source per pig tissue
# - "cardoso_<Tissue>"   -> Cardoso-Moreira RPKM matrix, that tissue column
# - "gse257558"          -> GSE257558 muscle counts
# - None                 -> no human data available
HUMAN_SOURCE: Dict[str, Optional[str]] = {
    "Muscle": "gse257558",
    "Brain": "cardoso_Brain",
    "Liver": "cardoso_Liver",
    "Lung": None,
    "Blood": None,
}

# Cardoso-Moreira postnatal stage bins
HUMAN_YOUNG = {"newborn", "infant", "toddler"}
HUMAN_OLD = {"youngAdult", "youngMidAge", "olderMidAge", "senior", "Senior"}

# Pig ordinal stages (5)
PIG_STAGES = ["Infant", "Early childhood", "Pre-pubertal",
              "Post-pubertal", "Adult"]
PIG_YOUNG = {"Infant", "Early childhood"}
PIG_OLD = {"Post-pubertal", "Adult"}

# ML config
SEEDS = list(range(42, 52))                # 10 seeds
N_FOLDS_PIG = 5                            # stratified K-fold
TOP_K = 100
SENSITIVITY_K = [50, 100, 200]
STABILITY_THRESHOLD = 8                    # >= 8 of 10 seeds

# Pig OrdinalLightGBM hyperparameters
# Mild regularisation but enough capacity to fit ~200-1000 samples
PIG_PARAMS: Dict[str, object] = dict(
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.05,
    n_estimators=200,
    min_data_in_leaf=10,
    feature_fraction=0.7,
    bagging_fraction=0.7,
    lambda_l1=0.1,
    lambda_l2=1.0,
    monotone_constraints=True,
)

# Human binary LightGBM hyperparameters
# Small-n regularisation per user spec
HUMAN_PARAMS_DEFAULT: Dict[str, object] = dict(
    objective="binary",
    metric="binary_logloss",
    num_leaves=8,
    max_depth=-1,
    learning_rate=0.05,
    n_estimators=200,
    min_data_in_leaf=3,
    feature_fraction=0.7,
    bagging_fraction=0.7,
    bagging_freq=5,
    lambda_l1=0.1,
    lambda_l2=1.0,
    verbosity=-1,
)
HUMAN_EARLY_STOPPING = 20

# Hyperparameter refinement sweep (used if joint set is empty on first pass)
HUMAN_PARAM_SWEEP = [
    dict(num_leaves=8,  min_data_in_leaf=3),   # default
    dict(num_leaves=6,  min_data_in_leaf=2),
    dict(num_leaves=12, min_data_in_leaf=5),
]

# Concordance
N_BOOTSTRAP = 1000
PSEUDO = 0.01

# Biological annotation categories
BIO_CATEGORIES: Dict[str, List[str]] = {
    "muscle_contraction": [
        "muscle contraction", "myofibril", "sarcomere", "actin", "myosin",
        "troponin", "contractile fiber", "skeletal muscle",
    ],
    "neuronal_synaptic": [
        "synap", "neuron", "axon", "dendrit", "myelin", "neurotransmit",
        "glutamatergic", "GABA", "nervous system",
    ],
    "liver_metabolic": [
        "lipid metabolism", "bile", "lipoprotein", "cholesterol",
        "cytochrome P450", "drug metabolism", "fatty acid",
        "gluconeogenesis", "xenobiotic",
    ],
    "immune_response": [
        "immune", "lymphocyte", "interleukin", "interferon",
        "complement", "chemokine", "antigen", "B cell", "T cell",
    ],
    "cell_cycle_proliferation": [
        "cell cycle", "mitosis", "cyclin", "kinetochore", "DNA replication",
        "proliferation", "G1/S transition",
    ],
    "extracellular_matrix": [
        "extracellular matrix", "collagen", "laminin", "elastin",
        "basement membrane", "fibrillar",
    ],
    "general_metabolism": [
        "metabol", "biosynth", "catabol", "glycolysis",
        "oxidative phosphorylation", "TCA cycle",
    ],
    "transcription_regulation": [
        "transcription factor", "DNA binding", "homeobox",
        "zinc finger", "chromatin",
    ],
}

# ============================================================================
# LOGGING
# ============================================================================
def init_log() -> None:
    if not LOG_FILE.exists():
        LOG_FILE.write_text("attempt_id\ttissue\tstep\tmetric_name\t"
                            "metric_value\tnote\n")

def log_row(tissue: str, step: str, metric_name: str,
            metric_value: object, note: str = "") -> None:
    aid = f"{int(time.time())}_{tissue}_{step}"
    line = f"{aid}\t{tissue}\t{step}\t{metric_name}\t{metric_value}\t{note}\n"
    with LOG_FILE.open("a") as fh:
        fh.write(line)


# ============================================================================
# ORTHOLOG LOADER (reused from cross_species_all_tissues.py output)
# ============================================================================
def load_orthologs() -> pd.DataFrame:
    df = pd.read_csv(ORTHO_CSV)
    df = df[df["is_one_to_one"]].copy() if "is_one_to_one" in df.columns else df
    df["symbol"] = df["symbol"].str.upper()
    return df[["symbol", "pig_gene_id", "human_gene_id"]].drop_duplicates()


# ============================================================================
# AGE PARSER (pig)
# ============================================================================
_AGE_UNITS = {"day": 1, "days": 1, "week": 7, "weeks": 7,
              "month": 30, "months": 30, "year": 365, "years": 365}

def parse_age_days(age_str) -> float:
    if pd.isna(age_str):
        return np.nan
    s = str(age_str).strip().lower()
    if s == "unknown" or s == "":
        return np.nan
    m = re.match(r"(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years)", s)
    if not m:
        return np.nan
    return float(m.group(1)) * _AGE_UNITS[m.group(2)]


def days_to_stage(d: float) -> Optional[str]:
    if pd.isna(d):
        return None
    if d <= 20:
        return "Infant"
    if d <= 59:
        return "Early childhood"
    if d <= 149:
        return "Pre-pubertal"
    if d <= 365:
        return "Post-pubertal"
    return "Adult"


# ============================================================================
# PIG DATA LOADER
# ============================================================================
def load_pig(tissue: str, ortho: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return log2(TPM+1) matrix [samples x ortholog_genes] indexed by pig_gene_id,
    and a Series of stage labels (5-level ordinal).
    Filters out stages with <5 samples to avoid degenerate stratified splits."""
    cache = CACHE_DIR / f"pig_{tissue}.parquet"
    cache_y = CACHE_DIR / f"pig_{tissue}_y.csv"
    cache_raw = CACHE_DIR / f"pig_{tissue}_rawTPM.parquet"
    if cache.exists() and cache_y.exists() and cache_raw.exists():
        X = pd.read_parquet(cache)
        y = pd.read_csv(cache_y, index_col=0).iloc[:, 0]
        raw = pd.read_parquet(cache_raw)
        return X, y, raw

    meta = pd.read_csv(PIG_META_CSV)
    meta = meta.rename(columns={
        "BioSample": "Sample_ID",
        "Tissue class": "Tissue",
        "Main categories": "Tissue_Main",
        "Sub categories": "Sub_categories",
    }).set_index("Sample_ID")
    meta["Age_Days"] = meta["Age"].apply(parse_age_days)
    meta["Stage"] = meta["Age_Days"].apply(days_to_stage)
    sub_meta = meta[
        (meta["Tissue"] == tissue) &
        (meta["Sub_categories"] == tissue) &
        meta["Stage"].notna()
    ].copy()

    tpm_path = PIG_TPM_DIR / f"{tissue}.expr_tpm.txt.gz"
    with gzip.open(tpm_path, "rt") as fh:
        expr = pd.read_csv(fh, sep="\t", index_col=0)

    # restrict to ortholog genes
    keep = [g for g in ortho["pig_gene_id"].values if g in expr.index]
    expr = expr.loc[keep]
    expr.index.name = "pig_gene_id"

    # restrict to samples with valid stage and present in matrix
    samples = [s for s in sub_meta.index if s in expr.columns]
    expr = expr[samples]
    y = sub_meta.loc[samples, "Stage"]

    # Drop stages with <5 samples - 5-fold stratified CV needs >=5 per class,
    # and OrdinalLightGBM's internal 80/20 split needs >=2 of each binary
    # class per fold. Keeping these tiny strata would make stratified
    # cross-validation degenerate.
    raw_counts = y.value_counts()
    viable_stages = raw_counts[raw_counts >= 5].index.tolist()
    if len(viable_stages) < 2:
        raise RuntimeError(
            f"{tissue}: <2 viable stages with >=5 samples (counts={raw_counts.to_dict()})")
    keep_mask = y.isin(viable_stages)
    dropped = raw_counts[~raw_counts.index.isin(viable_stages)].to_dict()
    if dropped:
        print(f"  load_pig({tissue}): dropping tiny stages {dropped}", flush=True)
    expr = expr.loc[:, keep_mask.values]
    y = y[keep_mask]

    # log2(TPM+1) matrix in [samples x genes]
    X = np.log2(expr.T.astype(float) + 1.0)
    X.index.name = "Sample_ID"
    X.columns.name = "pig_gene_id"

    raw = expr.T.astype(float)  # samples x genes (raw TPM)

    X.to_parquet(cache)
    raw.to_parquet(cache_raw)
    y.to_frame("Stage").to_csv(cache_y)
    return X, y, raw


# ============================================================================
# HUMAN DATA LOADERS
# ============================================================================
def load_human_cardoso(tissue: str, ortho: pd.DataFrame
                       ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """log2(RPKM+1) matrix + binary y (Infant vs Adult) + raw RPKM df."""
    cache = CACHE_DIR / f"human_cardoso_{tissue}.parquet"
    cache_y = CACHE_DIR / f"human_cardoso_{tissue}_y.csv"
    cache_raw = CACHE_DIR / f"human_cardoso_{tissue}_raw.parquet"
    if cache.exists() and cache_y.exists() and cache_raw.exists():
        X = pd.read_parquet(cache)
        y = pd.read_csv(cache_y, index_col=0).iloc[:, 0]
        raw = pd.read_parquet(cache_raw)
        return X, y, raw

    df = pd.read_csv(HUMAN_RPKM, sep="\t", index_col=0)
    cols = [c for c in df.columns if c.startswith(f"{tissue}.")]
    sub = df[cols].copy()
    stage_map = {c: c.split(".")[1] for c in cols}
    young_cols = [c for c, s in stage_map.items() if s in HUMAN_YOUNG]
    old_cols = [c for c, s in stage_map.items() if s in HUMAN_OLD]
    sel = young_cols + old_cols
    sub = sub[sel]

    # restrict to ortholog genes (human side)
    keep = [g for g in ortho["human_gene_id"].values if g in sub.index]
    sub = sub.loc[keep]
    sub.index.name = "human_gene_id"

    X = np.log2(sub.T.astype(float) + 1.0)  # samples x genes
    X.index.name = "Sample_ID"
    X.columns.name = "human_gene_id"
    y = pd.Series([0] * len(young_cols) + [1] * len(old_cols),
                  index=young_cols + old_cols, name="binary",
                  dtype=int)

    raw = sub.T.astype(float)

    X.to_parquet(cache)
    raw.to_parquet(cache_raw)
    y.to_frame("y").to_csv(cache_y)
    return X, y, raw


def load_human_gse257558(ortho: pd.DataFrame
                         ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Muscle counts indexed by gene SYMBOL; map to ortholog human_gene_id
    via the ortholog symbol column."""
    cache = CACHE_DIR / "human_muscle.parquet"
    cache_y = CACHE_DIR / "human_muscle_y.csv"
    cache_raw = CACHE_DIR / "human_muscle_raw.parquet"
    if cache.exists() and cache_y.exists() and cache_raw.exists():
        X = pd.read_parquet(cache)
        y = pd.read_csv(cache_y, index_col=0).iloc[:, 0]
        raw = pd.read_parquet(cache_raw)
        return X, y, raw

    df = pd.read_csv(HUMAN_MUSCLE_CSV, index_col=0)  # symbol-indexed counts
    df.index = df.index.str.upper()

    # Symbol -> human_gene_id via ortholog table
    sym_to_h = dict(zip(ortho["symbol"], ortho["human_gene_id"]))
    common_syms = [s for s in df.index if s in sym_to_h]
    df = df.loc[common_syms].copy()
    df.index = [sym_to_h[s] for s in df.index]
    df.index.name = "human_gene_id"

    # normalise to CPM then log2(CPM+1)
    libsize = df.sum(axis=0)
    cpm = df.divide(libsize, axis=1) * 1e6
    X = np.log2(cpm.T.astype(float) + 1.0)
    X.index.name = "Sample_ID"
    X.columns.name = "human_gene_id"

    y = pd.Series(
        [0 if c.startswith("I") else 1 for c in df.columns],
        index=df.columns, name="binary", dtype=int,
    )

    raw = cpm.T.astype(float)  # CPM-equivalent raw used for log2FC

    X.to_parquet(cache)
    raw.to_parquet(cache_raw)
    y.to_frame("y").to_csv(cache_y)
    return X, y, raw


def load_human(tissue: str, ortho: pd.DataFrame
               ) -> Optional[Tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    src = HUMAN_SOURCE[tissue]
    if src is None:
        return None
    if src == "gse257558":
        return load_human_gse257558(ortho)
    if src.startswith("cardoso_"):
        return load_human_cardoso(src.split("_", 1)[1], ortho)
    raise ValueError(f"Unknown human source: {src}")


# ============================================================================
# PIG ORDINAL TRAINER
# ============================================================================
def train_pig_seed(X: pd.DataFrame, y: pd.Series,
                   seed: int) -> Tuple[pd.Series, dict]:
    """5-fold CV with one seed -> mean-gain importance per gene + stats."""
    stages_present = [s for s in PIG_STAGES if s in y.unique()]
    # OrdinalLightGBM needs at least 2 classes (it builds class-1 binary models).
    # Pig has 5 stages everywhere; just guard for safety.
    if len(stages_present) < 2:
        raise RuntimeError("Need at least 2 stages")

    # Strat K-fold; if any class has fewer than n_splits, reduce
    counts = y.value_counts()
    n_splits = min(N_FOLDS_PIG, int(counts.min()))
    n_splits = max(2, n_splits)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_imps = []
    fold_perf = []
    for fi, (tr, te) in enumerate(skf.split(X.values, y.values)):
        model = OrdinalLightGBM(seed=seed + fi, **PIG_PARAMS)
        model.fit(X.iloc[tr], y.iloc[tr].values)
        imp = model.get_feature_importance("gain")
        fold_imps.append(imp)
        # eval on test fold
        y_pred = model.predict(X.iloc[te])
        # mean absolute ordinal error
        stage_idx = {s: i for i, s in enumerate(PIG_STAGES)}
        y_true_idx = np.array([stage_idx.get(s, -1) for s in y.iloc[te].values])
        y_pred_idx = np.array([stage_idx.get(s, -1) for s in y_pred])
        ok = (y_true_idx >= 0) & (y_pred_idx >= 0)
        mae = float(np.mean(np.abs(y_true_idx[ok] - y_pred_idx[ok]))) if ok.any() else None
        fold_perf.append(mae)

    # Align: features may shift index name across folds
    df = pd.concat(fold_imps, axis=1)
    df = df.fillna(0.0)
    seed_imp = df.mean(axis=1).sort_values(ascending=False)
    seed_imp.name = f"seed_{seed}_gain"
    stats_d = {
        "seed": seed,
        "n_folds": int(n_splits),
        "mean_MAE": float(np.mean([m for m in fold_perf if m is not None]))
                    if any(m is not None for m in fold_perf) else None,
    }
    return seed_imp, stats_d


# ============================================================================
# HUMAN BINARY TRAINER (leave-pair-out)
# ============================================================================
def _leave_pair_out_indices(y: np.ndarray, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate (train_idx, test_idx) pairs leaving 1 infant + 1 adult out.
    Pairs are deterministic but shuffled per seed."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng.shuffle(pos); rng.shuffle(neg)
    pairs = []
    for p in pos:
        for n in neg:
            te = np.array([p, n])
            tr = np.array([i for i in range(len(y)) if i not in (p, n)])
            pairs.append((tr, te))
    return pairs


def train_human_seed(X: pd.DataFrame, y: pd.Series, seed: int,
                     params_override: Optional[dict] = None
                     ) -> Tuple[pd.Series, dict]:
    """Leave-pair-out CV per seed. Returns mean gain importance + stats."""
    y_arr = y.values.astype(int)
    pairs = _leave_pair_out_indices(y_arr, seed)
    base = {**HUMAN_PARAMS_DEFAULT, "random_state": seed}
    if params_override:
        base.update(params_override)
    base["objective"] = "binary"
    base["metric"] = "binary_logloss"
    base["verbosity"] = -1

    # scale_pos_weight (per the OrdinalLightGBM convention)
    n_pos = int((y_arr == 1).sum()); n_neg = int((y_arr == 0).sum())
    base["scale_pos_weight"] = n_neg / n_pos if n_pos > 0 else 1.0

    fold_imps = []
    correct = 0
    total = 0
    early_stop_counts = []
    for tr, te in pairs:
        # 3-sample internal validation split inside training fold for early stopping
        n_tr = len(tr)
        # stratify so val has at least one of each class when feasible
        rng = np.random.default_rng(seed + total)
        tr_perm = tr.copy()
        rng.shuffle(tr_perm)
        # Pick 2 samples (1 each class if possible) for internal val
        tr_y = y_arr[tr_perm]
        val_idx_pos = [i for i in tr_perm if y_arr[i] == 1][:1]
        val_idx_neg = [i for i in tr_perm if y_arr[i] == 0][:1]
        val_idx = np.array(val_idx_pos + val_idx_neg)
        tr_only = np.array([i for i in tr_perm if i not in val_idx])

        train_set = lgb.Dataset(X.iloc[tr_only].values, label=y_arr[tr_only])
        val_set = lgb.Dataset(X.iloc[val_idx].values, label=y_arr[val_idx],
                              reference=train_set)
        callbacks = [lgb.early_stopping(HUMAN_EARLY_STOPPING, verbose=False),
                     lgb.log_evaluation(0)]
        model = lgb.train(
            base, train_set,
            num_boost_round=int(base.get("n_estimators", 200)),
            valid_sets=[val_set], callbacks=callbacks,
        )
        early_stop_counts.append(int(model.best_iteration or model.num_trees()))

        # gain importance
        names = model.feature_name() or [str(c) for c in X.columns]
        gains = model.feature_importance(importance_type="gain")
        # map back to ortholog gene IDs (X.columns)
        imp = pd.Series(gains, index=X.columns[:len(gains)])
        fold_imps.append(imp)

        # held-out accuracy
        prob = model.predict(X.iloc[te].values,
                             num_iteration=model.best_iteration)
        pred = (prob >= 0.5).astype(int)
        correct += int(np.sum(pred == y_arr[te]))
        total += 2

    df = pd.concat(fold_imps, axis=1).fillna(0.0)
    seed_imp = df.mean(axis=1).sort_values(ascending=False)
    seed_imp.name = f"seed_{seed}_gain"
    stats_d = {
        "seed": seed,
        "n_pairs": len(pairs),
        "lpo_accuracy": float(correct / total) if total else None,
        "mean_best_iteration": float(np.mean(early_stop_counts))
                               if early_stop_counts else None,
        "frac_under20_trees": float(np.mean(
            [c < 20 for c in early_stop_counts]
        )) if early_stop_counts else 0.0,
    }
    return seed_imp, stats_d


# ============================================================================
# STABILITY SET HELPERS
# ============================================================================
def stable_set(seed_imps: Dict[int, pd.Series], k: int,
               threshold: int) -> Tuple[List[str], pd.DataFrame]:
    """Return list of genes appearing in top-k in >= threshold seeds,
    and a tidy table with rank stats."""
    counter = defaultdict(int)
    rank_records = defaultdict(list)
    gain_records = defaultdict(list)
    for seed, imp in seed_imps.items():
        # Exclude features the model never used (gain == 0). Without this,
        # ties at zero get included via head(k) and become spuriously
        # "stable" across seeds because pandas sort_values is stable in
        # alphabetical-index order, so the same alphabet-first zero-gain
        # features appear in every seed's "top-K" by sort.
        nonzero = imp[imp > 0]
        top = nonzero.nlargest(k)
        for rank, (g, v) in enumerate(top.items(), start=1):
            counter[g] += 1
            rank_records[g].append(rank)
            gain_records[g].append(v)
    rows = []
    for g, c in counter.items():
        rows.append({
            "gene_id": g,
            "n_seeds_top": c,
            "mean_rank": float(np.mean(rank_records[g])),
            "median_rank": float(np.median(rank_records[g])),
            "mean_gain": float(np.mean(gain_records[g])),
        })
    tab = pd.DataFrame(rows).sort_values(
        ["n_seeds_top", "mean_gain"], ascending=[False, False]
    ).reset_index(drop=True)
    stable = tab[tab["n_seeds_top"] >= threshold]["gene_id"].tolist()
    return stable, tab


def aggregate_seed_imps(seed_imps: Dict[int, pd.Series]
                        ) -> pd.DataFrame:
    """Combine per-seed importance into a wide DataFrame (gene x seed)."""
    return pd.concat(seed_imps.values(), axis=1).fillna(0.0)


# ============================================================================
# CONCORDANCE
# ============================================================================
def fold_change(raw: pd.DataFrame, samples_young: List[str],
                samples_old: List[str]) -> pd.Series:
    """log2 fold change Infant->Adult on raw TPM/RPKM. Returns Series indexed by gene."""
    y = raw.loc[samples_young].mean(axis=0)
    o = raw.loc[samples_old].mean(axis=0)
    fc = np.log2((o + PSEUDO) / (y + PSEUDO))
    return fc


def bootstrap_pearson(x: np.ndarray, y: np.ndarray,
                      n: int = N_BOOTSTRAP, seed: int = 42
                      ) -> Tuple[float, float, float]:
    if len(x) < 5:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        xs, ys = x[idx], y[idx]
        if np.std(xs) > 0 and np.std(ys) > 0:
            r, _ = stats.pearsonr(xs, ys)
            if not np.isnan(r):
                rs.append(r)
    rs = np.array(rs)
    return (float(np.mean(rs)),
            float(np.percentile(rs, 2.5)),
            float(np.percentile(rs, 97.5)))


# ============================================================================
# ANNOTATION (mygene)
# ============================================================================
def load_annot_cache() -> dict:
    if ANNOT_CACHE.exists():
        return json.loads(ANNOT_CACHE.read_text())
    return {}


def save_annot_cache(c: dict) -> None:
    ANNOT_CACHE.write_text(json.dumps(c, indent=1))


def annotate_genes(symbols: List[str]) -> Dict[str, dict]:
    """For each symbol, return a dict with 'go_bp' (list) and 'category' (str)."""
    cache = load_annot_cache()
    todo = [s for s in symbols if s not in cache]
    if todo:
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            res = mg.querymany(todo, scopes="symbol",
                               fields="go.BP.term,name,summary",
                               species="human", returnall=False)
            for r in res:
                sym = (r.get("query") or "").upper()
                terms: List[str] = []
                go = r.get("go", {}) or {}
                bp = go.get("BP", []) if isinstance(go, dict) else []
                if isinstance(bp, dict):
                    bp = [bp]
                for t in bp:
                    if isinstance(t, dict) and t.get("term"):
                        terms.append(t["term"])
                cache[sym] = {
                    "go_bp": terms[:30],
                    "name": r.get("name", ""),
                    "summary": (r.get("summary", "") or "")[:300],
                }
            save_annot_cache(cache)
        except Exception as exc:  # noqa: BLE001
            for s in todo:
                if s not in cache:
                    cache[s] = {"go_bp": [], "name": "",
                                "summary": f"annot_error:{exc}"}
            save_annot_cache(cache)
    # classify
    classified = {}
    for sym in symbols:
        rec = cache.get(sym, {"go_bp": [], "name": "", "summary": ""})
        text = " | ".join(rec.get("go_bp", []) + [rec.get("name", "")]).lower()
        best = "other"
        best_hits = 0
        for cat, kws in BIO_CATEGORIES.items():
            hits = sum(kw.lower() in text for kw in kws)
            if hits > best_hits:
                best = cat
                best_hits = hits
        classified[sym] = {**rec, "category": best, "hits": best_hits}
    return classified


def annotation_summary(annot: Dict[str, dict]) -> Dict[str, int]:
    counts = defaultdict(int)
    for v in annot.values():
        counts[v["category"]] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# ============================================================================
# PER-TISSUE PIPELINE
# ============================================================================
def run_tissue(tissue: str, ortho: pd.DataFrame) -> dict:
    print(f"\n{'='*72}\n>> {tissue}\n{'='*72}", flush=True)
    t0 = time.time()

    # ---- Load pig
    print("  Loading pig matrix...", flush=True)
    pig_X, pig_y, pig_raw = load_pig(tissue, ortho)
    print(f"    pig: {pig_X.shape[0]} samples x {pig_X.shape[1]} ortholog genes",
          flush=True)
    print(f"    pig stage counts: {pig_y.value_counts().to_dict()}",
          flush=True)
    log_row(tissue, "load_pig", "n_samples", pig_X.shape[0])
    log_row(tissue, "load_pig", "n_genes", pig_X.shape[1])

    # ---- Train pig (5-fold CV per seed)
    print("  Pig ordinal: training 10 seeds x 5-fold CV...", flush=True)
    pig_seed_imp: Dict[int, pd.Series] = {}
    pig_seed_stats: List[dict] = []
    for seed in SEEDS:
        ts = time.time()
        imp, stat = train_pig_seed(pig_X, pig_y, seed)
        pig_seed_imp[seed] = imp
        pig_seed_stats.append(stat)
        print(f"    pig seed {seed}: top-5={list(imp.head(5).index)[:3]}... "
              f"MAE={stat['mean_MAE']:.2f} ({time.time()-ts:.0f}s)",
              flush=True)
        log_row(tissue, "pig_train", "seed_mae",
                stat["mean_MAE"], note=f"seed={seed}")

    pig_stable_100, pig_table = stable_set(pig_seed_imp, k=100,
                                           threshold=STABILITY_THRESHOLD)
    pig_stable_50, _ = stable_set(pig_seed_imp, k=50, threshold=STABILITY_THRESHOLD)
    pig_stable_200, _ = stable_set(pig_seed_imp, k=200, threshold=STABILITY_THRESHOLD)
    pig_table.to_csv(OUT_DIR / f"features_pig_{tissue}.csv", index=False)
    # Dump per-seed full importance for downstream K-sensitivity analysis
    pig_full_imp = aggregate_seed_imps(pig_seed_imp)
    pig_full_imp.columns = [f"seed_{s}" for s in pig_seed_imp.keys()]
    pig_full_imp.index.name = "pig_gene_id"
    pig_full_imp.to_parquet(OUT_DIR / f"seed_importance_pig_{tissue}.parquet")
    log_row(tissue, "pig_stable", "n_stable_100", len(pig_stable_100))

    # Seed-stability (Jaccard @100)
    pig_jaccards = []
    seed_list = list(pig_seed_imp.keys())
    for i in range(len(seed_list)):
        for j in range(i + 1, len(seed_list)):
            a = set(pig_seed_imp[seed_list[i]].head(100).index)
            b = set(pig_seed_imp[seed_list[j]].head(100).index)
            if a or b:
                pig_jaccards.append(len(a & b) / len(a | b))
    pig_jacc_mean = float(np.mean(pig_jaccards)) if pig_jaccards else float("nan")
    log_row(tissue, "pig_stability", "jaccard_top100_mean", pig_jacc_mean)

    # ---- Load human (if available)
    human_data = load_human(tissue, ortho)
    if human_data is None:
        print(f"  Human: NOT AVAILABLE for {tissue} (no Cardoso-Moreira / "
              f"GSE matrix)", flush=True)
        log_row(tissue, "human_load", "available", 0,
                note="no human postnatal matrix")
        # No joint set; still save pig-only output
        result = {
            "tissue": tissue,
            "human_source": HUMAN_SOURCE[tissue],
            "human_available": False,
            "n_orthologs_used": int(pig_X.shape[1]),
            "pig": {
                "n_samples": int(pig_X.shape[0]),
                "stage_counts": pig_y.value_counts().to_dict(),
                "seed_mae": [s["mean_MAE"] for s in pig_seed_stats],
                "stable_top100_n": len(pig_stable_100),
                "stable_top100": pig_stable_100[:200],
                "stable_top50_n": len(pig_stable_50),
                "stable_top200_n": len(pig_stable_200),
                "jaccard_top100_mean": pig_jacc_mean,
            },
            "joint_features": [],
            "joint_set_n": 0,
            "concordance": None,
            "annotation_summary": {},
            "elapsed_s": round(time.time() - t0, 1),
        }
        (OUT_DIR / f"results_{tissue}.json").write_text(
            json.dumps(result, indent=2, default=str))
        # Empty joint csv for completeness
        pd.DataFrame(columns=[
            "symbol", "pig_gene_id", "human_gene_id",
            "pig_mean_rank", "human_mean_rank",
            "pig_n_seeds_top100", "human_n_seeds_top100",
            "log2fc_pig", "log2fc_human", "category",
        ]).to_csv(OUT_DIR / f"joint_features_{tissue}.csv", index=False)
        return result

    human_X, human_y, human_raw = human_data
    print(f"    human: {human_X.shape[0]} samples (infant={int((human_y==0).sum())}, "
          f"adult={int((human_y==1).sum())}) x {human_X.shape[1]} ortholog genes",
          flush=True)
    log_row(tissue, "load_human", "n_samples", human_X.shape[0])
    log_row(tissue, "load_human", "n_infant", int((human_y == 0).sum()))
    log_row(tissue, "load_human", "n_adult", int((human_y == 1).sum()))

    # ---- Train human (10 seeds x leave-pair-out)
    print("  Human binary: training 10 seeds x leave-pair-out...", flush=True)
    human_seed_imp: Dict[int, pd.Series] = {}
    human_seed_stats: List[dict] = []
    params = HUMAN_PARAM_SWEEP[0]
    sweep_idx = 0
    for seed in SEEDS:
        ts = time.time()
        imp, stat = train_human_seed(human_X, human_y, seed,
                                     params_override=params)
        human_seed_imp[seed] = imp
        human_seed_stats.append(stat)
        print(f"    human seed {seed}: top-5={list(imp.head(5).index)[:3]}... "
              f"LPO_acc={stat['lpo_accuracy']:.2f} "
              f"bestIter={stat['mean_best_iteration']:.0f} "
              f"({time.time()-ts:.0f}s)",
              flush=True)
        log_row(tissue, "human_train", "seed_lpo_acc",
                stat["lpo_accuracy"], note=f"seed={seed}")

    human_stable_100, human_table = stable_set(
        human_seed_imp, k=100, threshold=STABILITY_THRESHOLD)
    human_stable_50, _ = stable_set(human_seed_imp, k=50, threshold=STABILITY_THRESHOLD)
    human_stable_200, _ = stable_set(human_seed_imp, k=200, threshold=STABILITY_THRESHOLD)
    human_table.to_csv(OUT_DIR / f"features_human_{tissue}.csv", index=False)
    # Dump per-seed full importance for downstream K-sensitivity analysis
    human_full_imp = aggregate_seed_imps(human_seed_imp)
    human_full_imp.columns = [f"seed_{s}" for s in human_seed_imp.keys()]
    human_full_imp.index.name = "human_gene_id"
    human_full_imp.to_parquet(OUT_DIR / f"seed_importance_human_{tissue}.parquet")
    log_row(tissue, "human_stable", "n_stable_100", len(human_stable_100))

    # Refinement sweep: if joint @ 100 is empty AND we have not used all sweeps
    def compute_joint(p_stable: List[str], h_stable: List[str]
                      ) -> List[Tuple[str, str]]:
        ortho_sub = ortho[
            ortho["pig_gene_id"].isin(p_stable) &
            ortho["human_gene_id"].isin(h_stable)
        ]
        return list(zip(ortho_sub["pig_gene_id"], ortho_sub["human_gene_id"]))

    joint_pairs = compute_joint(pig_stable_100, human_stable_100)
    if len(joint_pairs) == 0 and len(HUMAN_PARAM_SWEEP) > 1:
        print(f"  Joint @100 empty - running human param refinement", flush=True)
        for alt_idx in range(1, len(HUMAN_PARAM_SWEEP)):
            params = HUMAN_PARAM_SWEEP[alt_idx]
            sweep_idx = alt_idx
            human_seed_imp = {}
            human_seed_stats = []
            for seed in SEEDS:
                imp, stat = train_human_seed(human_X, human_y, seed,
                                             params_override=params)
                human_seed_imp[seed] = imp
                human_seed_stats.append(stat)
            human_stable_100, human_table = stable_set(
                human_seed_imp, k=100, threshold=STABILITY_THRESHOLD)
            human_stable_50, _ = stable_set(human_seed_imp, k=50,
                                            threshold=STABILITY_THRESHOLD)
            human_stable_200, _ = stable_set(human_seed_imp, k=200,
                                             threshold=STABILITY_THRESHOLD)
            human_table.to_csv(OUT_DIR / f"features_human_{tissue}.csv",
                               index=False)
            # Re-dump human full importance to match the kept sweep
            human_full_imp = aggregate_seed_imps(human_seed_imp)
            human_full_imp.columns = [f"seed_{s}" for s in human_seed_imp.keys()]
            human_full_imp.index.name = "human_gene_id"
            human_full_imp.to_parquet(
                OUT_DIR / f"seed_importance_human_{tissue}.parquet")
            joint_pairs = compute_joint(pig_stable_100, human_stable_100)
            print(f"    sweep {alt_idx} ({params}): joint @100 = {len(joint_pairs)}",
                  flush=True)
            log_row(tissue, "human_sweep", "joint_n",
                    len(joint_pairs), note=f"sweep={alt_idx}")
            if len(joint_pairs) > 0:
                break

    log_row(tissue, "joint", "n_joint_100", len(joint_pairs))

    # Sensitivity joint sets (top-50, top-200)
    joint_50 = compute_joint(pig_stable_50, human_stable_50)
    joint_200 = compute_joint(pig_stable_200, human_stable_200)

    # human Jaccard @100
    h_jaccs = []
    seed_list = list(human_seed_imp.keys())
    for i in range(len(seed_list)):
        for j in range(i + 1, len(seed_list)):
            a = set(human_seed_imp[seed_list[i]].head(100).index)
            b = set(human_seed_imp[seed_list[j]].head(100).index)
            if a or b:
                h_jaccs.append(len(a & b) / len(a | b))
    human_jacc_mean = float(np.mean(h_jaccs)) if h_jaccs else float("nan")
    log_row(tissue, "human_stability", "jaccard_top100_mean", human_jacc_mean)

    # ---- Concordance on joint set (log2FC infant->adult)
    if joint_pairs:
        pig_young_samples = pig_y[pig_y.isin(PIG_YOUNG)].index.tolist()
        pig_old_samples = pig_y[pig_y.isin(PIG_OLD)].index.tolist()
        human_young_samples = human_y[human_y == 0].index.tolist()
        human_old_samples = human_y[human_y == 1].index.tolist()

        pig_fc = fold_change(pig_raw, pig_young_samples, pig_old_samples)
        human_fc = fold_change(human_raw, human_young_samples, human_old_samples)

        pig_ids = [p for p, _ in joint_pairs]
        human_ids = [h for _, h in joint_pairs]
        x = pig_fc.reindex(pig_ids).values
        y_v = human_fc.reindex(human_ids).values
        ok = np.isfinite(x) & np.isfinite(y_v)
        x = x[ok]; y_v = y_v[ok]
        joint_pairs_ok = [joint_pairs[i] for i, k in enumerate(ok) if k]

        if len(x) >= 5 and np.std(x) > 0 and np.std(y_v) > 0:
            r_obs, p_obs = stats.pearsonr(x, y_v)
            r_mean, ci_lo, ci_hi = bootstrap_pearson(x, y_v)
            concordant = int(np.sum(np.sign(x) == np.sign(y_v)))
            pct_dir = 100.0 * concordant / len(x)
        else:
            r_obs = p_obs = r_mean = ci_lo = ci_hi = float("nan")
            concordant = 0; pct_dir = float("nan")

        concordance = {
            "n_joint_genes_for_fc": int(len(x)),
            "pearson_r": float(r_obs) if np.isfinite(r_obs) else None,
            "pearson_p": float(p_obs) if np.isfinite(p_obs) else None,
            "bootstrap_mean_r": float(r_mean) if np.isfinite(r_mean) else None,
            "ci_low": float(ci_lo) if np.isfinite(ci_lo) else None,
            "ci_high": float(ci_hi) if np.isfinite(ci_hi) else None,
            "n_concordant_direction": concordant,
            "directional_concordance_pct": float(pct_dir) if np.isfinite(pct_dir) else None,
        }
    else:
        joint_pairs_ok = []
        concordance = {
            "n_joint_genes_for_fc": 0,
            "pearson_r": None, "pearson_p": None,
            "bootstrap_mean_r": None, "ci_low": None, "ci_high": None,
            "n_concordant_direction": 0,
            "directional_concordance_pct": None,
        }
    log_row(tissue, "concordance", "pearson_r",
            concordance["pearson_r"], note=f"n={concordance['n_joint_genes_for_fc']}")

    # Build joint feature CSV (with symbols and FCs)
    pig_to_sym = dict(zip(ortho["pig_gene_id"], ortho["symbol"]))
    human_to_sym = dict(zip(ortho["human_gene_id"], ortho["symbol"]))
    pig_rank_map = {r["gene_id"]: r["mean_rank"] for _, r in pig_table.iterrows()}
    pig_seed_count = {r["gene_id"]: r["n_seeds_top"] for _, r in pig_table.iterrows()}
    human_rank_map = {r["gene_id"]: r["mean_rank"] for _, r in human_table.iterrows()}
    human_seed_count = {r["gene_id"]: r["n_seeds_top"]
                        for _, r in human_table.iterrows()}

    if joint_pairs:
        pig_young_samples = pig_y[pig_y.isin(PIG_YOUNG)].index.tolist()
        pig_old_samples = pig_y[pig_y.isin(PIG_OLD)].index.tolist()
        human_young_samples = human_y[human_y == 0].index.tolist()
        human_old_samples = human_y[human_y == 1].index.tolist()
        pig_fc_full = fold_change(pig_raw, pig_young_samples, pig_old_samples)
        human_fc_full = fold_change(human_raw, human_young_samples, human_old_samples)
    else:
        pig_fc_full = pd.Series(dtype=float)
        human_fc_full = pd.Series(dtype=float)

    rows = []
    for (pid, hid) in joint_pairs:
        sym = pig_to_sym.get(pid) or human_to_sym.get(hid) or ""
        rows.append({
            "symbol": sym,
            "pig_gene_id": pid,
            "human_gene_id": hid,
            "pig_n_seeds_top100": int(pig_seed_count.get(pid, 0)),
            "pig_mean_rank": float(pig_rank_map.get(pid, np.nan)),
            "human_n_seeds_top100": int(human_seed_count.get(hid, 0)),
            "human_mean_rank": float(human_rank_map.get(hid, np.nan)),
            "log2fc_pig": float(pig_fc_full.get(pid, np.nan)),
            "log2fc_human": float(human_fc_full.get(hid, np.nan)),
        })
    joint_df = pd.DataFrame(rows)

    # Annotate
    annot = {}
    if joint_pairs:
        syms = [r["symbol"] for r in rows if r["symbol"]]
        annot = annotate_genes(sorted(set(syms)))
    if joint_df.shape[0] > 0:
        joint_df["category"] = joint_df["symbol"].map(
            lambda s: annot.get(s, {}).get("category", "unknown"))
    else:
        joint_df["category"] = []
    joint_df = joint_df.sort_values(
        ["pig_n_seeds_top100", "human_n_seeds_top100", "pig_mean_rank"],
        ascending=[False, False, True]
    ).reset_index(drop=True)
    joint_df.to_csv(OUT_DIR / f"joint_features_{tissue}.csv", index=False)

    annot_summary = annotation_summary({s: a for s, a in annot.items()
                                        if s in joint_df["symbol"].values})

    result = {
        "tissue": tissue,
        "human_source": HUMAN_SOURCE[tissue],
        "human_available": True,
        "n_orthologs_used": int(pig_X.shape[1]),
        "human_params_used": params,
        "human_sweep_idx": sweep_idx,
        "pig": {
            "n_samples": int(pig_X.shape[0]),
            "stage_counts": pig_y.value_counts().to_dict(),
            "seed_mae": [s["mean_MAE"] for s in pig_seed_stats],
            "stable_top100_n": len(pig_stable_100),
            "stable_top50_n": len(pig_stable_50),
            "stable_top200_n": len(pig_stable_200),
            "jaccard_top100_mean": pig_jacc_mean,
        },
        "human": {
            "n_samples": int(human_X.shape[0]),
            "n_infant": int((human_y == 0).sum()),
            "n_adult": int((human_y == 1).sum()),
            "seed_lpo_accuracy": [s["lpo_accuracy"] for s in human_seed_stats],
            "stable_top100_n": len(human_stable_100),
            "stable_top50_n": len(human_stable_50),
            "stable_top200_n": len(human_stable_200),
            "jaccard_top100_mean": human_jacc_mean,
            "mean_best_iteration_seeds":
                [s["mean_best_iteration"] for s in human_seed_stats],
        },
        "joint_set_n": int(len(joint_pairs)),
        "joint_set_top50_n": int(len(joint_50)),
        "joint_set_top200_n": int(len(joint_200)),
        "joint_features": [
            {"symbol": r["symbol"],
             "pig_gene_id": r["pig_gene_id"],
             "human_gene_id": r["human_gene_id"],
             "pig_n_seeds": int(r["pig_n_seeds_top100"]),
             "human_n_seeds": int(r["human_n_seeds_top100"]),
             "log2fc_pig": float(r["log2fc_pig"]) if pd.notna(r["log2fc_pig"]) else None,
             "log2fc_human": float(r["log2fc_human"]) if pd.notna(r["log2fc_human"]) else None,
             "category": r["category"]}
            for _, r in joint_df.iterrows()
        ],
        "concordance": concordance,
        "annotation_summary": annot_summary,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / f"results_{tissue}.json").write_text(
        json.dumps(result, indent=2, default=str))
    print(f"  ==> joint set n={len(joint_pairs)}, "
          f"r={concordance.get('pearson_r')}, "
          f"elapsed={result['elapsed_s']}s", flush=True)
    return result


# ============================================================================
# MAIN
# ============================================================================
def main():
    init_log()
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"Loading orthologs from {ORTHO_CSV}")
    ortho = load_orthologs()
    print(f"  {len(ortho)} one-to-one ortholog pairs")
    log_row("ALL", "orthologs", "n_pairs", len(ortho))

    results = {}
    crashes = defaultdict(int)
    for tissue in TISSUES:
        if crashes[tissue] >= 3:
            print(f"SKIP {tissue}: 3 consecutive crashes")
            log_row(tissue, "main", "skipped", 0, note="3_consecutive_crashes")
            continue
        # Skip if a prior non-error result file exists
        prior = OUT_DIR / f"results_{tissue}.json"
        if prior.exists():
            try:
                pd_obj = json.loads(prior.read_text())
                if "error" not in pd_obj:
                    print(f"SKIP {tissue}: prior results_{tissue}.json exists")
                    results[tissue] = pd_obj
                    continue
            except Exception:  # noqa: BLE001
                pass
        try:
            results[tissue] = run_tissue(tissue, ortho)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            crashes[tissue] += 1
            print(f"!! {tissue} crashed: {exc}")
            log_row(tissue, "crash", "exception",
                    str(exc).replace("\t", " ")[:200],
                    note=f"attempt={crashes[tissue]}")
            results[tissue] = {"tissue": tissue, "error": str(exc)}
            continue

    summary = {
        "n_tissues": len(results),
        "tissues": list(results.keys()),
        "joint_set_sizes": {
            t: r.get("joint_set_n", 0) for t, r in results.items()
        },
        "concordance_r": {
            t: (r.get("concordance") or {}).get("pearson_r")
            for t, r in results.items()
        },
    }
    (OUT_DIR / "summary_all_tissues.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
