#!/usr/bin/env python3
"""Canonical within-pig 6-stage figure artifact (reproducible, local).

Produces ONE JSON consumed by Fig 2 (confound + inflation gap), Fig S3
(per-class recall under grouped CV) and Fig 4 panel D (within-pig ordinal rho):

  paper/figures/data/within_pig_6stage.json

Per tissue (6-stage scheme, drop-small < 10):
  n_samples, n_bioprojects, kept6{stage:n}, n_class, chance,
  single_stage_pct  (% of kept samples from single-stage BioProjects),
  stratified_6stage {BA_mean, BA_sd, rho_mean, rho_sd, per_seed_ba},
  grouped_6stage    {BA_mean, BA_sd, rho_mean, rho_sd, per_seed_ba,
                     per_class_recall {stage: {mean, sd}}}

Model: OrdinalPLS (default n_components=10) — the SAME default model the paper's
main-text PLS evaluation uses (reproducible from the repo; the Optuna-tuned
18-model HPC benchmark in Table S8 is a separate ranking protocol). 5 seeds.

Honest-eval CV regimes (both reported):
  stratified : StratifiedKFold              (batch-inclusive, in-distribution)
  grouped    : StratifiedGroupKFold(BioProject)  (leakage-controlled, deployment)

Per-class recall + ordinal rho are computed from POOLED out-of-fold predictions
within each seed (every sample predicted once), then averaged over seeds.
Real data only — no fabricated numbers.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/Users/tianyuan/Desktop/github_dev/pig-dev-stage")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # provenance dir: sibling modules
from machine_learning.data_processing.data_loader import DataLoader
from machine_learning.model_training.models import OrdinalPLS
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, recall_score
from scipy.stats import spearmanr

DATA_DIR = ROOT / "data" / "pigGTEx"
META = ROOT / "data" / "PigGTEx_v0.MetaTable.csv"
TISSUES = ["Muscle", "Brain", "Liver", "Blood", "Lung"]
MIN_PER_STAGE = 10
N_COMPONENTS = 10
SEEDS = [42, 43, 44, 45, 46]
N_FOLDS = 5

ORDER6 = ["Neonatal", "Suckling", "Juvenile", "Pre-pubertal", "Post-pubertal", "Adult"]


def stage6(d):
    if pd.isna(d): return None
    if d <= 3:   return "Neonatal"
    if d <= 28:  return "Suckling"
    if d <= 90:  return "Juvenile"
    if d <= 165: return "Pre-pubertal"
    if d <= 540: return "Post-pubertal"
    return "Adult"


def drop_small(labels: pd.Series, order):
    counts = labels.value_counts()
    return [s for s in order if counts.get(s, 0) >= MIN_PER_STAGE]


def single_stage_pct(stage: pd.Series, bp: pd.Series, kept) -> float:
    """% of kept-stage samples that come from single-stage BioProjects."""
    m = stage.isin(kept).values
    s = stage[m].reset_index(drop=True)
    g = bp[m].reset_index(drop=True)
    if len(s) == 0:
        return float("nan")
    df = pd.DataFrame({"stage": s, "bp": g})
    n_stage_per_bp = df.groupby("bp")["stage"].nunique()
    single_bps = set(n_stage_per_bp[n_stage_per_bp == 1].index)
    n_single = int(df["bp"].isin(single_bps).sum())
    return round(100.0 * n_single / len(df), 1)


def expected_class(model, X) -> np.ndarray:
    """Continuous expected stage index E[k]=sum_k k*P_k from predict_proba."""
    P = model.predict_proba(X)
    ks = np.arange(P.shape[1])
    return P @ ks


def eval_regime(Xk, yi, gk, kept, cv_kind):
    """Return per-seed BA list, per-seed rho list, and per-seed per-class
    recall (matrix seeds x K) using POOLED out-of-fold predictions per seed."""
    K = len(kept)
    per_seed_ba, per_seed_rho, per_seed_recall = [], [], []
    for seed in SEEDS:
        if cv_kind == "stratified":
            splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            split_iter = list(splitter.split(Xk, yi))
        else:
            ngrp = len(np.unique(gk))
            nf = min(N_FOLDS, ngrp)
            if nf < 2:
                return None
            splitter = StratifiedGroupKFold(n_splits=nf, shuffle=True, random_state=seed)
            split_iter = list(splitter.split(Xk, yi, groups=gk))
        oof_true = np.full(len(yi), -1, dtype=int)
        oof_pred = np.full(len(yi), -1, dtype=int)
        oof_score = np.full(len(yi), np.nan, dtype=float)
        fold_ba = []
        for tr, te in split_iter:
            if len(np.unique(yi[tr])) < 2:
                continue
            model = OrdinalPLS(n_components=N_COMPONENTS, seed=seed)
            model.fit(pd.DataFrame(Xk[tr]), yi[tr])
            pred = model.predict(pd.DataFrame(Xk[te]))
            pred = np.asarray(pred, dtype=int)
            score = expected_class(model, pd.DataFrame(Xk[te]))
            oof_true[te] = yi[te]
            oof_pred[te] = pred
            oof_score[te] = score
            fold_ba.append(balanced_accuracy_score(yi[te], pred))
        ok = oof_true >= 0
        if ok.sum() < 2 or not fold_ba:
            continue
        per_seed_ba.append(float(np.mean(fold_ba)))
        # ordinal rho from pooled OOF continuous score
        if np.unique(oof_true[ok]).size > 1 and np.nanstd(oof_score[ok]) > 0:
            rho = spearmanr(oof_true[ok], oof_score[ok]).correlation
            per_seed_rho.append(float(rho) if rho == rho else np.nan)
        else:
            per_seed_rho.append(np.nan)
        rec = recall_score(oof_true[ok], oof_pred[ok], labels=list(range(K)),
                           average=None, zero_division=0)
        per_seed_recall.append(rec)
    if not per_seed_ba:
        return None
    return {
        "per_seed_ba": per_seed_ba,
        "per_seed_rho": per_seed_rho,
        "recall_matrix": np.vstack(per_seed_recall),  # seeds x K
    }


def summarize(reg, kept, with_recall=False):
    rhos = np.array(reg["per_seed_rho"], dtype=float)
    valid = rhos[~np.isnan(rhos)]
    rho_mean = round(float(valid.mean()), 6) if valid.size else None  # None = degenerate
    rho_sd = round(float(valid.std()), 6) if valid.size else None
    out = {
        "BA_mean": float(np.mean(reg["per_seed_ba"])),
        "BA_sd": float(np.std(reg["per_seed_ba"])),
        "rho_mean": rho_mean,
        "rho_sd": rho_sd,
        "n_seeds": len(reg["per_seed_ba"]),
        "per_seed_ba": [round(b, 4) for b in reg["per_seed_ba"]],
        "per_seed_rho": [None if (r != r) else round(r, 4) for r in reg["per_seed_rho"]],
    }
    if with_recall:
        M = reg["recall_matrix"]  # seeds x K
        out["per_class_recall"] = {
            kept[k]: {"mean": round(float(M[:, k].mean()), 4),
                      "sd": round(float(M[:, k].std()), 4)}
            for k in range(len(kept))
        }
        # per-seed recall (for the supplementary figure's seed-spread jitter)
        out["per_class_recall_seeds"] = {
            kept[k]: [round(float(M[s, k]), 4) for s in range(M.shape[0])]
            for k in range(len(kept))
        }
    return out


def main():
    tissues = sys.argv[1:] or TISSUES
    dl = DataLoader(data_dir=DATA_DIR, metadata_path=META)
    dl.load_metadata()
    results = {}
    for tissue in tissues:
        if tissue == "Brain":
            import verify_brain_6stage as vb6
            meta = vb6._load_meta()
            expr = vb6._load_expr()
            fb_idx = vb6.select_forebrain(meta)
            samples = [s for s in fb_idx if s in expr.columns]
            expr_sub = expr[samples].T
            X = np.log2(expr_sub.values + 1.0)
            age = meta.loc[samples, "Age_Days"]
            bp = meta.loc[samples, "BioProject"].astype(str).reset_index(drop=True)
            y6 = age.map(stage6).reset_index(drop=True)
        else:
            expr, meta = dl.load_expression(tissue)
            X = np.log2(expr.T.values + 1.0)
            age = meta["Age_Days"] if "Age_Days" in meta else meta["Age"]
            age = pd.to_numeric(age, errors="coerce")
            bp = meta["BioProject"].astype(str).reset_index(drop=True)
            y6 = age.map(stage6).reset_index(drop=True)

        keep_age = age.notna().values
        X = X[keep_age]
        y6 = y6[keep_age].reset_index(drop=True)
        bp = bp[keep_age].reset_index(drop=True)

        kept6 = drop_small(y6, ORDER6)
        K = len(kept6)
        code = {s: i for i, s in enumerate(kept6)}
        mask = y6.isin(kept6).values
        Xk = X[mask]
        yk = y6[mask].values
        gk = bp[mask].values
        yi = np.array([code[v] for v in yk])

        tinfo = {
            "n_samples": int(keep_age.sum()),
            "n_bioprojects": int(bp.nunique()),
            "kept6": {s: int((y6 == s).sum()) for s in kept6},
            "n_class": K,
            "chance": round(1.0 / K, 4),
            "single_stage_pct": single_stage_pct(y6, bp, kept6),
        }
        for cv in ["stratified", "grouped"]:
            reg = eval_regime(Xk, yi, gk, kept6, cv)
            key = f"{cv}_6stage"
            if reg is None:
                tinfo[key] = {"BA_mean": None, "note": "not identifiable / single study"}
            else:
                tinfo[key] = summarize(reg, kept6, with_recall=(cv == "grouped"))
        results[tissue] = tinfo

        def fmt(k):
            v = tinfo[k]
            if v.get("BA_mean") is None:
                return "n/a"
            rho = v.get("rho_mean")
            rs = f"{rho:.3f}" if rho is not None else "degenerate"
            return f"BA {v['BA_mean']:.3f}±{v['BA_sd']:.3f} rho {rs}"
        print(f"\n=== {tissue}  (n={tinfo['n_samples']}, {tinfo['n_bioprojects']} BP, "
              f"K={K}, chance={tinfo['chance']}, single-stage {tinfo['single_stage_pct']}%) ===")
        print(f"  kept6: {tinfo['kept6']}")
        print(f"  STRATIFIED: {fmt('stratified_6stage')}")
        print(f"  GROUPED   : {fmt('grouped_6stage')}")
        if tinfo["grouped_6stage"].get("BA_mean") is not None:
            for s, r in tinfo["grouped_6stage"]["per_class_recall"].items():
                print(f"      recall {s:<14} {r['mean']:.2f}±{r['sd']:.2f}")
        sys.stdout.flush()

    out = ROOT / "paper" / "figures" / "data" / "within_pig_6stage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
