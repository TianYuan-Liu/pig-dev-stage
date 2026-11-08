#!/usr/bin/env python3
"""Recompute Supplementary Table S2 (stratified per-class P/R/F1) and
Table S4 (PCA variance decomposition Stage/BioProject) under the 6-stage scheme.

Reuses the canonical 6-stage helpers from paper/figures/provenance/transfer_6stage.py
(load_pig_tissue, stage6, assign_ordinal_stages) and the deployed OrdinalPLS
(n_components=10, the reproducible default used for the main-text honest eval).
No fabricated numbers -- everything computed from data/.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # provenance dir: sibling modules
import transfer_6stage as t6
from machine_learning.model_training.models import OrdinalPLS

TISSUES = ["Muscle", "Brain", "Liver", "Blood", "Lung"]
SEED = 42
NCOMP = 10


def s2_per_class(tissue, pig_X):
    """Stratified 5-fold OOF per-class precision/recall/F1 (6-stage, drop-small)."""
    stage, kept = t6.assign_ordinal_stages(pig_X["Age_Days"], "6stage")
    if len(kept) < 2:
        return None
    code = {s: i for i, s in enumerate(kept)}
    mask = stage.notna().values
    feat = pig_X.drop(columns=["Age_Days", "BioProject"]).loc[mask]
    cols = [f"g{i}" for i in range(feat.shape[1])]
    feat.columns = cols
    y = np.array([code[s] for s in stage[mask].values])
    n_splits = min(5, np.bincount(y).min())
    if n_splits < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.full(len(y), -1)
    for tr, te in skf.split(feat, y):
        m = OrdinalPLS(n_components=NCOMP, seed=SEED)
        m.fit(feat.iloc[tr], y[tr])
        oof[te] = np.rint(np.clip(m.predict(feat.iloc[te]), 0, len(kept) - 1)).astype(int)
    P, R, F, S = precision_recall_fscore_support(y, oof, labels=list(range(len(kept))), zero_division=0)
    return [{"stage": kept[i], "n": int(S[i]), "precision": round(float(P[i]), 3),
             "recall": round(float(R[i]), 3), "f1": round(float(F[i]), 3)} for i in range(len(kept))]


def s4_variance(tissue, pig_X):
    """PC1-PC5 R^2 wrt Stage (6-level), BioProject, and combined; top-5000 var genes."""
    stage = pig_X["Age_Days"].apply(t6.stage6)
    keep = stage.notna().values
    feat = pig_X.drop(columns=["Age_Days", "BioProject"]).loc[keep]
    stg = stage[keep].astype(str).values
    bp = pig_X["BioProject"].loc[keep].astype(str).values
    # top-5000 most-variable genes
    v = feat.values.astype(np.float64)
    idx = np.argsort(v.var(axis=0))[::-1][:5000]
    pcs = PCA(n_components=10, random_state=SEED).fit_transform(v[:, idx])

    def r2(pc, *factors):
        D = np.hstack([pd.get_dummies(f, drop_first=True).values for f in factors]).astype(float)
        if D.shape[1] == 0:
            return 0.0
        return float(LinearRegression().fit(D, pc).score(D, pc))

    rows = []
    for k in range(5):
        rows.append({"PC": f"PC{k+1}",
                     "r2_stage": round(r2(pcs[:, k], stg), 3),
                     "r2_bioproject": round(r2(pcs[:, k], bp), 3),
                     "r2_combined": round(r2(pcs[:, k], stg, bp), 3)})
    return {"rows": rows, "n_samples": int(keep.sum()), "n_projects": int(len(set(bp)))}


def main():
    out = {"S2": {}, "S4": {}}
    for tis in TISSUES:
        print(f"[{tis}] loading ...", flush=True)
        X = t6.load_pig_tissue(tis)
        out["S2"][tis] = s2_per_class(tis, X)
        out["S4"][tis] = s4_variance(tis, X)
        print(f"  S2 classes={len(out['S2'][tis]) if out['S2'][tis] else 0}; "
              f"S4 n={out['S4'][tis]['n_samples']} proj={out['S4'][tis]['n_projects']}")
    p = ROOT / "paper" / "figures" / "data" / "supp_s2_s4_6stage.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {p}")
    print("\n=== S2 (stratified per-class) ===")
    for tis in TISSUES:
        for r in (out["S2"][tis] or []):
            print(f"{tis:7} {r['stage']:16} n={r['n']:4} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
    print("\n=== S4 (variance) ===")
    for tis in TISSUES:
        for r in out["S4"][tis]["rows"]:
            print(f"{tis:7} {r['PC']} stage={r['r2_stage']:.3f} bp={r['r2_bioproject']:.3f} comb={r['r2_combined']:.3f}")


if __name__ == "__main__":
    main()
