#!/usr/bin/env python3
"""
Marker-free, leave-one-cohort-out (LOCO) developmental-boundary criterion.

Reconstructs the analysis described in paper.tex (Discussion): a genome-wide
"young-versus-old" contrast fit on TRAINING cohorts only (groups = BioProject),
swept across candidate age cut-points, scored by held-out ROC-AUC. Where the
held-out AUC peaks identifies where the transcriptome separates most cleanly,
i.e. where a developmental boundary belongs. This is "marker-free" (whole
transcriptome, no curated marker list) and leakage-controlled (no BioProject in
both train and held-out folds).

Real data only -- no fabricated numbers. Reads data/pigGTEx/<Tissue>.expr_tpm.txt.gz
+ data/PigGTEx_v0.MetaTable.csv. Output: review/analyses/results/boundary_auc_loco.json
"""
from __future__ import annotations
import gzip, json, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "pigGTEx"
META = ROOT / "data" / "PigGTEx_v0.MetaTable.csv"
OUT = ROOT / "review" / "analyses" / "results" / "boundary_auc_loco.json"


def age_days(s):
    if pd.isna(s):
        return None
    s = str(s).lower().strip()
    m = re.match(r"(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years)", s)
    if not m:
        return None
    v = float(m.group(1)); u = m.group(2)
    return v * (1 if "day" in u else 7 if "week" in u else 30 if "month" in u else 365)


def load_meta():
    md = pd.read_csv(META)
    md.columns = [c.strip().replace(" ", "_") for c in md.columns]
    md = md.rename(columns={"BioSample": "Sample_ID", "Tissue_class": "Tissue"}).set_index("Sample_ID")
    md["d"] = md["Age"].apply(age_days)
    return md


def load_tpm(t):
    p = DATA / f"{t}.expr_tpm.txt.gz"
    with gzip.open(p, "rt") as f:
        return np.log2(pd.read_csv(f, sep="\t", index_col=0) + 1.0)


def loco_auc_for_cut(X, ages, groups, cut):
    """Pooled held-out AUC of a genome-wide young(<cut)/old(>=cut) contrast,
    leave-one-BioProject-out, train-fold z-scoring + contrast direction."""
    y = (ages >= cut).astype(int).values
    if y.sum() < 3 or (len(y) - y.sum()) < 3:
        return None, int(y.sum()), int(len(y) - y.sum())
    oof_score = np.full(len(y), np.nan)
    uniq = pd.unique(groups)
    for g in uniq:
        te = (groups == g).values
        tr = ~te
        ytr = y[tr]
        if ytr.sum() < 2 or (tr.sum() - ytr.sum()) < 2:
            continue  # training fold needs both classes to define a contrast
        Xtr = X[tr]; Xte = X[te]
        mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
        Ztr = (Xtr - mu) / sd
        Zte = (Xte - mu) / sd
        w = Ztr[ytr == 1].mean(0) - Ztr[ytr == 0].mean(0)   # old-minus-young contrast
        nw = np.linalg.norm(w)
        if nw == 0:
            continue
        w = w / nw
        oof_score[te] = Zte @ w
    mask = ~np.isnan(oof_score)
    if mask.sum() < 6 or len(np.unique(y[mask])) < 2:
        return None, int(y.sum()), int(len(y) - y.sum())
    auc = roc_auc_score(y[mask], oof_score[mask])
    return float(auc), int(y.sum()), int(len(y) - y.sum())


def run_tissue(t, cuts):
    md = load_meta()
    e = load_tpm(t)
    common = [c for c in e.columns if c in md.index]
    sub = md.loc[common]
    keep = sub["d"].notna() & sub["BioProject"].notna()
    common = [c for c, k in zip(common, keep) if k]
    sub = md.loc[common]
    e = e[common]
    # express filter: genes detected in >20% of samples (matches velocity filter)
    expr = e[(e > 1).mean(axis=1) > 0.2]
    X = expr.T.values.astype(float)            # samples x genes
    ages = sub["d"]; groups = sub["BioProject"]
    rows = []
    for c in cuts:
        auc, n_old, n_young = loco_auc_for_cut(X, ages, groups, c)
        rows.append(dict(cut_d=c, auc=auc, n_young=n_young, n_old=n_old))
    return dict(tissue=t, n_samples=int(X.shape[0]), n_genes=int(X.shape[1]),
                n_bioprojects=int(groups.nunique()),
                age_min=float(ages.min()), age_max=float(ages.max()), sweep=rows)


def main():
    cuts = [3, 7, 14, 21, 28, 45, 60, 90, 110, 130, 165, 220, 300, 400, 540]
    res = {}
    for t in ["Muscle", "Liver"]:
        print(f"=== {t} ===", flush=True)
        r = run_tissue(t, cuts)
        res[t] = r
        for row in r["sweep"]:
            a = row["auc"]
            print(f"  cut {row['cut_d']:>4}d  AUC {a if a is None else round(a,4)}"
                  f"  (young={row['n_young']}, old={row['n_old']})", flush=True)
        valid = [row for row in r["sweep"] if row["auc"] is not None]
        if valid:
            best = max(valid, key=lambda x: x["auc"])
            print(f"  -> peak AUC {round(best['auc'],4)} at cut {best['cut_d']}d", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
