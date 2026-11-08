"""Variant B — Binary-pig retraining.

Retrain pig as binary LightGBM Young (Infant + Early childhood) vs Old
(Post-pubertal + Adult). Drop Pre-pubertal as transitional. Same 10 seeds
x 5-fold CV across all 5 pig tissues. Reuse existing human binary seed
importance parquets to compute strict joint set for Muscle/Brain/Liver.
For Lung and Blood we report pig-only stable set (no human matrix).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _utils as U  # noqa: E402

VARIANT = "B"
OUT_DIR = HERE
TISSUES = U.TISSUES
SEEDS = list(range(42, 52))
N_FOLDS = 5
TOP_K = 100

PIG_PARAMS: Dict[str, object] = dict(
    objective="binary",
    metric="binary_logloss",
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.05,
    n_estimators=200,
    min_data_in_leaf=10,
    feature_fraction=0.7,
    bagging_fraction=0.7,
    bagging_freq=5,
    lambda_l1=0.1,
    lambda_l2=1.0,
    verbosity=-1,
)


def filter_binary(X: pd.DataFrame, y: pd.Series):
    """Drop Pre-pubertal; map Young=0, Old=1."""
    young = y[y.isin(U.PIG_YOUNG)].index
    old = y[y.isin(U.PIG_OLD)].index
    keep = list(young) + list(old)
    Xb = X.loc[keep]
    yb = pd.Series([0] * len(young) + [1] * len(old),
                   index=list(young) + list(old), dtype=int)
    return Xb, yb


def train_seed(X: pd.DataFrame, y: pd.Series, seed: int) -> pd.Series:
    counts = y.value_counts()
    n_splits = min(N_FOLDS, int(counts.min()))
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_imps: List[pd.Series] = []
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    base = {**PIG_PARAMS, "random_state": seed,
            "scale_pos_weight": (n_neg / n_pos) if n_pos > 0 else 1.0}
    for fi, (tr, te) in enumerate(skf.split(X.values, y.values)):
        # internal val: 1 pos + 1 neg from training fold for early stopping
        tr_y = y.iloc[tr].values
        pos_idx = [tr[i] for i in range(len(tr)) if tr_y[i] == 1][:1]
        neg_idx = [tr[i] for i in range(len(tr)) if tr_y[i] == 0][:1]
        val_idx = np.array(pos_idx + neg_idx)
        tr_only = np.array([i for i in tr if i not in val_idx])
        if len(val_idx) < 2 or len(tr_only) < 4:
            tr_only = np.array(tr)
            val_idx = None
        train_set = lgb.Dataset(X.iloc[tr_only].values,
                                label=y.iloc[tr_only].values)
        callbacks = [lgb.log_evaluation(0)]
        if val_idx is not None:
            val_set = lgb.Dataset(X.iloc[val_idx].values,
                                  label=y.iloc[val_idx].values,
                                  reference=train_set)
            callbacks.insert(0, lgb.early_stopping(20, verbose=False))
            model = lgb.train(base, train_set,
                              num_boost_round=int(base.get("n_estimators", 200)),
                              valid_sets=[val_set], callbacks=callbacks)
        else:
            model = lgb.train(base, train_set,
                              num_boost_round=int(base.get("n_estimators", 200)),
                              callbacks=callbacks)
        gains = model.feature_importance(importance_type="gain")
        imp = pd.Series(gains, index=X.columns[:len(gains)])
        fold_imps.append(imp)
    df = pd.concat(fold_imps, axis=1).fillna(0.0)
    return df.mean(axis=1).sort_values(ascending=False)


def run_tissue(tissue: str, ortho: pd.DataFrame, annot: dict) -> dict:
    t0 = time.time()
    Xp, yp, raw_p = U.load_pig_cached(tissue)
    Xpb, ypb = filter_binary(Xp, yp)
    print(f"[{VARIANT}] {tissue}: binary pig n={len(ypb)} "
          f"young={(ypb==0).sum()} old={(ypb==1).sum()}", flush=True)
    U.log_row(VARIANT, tissue, "load", "n_binary", len(ypb))
    if (ypb == 0).sum() < 4 or (ypb == 1).sum() < 4:
        return {"variant": VARIANT, "tissue": tissue,
                "error": "too few samples for binary"}

    seed_imps: Dict[int, pd.Series] = {}
    for seed in SEEDS:
        ts = time.time()
        imp = train_seed(Xpb, ypb, seed)
        seed_imps[seed] = imp
        nz = int((imp > 0).sum())
        print(f"  seed {seed}: nonzero={nz} top-3={list(imp.head(3).index)} "
              f"({time.time()-ts:.0f}s)", flush=True)
        U.log_row(VARIANT, tissue, "train", "n_nonzero", nz,
                  note=f"seed={seed}")

    pig_full = pd.concat(seed_imps.values(), axis=1).fillna(0.0)
    pig_full.columns = [f"seed_{s}" for s in seed_imps.keys()]
    pig_full.index.name = "pig_gene_id"
    pig_full.to_parquet(OUT_DIR / f"seed_importance_pig_{tissue}.parquet")

    pig_stable, pig_tab = U.stable_set_from_seeds(pig_full, TOP_K)
    pig_tab.to_csv(OUT_DIR / f"features_pig_{tissue}.csv", index=False)
    U.log_row(VARIANT, tissue, "pig_stable", "n_stable_100", len(pig_stable))

    # Concordance scaffolding
    pig_fc = U.pig_fold_change(tissue)
    human_fc = U.human_fold_change(tissue)
    human_seed = U.load_human_seed_imp(tissue)
    pid_to_hid = dict(zip(ortho["pig_gene_id"], ortho["human_gene_id"]))
    pig_to_sym = dict(zip(ortho["pig_gene_id"], ortho["symbol"]))
    human_to_sym = dict(zip(ortho["human_gene_id"], ortho["symbol"]))

    joint_pairs: List = []
    human_stable: List[str] = []
    expected_n = None
    if human_seed is not None:
        human_stable, _ = U.stable_set_from_seeds(human_seed, TOP_K)
        human_set = set(human_stable)
        for pid in pig_stable:
            hid = pid_to_hid.get(pid)
            if hid and hid in human_set:
                joint_pairs.append((pid, hid))
        expected_n = U.expected_n_by_chance(
            len(pig_stable), len(human_stable), n_orthologs=pig_full.shape[0])

    conc = U.compute_concordance(joint_pairs, pig_fc, human_fc)

    rows = []
    for pid, hid in joint_pairs:
        sym = pig_to_sym.get(pid) or human_to_sym.get(hid) or ""
        rows.append({
            "symbol": sym,
            "pig_gene_id": pid,
            "human_gene_id": hid,
            "log2fc_pig": float(pig_fc.get(pid, np.nan))
                          if pig_fc is not None else None,
            "log2fc_human": float(human_fc.get(hid, np.nan))
                            if human_fc is not None else None,
            "category": U.classify_symbol(sym, annot),
        })
    joint_df = pd.DataFrame(rows)
    joint_df.to_csv(OUT_DIR / f"joint_features_{tissue}.csv", index=False)

    annot_summary = U.annotation_summary(
        [r["symbol"] for r in rows if r["symbol"]], annot)

    result = {
        "variant": VARIANT,
        "tissue": tissue,
        "n_orthologs_used": int(pig_full.shape[0]),
        "n_pig_samples_binary": int(len(ypb)),
        "n_pig_young": int((ypb == 0).sum()),
        "n_pig_old": int((ypb == 1).sum()),
        "n_pig_seeds": len(SEEDS),
        "pig_stable_n": len(pig_stable),
        "human_available": human_seed is not None,
        "human_stable_n": len(human_stable),
        "joint_set_n": len(joint_pairs),
        "expected_by_chance": expected_n,
        "ratio_observed_to_chance":
            (len(joint_pairs) / expected_n) if expected_n else None,
        "concordance": conc,
        "joint_features": rows,
        "annotation_summary": annot_summary,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / f"results_{tissue}.json").write_text(
        json.dumps(result, indent=2, default=str))
    U.log_row(VARIANT, tissue, "joint", "joint_n", len(joint_pairs))
    U.log_row(VARIANT, tissue, "joint", "pearson_r", conc.get("pearson_r"))
    print(f"  ==> joint_n={len(joint_pairs)}, r={conc.get('pearson_r')}, "
          f"{time.time()-t0:.0f}s\n", flush=True)
    return result


def main():
    ortho = U.load_ortho()
    annot = U.load_annot_cache()
    summary = {}
    for tissue in TISSUES:
        prior = OUT_DIR / f"results_{tissue}.json"
        if prior.exists():
            try:
                pj = json.loads(prior.read_text())
                if "error" not in pj:
                    print(f"[{VARIANT}] SKIP {tissue}: prior result exists")
                    summary[tissue] = pj
                    continue
            except Exception:  # noqa: BLE001
                pass
        try:
            r = run_tissue(tissue, ortho, annot)
            summary[tissue] = r
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            U.log_row(VARIANT, tissue, "crash", "exception",
                      str(exc).replace("\t", " ")[:200])
            summary[tissue] = {"variant": VARIANT, "tissue": tissue,
                               "error": str(exc)}
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print("\nDone.")


if __name__ == "__main__":
    main()
