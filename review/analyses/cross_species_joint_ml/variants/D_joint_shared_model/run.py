"""Variant D — Joint shared model.

Combine pig + human ortholog matrices per tissue into a single binary
LightGBM training matrix (Young vs Old) with a species indicator feature
and species-balancing sample weights. Train 10 seeds x 5-fold stratified
CV (stratified by species*label) and extract gain importance. The
stable@100 (>= 8/10 seeds) set IS the joint conserved feature set.
Only runs for tissues with human data (Muscle, Brain, Liver).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _utils as U  # noqa: E402

VARIANT = "D"
OUT_DIR = HERE
TISSUES_WITH_HUMAN = U.HUMAN_TISSUES
SEEDS = list(range(42, 52))
N_FOLDS = 5
TOP_K = 100

PARAMS_DEFAULT: Dict[str, object] = dict(
    objective="binary",
    metric="binary_logloss",
    num_leaves=15,  # midway between pig (31) and human (8) configs
    max_depth=-1,
    learning_rate=0.05,
    n_estimators=200,
    min_data_in_leaf=5,
    feature_fraction=0.7,
    bagging_fraction=0.7,
    bagging_freq=5,
    lambda_l1=0.1,
    lambda_l2=1.0,
    verbosity=-1,
)


def build_joint_matrix(tissue: str, ortho: pd.DataFrame
                       ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Return (X, y, species_arr) where:
      X: (n_pig_binary + n_human) x (n_orthologs + 1 species indicator)
      y: binary Young (0) / Old (1)
      species_arr: 0 = pig, 1 = human (matches X.index)
    """
    # ---- Pig binary
    Xp, yp, _ = U.load_pig_cached(tissue)
    pig_young = yp[yp.isin(U.PIG_YOUNG)].index
    pig_old = yp[yp.isin(U.PIG_OLD)].index
    pig_keep = list(pig_young) + list(pig_old)
    Xp = Xp.loc[pig_keep]
    yp_bin = pd.Series([0] * len(pig_young) + [1] * len(pig_old),
                       index=pig_keep, dtype=int)
    # ---- Human binary
    hd = U.load_human_cached(tissue)
    if hd is None:
        raise RuntimeError(f"No human data for {tissue}")
    Xh, yh, _ = hd
    # Pig matrix is indexed by pig_gene_id; human by human_gene_id.
    # Align both onto symbols (common space).
    pid_to_sym = dict(zip(ortho["pig_gene_id"], ortho["symbol"]))
    hid_to_sym = dict(zip(ortho["human_gene_id"], ortho["symbol"]))
    Xp_sym = Xp.copy()
    Xp_sym.columns = [pid_to_sym.get(c, None) for c in Xp_sym.columns]
    Xp_sym = Xp_sym.loc[:, [c for c in Xp_sym.columns if c is not None]]
    # collapse duplicate symbols if any (one-to-one orthologs, so unlikely)
    if Xp_sym.columns.duplicated().any():
        Xp_sym = Xp_sym.groupby(level=0, axis=1).mean()
    Xh_sym = Xh.copy()
    Xh_sym.columns = [hid_to_sym.get(c, None) for c in Xh_sym.columns]
    Xh_sym = Xh_sym.loc[:, [c for c in Xh_sym.columns if c is not None]]
    if Xh_sym.columns.duplicated().any():
        Xh_sym = Xh_sym.groupby(level=0, axis=1).mean()
    # Common symbols
    common = sorted(set(Xp_sym.columns) & set(Xh_sym.columns))
    Xp_sym = Xp_sym.loc[:, common]
    Xh_sym = Xh_sym.loc[:, common]
    # Concatenate (samples x genes)
    X = pd.concat([Xp_sym, Xh_sym], axis=0)
    species = np.array([0] * Xp_sym.shape[0] + [1] * Xh_sym.shape[0],
                       dtype=int)
    X["__species_indicator__"] = species
    y = pd.concat([yp_bin, yh.reindex(Xh.index).astype(int)], axis=0)
    y = y.reindex(X.index)
    return X, y, species


def per_species_weight(species: np.ndarray) -> np.ndarray:
    n_pig = int((species == 0).sum())
    n_human = int((species == 1).sum())
    total = n_pig + n_human
    w = np.where(species == 0, total / max(n_pig, 1), total / max(n_human, 1))
    return w.astype(float)


def train_seed(X: pd.DataFrame, y: pd.Series, species: np.ndarray,
               weights: np.ndarray, seed: int,
               params_override: Dict[str, object] | None = None
               ) -> Tuple[pd.Series, float]:
    """5-fold stratified-by-(species, label) CV. Return mean gain importance
    and held-out balanced accuracy."""
    # Compose strat groups
    grp = species * 2 + y.values
    counts = pd.Series(grp).value_counts()
    n_splits = min(N_FOLDS, int(counts.min()))
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_imps: List[pd.Series] = []
    fold_acc: List[float] = []
    base = {**PARAMS_DEFAULT, "random_state": seed}
    if params_override:
        base.update(params_override)
    base["objective"] = "binary"
    base["verbosity"] = -1
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    base["scale_pos_weight"] = (n_neg / n_pos) if n_pos > 0 else 1.0

    for fi, (tr, te) in enumerate(skf.split(X.values, grp)):
        # internal val split for early stopping
        tr_grp = grp[tr]
        try:
            # one example per group for val
            unique_g = list(set(tr_grp.tolist()))
            val_local = [int(np.where(tr_grp == g)[0][0]) for g in unique_g]
            val_idx = tr[val_local]
            tr_only = np.array([i for i in tr if i not in val_idx])
        except Exception:
            tr_only = np.array(tr)
            val_idx = None
        train_set = lgb.Dataset(
            X.iloc[tr_only].values,
            label=y.iloc[tr_only].values,
            weight=weights[tr_only],
        )
        callbacks = [lgb.log_evaluation(0)]
        if val_idx is not None and len(val_idx) >= 2:
            val_set = lgb.Dataset(
                X.iloc[val_idx].values,
                label=y.iloc[val_idx].values,
                weight=weights[val_idx],
                reference=train_set,
            )
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
        prob = model.predict(X.iloc[te].values,
                             num_iteration=model.best_iteration)
        pred = (prob >= 0.5).astype(int)
        # balanced acc
        y_te = y.iloc[te].values
        tp = ((pred == 1) & (y_te == 1)).sum()
        tn = ((pred == 0) & (y_te == 0)).sum()
        p = max((y_te == 1).sum(), 1)
        n = max((y_te == 0).sum(), 1)
        bal_acc = 0.5 * (tp / p + tn / n)
        fold_acc.append(float(bal_acc))
    df = pd.concat(fold_imps, axis=1).fillna(0.0)
    imp = df.mean(axis=1).sort_values(ascending=False)
    return imp, float(np.mean(fold_acc))


def run_tissue(tissue: str, ortho: pd.DataFrame, annot: dict,
               params_override: Dict[str, object] | None = None
               ) -> dict:
    t0 = time.time()
    X, y, species = build_joint_matrix(tissue, ortho)
    weights = per_species_weight(species)
    print(f"[{VARIANT}] {tissue}: joint X={X.shape}, pig={int((species==0).sum())}, "
          f"human={int((species==1).sum())}, y young={(y==0).sum()} "
          f"old={(y==1).sum()}", flush=True)
    U.log_row(VARIANT, tissue, "load", "n_joint", int(len(y)))

    seed_imps: Dict[int, pd.Series] = {}
    bal_accs: List[float] = []
    for seed in SEEDS:
        ts = time.time()
        imp, bal = train_seed(X, y, species, weights, seed, params_override)
        seed_imps[seed] = imp
        bal_accs.append(bal)
        nz = int((imp > 0).sum())
        print(f"  seed {seed}: nonzero={nz}, bal_acc={bal:.2f} "
              f"({time.time()-ts:.0f}s)", flush=True)
        U.log_row(VARIANT, tissue, "train", "bal_acc", round(bal, 4),
                  note=f"seed={seed}")

    sym_full = pd.concat(seed_imps.values(), axis=1).fillna(0.0)
    sym_full.columns = [f"seed_{s}" for s in seed_imps.keys()]
    sym_full.index.name = "symbol"
    # drop species indicator from saved importance
    if "__species_indicator__" in sym_full.index:
        sym_full = sym_full.drop("__species_indicator__")
    sym_full.to_parquet(OUT_DIR / f"seed_importance_symbol_{tissue}.parquet")

    # Stable set in symbol space
    stable_syms, tab = U.stable_set_from_seeds(sym_full, TOP_K)
    tab.to_csv(OUT_DIR / f"features_joint_{tissue}.csv", index=False)
    U.log_row(VARIANT, tissue, "stable", "n_stable_100", len(stable_syms))

    # Map symbols back to pig+human ortholog IDs and compute concordance
    pig_fc = U.pig_fold_change(tissue)
    human_fc = U.human_fold_change(tissue)
    sym_to_pid = dict(zip(ortho["symbol"], ortho["pig_gene_id"]))
    sym_to_hid = dict(zip(ortho["symbol"], ortho["human_gene_id"]))

    joint_pairs: List = []
    for s in stable_syms:
        pid = sym_to_pid.get(s)
        hid = sym_to_hid.get(s)
        if pid and hid:
            joint_pairs.append((pid, hid))
    conc = U.compute_concordance(joint_pairs, pig_fc, human_fc)

    rows = []
    for s in stable_syms:
        pid = sym_to_pid.get(s)
        hid = sym_to_hid.get(s)
        rows.append({
            "symbol": s,
            "pig_gene_id": pid,
            "human_gene_id": hid,
            "log2fc_pig": (float(pig_fc.get(pid, np.nan))
                           if pig_fc is not None and pid in pig_fc.index else None),
            "log2fc_human": (float(human_fc.get(hid, np.nan))
                             if human_fc is not None
                             and hid in human_fc.index else None),
            "category": U.classify_symbol(s, annot),
        })
    joint_df = pd.DataFrame(rows)
    joint_df.to_csv(OUT_DIR / f"joint_features_{tissue}.csv", index=False)

    expected_n = U.expected_n_by_chance(TOP_K, TOP_K,
                                        n_orthologs=sym_full.shape[0])

    result = {
        "variant": VARIANT,
        "tissue": tissue,
        "n_pig_samples": int((species == 0).sum()),
        "n_human_samples": int((species == 1).sum()),
        "n_orthologs_used": int(sym_full.shape[0]),
        "balanced_accuracy_per_seed": [round(b, 4) for b in bal_accs],
        "mean_balanced_accuracy": float(np.mean(bal_accs)),
        "joint_set_n": len(stable_syms),
        "expected_by_chance": expected_n,
        "ratio_observed_to_chance":
            (len(stable_syms) / expected_n) if expected_n else None,
        "concordance": conc,
        "joint_features": rows,
        "annotation_summary": U.annotation_summary(stable_syms, annot),
        "params": params_override or PARAMS_DEFAULT,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / f"results_{tissue}.json").write_text(
        json.dumps(result, indent=2, default=str))
    U.log_row(VARIANT, tissue, "joint", "joint_n", len(stable_syms))
    U.log_row(VARIANT, tissue, "joint", "pearson_r", conc.get("pearson_r"))
    print(f"  ==> joint_n={len(stable_syms)}, r={conc.get('pearson_r')}, "
          f"{time.time()-t0:.0f}s\n", flush=True)
    return result


def main():
    ortho = U.load_ortho()
    annot = U.load_annot_cache()
    summary = {}
    # Adaptive sweep params for fallback
    sweeps = [
        None,  # default
        dict(num_leaves=8, min_data_in_leaf=3),
        dict(num_leaves=23, min_data_in_leaf=3),
    ]
    for tissue in TISSUES_WITH_HUMAN:
        prior = OUT_DIR / f"results_{tissue}.json"
        if prior.exists():
            try:
                pj = json.loads(prior.read_text())
                if "error" not in pj and pj.get("joint_set_n", 0) > 0:
                    print(f"[{VARIANT}] SKIP {tissue}: prior result exists")
                    summary[tissue] = pj
                    continue
            except Exception:  # noqa: BLE001
                pass
        success = False
        for sw_idx, sw in enumerate(sweeps):
            try:
                r = run_tissue(tissue, ortho, annot, params_override=sw)
                summary[tissue] = r
                if r.get("joint_set_n", 0) > 0:
                    success = True
                    break
                print(f"[{VARIANT}] {tissue} sweep {sw_idx} produced joint_n=0, "
                      f"trying next sweep", flush=True)
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                U.log_row(VARIANT, tissue, "crash", "exception",
                          str(exc).replace("\t", " ")[:200],
                          note=f"sweep={sw_idx}")
                summary[tissue] = {"variant": VARIANT, "tissue": tissue,
                                   "error": str(exc)}
                break
        if not success:
            print(f"[{VARIANT}] {tissue}: all sweeps exhausted", flush=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print("\nDone.")


if __name__ == "__main__":
    main()
