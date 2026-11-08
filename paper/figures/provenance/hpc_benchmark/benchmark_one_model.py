#!/usr/bin/env python3
"""Within-pig model comparison under grouped-CV (one model per invocation).

Runs ONE model across the 5-tissue panel under TWO cross-validation regimes,
using the EXACT model classes from the paper's machine_learning package:

  grouped     StratifiedGroupKFold on BioProject  (leakage-controlled; the
              HONEST deployment estimate; only Muscle is truly identifiable)
  stratified  StratifiedKFold                      (batch-inclusive; inflated)

Staging: current 5-stage scheme (Infant 0-28 | Early childhood 29-90 |
Pre-pubertal 91-165 | Post-pubertal 166-540 | Adult >540), drop-small
(>=10 samples/stage), integer-coded in ordinal order. Features: log2(TPM+1)
over all expressed genes. Multi-seed (5 seeds); report mean +/- SD over seeds.

Each (tissue) result is written incrementally so a SLURM TIMEOUT preserves
completed tissues. Peak RSS recorded via getrusage at the end (also captured
per-task by SLURM MaxRSS). Defaults only -- no per-fold hyperparameter tuning.
Real data only.
"""
from __future__ import annotations
import sys, os, json, time, resource, warnings, inspect, traceback
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
WS = Path(os.environ.get("BENCH_WS", "/storage/gge/Tian/pig_ml_benchmark"))
sys.path.insert(0, str(WS / "code"))

from machine_learning.data_processing.data_loader import DataLoader
from machine_learning.model_training.model_factory import get_model_class, MODEL_FIXED_PARAMS
from machine_learning.model_training.models import _OrdinalBase
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

DATA_DIR = WS / "data" / "pigGTEx"
META = WS / "data" / "PigGTEx_v0.MetaTable.csv"
RESULTS = WS / os.environ.get("BENCH_RESULTS", "results")
RESULTS.mkdir(parents=True, exist_ok=True)

TISSUES = os.environ.get("BENCH_TISSUES", "Muscle,Brain,Liver,Blood,Lung").split(",")
SEEDS = [int(s) for s in os.environ.get("BENCH_SEEDS", "42,43,44,45,46").split(",")]
NFOLDS = 5
MIN_PER_STAGE = 10
N_COMP_PLS = 10
SCHEME = os.environ.get("BENCH_SCHEME", "5stage")
ORDER5 = ["Infant", "Early childhood", "Pre-pubertal", "Post-pubertal", "Adult"]
# 6-stage perinatal split: Infant 0-28 -> Neonatal 0-3 + Suckling 4-28
ORDER6 = ["Neonatal", "Suckling", "Juvenile", "Pre-pubertal", "Post-pubertal", "Adult"]


def stage5(d):
    if pd.isna(d): return None
    if d <= 28:  return "Infant"
    if d <= 90:  return "Early childhood"
    if d <= 165: return "Pre-pubertal"
    if d <= 540: return "Post-pubertal"
    return "Adult"


def stage6(d):
    if pd.isna(d): return None
    if d <= 3:   return "Neonatal"
    if d <= 28:  return "Suckling"
    if d <= 90:  return "Juvenile"
    if d <= 165: return "Pre-pubertal"
    if d <= 540: return "Post-pubertal"
    return "Adult"


STAGE_FN = stage6 if SCHEME == "6stage" else stage5
ORDER = ORDER6 if SCHEME == "6stage" else ORDER5


# ---- "original PLS": classic PLS-DA (one-hot Y, argmax). Contrasts with the
#      paper's OrdinalPLS (regression on the integer stage index + softmax). ----
class PLSDAOriginal(_OrdinalBase):
    def __init__(self, n_components: int = 10, seed: int = 42):
        self.n_components = n_components
        self.seed = seed
        self.model_ = None
        self._mean = None
        self._scale = None

    def fit(self, X, y):
        from sklearn.cross_decomposition import PLSRegression
        X_df, y_enc = self._fit_setup(X, y)
        Xv = np.ascontiguousarray(X_df.values, dtype=np.float64)
        self._mean = Xv.mean(axis=0)
        self._scale = Xv.std(axis=0)
        self._scale[self._scale == 0] = 1.0
        Xz = (Xv - self._mean) / self._scale
        Y = np.zeros((len(y_enc), self.n_classes_), dtype=float)
        Y[np.arange(len(y_enc)), y_enc] = 1.0
        n_comp = max(1, min(self.n_components, Xz.shape[1], Xz.shape[0] - 1))
        self.model_ = PLSRegression(n_components=n_comp, scale=False)
        self.model_.fit(Xz, Y)
        return self

    def predict_proba(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        Xv = np.ascontiguousarray(X_df.values, dtype=np.float64)
        Xz = (Xv - self._mean) / self._scale
        S = np.asarray(self.model_.predict(Xz))
        if S.ndim == 1:
            S = S[:, None]
        S = S - S.max(axis=1, keepdims=True)
        ex = np.exp(S)
        return ex / ex.sum(axis=1, keepdims=True)


def make_model(name, seed):
    if name == "chance":
        from sklearn.dummy import DummyClassifier
        return DummyClassifier(strategy="stratified", random_state=seed)
    if name == "plsda":
        return PLSDAOriginal(n_components=N_COMP_PLS, seed=seed)
    cls = get_model_class(name)
    fixed = dict(MODEL_FIXED_PARAMS.get(name, {}))
    sig = inspect.signature(cls.__init__)
    kwargs = {k: v for k, v in fixed.items() if k in sig.parameters}
    if "seed" in sig.parameters:
        kwargs["seed"] = seed
    return cls(**kwargs)


def run_cv(X, yi, groups, cv_kind, seed, name):
    if cv_kind == "stratified":
        sk = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=seed)
        it = sk.split(X, yi)
    else:
        ng = len(np.unique(groups))
        nf = min(NFOLDS, ng)
        if nf < 2:
            return None, None
        sk = StratifiedGroupKFold(n_splits=nf, shuffle=True, random_state=seed)
        it = sk.split(X, yi, groups=groups)
    cols = [f"g{i}" for i in range(X.shape[1])]  # string feature names (LightGBM rejects int names)
    bas, fts = [], []
    for tr, te in it:
        if len(np.unique(yi[tr])) < 2:
            continue
        m = make_model(name, seed)
        t0 = time.perf_counter()
        m.fit(pd.DataFrame(X[tr], columns=cols), yi[tr])
        fts.append(time.perf_counter() - t0)
        pred = np.asarray(m.predict(pd.DataFrame(X[te], columns=cols))).astype(int)
        bas.append(balanced_accuracy_score(yi[te], pred))
    if not bas:
        return None, None
    return float(np.mean(bas)), float(np.median(fts))


def peak_rss_mb():
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(kb / 1024.0, 1)  # Linux ru_maxrss is KB


def prepare_tissue(dl, tissue):
    """Load + stage + drop-small for ONE tissue. Deterministic; no global state.
    Returns (X, yi, gk, tinfo_base). Used IDENTICALLY by the monolith (main) and
    the per-seed split harness (benchmark_split.py) so numbers are bit-identical.
    """
    expr, meta = dl.load_expression(tissue)             # genes x samples
    X = np.log2(expr.T.values.astype(np.float64) + 1.0)
    age = meta["Age_Days"] if "Age_Days" in meta else meta["Age"]
    age = pd.to_numeric(age, errors="coerce").reset_index(drop=True)
    bp = meta["BioProject"].astype(str).reset_index(drop=True)
    y = age.map(STAGE_FN)
    keep_age = age.notna().values
    X = X[keep_age]; y = y[keep_age].reset_index(drop=True); bp = bp[keep_age].reset_index(drop=True)
    counts = y.value_counts()
    kept = [s for s in ORDER if counts.get(s, 0) >= MIN_PER_STAGE]
    mask = y.isin(kept).values
    X = X[mask]
    code = {s: i for i, s in enumerate(kept)}
    yi = np.array([code[v] for v in y[mask].values])
    gk = bp[mask].values
    del expr
    tinfo = {"n": int(mask.sum()), "n_bioprojects": int(pd.unique(gk).size),
             "n_classes": len(kept), "kept": {s: int((y[mask] == s).sum()) for s in kept},
             "chance": round(1.0 / len(kept), 4)}
    return X, yi, gk, tinfo


def main():
    name = sys.argv[1]
    out_path = RESULTS / f"{name}.json"
    result = {"model": name, "scheme": SCHEME, "seeds": SEEDS, "min_per_stage": MIN_PER_STAGE,
              "n_components_pls": N_COMP_PLS, "tissues": {}}

    dl = DataLoader(data_dir=DATA_DIR, metadata_path=META)
    dl.load_metadata()

    for tissue in TISSUES:
        t_start = time.perf_counter()
        try:
            X, yi, gk, tinfo = prepare_tissue(dl, tissue)
            for cv in ["grouped", "stratified"]:
                seed_ba, seed_ft = [], []
                for seed in SEEDS:
                    try:
                        ba, ft = run_cv(X, yi, gk, cv, seed, name)
                    except Exception as e:
                        ba, ft = None, None
                        tinfo.setdefault("errors", []).append(f"{cv}/seed{seed}: {type(e).__name__}: {e}")
                    if ba is not None:
                        seed_ba.append(ba); seed_ft.append(ft)
                if seed_ba:
                    tinfo[cv] = {"BA_mean": round(float(np.mean(seed_ba)), 4),
                                 "BA_sd": round(float(np.std(seed_ba)), 4),
                                 "n_seeds": len(seed_ba),
                                 "fit_time_s": round(float(np.median(seed_ft)), 4),
                                 "per_seed": [round(b, 4) for b in seed_ba]}
                else:
                    tinfo[cv] = {"BA_mean": None, "note": "not identifiable / failed"}
            tinfo["wall_s"] = round(time.perf_counter() - t_start, 1)
            result["tissues"][tissue] = tinfo
            print(f"[{name}] {tissue}: grouped={tinfo['grouped'].get('BA_mean')} "
                  f"stratified={tinfo['stratified'].get('BA_mean')} "
                  f"({tinfo['wall_s']}s)", flush=True)
        except Exception as e:
            result["tissues"][tissue] = {"error": f"{type(e).__name__}: {e}",
                                         "trace": traceback.format_exc()[-800:]}
            print(f"[{name}] {tissue}: ERROR {e}", flush=True)
        result["peak_rss_mb"] = peak_rss_mb()
        out_path.write_text(json.dumps(result, indent=2))  # incremental save

    result["peak_rss_mb"] = peak_rss_mb()
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[{name}] DONE peak_rss={result['peak_rss_mb']}MB -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
