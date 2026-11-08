#!/usr/bin/env python3
"""Compute ONE (model, tissue, cv, seed, fold) — fold-level split for max parallelism.

Replicates benchmark_one_model.run_cv()'s splitter construction + skip logic
EXACTLY for a single fold, reusing prepare_tissue() and make_model() verbatim.
Merged across folds (merge_split.py) the result is bit-identical to the monolith
-- proven by validate_diff.py before any number is trusted. Records the per-fold
fit time so the merge can report correct single-fit / total-CPU timing.

Pinned to 1 thread (set in the sbatch) => deterministic + correct 1-core cost.
Args: <model> <tissue> <cv> <seed> <fold> [outdir]
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("BENCH_SCHEME", "6stage")
WS = Path(os.environ.get("BENCH_WS", "/storage/gge/Tian/pig_ml_benchmark"))
sys.path.insert(0, str(WS / "code"))
import benchmark_one_model as B                       # noqa: E402
from machine_learning.data_processing.data_loader import DataLoader  # noqa: E402
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold  # noqa: E402
from sklearn.metrics import balanced_accuracy_score   # noqa: E402

name, tissue, cv, seed, fold = (sys.argv[1], sys.argv[2], sys.argv[3],
                                int(sys.argv[4]), int(sys.argv[5]))
outdir = Path(sys.argv[6]) if len(sys.argv) > 6 else (WS / "results_split6")
outdir.mkdir(parents=True, exist_ok=True)

dl = DataLoader(data_dir=B.DATA_DIR, metadata_path=B.META)
dl.load_metadata()
X, yi, gk, tinfo = B.prepare_tissue(dl, tissue)        # identical data prep as monolith

# --- replicate run_cv's splitter construction EXACTLY ---
if cv == "stratified":
    sk = StratifiedKFold(n_splits=B.NFOLDS, shuffle=True, random_state=seed)
    splits = list(sk.split(X, yi))
else:
    ng = len(np.unique(gk)); nf = min(B.NFOLDS, ng)
    splits = (None if nf < 2 else
              list(StratifiedGroupKFold(n_splits=nf, shuffle=True,
                                        random_state=seed).split(X, yi, groups=gk)))

status, ba, ft = "ok", None, None
if splits is None:
    status = "cv_not_identifiable"
elif fold >= len(splits):
    status = "fold_absent"                              # grouped nf<NFOLDS
else:
    tr, te = splits[fold]
    if len(np.unique(yi[tr])) < 2:
        status = "skipped_lt2classes"                   # same skip as run_cv
    else:
        cols = [f"g{i}" for i in range(X.shape[1])]
        m = B.make_model(name, seed)                    # SAME seed all folds (as run_cv)
        t0 = time.perf_counter()
        m.fit(pd.DataFrame(X[tr], columns=cols), yi[tr])
        ft = float(time.perf_counter() - t0)
        pred = np.asarray(m.predict(pd.DataFrame(X[te], columns=cols))).astype(int)
        ba = float(balanced_accuracy_score(yi[te], pred))

out = {"model": name, "tissue": tissue, "cv": cv, "seed": seed, "fold": fold,
       "ba": ba, "ft": ft, "status": status,
       "n": tinfo["n"], "n_classes": tinfo["n_classes"],
       "chance": tinfo["chance"], "kept": tinfo["kept"]}
(outdir / f"{name}__{tissue}__{cv}__s{seed}__f{fold}.json").write_text(json.dumps(out))
print(f"{name} {tissue} {cv} s{seed} f{fold}: {status} ba={ba} ft={ft}", flush=True)
