#!/usr/bin/env python3
"""Merge fold-level partials -> monolith per-model JSON (two-level aggregation).

Replicates benchmark_one_model EXACTLY:
  run_cv : seed_ba = mean over OK folds ; seed_ft = median over OK folds
  main   : BA_mean = round(mean(seed_ba),4) ; BA_sd = round(std,4) ;
           fit_time_s = round(median(seed_ft),4) ; per_seed = [round(b,4)]
So merged BA fields are identical to a monolith run (validated to the digit).

Also records CORRECT timing for the efficiency table:
  fit_time_s     : median single-fit (same metric as the 18 done models)
  muscle_fit_s   : median Muscle single-fit
  total_fit_cpu_s: SUM of every fold's fit time = total model compute (1-core =>
                   CPU-seconds; parallelism-invariant; for single-threaded models
                   this equals the multi-core cost).

Args: <comma-models> <split_dir> <out_dir>
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

models = sys.argv[1].split(",")
SPLIT = Path(sys.argv[2]); OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
TISSUES = ["Muscle", "Brain", "Liver", "Blood", "Lung"]
SEEDS = [42, 43, 44, 45, 46]
NFOLDS = 5

for model in models:
    result = {"model": model, "scheme": "6stage", "seeds": SEEDS, "min_per_stage": 10,
              "n_components_pls": 10, "merged_from_split": "fold", "tissues": {}}
    all_fit_times = []          # every OK fold's fit time, across the whole model
    muscle_fits = []
    for tissue in TISSUES:
        base = None; tres = {}
        for cv in ["grouped", "stratified"]:
            seed_ba, seed_ft = [], []
            for s in SEEDS:
                fold_bas, fold_fts = [], []
                for fdx in range(NFOLDS):
                    p = SPLIT / f"{model}__{tissue}__{cv}__s{s}__f{fdx}.json"
                    if not p.exists():
                        continue
                    d = json.loads(p.read_text())
                    if base is None:
                        base = {"n": d["n"], "n_classes": d["n_classes"],
                                "chance": d["chance"], "kept": d["kept"]}
                    if d["status"] == "ok" and d["ba"] is not None:
                        fold_bas.append(d["ba"]); fold_fts.append(d["ft"])
                        all_fit_times.append(d["ft"])
                        if tissue == "Muscle":
                            muscle_fits.append(d["ft"])
                if fold_bas:                                   # == run_cv: skip empty
                    seed_ba.append(float(np.mean(fold_bas)))
                    seed_ft.append(float(np.median(fold_fts)))
            if seed_ba:
                tres[cv] = {"BA_mean": round(float(np.mean(seed_ba)), 4),
                            "BA_sd": round(float(np.std(seed_ba)), 4),
                            "n_seeds": len(seed_ba),
                            "fit_time_s": round(float(np.median(seed_ft)), 4),
                            "per_seed": [round(b, 4) for b in seed_ba]}
            else:
                tres[cv] = {"BA_mean": None, "note": "not identifiable / failed"}
        result["tissues"][tissue] = ({**base, **tres} if base else {"error": "no partials"})
    result["timing"] = {
        "single_fit_s_median": round(float(np.median(all_fit_times)), 3) if all_fit_times else None,
        "muscle_fit_s_median": round(float(np.median(muscle_fits)), 1) if muscle_fits else None,
        "total_fit_cpu_s": round(float(np.sum(all_fit_times)), 1) if all_fit_times else None,
        "n_fits": len(all_fit_times),
        "threads": 1,
    }
    (OUT / f"{model}.json").write_text(json.dumps(result, indent=2))
    tm = result["timing"]
    tot = tm.get("total_fit_cpu_s")
    cpuh = f"{tot/3600:.2f}" if tot else "n/a"
    print(f"merged {model}: total_fit_cpu={tot}s ({cpuh} CPU-h), "
          f"muscle_fit~{tm.get('muscle_fit_s_median')}s, n_fits={tm['n_fits']} -> {OUT/f'{model}.json'}")
