#!/usr/bin/env python3
"""Loop B — ML framework sweep (LOCAL).

Runs every (framework x ML-eligible tissue) cell of the developmental-stage
classifier through the EXISTING nested-CV pipeline (machine_learning/run_pipeline
via autoresearch/run_rvalue), collects balanced accuracy, and picks the winning
framework: highest mean BA across tissues with NO tissue below 0.80.

Frameworks and the eligible-tissue panel are the published ML set. Each cell is
an isolated subprocess (a crash in one cell does not abort the sweep). Threads
per cell are capped so concurrent cells don't oversubscribe the CPU.

Outputs:
  autoresearch/ml_sweep_simple/cells.tsv     — one row per (framework, tissue)
  autoresearch/ml_sweep_simple/winner.json   — winning framework + per-tissue BA
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "autoresearch" / "ml_sweep_simple"
MODEL_OUT = ROOT / "machine_learning" / "model_outputs"

FRAMEWORKS = ["ordinal_lgb", "ridge_continuous", "random_forest", "elastic_net_lr", "multiclass_lgb"]
PANEL = ["Muscle", "Brain", "Liver", "Blood", "Lung"]   # ML-eligible (samples-per-class)

N_TRIALS = 5
N_CV = 5
N_INNER = 3
MAX_WORKERS = 4          # concurrent cells
THREADS_PER_CELL = "2"   # OMP/MKL threads per subprocess


def run_cell(fw: str, tissue: str) -> dict:
    suffix = f"_loopB_{fw}"
    out_json = MODEL_OUT / f"{tissue}{suffix}_results.json"
    log = OUTDIR / f"{tissue}_{fw}.log"
    cmd = [
        sys.executable, "-m", "autoresearch.run_rvalue",
        "--tissues", tissue, "--model-type", fw,
        "--n-trials", str(N_TRIALS), "--n-cv-folds", str(N_CV),
        "--n-inner-folds", str(N_INNER), "--output-suffix", suffix,
        "--log-level", "WARNING",
    ]
    env = dict(os.environ, OMP_NUM_THREADS=THREADS_PER_CELL,
               MKL_NUM_THREADS=THREADS_PER_CELL, OPENBLAS_NUM_THREADS=THREADS_PER_CELL)
    rec = {"framework": fw, "tissue": tissue, "BA": None, "scheme": None, "status": "crash"}
    try:
        with open(log, "w") as fh:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                           timeout=3600, check=True)
        if out_json.exists():
            d = json.load(open(out_json))
            rec["BA"] = float(d["metrics"]["balanced_accuracy"])
            rec["scheme"] = d.get("scheme") or d.get("n_classes") or d.get("classification_scheme")
            rec["status"] = "ok"
        else:
            rec["status"] = "crash"
            rec["note"] = "no output json (tissue likely ineligible)"
    except subprocess.CalledProcessError as e:
        rec["note"] = f"nonzero exit {e.returncode}"
    except subprocess.TimeoutExpired:
        rec["note"] = "timeout >3600s"
    except Exception as e:  # noqa: BLE001
        rec["note"] = f"{type(e).__name__}: {e}"
    return rec


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cells = [(fw, t) for fw in FRAMEWORKS for t in PANEL]
    print(f"Loop B sweep: {len(FRAMEWORKS)} frameworks x {len(PANEL)} tissues = {len(cells)} cells "
          f"(n_trials={N_TRIALS}, {N_CV}-fold outer, {MAX_WORKERS} concurrent)")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_cell, fw, t): (fw, t) for fw, t in cells}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            ba = f"{r['BA']:.4f}" if r["BA"] is not None else "----"
            print(f"  {r['framework']:16s} {r['tissue']:8s} BA={ba} [{r['status']}]", flush=True)

    # ---- per-cell TSV ----
    cells_tsv = OUTDIR / "cells.tsv"
    with open(cells_tsv, "w") as fh:
        fh.write("framework\ttissue\tBA\tscheme\tstatus\tnote\n")
        for r in sorted(results, key=lambda x: (x["framework"], x["tissue"])):
            fh.write(f"{r['framework']}\t{r['tissue']}\t"
                     f"{r['BA'] if r['BA'] is not None else ''}\t{r.get('scheme')}\t"
                     f"{r['status']}\t{r.get('note', '')}\n")

    # ---- aggregate per framework ----
    agg = {}
    for fw in FRAMEWORKS:
        bas = [r["BA"] for r in results if r["framework"] == fw and r["BA"] is not None]
        n_ok = len(bas)
        agg[fw] = {
            "mean_BA": sum(bas) / n_ok if bas else None,
            "min_BA": min(bas) if bas else None,
            "max_BA": max(bas) if bas else None,
            "n_tissues_ok": n_ok,
            "all_above_080": bool(bas) and min(bas) >= 0.80,
            "per_tissue": {r["tissue"]: r["BA"] for r in results if r["framework"] == fw},
        }

    print("\nframework         mean    min     max    n_ok  all>=0.80")
    for fw in FRAMEWORKS:
        a = agg[fw]
        m = f"{a['mean_BA']:.4f}" if a["mean_BA"] is not None else "----"
        mn = f"{a['min_BA']:.4f}" if a["min_BA"] is not None else "----"
        mx = f"{a['max_BA']:.4f}" if a["max_BA"] is not None else "----"
        print(f"  {fw:16s} {m}  {mn}  {mx}   {a['n_tissues_ok']}     {a['all_above_080']}")

    # ---- winner: highest mean BA among frameworks with all tissues >= 0.80;
    #      fall back to highest mean BA overall if none clears the bar ----
    qualified = {fw: a for fw, a in agg.items()
                 if a["all_above_080"] and a["n_tissues_ok"] == len(PANEL)}
    pool = qualified or {fw: a for fw, a in agg.items() if a["mean_BA"] is not None}
    winner_fw = max(pool, key=lambda fw: pool[fw]["mean_BA"])
    winner = {
        "winner": winner_fw,
        "selection_rule": "max mean BA among frameworks with all tissues >= 0.80 (else max mean BA)",
        "qualified_frameworks": sorted(qualified),
        "panel": PANEL,
        "config": {"n_trials": N_TRIALS, "n_cv_folds": N_CV, "n_inner_folds": N_INNER, "seed": 42},
        "aggregate": agg,
    }
    (OUTDIR / "winner.json").write_text(json.dumps(winner, indent=2))
    print(f"\nWINNER: {winner_fw}  (mean BA={agg[winner_fw]['mean_BA']:.4f}, "
          f"min={agg[winner_fw]['min_BA']:.4f})")
    print(f"Wrote {OUTDIR / 'winner.json'} and {cells_tsv}")

    # ---- append to results.tsv (loop=B, one row per cell) ----
    import subprocess as sp
    sha = sp.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).decode().strip()
    tsv = ROOT / "autoresearch" / "autoresearch_results.tsv"
    with open(tsv, "a") as fh:
        for r in sorted(results, key=lambda x: (x["framework"], x["tissue"])):
            ba = f"{r['BA']:.4f}" if r["BA"] is not None else ""
            status = "keep" if r["status"] == "ok" else "crash"
            desc = f"{r['framework']} dev-stage classifier BA on {r['tissue']}"
            if r.get("note"):
                desc += f"; {r['note']}"
            fh.write("\t".join([sha, "B", f"{r['framework']}|{r['tissue']}", ba, "", "",
                                "1", status, desc]) + "\n")
    print(f"Appended {len(results)} Loop B rows to {tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
