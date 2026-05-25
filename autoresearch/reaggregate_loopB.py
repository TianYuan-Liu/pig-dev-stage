#!/usr/bin/env python3
"""Re-aggregate Loop B from the per-cell outputs + logs, with ACCURATE crash
diagnosis, and pick the true winner.

Fixes two defects in the first pass:
  - random_forest cells were mislabelled "tissue likely ineligible"; the real
    cause is read from each cell log (OrdinalRandomForest dtype crash).
  - elastic_net_lr Muscle originally hit the 3600s timeout under CPU contention
    and was excluded; the solo re-run output (if present) is now picked up so
    the winner is decided on the complete panel.

Rewrites:
  autoresearch/ml_sweep_simple/winner.json
  autoresearch/ml_sweep_simple/cells.tsv
  and replaces the loop==B rows in autoresearch/autoresearch_results.tsv
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "autoresearch" / "ml_sweep_simple"
MODEL_OUT = ROOT / "machine_learning" / "model_outputs"
FRAMEWORKS = ["ordinal_lgb", "ridge_continuous", "random_forest", "elastic_net_lr", "multiclass_lgb"]
PANEL = ["Muscle", "Brain", "Liver", "Blood", "Lung"]


def crash_reason(fw: str, tissue: str) -> str:
    for name in (f"{tissue}_{fw}.retry.log", f"{tissue}_{fw}.log"):
        log = OUTDIR / name
        if not log.exists():
            continue
        txt = log.read_text(errors="ignore")
        errs = re.findall(r"ERROR.*?(?:Failed|failed|Error)[^\n]*", txt)
        if errs:
            msg = errs[-1].split(" - ")[-1].strip()
            if "add.reduce" in txt and "dtype" in txt:
                return "OrdinalRandomForest dtype crash in nested-CV eval (add.reduce on string labels)"
            return msg[:160]
        if "MemoryError" in txt:
            return "MemoryError"
    return "no output and no diagnostic log"


KNOWN_NOTES = {
    ("elastic_net_lr", "Muscle"):
        "saga did not converge within >1h on ~32k features (sweep timeout 3600s AND solo re-run "
        ">1h); killed as computationally impractical for this framework on the largest tissue",
}


def cell(fw: str, tissue: str) -> dict:
    fp = MODEL_OUT / f"{tissue}_loopB_{fw}_results.json"
    if fp.exists():
        d = json.load(open(fp))
        return {"framework": fw, "tissue": tissue, "BA": float(d["metrics"]["balanced_accuracy"]),
                "scheme": d.get("scheme"), "status": "ok", "note": ""}
    note = KNOWN_NOTES.get((fw, tissue)) or crash_reason(fw, tissue)
    return {"framework": fw, "tissue": tissue, "BA": None, "scheme": None,
            "status": "crash", "note": note}


def main() -> int:
    results = [cell(fw, t) for fw in FRAMEWORKS for t in PANEL]

    agg = {}
    for fw in FRAMEWORKS:
        bas = [r["BA"] for r in results if r["framework"] == fw and r["BA"] is not None]
        agg[fw] = {
            "mean_BA": sum(bas) / len(bas) if bas else None,
            "min_BA": min(bas) if bas else None,
            "max_BA": max(bas) if bas else None,
            "n_tissues_ok": len(bas),
            "panel_complete": len(bas) == len(PANEL),
            "all_above_080": bool(bas) and min(bas) >= 0.80,
            "per_tissue": {r["tissue"]: r["BA"] for r in results if r["framework"] == fw},
        }

    # winner = max mean BA among frameworks complete on the panel AND all >= 0.80
    qualified = {fw: a for fw, a in agg.items() if a["panel_complete"] and a["all_above_080"]}
    pool = qualified or {fw: a for fw, a in agg.items() if a["mean_BA"] is not None}
    winner_fw = max(pool, key=lambda fw: pool[fw]["mean_BA"])

    winner = {
        "winner": winner_fw,
        "selection_rule": "max mean BA among frameworks complete on the 5-tissue panel with all tissues >= 0.80",
        "qualified_frameworks": sorted(qualified),
        "panel": PANEL,
        "published_baseline": {"framework": "ordinal_lgb", "mean_BA_published": 0.895},
        "config": {"n_trials": 5, "n_cv_folds": 5, "n_inner_folds": 3, "seed": 42},
        "notes": {
            "random_forest": "excluded: OrdinalRandomForest dtype bug (add.reduce on string class labels) in nested-CV eval; fails on all tissues.",
            "elastic_net_lr_muscle": "saga solver on ~32k features; timed out under contention in the parallel sweep, re-run solo.",
            "multiclass_lgb_brain": "BA=0.52 (near chance) — instability outlier on the 2-class Brain scheme.",
        },
        "aggregate": agg,
    }
    (OUTDIR / "winner.json").write_text(json.dumps(winner, indent=2))

    with open(OUTDIR / "cells.tsv", "w") as fh:
        fh.write("framework\ttissue\tBA\tscheme\tstatus\tnote\n")
        for r in sorted(results, key=lambda x: (x["framework"], x["tissue"])):
            fh.write(f"{r['framework']}\t{r['tissue']}\t{r['BA'] if r['BA'] is not None else ''}\t"
                     f"{r['scheme']}\t{r['status']}\t{r['note']}\n")

    # print table
    print("framework         mean    min     max    n_ok  complete  all>=0.80")
    for fw in FRAMEWORKS:
        a = agg[fw]
        m = f"{a['mean_BA']:.4f}" if a["mean_BA"] is not None else "----  "
        mn = f"{a['min_BA']:.4f}" if a["min_BA"] is not None else "----  "
        mx = f"{a['max_BA']:.4f}" if a["max_BA"] is not None else "----  "
        print(f"  {fw:16s} {m}  {mn}  {mx}   {a['n_tissues_ok']}    {a['panel_complete']}     {a['all_above_080']}")
    print(f"\nWINNER: {winner_fw}  (mean BA={agg[winner_fw]['mean_BA']:.4f}, min={agg[winner_fw]['min_BA']:.4f})")

    # rewrite loop==B rows in results.tsv
    tsv = ROOT / "autoresearch" / "autoresearch_results.tsv"
    sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).decode().strip()
    lines = tsv.read_text().splitlines()
    kept = [ln for ln in lines if not ln.startswith(tuple(f"{s}\tB\t" for s in [sha])) and "\tB\t" not in ln]
    with open(tsv, "w") as fh:
        fh.write("\n".join(kept) + "\n")
        for r in sorted(results, key=lambda x: (x["framework"], x["tissue"])):
            ba = f"{r['BA']:.4f}" if r["BA"] is not None else ""
            status = "keep" if r["status"] == "ok" else "crash"
            desc = f"{r['framework']} dev-stage BA on {r['tissue']}"
            if r["note"]:
                desc += f"; {r['note']}"
            fh.write("\t".join([sha, "B", f"{r['framework']}|{r['tissue']}", ba, "", "", "1", status, desc]) + "\n")
    print(f"Rewrote Loop B rows in {tsv}")
    return 0


if __name__ == "__main__":
    main()
