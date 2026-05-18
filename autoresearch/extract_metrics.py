#!/usr/bin/env python3
"""Extract metrics from canonical pipeline output paths and append a row to
autoresearch_results.tsv.

Usage:
    python autoresearch/extract_metrics.py --description "baseline" --status keep

Reads:
  - machine_learning/model_outputs/{Tissue}_results.json  (per-tissue BA)
  - paper/figures/output/stats/fig4_summary.txt           (muscle r, n)
  - review/analyses/results/cross_species_all_tissues.json (brain/liver r, n)

Computes:
  joint_score = mean_BA + muscle_r + brain_r + liver_r
  (liver_r defaults to 0.0 if strict null AND pig-anchored directional concordance < 50%)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_OUTPUTS = ROOT / "machine_learning" / "model_outputs"
FIG4_SUMMARY = ROOT / "paper" / "figures" / "output" / "stats" / "fig4_summary.txt"
ALLTIS_JSON = ROOT / "review" / "analyses" / "results" / "cross_species_all_tissues.json"
TSV = ROOT / "autoresearch_results.tsv"

TISSUES = ["Muscle", "Brain", "Liver", "Lung", "Blood"]


def read_ba_per_tissue() -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for t in TISSUES:
        p = MODEL_OUTPUTS / f"{t}_results.json"
        if not p.exists():
            out[t] = None
            continue
        d = json.loads(p.read_text())
        out[t] = float(d["metrics"]["balanced_accuracy"])
    return out


def read_muscle_r() -> tuple[float | None, int | None]:
    if not FIG4_SUMMARY.exists():
        return None, None
    txt = FIG4_SUMMARY.read_text()
    r = re.search(r"Pearson R:\s*([0-9.\-]+)", txt)
    n = re.search(r"Number of genes:\s*([0-9]+)", txt)
    return (float(r.group(1)) if r else None, int(n.group(1)) if n else None)


def read_alltis_r(alltis_path: Path = ALLTIS_JSON) -> dict[str, tuple[float | None, int | None]]:
    """Return per-tissue (r, n) for ALL tissues in the JSON using strict if available
    else pig-anchored gated on directional concordance >= 50% (otherwise treat as null)."""
    if not alltis_path.exists():
        return {}
    d = json.loads(alltis_path.read_text())
    per = d.get("per_tissue", {})
    out: dict[str, tuple[float | None, int | None]] = {}
    for tissue, row in per.items():
        if not isinstance(row, dict) or row.get("skipped") or row.get("error"):
            out[tissue] = (0.0, 0)
            continue
        strict = row.get("strict", {}) or {}
        anchored = row.get("pig_anchored", {}) or {}
        if strict.get("n_genes", 0) and strict.get("pearson_r") is not None:
            out[tissue] = (float(strict["pearson_r"]), int(strict["n_genes"]))
            continue
        dc = anchored.get("directional_concordance")
        if anchored.get("n_genes", 0) and anchored.get("pearson_r") is not None \
                and dc is not None and dc >= 50.0:
            out[tissue] = (float(anchored["pearson_r"]), int(anchored["n_genes"]))
            continue
        out[tissue] = (0.0, 0)
    return out


def git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "nogit"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--description", required=True, help="Short text describing the experiment")
    ap.add_argument("--status", choices=["keep", "discard", "crash"], required=True)
    ap.add_argument("--wall-min", type=float, default=0.0,
                    help="Wall-clock minutes for this experiment (informational)")
    ap.add_argument("--alltis-json", type=str, default=str(ALLTIS_JSON),
                    help="Override path to cross_species_all_tissues.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the row without appending")
    args = ap.parse_args()

    ba = read_ba_per_tissue()
    muscle_r, muscle_n = read_muscle_r()
    alltis = read_alltis_r(Path(args.alltis_json))
    brain_r, brain_n = alltis.get("Brain", (0.0, 0))
    liver_r, liver_n = alltis.get("Liver", (0.0, 0))
    # Extra tissues beyond Brain/Liver/Muscle (so we don't double-count Muscle)
    extra_tissues = {t: rn for t, rn in alltis.items()
                     if t not in ("Brain", "Liver", "Muscle")}

    if any(ba[t] is None for t in TISSUES):
        print(f"WARNING: missing BA for {[t for t in TISSUES if ba[t] is None]}", file=sys.stderr)
    ba_vals = [v for v in ba.values() if v is not None]
    mean_ba = sum(ba_vals) / len(ba_vals) if ba_vals else None

    def _z(x):
        return 0.0 if x is None else float(x)

    joint = _z(mean_ba) + _z(muscle_r) + _z(brain_r) + _z(liver_r) \
            + sum(_z(r) for r, _n in extra_tissues.values())

    # Build extra-tissues compact description for visibility (eg "Testis:0.42(n=1327)")
    extras_str = ";".join(f"{t}:{_z(r):.3f}(n={n})"
                          for t, (r, n) in sorted(extra_tissues.items())
                          if r is not None and r > 0)

    row = [
        git_short_sha(),
        f"{_z(mean_ba):.6f}",
        f"{_z(ba.get('Muscle')):.6f}",
        f"{_z(ba.get('Brain')):.6f}",
        f"{_z(ba.get('Liver')):.6f}",
        f"{_z(ba.get('Lung')):.6f}",
        f"{_z(ba.get('Blood')):.6f}",
        f"{_z(muscle_r):.6f}",
        str(muscle_n if muscle_n is not None else 0),
        f"{_z(brain_r):.6f}",
        str(brain_n if brain_n is not None else 0),
        f"{_z(liver_r):.6f}",
        str(liver_n if liver_n is not None else 0),
        f"{joint:.6f}",
        f"{args.wall_min:.1f}",
        args.status,
        (args.description + (f" | extra: {extras_str}" if extras_str else "")
         ).replace("\t", " ").replace("\n", " "),
    ]
    line = "\t".join(row) + "\n"
    if args.dry_run:
        print(line, end="")
        return 0
    with TSV.open("a") as f:
        f.write(line)
    print(line, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
