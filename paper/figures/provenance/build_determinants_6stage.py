#!/usr/bin/env python3
"""Assemble determinants_6stage.json (the per-cell cross-species transfer payload
behind Fig 3 / Fig 4 and the determinant statistics) from the two tracked
artifacts written by `transfer_6stage.py --exp matrix`:

  paper/figures/data/transfer_6stage/transfer_matrix_6stage.csv      -> the 40 `points`
  paper/figures/data/transfer_6stage/determinants_tests_6stage.json  -> the `tests` block

The brain rows are forebrain-aggregated BY CONSTRUCTION: transfer_6stage.load_pig_tissue
returns the forebrain set for Brain (n_pig=170, 4 stages), so the matrix already carries
the forebrain brain rhos and a forebrain-consistent expression breadth. There is no
post-hoc brain swap and NO hardcoded constants (this replaces the old circular
recompute_determinants_aggbrain.py, which read its own output and re-applied a hardcoded
AGG_BRAIN dict, leaving the brain breadth stale).

Reproduce end-to-end:
    python3 paper/figures/provenance/transfer_6stage.py --exp matrix
    python3 paper/figures/provenance/build_determinants_6stage.py

Real data only; deterministic (seed 42 inside transfer_6stage).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRATCH = ROOT / "paper" / "figures" / "data" / "transfer_6stage"
MATRIX = SCRATCH / "transfer_matrix_6stage.csv"
TESTS = SCRATCH / "determinants_tests_6stage.json"
OUT = ROOT / "paper" / "figures" / "data" / "determinants_6stage.json"


def main() -> None:
    m = pd.read_csv(MATRIX)
    tests = json.loads(TESTS.read_text())

    points = []
    for _, r in m.iterrows():
        points.append({
            "organ": r["organ"],
            "species": r["species"],
            "rho": round(float(r["rho"]), 4),
            "divergence_mya": float(r["divergence_mya"]),
            "n_target": int(r["n_target"]),
            "n_orthologs": int(r["n_orth"]),
            "scramble_p_systemA": round(float(r["scramble_p"]), 4),
            "kept_pig_stages": r["kept_pig_stages"],
            "K": int(r["K"]),
            "n_pig": int(r["n_pig"]),
        })

    out = {
        "basis": "6-stage ordinal PLS score projection (Spearman rho vs 1000x ortholog-scramble null)",
        "seed": 42,
        "scheme": "6-stage",
        "brain_definition": "forebrain-aggregated (cerebrum minus cerebellum; n_pig=170, 4 stages)",
        "points": points,
        "tests": tests,
    }
    OUT.write_text(json.dumps(out, indent=2))
    bd = tests["H2_pig_breadth"]["Brain"]
    cx = tests["H2_rho_vs_complexity"]["spearman_r"]
    print(f"Wrote {OUT}")
    print(f"  {len(points)} points | brain breadth = {bd} (forebrain) | "
          f"complexity Spearman = {cx:+.4f}")


if __name__ == "__main__":
    main()
