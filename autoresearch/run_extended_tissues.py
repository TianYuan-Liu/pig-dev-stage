#!/usr/bin/env python3
"""Extend cross-species comparison to additional tissues available in both
PigGTEx and Cardoso-Moreira 2019:
  - Cerebellum (PigGTEx: ~5 cerebellum samples in Brain TPM file -> SKIP)
  - Heart      (PigGTEx Heart.expr_tpm.txt.gz, n~192 ; CM Heart n=12 postnatal)
  - Testis     (PigGTEx Testis.expr_tpm.txt.gz, n~225; CM Testis n=12 postnatal)
  - Kidney     (PigGTEx Kidney.expr_tpm.txt.gz, n~60 ; CM Kidney n=9 marginal)

Writes a JSON file in the cross_species_all_tissues.json schema, with
Brain/Liver copied unchanged from the published file and the new tissues
appended.

Usage:
  python autoresearch/run_extended_tissues.py \
         --tissues Heart Testis Kidney \
         --out review/analyses/results/cross_species_extended_tissues.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "review" / "analyses"))

import cross_species_all_tissues as xs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tissues", nargs="+", required=True,
                    help="Tissue names matching both PigGTEx file and Cardoso-Moreira column prefix")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    ortho = xs.build_one_to_one_orthologs()

    base = json.loads(xs.JSON_OUT.read_text())
    all_gene_rows: list[dict] = []
    new_results: dict = {}
    for t in args.tissues:
        print(f"\n--- {t} ---")
        try:
            res = xs.analyse_tissue(t, t, ortho, all_gene_rows)
            new_results[t] = res
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {type(e).__name__}: {e}")
            new_results[t] = {"tissue": t, "error": str(e)}

    # Keep Brain/Liver as-is, add new tissues
    base["per_tissue"].update(new_results)
    base["analysis"] = "cross_species_extended_tissues"
    base["extended_tissues"] = list(new_results.keys())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(base, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
