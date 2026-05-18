#!/usr/bin/env python3
"""Cross-species comparison for additional tissues, with configurable stage bins
to rescue tissues with sparse postnatal cohorts (e.g. Heart only has 2 samples
in the strict OLD bin but 3 if teenager is included).

Stage bins (configurable via --extend-old):
  - default      : YOUNG = {newborn, infant, toddler}
                   OLD   = {youngAdult, youngMidAge, olderMidAge, senior, Senior}
  - +teenager    : OLD also includes {teenager, oldTeenager}
  - +schoolyoung : YOUNG also includes {school, youngTeenager}

This is more biologically consistent with the porcine stage scheme, where the
pig OLD bin (Post-pubertal 150-365d + Adult >365d) corresponds to a human age
range that spans teenager+adult, not just adult.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "review" / "analyses"))

import cross_species_all_tissues as xs  # noqa: E402

BINS = {
    "default": {  # The PUBLISHED mapping (broad OLD, includes aging samples)
        "YOUNG": {"newborn", "infant", "toddler"},
        "OLD": {"youngAdult", "youngMidAge", "olderMidAge", "senior", "Senior"},
    },
    "extended_old": {  # Adds teenagers to OLD (still includes aging)
        "YOUNG": {"newborn", "infant", "toddler"},
        "OLD": {"teenager", "oldTeenager", "youngAdult", "youngMidAge",
                "olderMidAge", "senior", "Senior"},
    },
    "extended_both": {
        "YOUNG": {"newborn", "infant", "toddler", "school", "youngTeenager"},
        "OLD": {"teenager", "oldTeenager", "youngAdult", "youngMidAge",
                "olderMidAge", "senior", "Senior"},
    },
    # CORRECTED developmental mappings (exclude aging samples)
    "developmental": {  # Pig Adult ~= human youngAdult+youngMidAge (20-50 y)
        "YOUNG": {"newborn", "infant", "toddler"},
        "OLD": {"youngAdult", "youngMidAge"},
    },
    "developmental_postpub": {  # For Heart/Lung (no pig Adult, only Post-pubertal)
        "YOUNG": {"newborn", "infant", "toddler"},
        "OLD": {"teenager", "oldTeenager"},
    },
    "developmental_strict": {  # Only youngAdult (matches pig 2-year-old "young adult")
        "YOUNG": {"newborn", "infant", "toddler"},
        "OLD": {"youngAdult"},
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tissues", nargs="+", required=True)
    ap.add_argument("--bins", choices=list(BINS.keys()), default="developmental")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    # Patch the module-level stage sets so analyse_tissue uses them
    bin_def = BINS[args.bins]
    xs.HUMAN_YOUNG = bin_def["YOUNG"]
    xs.HUMAN_OLD = bin_def["OLD"]
    print(f"Using bins '{args.bins}':")
    print(f"  YOUNG = {sorted(xs.HUMAN_YOUNG)}")
    print(f"  OLD   = {sorted(xs.HUMAN_OLD)}")

    ortho = xs.build_one_to_one_orthologs()
    base = json.loads(xs.JSON_OUT.read_text())
    new_results: dict = {}
    all_gene_rows: list[dict] = []
    for t in args.tissues:
        print(f"\n--- {t} ({args.bins}) ---")
        try:
            res = xs.analyse_tissue(t, t, ortho, all_gene_rows)
            new_results[t] = res
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {type(e).__name__}: {e}")
            new_results[t] = {"tissue": t, "error": str(e)}

    base["per_tissue"].update(new_results)
    base["analysis"] = f"cross_species_extended_{args.bins}"
    base["extended_tissues"] = list(new_results.keys())
    base["bin_config"] = args.bins
    base["bins_used"] = {"YOUNG": sorted(xs.HUMAN_YOUNG), "OLD": sorted(xs.HUMAN_OLD)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(base, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
