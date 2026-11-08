#!/usr/bin/env python3
"""Brain decision (6-stage): plain Sub_categories=="Brain" vs forebrain-aggregated.

Reuses the EXACT canonical transfer protocol from paper/figures/provenance/transfer_6stage.py
(run_cell_pls_only: drop-small 6-stage -> OrdinalPLS(n_components=10, seed=42) on
independently z-scored 1:1 orthologs -> project -> Spearman vs target level ->
1000x ortholog-scramble null). The ONLY thing that changes between the two arms
is which PigGTEx brain samples define the pig developmental axis.

Self-check: the plain arm must reproduce the committed transfer_matrix_6stage.csv
brain row (Human rho=0.3745) before the aggregated number is trusted.
"""
from __future__ import annotations
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # provenance dir: sibling modules

import transfer_6stage as t6  # canonical protocol + config

# Forebrain aggregation (constants inlined below; matches the retired run_brain_aggregated).
FOREBRAIN = ["Brain", "Frontal cortex", "Cerebral cortex", "Frontal lobe",
             "Cerebrum", "Hippocampus", "Amygdala", "Striatum region"]
SPECIES = ["Human", "Macaque", "Mouse", "Rat", "Rabbit", "Opossum", "Chicken"]


def _load_meta() -> pd.DataFrame:
    meta = pd.read_csv(t6.PIG_META).rename(columns={
        "BioSample": "Sample_ID", "Tissue class": "Tissue",
        "Sub categories": "Sub_categories"}).set_index("Sample_ID")
    meta["Age_Days"] = meta["Age"].apply(t6.parse_age_days)
    return meta


def _load_expr() -> pd.DataFrame:
    with gzip.open(t6.PIG_TPM_DIR / "Brain.expr_tpm.txt.gz", "rt") as fh:
        return pd.read_csv(fh, sep="\t", index_col=0)  # genes x samples


def _build_pig_X(meta: pd.DataFrame, expr: pd.DataFrame, sample_idx) -> pd.DataFrame:
    samples = [s for s in sample_idx if s in expr.columns]
    X = np.log2(expr[samples].T + 1.0)
    X["Age_Days"] = meta.loc[samples, "Age_Days"].values
    X["BioProject"] = meta.loc[samples, "BioProject"].astype(str).values
    return X


def inspect(meta: pd.DataFrame, expr: pd.DataFrame) -> None:
    in_expr = meta.index.isin(expr.columns)
    sub = meta[in_expr]
    print("=== Brain expr file: samples present in metadata ===")
    print(f"  total samples in Brain.expr file with metadata: {in_expr.sum()}")
    print("\n  'Tissue class' values (samples in expr file):")
    print(sub["Tissue"].value_counts().to_string())
    print("\n  'Sub categories' values (samples in expr file):")
    print(sub["Sub_categories"].value_counts().to_string())
    # forebrain membership via either column
    fb = sub[sub["Tissue"].isin(FOREBRAIN) | sub["Sub_categories"].isin(FOREBRAIN)]
    print(f"\n  forebrain-set members (either column): n={len(fb)}")
    print("  6-stage distribution (forebrain, drop-small applies later):")
    print(fb["Age_Days"].apply(t6.stage6).value_counts().to_string())


def select_plain(meta: pd.DataFrame) -> pd.Index:
    return meta[(meta["Tissue"] == "Brain") & (meta["Sub_categories"] == "Brain")].index


def select_forebrain(meta: pd.DataFrame) -> pd.Index:
    return meta[meta["Tissue"].isin(FOREBRAIN) | meta["Sub_categories"].isin(FOREBRAIN)].index


def run_arm(pig_X: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label}: pig brain n(raw)={len(pig_X)} ===")
    stage, kept = t6.assign_ordinal_stages(pig_X["Age_Days"], "6stage")
    print(f"  kept 6-stages (>=10): {kept}  (n staged={int(stage.notna().sum())})")
    print(f"{'species':<9} {'rho':>8} {'scramble_p':>11} {'n_orth':>7} {'n_tgt':>6} {'n_pig':>6}")
    for sp in SPECIES:
        ortho = t6.build_ortholog_map(sp)
        tgt_expr, tgt_level = t6.load_cardoso_target(sp, "Brain")
        r = t6.run_cell_pls_only(pig_X, ortho, tgt_expr, tgt_level, "6stage", do_scramble=True)
        if r is None:
            print(f"{sp:<9} {'SKIP (not fittable)':>30}")
            continue
        print(f"{sp:<9} {r['rho_pls']:>8.4f} {r.get('scramble_p_pls', float('nan')):>11.4f} "
              f"{r['n_orthologs']:>7} {r['n_target_samples']:>6} {r['n_pig_samples']:>6}")


def main() -> None:
    meta = _load_meta()
    expr = _load_expr()
    inspect(meta, expr)

    plain_X = _build_pig_X(meta, expr, select_plain(meta))
    agg_X = _build_pig_X(meta, expr, select_forebrain(meta))

    print("\n" + "=" * 70)
    print("ARM 1 — PLAIN  (Sub_categories=='Brain'; canonical, committed)")
    print("  self-check: Human rho should reproduce 0.3745 from "
          "transfer_matrix_6stage.csv")
    print("=" * 70)
    run_arm(plain_X, "PLAIN brain")

    print("\n" + "=" * 70)
    print("ARM 2 — AGGREGATED forebrain (Brain+cortex+hippo+amygdala+striatum)")
    print("=" * 70)
    run_arm(agg_X, "AGGREGATED forebrain")


if __name__ == "__main__":
    main()
