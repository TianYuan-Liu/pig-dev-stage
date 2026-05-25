"""dGTEx data loader: reads per-tissue gene_tpm gct.gz + metadata, returns young/old splits.

Joins sample-attributes → subject-phenotypes via SUBJECT_ID, applies v2 binning,
pools all dGTEx sub-regions (SMTSD variants) under the SMTS umbrella, and returns
an expression matrix with samples assigned to young/old bins.

Reuses ortholog table at review/analyses/results/pig_human_one_to_one_orthologs.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from autoresearch.stage_mapping_v2 import human_bin

ROOT = Path(__file__).resolve().parents[1]
DGTEX_META_DIR = ROOT / "data" / "dgtex" / "metadata"
DGTEX_EXPR_DIR = ROOT / "data" / "dgtex" / "expression"

SAMPLE_ATTR_FILE = "dGTEx_Analysis_2026-01-30_v1_Sample_Attributes_Dataset_DS_open.xlsx"
SUBJECT_PHENO_FILE = "dGTEx_Analysis_2026-01-30_v1_Subject_Phenotypes_Dataset_DS_open.xlsx"


def load_sample_metadata() -> pd.DataFrame:
    """Cross-join sample × subject and return one row per RNA-seq sample.

    Filters to RNA samples only (SMAFRZE='RNASEQ'); EXCLUDE'd samples dropped.
    Adds 'bin' column ('young' / 'old' / None) per stage_mapping_v2.
    """
    samples = pd.read_excel(DGTEX_META_DIR / SAMPLE_ATTR_FILE)
    subjects = pd.read_excel(DGTEX_META_DIR / SUBJECT_PHENO_FILE)

    rna = samples[
        (samples["ANALYTE_TYPE"].astype(str).str.contains("RNA", case=False, na=False))
        & (samples["SMAFRZE"] == "RNASEQ")
    ].copy()
    joined = rna.merge(subjects, on="SUBJECT_ID", how="left")
    joined["bin"] = joined["AGECOHORT"].astype("Int64").map(
        lambda x: human_bin(int(x)) if pd.notna(x) else None
    )
    return joined


def load_tissue_expression(tissue_file_name: str) -> pd.DataFrame:
    """Read gene_tpm_dgtex_v1_<tissue>.gct.gz into DataFrame (genes × samples).

    gct format:
      Line 1: '#1.2'
      Line 2: '<n_rows> <n_cols>'
      Line 3: header (Name, Description, sample_id, sample_id, ...)
      Lines 4+: ENSG_id, gene_symbol, tpm_values...

    Returns DataFrame indexed by un-versioned ENSG ID, columns = sample IDs.
    """
    path = DGTEX_EXPR_DIR / f"gene_tpm_dgtex_v1_{tissue_file_name}.gct.gz"
    df = pd.read_csv(path, sep="\t", skiprows=2, compression="gzip")
    df = df.rename(columns={"Name": "ensg_versioned", "Description": "symbol"})
    df["ensg"] = df["ensg_versioned"].str.replace(r"\.\d+$", "", regex=True)
    df = df.drop_duplicates(subset="ensg", keep="first").set_index("ensg")
    sample_cols = [c for c in df.columns if c not in ("ensg_versioned", "symbol")]
    return df[sample_cols]


def load_dgtex_tissue(
    dgtex_smts_label: str,
    tissue_file_name: str,
) -> Tuple[pd.DataFrame, list[str], list[str]]:
    """Load one dGTEx tissue with young/old sample lists.

    Args:
        dgtex_smts_label: SMTS value (e.g., 'Heart', 'Adipose Tissue') — used to
            filter samples to the right tissue from the metadata cross-join.
        tissue_file_name: filename slug for the gct.gz (e.g., 'heart',
            'adipose_tissue', 'small_intestine').

    Returns:
        (expression_df, young_sample_ids, old_sample_ids)
        expression_df is indexed by un-versioned ENSG ID, columns = all SAMPLE_IDs
        that survived RNA-seq + AGECOHORT binning.
        Sub-regions (SMTSD variants) are pooled under the SMTS umbrella.
    """
    meta = load_sample_metadata()
    meta_t = meta[(meta["SMTS"] == dgtex_smts_label) & meta["bin"].notna()].copy()

    expr = load_tissue_expression(tissue_file_name)

    sample_ids = [s for s in meta_t["SAMPLE_ID"] if s in expr.columns]
    expr_t = expr[sample_ids]
    meta_t = meta_t[meta_t["SAMPLE_ID"].isin(sample_ids)]

    young = meta_t.loc[meta_t["bin"] == "young", "SAMPLE_ID"].tolist()
    old = meta_t.loc[meta_t["bin"] == "old", "SAMPLE_ID"].tolist()
    return expr_t, young, old


if __name__ == "__main__":
    from autoresearch.stage_mapping_v2 import INCLUDED_TISSUES

    print("dGTEx tissue × sample-count smoke test")
    print("=" * 60)
    for smts, _pig, fname in INCLUDED_TISSUES:
        try:
            expr, young, old = load_dgtex_tissue(smts, fname)
            print(f"  {smts:20s} expr={expr.shape}  young={len(young):3d}  old={len(old):3d}")
        except FileNotFoundError as e:
            print(f"  {smts:20s} FILE MISSING: {e}")
