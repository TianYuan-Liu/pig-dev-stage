#!/usr/bin/env python3
"""Re-extract PLS-VIP developmental biomarkers at 6-STAGE (Fig 3 / Table S5 / figS5).

Self-contained: the old inputs (autoresearch/ml_benchmark/pls_final/*) are gone,
so this fits the final OrdinalPLS per tissue from scratch, reusing the pig
6-stage loader / staging from paper/figures/provenance/transfer_6stage.py.

Per tissue (Muscle, Brain, Liver, Blood, Lung):
  1. Load pig expression, assign 6 stages (boundaries 3/28/90/165/540, drop-small
     <10), expression = log2(TPM+1).
  2. Fit OrdinalPLS(n_components=10) on ALL samples (ordinal kept-stage labels).
  3. Rank genes by VIP (pls.get_feature_importance()); take the top 20 per tissue.
     Sign each by the PLS regression coefficient onto the stage index
     (pls.model_.coef_): + = accelerator (rises toward later stage), - =
     decelerator (falls toward later stage).
  4. Map pig Ensembl id -> symbol via data/cardoso_moreira_2019/_pig_id_to_symbol
     .json; genes with no symbol are flagged novel/unannotated (ENSSSCG...).
  5. Per-stage mean log2(TPM+1) trajectory + peak stage.

OUTPUT SCHEMA (read by paper/figures/R/figS5_tissue_features.R and the
auto-generated tableS7_tissue_features.tex):
    tissue, rank, ensembl, gene, symbol, vip, pls_coef, sign, trend, function,
    annotation_status, Neonatal_TPM, Suckling_TPM, Juvenile_TPM, Pre-pubertal_TPM,
    Post-pubertal_TPM, Adult_TPM, peak_stage, fold_change_peak_vs_trough

STAGING NOTE:
    Gene SELECTION, VIP and SIGN come from the 6-stage OrdinalPLS fit. The
    per-stage trajectory columns + peak_stage are reported on the SAME six stages
    (boundaries 3/28/90/165/540 d): Neonatal/Suckling/Juvenile/Pre-pubertal/
    Post-pubertal/Adult. A stage column is NaN if that stage was dropped by the
    6-stage drop-small (<10) filter for the tissue (so a 2-stage tissue shows a
    2-point trajectory).

FUNCTION COLUMN: this script does NOT assign gene functions. For a gene that was
already curated in the committed CSV (matched on tissue+ensembl) the existing
`function` + `annotation_status` are carried over verbatim; NEW genes get an
empty `function` and annotation_status = 'novel' (no symbol) or 'pending' (has
symbol, awaiting PubMed/UniProt verification).

Deterministic (seed=42 inside OrdinalPLS). Real data only.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path("/Users/tianyuan/Desktop/github_dev/pig-dev-stage")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper" / "figures" / "provenance"))

from transfer_6stage import (  # noqa: E402
    assign_ordinal_stages, load_pig_tissue, N_COMPONENTS,
)
from machine_learning.model_training.models import OrdinalPLS  # noqa: E402

PIG_SYM = ROOT / "data" / "cardoso_moreira_2019" / "_pig_id_to_symbol.json"
RESULTS_DIR = ROOT / "review" / "analyses" / "results"
CSV_PATH = RESULTS_DIR / "pls_vip_features.csv"
BACKUP_PATH = RESULTS_DIR / "pls_vip_features.5stage.bak"

TISSUES = ["Muscle", "Brain", "Liver", "Blood", "Lung"]
TOP_N = 20
SCHEME = "6stage"

# 6-stage display bins (boundaries 3/28/90/165/540 d): one per developmental
# stage, so the per-stage TPM trajectory + peak_stage are reported on the same
# six stages as the rest of the paper (Neonatal/Suckling/Juvenile/Pre-pubertal/
# Post-pubertal/Adult).
DISPLAY_BINS = [
    ("Neonatal_TPM", "Neonatal", 0, 3),
    ("Suckling_TPM", "Suckling", 4, 28),
    ("Juvenile_TPM", "Juvenile", 29, 90),
    ("Pre-pubertal_TPM", "Pre-pubertal", 91, 165),
    ("Post-pubertal_TPM", "Post-pubertal", 166, 540),
    ("Adult_TPM", "Adult", 541, 10**9),
]
# Each display bin maps to its own 6-stage label (identity); a bin is shown only
# if that stage survived the drop-small (<10) filter.
BIN_TO_STAGE6 = {
    "Neonatal": ["Neonatal"], "Suckling": ["Suckling"], "Juvenile": ["Juvenile"],
    "Pre-pubertal": ["Pre-pubertal"], "Post-pubertal": ["Post-pubertal"],
    "Adult": ["Adult"],
}


def load_symbol_map() -> Dict[str, str]:
    with open(PIG_SYM) as f:
        return json.load(f)


def load_curated() -> Dict[tuple, dict]:
    """Carry-over map: (tissue, ensembl) -> {function, annotation_status} from
    the committed CSV (so existing online-verified functions are not lost)."""
    if not CSV_PATH.exists():
        return {}
    old = pd.read_csv(CSV_PATH)
    out: Dict[tuple, dict] = {}
    for _, r in old.iterrows():
        out[(r["tissue"], r["ensembl"])] = {
            "function": r.get("function", ""),
            "annotation_status": r.get("annotation_status", ""),
        }
    return out


def extract_tissue(tissue: str, sym_map: Dict[str, str],
                   curated: Dict[tuple, dict]) -> List[dict]:
    pig_X = load_pig_tissue(tissue)
    stage, kept = assign_ordinal_stages(pig_X["Age_Days"], SCHEME)
    code = {s: i for i, s in enumerate(kept)}
    mask = stage.notna().values
    feat = pig_X.drop(columns=["Age_Days", "BioProject"]).loc[mask]  # log2(TPM+1)
    age = pig_X["Age_Days"].loc[mask].values
    y = np.array([code[s] for s in stage[mask].values])

    # --- fit OrdinalPLS on the 6-stage ordinal labels ----------------------
    pls = OrdinalPLS(n_components=N_COMPONENTS, seed=42)
    pls.fit(feat, y)
    vip = pls.get_feature_importance()                 # Series idx=gene, desc
    coef = pd.Series(np.asarray(pls.model_.coef_).ravel(), index=feat.columns)

    # QC: VIP must align with feature order (rho of |coef| ranking is a sanity
    # check, not a constraint). Skipped here; the wrapper guarantees alignment.

    # --- which DISPLAY bins survive the 6-stage drop-small filter ----------
    kept_set = set(kept)
    bin_kept = {bin_name: any(s in kept_set for s in BIN_TO_STAGE6[bin_name])
                for _col, bin_name, _lo, _hi in DISPLAY_BINS}

    # --- per display-bin mean log2(TPM+1) over the RETAINED samples --------
    # (samples already restricted to kept 6-stages via `mask`.)
    bin_means: Dict[str, Dict[str, float]] = {}  # gene -> {col: mean}
    bin_assign = np.full(len(age), None, dtype=object)
    for col, bin_name, lo, hi in DISPLAY_BINS:
        in_bin = (age >= lo) & (age <= hi)
        bin_assign[in_bin] = col

    top = list(vip.index[:TOP_N])
    rows: List[dict] = []
    for rank, gene in enumerate(top, start=1):
        gvals = feat[gene].values
        means: Dict[str, float] = {}
        for col, bin_name, _lo, _hi in DISPLAY_BINS:
            if not bin_kept[bin_name]:
                means[col] = np.nan          # bin dropped by 6-stage filter
                continue
            sel = bin_assign == col
            means[col] = float(np.mean(gvals[sel])) if sel.sum() > 0 else np.nan
        # peak stage among non-NaN display bins
        valid = {c: v for c, v in means.items() if not np.isnan(v)}
        peak_col = max(valid, key=valid.get) if valid else None
        peak_stage = next(b for c, b, _, _ in DISPLAY_BINS if c == peak_col) if peak_col else ""
        trough = min(valid.values()) if valid else np.nan
        peak = max(valid.values()) if valid else np.nan
        fc = float(peak - trough) if valid else np.nan

        symbol = sym_map.get(gene, "")
        c = float(coef[gene])
        sign = "+" if c >= 0 else "-"
        trend = "accelerator" if c >= 0 else "decelerator"

        carry = curated.get((tissue, gene), {})
        if carry:
            function = carry.get("function", "")
            annotation_status = carry.get("annotation_status", "")
        else:
            function = ""
            annotation_status = "novel" if not symbol else "pending"

        rows.append({
            "tissue": tissue,
            "rank": rank,
            "ensembl": gene,
            "gene": symbol if symbol else gene,
            "symbol": symbol,
            "vip": round(float(vip[gene]), 3),
            "pls_coef": round(c, 5),
            "sign": sign,
            "trend": trend,
            "function": function,
            "annotation_status": annotation_status,
            "Neonatal_TPM": means["Neonatal_TPM"],
            "Suckling_TPM": means["Suckling_TPM"],
            "Juvenile_TPM": means["Juvenile_TPM"],
            "Pre-pubertal_TPM": means["Pre-pubertal_TPM"],
            "Post-pubertal_TPM": means["Post-pubertal_TPM"],
            "Adult_TPM": means["Adult_TPM"],
            "peak_stage": peak_stage,
            "fold_change_peak_vs_trough": fc,
        })
    print(f"  {tissue:7s} kept6={kept}  n={int(mask.sum())}  "
          f"novel(no symbol)={sum(1 for r in rows if not r['symbol'])}/{TOP_N}",
          flush=True)
    return rows


def main() -> None:
    print("=" * 78)
    print("PLS-VIP biomarker re-extraction at 6-stage (Fig 3 / Table S5 / figS5)")
    print("=" * 78)

    if CSV_PATH.exists() and not BACKUP_PATH.exists():
        shutil.copy2(CSV_PATH, BACKUP_PATH)
        print(f"Backed up existing 5-stage CSV -> {BACKUP_PATH}")

    sym_map = load_symbol_map()
    curated = load_curated()

    all_rows: List[dict] = []
    for tissue in TISSUES:
        all_rows.extend(extract_tissue(tissue, sym_map, curated))

    cols = ["tissue", "rank", "ensembl", "gene", "symbol", "vip", "pls_coef",
            "sign", "trend", "function", "annotation_status",
            "Neonatal_TPM", "Suckling_TPM", "Juvenile_TPM", "Pre-pubertal_TPM",
            "Post-pubertal_TPM", "Adult_TPM", "peak_stage",
            "fold_change_peak_vs_trough"]
    df = pd.DataFrame(all_rows)[cols]
    df.to_csv(CSV_PATH, index=False)
    print(f"\nWrote {CSV_PATH}  ({len(df)} rows = {len(TISSUES)}x{TOP_N})")


if __name__ == "__main__":
    main()
