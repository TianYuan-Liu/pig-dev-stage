#!/usr/bin/env python3
"""Dump per-sample (projected pig PLS score, target developmental level) pairs for
every cross-species transfer cell, so the Spearman rho behind Fig 4 can be shown
as an actual correlation scatter (supplementary figure).

This reuses the CANONICAL transfer protocol from paper/figures/provenance/transfer_6stage.py
verbatim (same drop-small 6-stage pig labels, same OrdinalPLS(n_components=10,
seed=42) on independently z-scored 1:1 orthologs, same projection via the inner
sklearn estimator, same postnatal-only Cardoso target, same dGTEx AGECOHORT
ordinal). The ONLY thing added is that the per-target-sample projected score and
its ordinal level are written out, instead of being collapsed to a single rho.

Brain uses the FOREBRAIN-AGGREGATED pig matrix (the construct behind the primary
brain row of determinants_6stage.json: n_pig=170, 4 stages), reproduced exactly
from paper/figures/provenance/verify_brain_6stage.py ARM 2.

No scramble null is computed here (it is not needed for the scatter); the rho is
deterministic. Each cell's recomputed rho is checked against the committed
determinants_6stage.json (Cardoso) / supp_table_dGTEx_only_6stage.csv (dGTEx) and
the script aborts if any cell drifts by more than 1e-2, so the figure cannot show
a curve that disagrees with the published number.

Real data only; seed=42.  Output: paper/figures/data/transfer_scatter_6stage.csv
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # provenance dir: sibling modules

import transfer_6stage as t6  # canonical protocol + config

OUT_CSV = ROOT / "paper" / "figures" / "data" / "transfer_scatter_6stage.csv"
DET_JSON = ROOT / "paper" / "figures" / "data" / "determinants_6stage.json"
DGTEX_CSV = ROOT / "paper" / "figures" / "data" / "supp_table_dGTEx_only_6stage.csv"

# Forebrain aggregation set (verify_brain_6stage.py).
FOREBRAIN = ["Brain", "Frontal cortex", "Cerebral cortex", "Frontal lobe",
             "Cerebrum", "Hippocampus", "Amygdala", "Striatum region"]

CARDOSO_ORGANS = t6.CARDOSO_ORGANS          # Brain, Heart, Kidney, Liver, Ovary, Testis
CARDOSO_SPECIES = t6.CARDOSO_SPECIES        # Human ... Chicken (div order in DIVERGENCE_MYA)
DGTEX_SUBSTRATES = t6.DGTEX_TABLE_SUBSTRATES


# ---------------------------------------------------------------------------
# pig forebrain-aggregate matrix (brain row only); mirrors verify_brain_6stage
# ---------------------------------------------------------------------------
def load_pig_forebrain() -> pd.DataFrame:
    meta = pd.read_csv(t6.PIG_META).rename(columns={
        "BioSample": "Sample_ID", "Tissue class": "Tissue",
        "Sub categories": "Sub_categories"}).set_index("Sample_ID")
    meta["Age_Days"] = meta["Age"].apply(t6.parse_age_days)
    with gzip.open(t6.PIG_TPM_DIR / "Brain.expr_tpm.txt.gz", "rt") as fh:
        expr = pd.read_csv(fh, sep="\t", index_col=0)  # genes x samples
    sel = meta[meta["Tissue"].isin(FOREBRAIN) | meta["Sub_categories"].isin(FOREBRAIN)].index
    samples = [s for s in sel if s in expr.columns]
    X = np.log2(expr[samples].T + 1.0)
    X["Age_Days"] = meta.loc[samples, "Age_Days"].values
    X["BioProject"] = meta.loc[samples, "BioProject"].astype(str).values
    return X


# ---------------------------------------------------------------------------
# per-sample projection (deterministic core of t6.run_cell_pls_only)
# ---------------------------------------------------------------------------
def project_cell(pig_X, ortho, target_expr, target_level):
    """Return (scores, levels, samples, rho, n_orth, kept) for one transfer cell,
    or None if not fittable. Identical maths to t6.run_cell_pls_only minus the
    scramble null; the per-target-sample score vector is returned rather than
    discarded."""
    age = pig_X["Age_Days"]
    stage, kept = t6.assign_ordinal_stages(age, "6stage")
    if len(kept) < 2:
        return None
    code = {s: i for i, s in enumerate(kept)}
    mask = stage.notna().values
    pig_feat = pig_X.drop(columns=["Age_Days", "BioProject"]).loc[mask]
    y = np.array([code[s] for s in stage[mask].values])

    o = ortho[ortho["pig_gene_id"].isin(pig_feat.columns) &
              ortho["target_gene_id"].isin(target_expr.index)].copy()
    o = o.drop_duplicates("pig_gene_id").drop_duplicates("target_gene_id")
    if len(o) < 10:
        return None
    pig_ids = o["pig_gene_id"].tolist()
    tgt_ids = o["target_gene_id"].tolist()

    Zpig = t6._zscore_fit(pig_feat[pig_ids].values.astype(np.float64))
    Zpig_df = pd.DataFrame(Zpig, columns=pig_ids)
    Ztgt = t6._zscore_fit(target_expr.loc[tgt_ids].T.values.astype(np.float64))
    samples = list(target_expr.columns)
    lvl = target_level.loc[samples].values.astype(float)

    pls = t6.OrdinalPLS(n_components=t6.N_COMPONENTS, seed=t6.SEED)
    pls.fit(Zpig_df, y)
    scores = pls.model_.predict(Ztgt).ravel()
    rho = t6._spearman(scores, lvl)
    return scores, lvl, samples, rho, int(len(o)), ";".join(kept)


def cardoso_target(cardoso_df, species, organ):
    """Postnatal-only (genes x samples) submatrix + ordinal level for one organ.
    Mirrors t6.cardoso_cell_from_matrix's target construction exactly."""
    organ_cols = [c for c in cardoso_df.columns if c.split(".")[0] == organ]
    if not organ_cols:
        return None
    ranks, keep = {}, []
    for c in organ_cols:
        parts = c.split(".")
        stage_token = ".".join(parts[1:-1]) if len(parts) >= 3 else parts[1]
        r, is_post = t6.cardoso_stage_rank(stage_token, species=species)
        if r is not None and is_post:
            ranks[c] = r
            keep.append(c)
    if len(keep) < 3:
        return None
    return np.log2(cardoso_df[keep] + 1.0), pd.Series(ranks)


def main() -> None:
    det = json.loads(DET_JSON.read_text())
    det_rho = {(p["organ"], p["species"]): p["rho"] for p in det["points"]}
    dgtex_ref = pd.read_csv(DGTEX_CSV)
    dgtex_rho = {(r["organ"], "Human"): r["rho"] for _, r in dgtex_ref.iterrows()}

    rows = []
    checks = []  # (cell, rho_recomputed, rho_reference)

    # cache pig source matrices (one per organ; brain = forebrain aggregate)
    print("[pig] loading source tissues ...", flush=True)
    pig_cache = {}
    for organ in CARDOSO_ORGANS:
        pig_cache[organ] = (load_pig_forebrain() if organ == "Brain"
                            else t6.load_pig_tissue(organ))

    # ---- Cardoso: 7 species x 6 organs ----------------------------------
    print("\n[Cardoso] projecting 7 species x 6 organs (PLS 6-stage, postnatal)")
    for species in CARDOSO_SPECIES:
        print(f"  [load] {species} ...", flush=True)
        cardoso_df = t6._read_cardoso_matrix(species)
        ortho = t6.build_ortholog_map(species)
        for organ in CARDOSO_ORGANS:
            tgt = cardoso_target(cardoso_df, species, organ)
            if tgt is None:
                continue
            res = project_cell(pig_cache[organ], ortho, tgt[0], tgt[1])
            if res is None:
                continue
            scores, lvl, samples, rho, n_orth, kept = res
            dev_rank = rankdata(lvl, method="dense")
            for s, sc, lv, dr in zip(samples, scores, lvl, dev_rank):
                rows.append({"substrate": "Cardoso", "organ": organ,
                             "species": species, "sample": s,
                             "level": float(lv), "dev_rank": int(dr),
                             "score": float(sc)})
            ref = det_rho.get((organ, species), np.nan)
            checks.append((f"Cardoso {organ}/{species}", rho, ref))
            print(f"    {organ:7s} {species:8s} rho={rho:+.4f}  ref={ref:+.4f}  "
                  f"n_tgt={len(lvl)}  n_orth={n_orth}", flush=True)
        del cardoso_df

    # ---- dGTEx: human, AGECOHORT ordinal --------------------------------
    print("\n[dGTEx] projecting human organs (PLS 6-stage, AGECOHORT 1-4)")
    ortho_h = t6.build_ortholog_map("Human")
    for pig_tissue, smts, slug in DGTEX_SUBSTRATES:
        pig_X = (load_pig_forebrain() if pig_tissue == "Brain"
                 else t6.load_pig_tissue(pig_tissue))
        tgt_expr, tgt_level = t6.load_dgtex_target(smts, slug)
        res = project_cell(pig_X, ortho_h, tgt_expr, tgt_level)
        if res is None:
            print(f"    {pig_tissue:8s} -- not fittable")
            continue
        scores, lvl, samples, rho, n_orth, kept = res
        dev_rank = rankdata(lvl, method="dense")
        for s, sc, lv, dr in zip(samples, scores, lvl, dev_rank):
            rows.append({"substrate": "dGTEx", "organ": pig_tissue,
                         "species": "Human", "sample": s,
                         "level": float(lv), "dev_rank": int(dr),
                         "score": float(sc)})
        ref = dgtex_rho.get((pig_tissue, "Human"), np.nan)
        checks.append((f"dGTEx {pig_tissue}", rho, ref))
        print(f"    {pig_tissue:8s} rho={rho:+.4f}  ref={ref:+.4f}  "
              f"n_tgt={len(lvl)}  n_orth={n_orth}", flush=True)

    # ---- fidelity gate --------------------------------------------------
    print("\n[verify] recomputed rho vs committed reference (|drift| must be <= 1e-2)")
    worst = 0.0
    for cell, rho, ref in checks:
        if not np.isnan(ref):
            d = abs(rho - ref)
            worst = max(worst, d)
            flag = "" if d <= 1e-2 else "  <-- DRIFT"
            if flag:
                print(f"    {cell:24s} drift={d:.4f}{flag}")
    print(f"  worst |drift| across {len(checks)} cells = {worst:.4f}")
    if worst > 1e-2:
        raise SystemExit("ABORT: a cell's recomputed rho drifted from the "
                         "published value; the scatter would not match Fig 4.")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(out)} target samples across "
          f"{out.groupby(['substrate','organ','species']).ngroups} cells)")


if __name__ == "__main__":
    main()
