#!/usr/bin/env python3
"""Experiment A — Cross-species developmental-score transfer: PLS vs LightGBM.

This is the GATE experiment for the "linear-conservation" thesis. It asks a
single, sharp question:

    Does a pig-trained *linear* PLS developmental score transfer across species
    while a pig-trained *gradient-boosted tree* (LightGBM) does NOT, when both
    models see IDENTICAL orthologs and IDENTICAL independent z-scoring?

If yes, the only thing that differs between the two arms is linear-vs-tree, so a
collapse of the LightGBM arm (rho ~ 0 / fails the ortholog-scramble null) where
PLS holds is direct evidence that the conserved developmental signal is *linear*
in ortholog log-expression.

METHOD (implemented faithfully to the approved plan)
----------------------------------------------------
Pig side
    * Load data/pigGTEx/<Tissue>.expr_tpm.txt.gz (genes x samples) -> transpose
      to samples x genes; merge data/PigGTEx_v0.MetaTable.csv on BioSample for
      age-in-days + BioProject.
    * 6 stages by day boundaries [3,28,90,165,540]:
        Neonatal 0-3 | Suckling 4-28 | Juvenile 29-90 | Pre-pubertal 91-165 |
        Post-pubertal 166-540 | Adult >540
      drop-small <10 (DROP, never merge). Labels are ORDINAL across all kept
      stages (integer rank in developmental order).
    * Expression = log2(TPM + 1).
    * 5-stage scheme [28,90,165,540] is also computed for the staging-invariance
      Delta-rho readout.

Ortholog mapping (symbol-based 1:1)
    * From cached data/cardoso_moreira_2019/_pig_id_to_symbol.json and
      _<species>_id_to_symbol.json: keep symbols that appear EXACTLY ONCE in
      each species, build pig_geneid <-> target_geneid via the shared unique
      symbol. No network / Ensembl calls.

Fit + project (independent z-scoring, the only model-agnostic difference)
    * Restrict pig to the shared pig gene-ids (column order = the shared order).
    * Z-score EACH GENE INDEPENDENTLY on the pig matrix (pig-fit scaler) -> Zpig,
      and independently on the target matrix (target-fit scaler) -> Ztarget,
      with Ztarget columns ordered to match the shared pig-gene order.
    * Fit OrdinalPLS(n_components=10) and OrdinalLightGBM() on (Zpig, ordinal
      labels). Both `.fit(X, y)` take a samples x genes DataFrame.
    * Project onto the target using the SAME z-scored shared-ortholog matrices:
        - PLS continuous score   = pls.model_.predict(Ztarget).ravel()
          (the inner sklearn PLSRegression has scale=False, so feeding an
          already-z-scored matrix is exactly the "independent target-fit
          z-scoring" the Methods specify; the wrapper's pig-fit scaler is
          intentionally bypassed.)
        - LightGBM expected-stage = (lgb.predict_proba(Ztarget) * arange(K)).sum(1)
      Because both models consume the identical Zpig / Ztarget matrices, the
      ONLY difference between the two arms is linear-PLS vs tree-LightGBM.

Target ordinal developmental level
    * Cardoso columns are "<Organ>.<stage>.<rep>". Ordinal rank from the stage
      token: weeks-post-conception (#wpc, ordered numerically and embryonic)
      come before postnatal named stages
      (newborn < infant < toddler < school < youngTeenager < teenager <
       oldTeenager < youngAdult < youngMidAge < olderMidAge < senior < Senior).
      Embryonic e/E#.# tokens are supported for non-human species (the loader is
      species-parameterised) but do not occur in the human matrix.
    * dGTEx: AGECOHORT (1-4) from the Subject_Phenotypes xlsx joined to samples
      (load_dgtex) is the ordinal.
    * Transfer rho = Spearman(projected score, target ordinal level).

Ortholog-scramble null (1000x)
    * Permute which TARGET gene aligns to which pig weight (shuffle the shared
      column alignment of Ztarget), recompute the projected score + rho. The
      fitted model is reused; only the target column alignment is permuted, so
      the null is cheap. Empirical p = (#{|rho_null| >= |rho_obs|} + 1) / (N+1).
      Computed for BOTH models.

OUTPUTS
    paper/figures/data/transfer_6stage/expA_results.csv

The module is parameterised (TARGET_SPECIES_FILES, substrate tables) so it can
later extend to all 7 Cardoso species without code changes.

Author: Experiment A (linear-conservation gate). Real data only; seed=42.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
# Silence LightGBM's per-round early-stopping chatter (it floods stdout when we
# fit hundreds of Frank-Hall binary models across the scramble null).
logging.getLogger("lightgbm").setLevel(logging.ERROR)

# ----------------------------------------------------------------------------
# Paths + imports from the repo (run from repo root so these resolve)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # provenance dir: sibling modules

from machine_learning.model_training.models import OrdinalLightGBM, OrdinalPLS  # noqa: E402
from load_dgtex import load_sample_metadata  # noqa: E402

DATA = ROOT / "data"
PIG_META = DATA / "PigGTEx_v0.MetaTable.csv"
PIG_TPM_DIR = DATA / "pigGTEx"
CARDOSO_DIR = DATA / "cardoso_moreira_2019"
PIG_SYM_CACHE = CARDOSO_DIR / "_pig_id_to_symbol.json"

OUT_DIR = ROOT / "paper" / "figures" / "data" / "transfer_6stage"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "expA_results.csv"

SEED = 42
N_COMPONENTS = 10
MIN_PER_STAGE = 10
N_SCRAMBLE = 1000

# 6-stage and 5-stage day boundaries (upper-inclusive cut points)
ORDER6 = ["Neonatal", "Suckling", "Juvenile", "Pre-pubertal", "Post-pubertal", "Adult"]
ORDER5 = ["Infant", "Early childhood", "Pre-pubertal", "Post-pubertal", "Adult"]

# Cardoso target-species RPKM matrices (gene_id col, tab-delimited) + symbol map.
# Only Human is exercised in this gate run, but the table is here so the same
# code extends to the other six species verbatim.
TARGET_SPECIES_FILES: Dict[str, Dict[str, str]] = {
    "Human": {"rpkm": "Human.RPKM.tsv", "symbol_map": "_human_id_to_symbol.json"},
    "Macaque": {"rpkm": "Macaque_rpkm.txt", "symbol_map": "_macaque_id_to_symbol.json"},
    "Mouse": {"rpkm": "Mouse_rpkm.txt", "symbol_map": "_mouse_id_to_symbol.json"},
    "Rat": {"rpkm": "Rat_rpkm.txt", "symbol_map": "_rat_id_to_symbol.json"},
    "Rabbit": {"rpkm": "Rabbit_rpkm.txt", "symbol_map": "_rabbit_id_to_symbol.json"},
    "Opossum": {"rpkm": "Opossum_rpkm.txt", "symbol_map": "_opossum_id_to_symbol.json"},
    "Chicken": {"rpkm": "Chicken_rpkm.txt", "symbol_map": "_chicken_id_to_symbol.json"},
}

# Cardoso-Moreira substrates for this run: (pig_tissue, cardoso_organ).
CARDOSO_SUBSTRATES: List[Tuple[str, str]] = [
    ("Brain", "Brain"),
    ("Heart", "Heart"),
    ("Liver", "Liver"),
    ("Kidney", "Kidney"),
    ("Testis", "Testis"),
]

# dGTEx substrates: (pig_tissue, dgtex_SMTS_label, dgtex_file_slug).
DGTEX_SUBSTRATES: List[Tuple[str, str, str]] = [
    ("Lung", "Lung", "lung"),
    ("Muscle", "Muscle", "muscle"),
    ("Adipose", "Adipose Tissue", "adipose_tissue"),
]

# Published 5-stage PLS rho values for the validation sanity-check.
PUBLISHED_PLS_RHO = {
    ("dGTEx", "Lung"): 0.69,
    ("dGTEx", "Muscle"): 0.63,
    ("dGTEx", "Adipose"): 0.68,
    ("Cardoso-Human", "Brain"): 0.77,
    ("Cardoso-Human", "Heart"): 0.89,
    ("Cardoso-Human", "Liver"): 0.19,
    ("Cardoso-Human", "Kidney"): 0.63,
    ("Cardoso-Human", "Testis"): 0.98,
}


# ============================================================================
# FULL 6-STAGE TRANSFER MATRIX CONFIG (Fig 4 / Table S7 data layer)
# ============================================================================
# Cardoso-Moreira 2019 species divergence from pig (MYA). Eutherians that
# diverged from the pig lineage ~94 MYA (Boreoeutheria split); Opossum
# (marsupial) ~159 MYA; Chicken (sauropsid) ~319 MYA. (Same values as the
# committed panel_e_rho_vs_divergence.csv / determinants_tests.json.)
DIVERGENCE_MYA = {
    "Human": 94, "Macaque": 94, "Mouse": 94, "Rat": 94, "Rabbit": 94,
    "Opossum": 159, "Chicken": 319,
}
EUTHERIAN_SPECIES = ("Human", "Macaque", "Mouse", "Rat", "Rabbit")

# Cardoso organs (Fig 4 rows) and the pig source tissue each maps to. The pig
# PigGTEx tissue name equals the Cardoso organ column-prefix for all six.
CARDOSO_ORGANS = ["Brain", "Heart", "Kidney", "Liver", "Ovary", "Testis"]
CARDOSO_SPECIES = ["Human", "Macaque", "Mouse", "Rat", "Rabbit", "Opossum", "Chicken"]
# Ovary is ABSENT from the Human and Macaque Cardoso matrices -> n=5 species
# for that organ; every other organ has all 7.

# dGTEx organs for the supplementary table (matches supp_table_dGTEx_only.csv).
# (pig_tissue, dgtex_SMTS_label, dgtex_file_slug)
DGTEX_TABLE_SUBSTRATES: List[Tuple[str, str, str]] = [
    ("Lung",    "Lung",            "lung"),
    ("Muscle",  "Muscle",          "muscle"),
    ("Adipose", "Adipose Tissue",  "adipose_tissue"),
    ("Heart",   "Heart",           "heart"),
    ("Liver",   "Liver",           "liver"),
    ("Spleen",  "Spleen",          "spleen"),
    ("Testis",  "Testis",          "testis"),
]

# Committed Fig-4 panel-e values for the staging-robustness validation check.
PANEL_E_CSV = ROOT / "paper" / "figures" / "data" / "panel_e_rho_vs_divergence.csv"

MATRIX_OUT_CSV = OUT_DIR / "transfer_matrix_6stage.csv"
DETERMINANTS_OUT_JSON = OUT_DIR / "determinants_tests_6stage.json"
DGTEX_TABLE_OUT_CSV = OUT_DIR / "supp_table_dGTEx_only_6stage.csv"


# ----------------------------------------------------------------------------
# Pig loading + age parsing (idioms reused from
# review/analyses/cross_species_all_tissues.py and
# perinatal_split_eval.py (retired scratch))
# ----------------------------------------------------------------------------
def parse_age_days(age_str) -> float:
    """Parse '0 day', '20 days', '6 months', '2 years', '10 weeks' -> days."""
    if pd.isna(age_str) or str(age_str).strip().lower() == "unknown":
        return np.nan
    conv = {"day": 1, "days": 1, "week": 7, "weeks": 7,
            "month": 30, "months": 30, "year": 365, "years": 365}
    m = re.match(r"(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years)",
                 str(age_str).lower().strip())
    if not m:
        return np.nan
    return float(m.group(1)) * conv[m.group(2)]


def stage6(d: float) -> Optional[str]:
    """6-stage cut, boundaries [3,28,90,165,540]."""
    if pd.isna(d):
        return None
    if d <= 3:
        return "Neonatal"
    if d <= 28:
        return "Suckling"
    if d <= 90:
        return "Juvenile"
    if d <= 165:
        return "Pre-pubertal"
    if d <= 540:
        return "Post-pubertal"
    return "Adult"


def stage5(d: float) -> Optional[str]:
    """5-stage cut, boundaries [28,90,165,540] (Infant 0-28)."""
    if pd.isna(d):
        return None
    if d <= 28:
        return "Infant"
    if d <= 90:
        return "Early childhood"
    if d <= 165:
        return "Pre-pubertal"
    if d <= 540:
        return "Post-pubertal"
    return "Adult"


def load_pig_tissue(pig_tissue: str) -> pd.DataFrame:
    """Return samples x genes log2(TPM+1) DataFrame with 'Age_Days' + 'BioProject'.

    Uses the 3-column-style tissue filter (Tissue class == Sub categories ==
    pig_tissue), mirroring the published cross-species script.
    """
    meta = pd.read_csv(PIG_META).rename(columns={
        "BioSample": "Sample_ID",
        "Tissue class": "Tissue",
        "Sub categories": "Sub_categories",
    })
    meta = meta.set_index("Sample_ID")
    meta["Age_Days"] = meta["Age"].apply(parse_age_days)
    if pig_tissue == "Brain":
        FOREBRAIN = ["Brain", "Frontal cortex", "Cerebral cortex", "Frontal lobe",
                     "Cerebrum", "Hippocampus", "Amygdala", "Striatum region"]
        sub_meta = meta[meta["Tissue"].isin(FOREBRAIN) | meta["Sub_categories"].isin(FOREBRAIN)].copy()
    else:
        sub_meta = meta[(meta["Tissue"] == pig_tissue) &
                        (meta["Sub_categories"] == pig_tissue)].copy()

    tpm_path = PIG_TPM_DIR / f"{pig_tissue}.expr_tpm.txt.gz"
    with gzip.open(tpm_path, "rt") as fh:
        expr = pd.read_csv(fh, sep="\t", index_col=0)  # genes x samples

    samples = [s for s in sub_meta.index if s in expr.columns]
    expr = expr[samples]
    X = np.log2(expr.T + 1.0)  # samples x genes
    X["Age_Days"] = sub_meta.loc[samples, "Age_Days"].values
    X["BioProject"] = sub_meta.loc[samples, "BioProject"].astype(str).values
    return X


def assign_ordinal_stages(age_days: pd.Series, scheme: str) -> Tuple[pd.Series, List[str]]:
    """Map ages to ordinal stage labels for the given scheme, drop-small <10.

    Returns (string-stage Series aligned to age_days.index, kept-stage order).
    Samples whose stage is dropped (or NaN) become NaN in the returned Series.
    """
    order = ORDER6 if scheme == "6stage" else ORDER5
    cut = stage6 if scheme == "6stage" else stage5
    stage = age_days.apply(cut)
    counts = stage.value_counts()
    kept = [s for s in order if counts.get(s, 0) >= MIN_PER_STAGE]
    stage = stage.where(stage.isin(kept), other=np.nan)
    return stage, kept


# ----------------------------------------------------------------------------
# Ortholog mapping (symbol-based 1:1, cached maps only)
# ----------------------------------------------------------------------------
def _unique_symbol_to_id(id_to_symbol: Dict[str, str]) -> Dict[str, str]:
    """symbol(upper) -> gene_id, restricted to symbols that appear exactly once."""
    counts = Counter(s.upper() for s in id_to_symbol.values() if s)
    return {s.upper(): gid for gid, s in id_to_symbol.items()
            if s and counts[s.upper()] == 1}


def build_ortholog_map(target_species: str) -> pd.DataFrame:
    """Return DataFrame[symbol, pig_gene_id, target_gene_id] of 1:1 orthologs.

    A symbol is kept iff it is unique in BOTH the pig and the target symbol maps.
    """
    with open(PIG_SYM_CACHE) as f:
        pig_map = json.load(f)
    sym_file = CARDOSO_DIR / TARGET_SPECIES_FILES[target_species]["symbol_map"]
    with open(sym_file) as f:
        tgt_map = json.load(f)

    pig_sym2id = _unique_symbol_to_id(pig_map)
    tgt_sym2id = _unique_symbol_to_id(tgt_map)
    shared = sorted(set(pig_sym2id) & set(tgt_sym2id))
    return pd.DataFrame({
        "symbol": shared,
        "pig_gene_id": [pig_sym2id[s] for s in shared],
        "target_gene_id": [tgt_sym2id[s] for s in shared],
    })


# ----------------------------------------------------------------------------
# Target matrices + ordinal developmental level
# ----------------------------------------------------------------------------
# Postnatal stage order shared by all Cardoso species (developmental rank).
_CARDOSO_POSTNATAL_ORDER = [
    "newborn", "infant", "toddler", "school",
    "youngTeenager", "teenager", "oldTeenager",
    "youngAdult", "youngMidAge", "olderMidAge", "senior",
]


def cardoso_stage_rank(
    stage_token: str,
    species: str = "Human",
) -> Tuple[Optional[float], Optional[bool]]:
    """Ordinal developmental rank for a Cardoso stage token.

    Cardoso-Moreira 2019 encodes developmental time DIFFERENTLY per species:
      * Human   : weeks-post-conception '<n>wpc' (prenatal) then NAMED postnatal
                  stages (newborn < infant < ... < senior).
      * Mouse / Rat / Rabbit / Macaque / Chicken : embryonic days 'e<n>'
                  (prenatal) then postnatal days 'P<n>' (P0 = birth; ranked by n).
      * Opossum : a single bare-number 'days-post-conception' series (e.g. 13.5,
                  14, ... 194). Opossum is a marsupial born at ~13 d p.c., so the
                  ENTIRE sampled series (which starts at 13.5) is post-birth /
                  pouch development -> treated as postnatal, ranked by the day.

    Returns (rank, is_postnatal). ``rank`` is monotone in developmental time on a
    single scale (prenatal < 0 <= postnatal); only the WITHIN-species ordering
    matters because each (organ, species) cell is scored by its own Spearman.
    ``is_postnatal`` is True for postnatal tokens (named stages, 'P<n>', opossum
    bare days), False for embryonic 'e<n>' / '<n>wpc'. Both None if unrecognised.
    """
    tok = stage_token.strip()
    # --- prenatal: embryonic days e10.5 / E10.5 ------------------------------
    m = re.fullmatch(r"[eE](\d+(?:\.\d+)?)", tok)
    if m:
        return -1000.0 + float(m.group(1)), False
    # --- prenatal: weeks post conception 4wpc, 20wpc -------------------------
    m = re.fullmatch(r"(\d+(?:\.\d+)?)wpc", tok)
    if m:
        return -100.0 + float(m.group(1)), False
    # --- postnatal days: P0, P14, P3285 (mouse/rat/rabbit/macaque/chicken) ---
    m = re.fullmatch(r"[Pp](\d+(?:\.\d+)?)", tok)
    if m:
        return float(m.group(1)), True               # P-day count, all postnatal
    # --- Human named postnatal stages (case-insensitive) ---------------------
    low = tok.lower()
    order_low = [s.lower() for s in _CARDOSO_POSTNATAL_ORDER]
    if low in order_low:
        return float(order_low.index(low)), True      # 0..N
    # --- opossum bare-number days post conception (whole series is postnatal) -
    if species == "Opossum":
        m = re.fullmatch(r"(\d+(?:\.\d+)?)", tok)
        if m:
            return float(m.group(1)), True
    return None, None


def _read_cardoso_matrix(target_species: str) -> pd.DataFrame:
    """Load a Cardoso RPKM matrix robustly across the two on-disk formats.

    Human.RPKM.tsv is tab-delimited with a clean 'gene_id' header. The six
    non-human '<Species>_rpkm.txt' files are whitespace-delimited with EVERY
    field wrapped in literal double quotes (header 'Names' and all gene ids /
    column names). We strip the quotes from both the index and the columns so
    downstream code sees plain ENSEMBL ids and '<Organ>.<stage>.<rep>' columns.
    """
    path = CARDOSO_DIR / TARGET_SPECIES_FILES[target_species]["rpkm"]
    if path.suffix == ".tsv":
        df = pd.read_csv(path, sep="\t", index_col=0)
    else:
        df = pd.read_csv(path, sep=r"\s+", engine="python", index_col=0)
        df.index = df.index.astype(str).str.strip().str.strip('"')
        df.columns = [str(c).strip().strip('"') for c in df.columns]
    df.index.name = "gene_id"
    return df


def load_cardoso_target(
    target_species: str,
    organ: str,
    postnatal_only: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (genes x samples log2(RPKM+1) df, ordinal-level Series per sample).

    Columns kept are those matching '<organ>.<stage>.<rep>' with a recognised
    stage token. The ordinal level is the cardoso_stage_rank of the token.

    postnatal_only (default True): keep ONLY postnatal samples (newborn ->
        senior). This is required for a valid transfer: the pig model is trained
        exclusively on POSTNATAL pig samples (PigGTEx has no prenatal samples for
        these tissues; the youngest pig stage is Neonatal/0-3 d). Cardoso-Moreira
        matrices are dominated by prenatal samples (weeks-post-conception and, in
        non-human species, embryonic days) — typically 26-34 of ~40-53 columns
        per organ. Asking a postnatal-trained model to ORDER prenatal target
        samples is out-of-domain extrapolation and injects a large non-monotone
        component that spuriously deflates / sign-flips the Spearman rho (e.g.
        Cardoso-Human Liver goes from rho=+0.28 postnatal to rho=-0.13 with
        prenatal included; Testis +0.86 -> +0.48). Restricting to postnatal
        aligns the target developmental axis with the model's training domain and
        reproduces the published postnatal transfer rho (which used postnatal
        n_target = 9-15 per human organ). Set False only for prenatal-inclusive
        diagnostics.
    """
    df = _read_cardoso_matrix(target_species)
    cols = [c for c in df.columns if c.split(".")[0] == organ]
    ranks: Dict[str, float] = {}
    keep_cols: List[str] = []
    for c in cols:
        parts = c.split(".")
        if len(parts) < 2:
            continue
        # The stage token may itself contain a dot ('e10.5', opossum '13.5'):
        # it is everything between the organ prefix and the trailing replicate
        # index. '<Organ>.<stage>.<rep>' -> parts[1:-1]; '<Organ>.<stage>' (no
        # rep) -> parts[1].
        stage_token = ".".join(parts[1:-1]) if len(parts) >= 3 else parts[1]
        r, is_post = cardoso_stage_rank(stage_token, species=target_species)
        if r is None:
            continue
        if postnatal_only and not is_post:
            continue
        ranks[c] = r
        keep_cols.append(c)
    sub = np.log2(df[keep_cols] + 1.0)  # genes x samples
    level = pd.Series({c: ranks[c] for c in keep_cols})
    return sub, level


def load_dgtex_target(smts_label: str, file_slug: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (genes x samples log2(TPM+1) df, AGECOHORT ordinal Series).

    Pools all sub-regions under the SMTS umbrella; AGECOHORT (1-4) is the
    ordinal developmental level. Reuses load_dgtex idioms but keeps
    ALL AGECOHORT cohorts (1-4), not just the young/old binning.
    """
    expr_path = (DATA / "dgtex" / "expression" /
                 f"gene_tpm_dgtex_v1_{file_slug}.gct.gz")
    expr = pd.read_csv(expr_path, sep="\t", skiprows=2, compression="gzip")
    expr = expr.rename(columns={"Name": "ensg_versioned", "Description": "symbol"})
    expr["ensg"] = expr["ensg_versioned"].str.replace(r"\.\d+$", "", regex=True)
    expr = expr.drop_duplicates(subset="ensg", keep="first").set_index("ensg")
    sample_cols = [c for c in expr.columns
                   if c not in ("ensg_versioned", "symbol")]
    expr = expr[sample_cols]

    meta = load_sample_metadata()
    meta_t = meta[(meta["SMTS"] == smts_label) & meta["AGECOHORT"].notna()].copy()
    samples = [s for s in meta_t["SAMPLE_ID"] if s in expr.columns]
    sub = np.log2(expr[samples] + 1.0)  # genes x samples
    level = pd.Series(
        meta_t.set_index("SAMPLE_ID").loc[samples, "AGECOHORT"].astype(float).values,
        index=samples,
    )
    return sub, level


# ----------------------------------------------------------------------------
# Core transfer computation
# ----------------------------------------------------------------------------
def _zscore_fit(M: np.ndarray) -> np.ndarray:
    """Per-column (per-gene) z-score; constant genes -> scale 1 (=> all zeros)."""
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (M - mu) / sd


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(stats.spearmanr(a, b).correlation)


def run_cell_pls_only(
    pig_X: pd.DataFrame,
    ortho: pd.DataFrame,
    target_expr: pd.DataFrame,
    target_level: pd.Series,
    scheme: str = "6stage",
    do_scramble: bool = True,
) -> Optional[dict]:
    """PLS-ONLY transfer for one (pig tissue -> target) cell.

    Identical PLS pipeline to ``run_cell`` (same 6-stage labels, same independent
    z-scoring of the shared-ortholog matrices, same projection via the inner
    sklearn estimator, same 1000x ortholog-scramble null) but it does NOT fit
    LightGBM. This is the data-layer path for Fig 4 / Table S7, where only the
    linear PLS developmental score is needed; skipping the (slow) tree fit +
    tree-scramble makes the full 47-cell matrix tractable. Returns
    {rho_pls, scramble_p_pls, n_orthologs, n_target_samples, n_pig_samples,
     kept_stages, K} or None if not fittable.
    """
    age = pig_X["Age_Days"]
    stage, kept = assign_ordinal_stages(age, scheme)
    if len(kept) < 2:
        return None
    code = {s: i for i, s in enumerate(kept)}
    mask = stage.notna().values
    pig_feat = pig_X.drop(columns=["Age_Days", "BioProject"]).loc[mask]
    y = np.array([code[s] for s in stage[mask].values])
    K = len(kept)

    o = ortho[ortho["pig_gene_id"].isin(pig_feat.columns) &
              ortho["target_gene_id"].isin(target_expr.index)].copy()
    o = o.drop_duplicates("pig_gene_id").drop_duplicates("target_gene_id")
    if len(o) < 10:
        return None
    pig_ids = o["pig_gene_id"].tolist()
    tgt_ids = o["target_gene_id"].tolist()

    Zpig = _zscore_fit(pig_feat[pig_ids].values.astype(np.float64))
    Zpig_df = pd.DataFrame(Zpig, columns=pig_ids)
    Ztgt = _zscore_fit(target_expr.loc[tgt_ids].T.values.astype(np.float64))
    lvl = target_level.loc[target_expr.columns].values.astype(float)

    pls = OrdinalPLS(n_components=N_COMPONENTS, seed=SEED)
    pls.fit(Zpig_df, y)
    rho = _spearman(pls.model_.predict(Ztgt).ravel(), lvl)

    result = {
        "rho_pls": rho,
        "n_orthologs": int(len(o)),
        "n_target_samples": int(len(lvl)),
        "n_pig_samples": int(mask.sum()),
        "kept_stages": ";".join(kept),
        "K": K,
    }
    if do_scramble:
        rng = np.random.default_rng(SEED)
        n_genes = Ztgt.shape[1]
        ge = 0
        for _ in range(N_SCRAMBLE):
            perm = rng.permutation(n_genes)
            r = _spearman(pls.model_.predict(Ztgt[:, perm]).ravel(), lvl)
            if not np.isnan(rho) and not np.isnan(r) and abs(r) >= abs(rho):
                ge += 1
        result["scramble_p_pls"] = (np.nan if np.isnan(rho)
                                    else (ge + 1) / (N_SCRAMBLE + 1))
    return result


def run_cell(
    pig_X: pd.DataFrame,
    ortho: pd.DataFrame,
    target_expr: pd.DataFrame,
    target_level: pd.Series,
    scheme: str,
    do_scramble: bool,
) -> Optional[dict]:
    """Run one (pig tissue, target organ) transfer for one staging scheme.

    Returns a dict of PLS/LightGBM rho (+ scramble p when requested) and sizes,
    or None if the cell is not fittable (e.g. < 2 kept stages).
    """
    # --- pig labels (ordinal over kept stages) -------------------------------
    age = pig_X["Age_Days"]
    stage, kept = assign_ordinal_stages(age, scheme)
    if len(kept) < 2:
        return None
    code = {s: i for i, s in enumerate(kept)}
    mask = stage.notna().values
    pig_feat = pig_X.drop(columns=["Age_Days", "BioProject"]).loc[mask]
    y = np.array([code[s] for s in stage[mask].values])
    K = len(kept)

    # --- shared 1:1 orthologs present in BOTH matrices -----------------------
    o = ortho[ortho["pig_gene_id"].isin(pig_feat.columns) &
              ortho["target_gene_id"].isin(target_expr.index)].copy()
    o = o.drop_duplicates("pig_gene_id").drop_duplicates("target_gene_id")
    if len(o) < 10:
        return None
    pig_ids = o["pig_gene_id"].tolist()
    tgt_ids = o["target_gene_id"].tolist()  # column-ordered to match pig order

    # --- build the IDENTICAL z-scored shared-ortholog matrices ---------------
    # Pig fit matrix: samples x shared-genes, independently z-scored on pig.
    Xpig = pig_feat[pig_ids].values.astype(np.float64)
    Zpig = _zscore_fit(Xpig)
    Zpig_df = pd.DataFrame(Zpig, columns=pig_ids)

    # Target matrix: samples x shared-genes (same gene order), independently
    # z-scored on the target.
    Xtgt = target_expr.loc[tgt_ids].T.values.astype(np.float64)  # samples x genes
    Ztgt = _zscore_fit(Xtgt)
    lvl = target_level.loc[target_expr.columns].values.astype(float)

    # --- fit both models on the SAME (Zpig, y) -------------------------------
    pls = OrdinalPLS(n_components=N_COMPONENTS, seed=SEED)
    pls.fit(Zpig_df, y)
    lgb = OrdinalLightGBM(seed=SEED)
    lgb.fit(Zpig_df, y)

    # --- project onto target (PLS bypasses wrapper's pig scaler) -------------
    pls_score = pls.model_.predict(Ztgt).ravel()
    lgb_score = (lgb.predict_proba(Ztgt) * np.arange(K)).sum(axis=1)

    rho_pls = _spearman(pls_score, lvl)
    rho_lgb = _spearman(lgb_score, lvl)

    result = {
        "scheme": scheme,
        "n_orthologs": int(len(o)),
        "n_target_samples": int(len(lvl)),
        "n_pig_samples": int(mask.sum()),
        "kept_stages": ";".join(kept),
        "K": K,
        "rho_pls": rho_pls,
        "rho_lgb": rho_lgb,
    }

    # --- ortholog-scramble null (permute target column alignment) ------------
    # An UNDEFINED observed rho (nan) means the projected score is constant
    # across all target samples, i.e. the model produces ZERO discriminative
    # transfer. That is the strongest possible collapse, NOT a significant
    # result, so its scramble-p is fixed to 1.0 rather than estimated against an
    # also-degenerate null (where the permuted scores are likewise constant).
    if do_scramble:
        rng = np.random.default_rng(SEED)
        n_genes = Ztgt.shape[1]
        ge_pls = 0
        ge_lgb = 0
        for _ in range(N_SCRAMBLE):
            perm = rng.permutation(n_genes)
            Zp = Ztgt[:, perm]  # target genes shuffled vs fitted pig weights
            s_pls = pls.model_.predict(Zp).ravel()
            s_lgb = (lgb.predict_proba(Zp) * np.arange(K)).sum(axis=1)
            r_pls = _spearman(s_pls, lvl)
            r_lgb = _spearman(s_lgb, lvl)
            if not np.isnan(rho_pls) and not np.isnan(r_pls) and abs(r_pls) >= abs(rho_pls):
                ge_pls += 1
            if not np.isnan(rho_lgb) and not np.isnan(r_lgb) and abs(r_lgb) >= abs(rho_lgb):
                ge_lgb += 1
        result["scramble_p_pls"] = (
            np.nan if np.isnan(rho_pls) else (ge_pls + 1) / (N_SCRAMBLE + 1))
        result["scramble_p_lgb"] = (
            np.nan if np.isnan(rho_lgb) else (ge_lgb + 1) / (N_SCRAMBLE + 1))
    return result


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print("Experiment A — cross-species transfer: PLS vs LightGBM (linear gate)")
    print("=" * 78)

    rows: List[dict] = []

    # cache pig tissue matrices (loaded once each)
    pig_cache: Dict[str, pd.DataFrame] = {}

    def get_pig(tissue: str) -> pd.DataFrame:
        if tissue not in pig_cache:
            print(f"  [pig] loading {tissue} ...", flush=True)
            pig_cache[tissue] = load_pig_tissue(tissue)
        return pig_cache[tissue]

    # ---- dGTEx substrates (pig -> human dGTEx) ------------------------------
    print("\n[dGTEx substrates]")
    for pig_tissue, smts, slug in DGTEX_SUBSTRATES:
        substrate = "dGTEx"
        print(f"\n--- {substrate}: pig {pig_tissue} -> {smts} ---", flush=True)
        ortho = build_ortholog_map("Human")  # dGTEx uses human ENSG ids
        pig_X = get_pig(pig_tissue)
        tgt_expr, tgt_level = load_dgtex_target(smts, slug)

        r6 = run_cell(pig_X, ortho, tgt_expr, tgt_level, "6stage", do_scramble=True)
        r5 = run_cell(pig_X, ortho, tgt_expr, tgt_level, "5stage", do_scramble=False)
        _emit(rows, substrate, pig_tissue, r6, r5)

    # ---- Cardoso-Human substrates ------------------------------------------
    print("\n[Cardoso-Human substrates]")
    ortho_h = build_ortholog_map("Human")
    for pig_tissue, organ in CARDOSO_SUBSTRATES:
        substrate = "Cardoso-Human"
        print(f"\n--- {substrate}: pig {pig_tissue} -> {organ} ---", flush=True)
        pig_X = get_pig(pig_tissue)
        tgt_expr, tgt_level = load_cardoso_target("Human", organ)

        r6 = run_cell(pig_X, ortho_h, tgt_expr, tgt_level, "6stage", do_scramble=True)
        r5 = run_cell(pig_X, ortho_h, tgt_expr, tgt_level, "5stage", do_scramble=False)
        _emit(rows, substrate, pig_tissue, r6, r5)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(out)} cells)")
    print("\n" + out.to_string(index=False))


def _emit(rows: List[dict], substrate: str, organ_label: str,
          r6: Optional[dict], r5: Optional[dict]) -> None:
    """Merge the 6-stage (with scramble) and 5-stage (rho only) results into a
    single output row and print a live summary line."""
    if r6 is None:
        print(f"    SKIP (not fittable: <2 kept stages or <10 orthologs)")
        rows.append({"substrate": substrate, "organ": organ_label,
                     "fittable": False})
        return
    rho5 = r5["rho_pls"] if r5 is not None else np.nan
    pub = PUBLISHED_PLS_RHO.get((substrate, organ_label), np.nan)
    row = {
        "substrate": substrate,
        "organ": organ_label,
        "fittable": True,
        "n_orthologs": r6["n_orthologs"],
        "n_target_samples": r6["n_target_samples"],
        "n_pig_samples": r6["n_pig_samples"],
        "kept_stages_6": r6["kept_stages"],
        "K6": r6["K"],
        "rho_pls_6stage": round(r6["rho_pls"], 4),
        "rho_lgb_6stage": round(r6["rho_lgb"], 4),
        "scramble_p_pls": round(r6["scramble_p_pls"], 4),
        "scramble_p_lgb": round(r6["scramble_p_lgb"], 4),
        "rho_pls_5stage": round(rho5, 4) if not np.isnan(rho5) else np.nan,
        "delta_rho_5v6_pls": (round(r6["rho_pls"] - rho5, 4)
                              if not np.isnan(rho5) else np.nan),
        "published_pls_rho_5stage": pub,
    }
    rows.append(row)
    print(f"    n_orthologs={row['n_orthologs']}  n_target={row['n_target_samples']}  "
          f"kept6=[{row['kept_stages_6']}]")
    print(f"    PLS rho(6)={row['rho_pls_6stage']:+.3f} (scramble p={row['scramble_p_pls']:.3f})  "
          f"|  LGB rho(6)={row['rho_lgb_6stage']:+.3f} (scramble p={row['scramble_p_lgb']:.3f})")
    print(f"    PLS rho(5)={row['rho_pls_5stage']}  published(5)={pub}  "
          f"Δrho(5->6)={row['delta_rho_5v6_pls']}")


# ============================================================================
# EXPERIMENT B — the nonlinearity sweep (decisive aggregate test)
# ============================================================================
# A ladder of models of INCREASING nonlinearity is run on two axes:
#   A. within-pig accuracy  (pig Muscle, the only group-identifiable tissue):
#        grouped CV (StratifiedGroupKFold on BioProject), 6-stage, 5 seeds,
#        mean +/- SD balanced accuracy. Uses the native HPC-benchmark
#        preprocessing path (DataLoader 3-column tissue filter; log2 TPM) so
#        the numbers land on the published anchors (PLS ~ 0.43, LGB-unlimited
#        ~ 0.54).
#   B. cross-species transfer rho (+ scramble p), computed EXACTLY as Exp A:
#        every model consumes the IDENTICAL z-scored shared-ortholog matrices
#        (Zpig to fit, Ztgt to project), so the ONLY thing that changes along
#        the ladder is model nonlinearity.
#
# Ladder (linear -> unlimited tree):
#   pls            OrdinalPLS(n_components=10)          linear (latent)
#   ridge          OrdinalRidge                         linear (ridge)
#   lgb_depth1     OrdinalLightGBM(max_depth=1)         additive stumps (no interactions)
#   lgb_depth2     OrdinalLightGBM(max_depth=2)         depth-2 interactions
#   lgb_depth4     OrdinalLightGBM(max_depth=4)         depth-4 interactions
#   lgb_unlimited  OrdinalLightGBM(max_depth=-1)        unlimited (= Exp A tree)
#
# Well-powered transfer cells (the aggregate set): dGTEx {Lung, Muscle,
# Adipose} + Cardoso-Human {Heart, Testis}. Mean transfer rho is reported over
# all 5 cells AND ex-Muscle (4 cells), since Muscle is the Exp A exception.

from machine_learning.model_training.models import OrdinalRidge  # noqa: E402

# (model_key, kind, builder).  kind drives the projection:
#   "linear" -> continuous score = model.model_.predict(Ztgt).ravel()
#               (PLS bypasses its wrapper scaler; Ridge has no internal scaler,
#                so feeding the target-fit-z-scored Ztgt is symmetric & correct)
#   "tree"   -> expected stage   = (model.predict_proba(Ztgt) * arange(K)).sum(1)
LADDER: List[Tuple[str, str, "callable"]] = [
    ("pls",           "linear", lambda s: OrdinalPLS(n_components=N_COMPONENTS, seed=s)),
    ("ridge",         "linear", lambda s: OrdinalRidge(seed=s)),
    ("lgb_depth1",    "tree",   lambda s: OrdinalLightGBM(max_depth=1, seed=s)),
    ("lgb_depth2",    "tree",   lambda s: OrdinalLightGBM(max_depth=2, seed=s)),
    ("lgb_depth4",    "tree",   lambda s: OrdinalLightGBM(max_depth=4, seed=s)),
    ("lgb_unlimited", "tree",   lambda s: OrdinalLightGBM(max_depth=-1, seed=s)),
]

# Well-powered transfer cells: (substrate, pig_tissue, target_loader_spec).
EXPB_CELLS: List[Tuple[str, str, dict]] = [
    ("dGTEx",         "Lung",    {"kind": "dgtex", "smts": "Lung",           "slug": "lung"}),
    ("dGTEx",         "Muscle",  {"kind": "dgtex", "smts": "Muscle",         "slug": "muscle"}),
    ("dGTEx",         "Adipose", {"kind": "dgtex", "smts": "Adipose Tissue", "slug": "adipose_tissue"}),
    ("Cardoso-Human", "Heart",   {"kind": "cardoso", "organ": "Heart"}),
    ("Cardoso-Human", "Testis",  {"kind": "cardoso", "organ": "Testis"}),
]

EXPB_SCHEME = "6stage"
EXPB_OUT_CSV = OUT_DIR / "expB_sweep.csv"

# within-pig axis config
WITHIN_TISSUE = "Muscle"
WITHIN_SEEDS = [42, 43, 44, 45, 46]
WITHIN_NFOLDS = 5


def _project(model, kind: str, Ztgt: np.ndarray, K: int) -> np.ndarray:
    """Continuous transfer score for a fitted model on the z-scored target."""
    if kind == "linear":
        return np.asarray(model.model_.predict(Ztgt)).ravel()
    return (model.predict_proba(Ztgt) * np.arange(K)).sum(axis=1)


def transfer_rho_all_models(
    pig_X: pd.DataFrame,
    ortho: pd.DataFrame,
    target_expr: pd.DataFrame,
    target_level: pd.Series,
    do_scramble: bool,
) -> Optional[Dict[str, dict]]:
    """Fit EVERY ladder model on the same Zpig and project to the same Ztgt.

    Returns {model_key: {rho, scramble_p, n_orthologs, n_target_samples, K}}.
    The z-scored shared-ortholog matrices are built ONCE and shared across all
    models, so the only difference between models is the model itself (this is
    the Exp A invariant, extended to the whole ladder).
    """
    age = pig_X["Age_Days"]
    stage, kept = assign_ordinal_stages(age, EXPB_SCHEME)
    if len(kept) < 2:
        return None
    code = {s: i for i, s in enumerate(kept)}
    mask = stage.notna().values
    pig_feat = pig_X.drop(columns=["Age_Days", "BioProject"]).loc[mask]
    y = np.array([code[s] for s in stage[mask].values])
    K = len(kept)

    o = ortho[ortho["pig_gene_id"].isin(pig_feat.columns) &
              ortho["target_gene_id"].isin(target_expr.index)].copy()
    o = o.drop_duplicates("pig_gene_id").drop_duplicates("target_gene_id")
    if len(o) < 10:
        return None
    pig_ids = o["pig_gene_id"].tolist()
    tgt_ids = o["target_gene_id"].tolist()

    Zpig = _zscore_fit(pig_feat[pig_ids].values.astype(np.float64))
    Zpig_df = pd.DataFrame(Zpig, columns=pig_ids)
    Ztgt = _zscore_fit(target_expr.loc[tgt_ids].T.values.astype(np.float64))
    lvl = target_level.loc[target_expr.columns].values.astype(float)

    out: Dict[str, dict] = {}
    for key, kind, build in LADDER:
        model = build(SEED)
        model.fit(Zpig_df, y)
        score = _project(model, kind, Ztgt, K)
        rho = _spearman(score, lvl)
        rec = {"rho": rho, "n_orthologs": int(len(o)),
               "n_target_samples": int(len(lvl)), "K": K, "kind": kind}
        if do_scramble:
            rng = np.random.default_rng(SEED)
            n_genes = Ztgt.shape[1]
            ge = 0
            for _ in range(N_SCRAMBLE):
                perm = rng.permutation(n_genes)
                s = _project(model, kind, Ztgt[:, perm], K)
                r = _spearman(s, lvl)
                if not np.isnan(rho) and not np.isnan(r) and abs(r) >= abs(rho):
                    ge += 1
            rec["scramble_p"] = (np.nan if np.isnan(rho)
                                 else (ge + 1) / (N_SCRAMBLE + 1))
        out[key] = rec
    return out


def within_pig_muscle_ba() -> Dict[str, dict]:
    """Within-pig Muscle 6-stage grouped-CV balanced accuracy for every ladder
    model (mean +/- SD over WITHIN_SEEDS). Uses the HPC-benchmark preprocessing
    path (DataLoader 3-column tissue filter) so numbers match the published
    anchors. Returns {model_key: {ba_mean, ba_sd, per_seed, n, n_bioprojects,
    kept}}.
    """
    from machine_learning.data_processing.data_loader import DataLoader
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import balanced_accuracy_score

    dl = DataLoader(data_dir=PIG_TPM_DIR, metadata_path=PIG_META)
    dl.load_metadata()
    expr, meta = dl.load_expression(WITHIN_TISSUE)  # genes x samples, 3-col filter
    X = np.log2(expr.T.values.astype(np.float64) + 1.0)
    age = pd.to_numeric(meta["Age_Days"] if "Age_Days" in meta else meta["Age"],
                        errors="coerce").reset_index(drop=True)
    bp = meta["BioProject"].astype(str).reset_index(drop=True)
    y = age.apply(stage6)
    keep = age.notna().values
    X = X[keep]; y = y[keep].reset_index(drop=True); bp = bp[keep].reset_index(drop=True)
    counts = y.value_counts()
    kept = [s for s in ORDER6 if counts.get(s, 0) >= MIN_PER_STAGE]
    mask = y.isin(kept).values
    X = X[mask]
    code = {s: i for i, s in enumerate(kept)}
    yi = np.array([code[v] for v in y[mask].values])
    gk = bp[mask].values
    cols = [f"g{i}" for i in range(X.shape[1])]  # str names (LightGBM rejects int)
    n_bp = int(pd.unique(gk).size)
    kept_counts = {s: int((y[mask] == s).sum()) for s in kept}
    print(f"  [within-pig] Muscle n={int(mask.sum())} BioProjects={n_bp} "
          f"kept6={kept_counts}", flush=True)

    out: Dict[str, dict] = {}
    for key, _kind, build in LADDER:
        per_seed = []
        for seed in WITHIN_SEEDS:
            ng = len(np.unique(gk))
            nf = min(WITHIN_NFOLDS, ng)
            sk = StratifiedGroupKFold(n_splits=nf, shuffle=True, random_state=seed)
            fold_ba = []
            for tr, te in sk.split(X, yi, groups=gk):
                if len(np.unique(yi[tr])) < 2:
                    continue
                m = build(seed)
                m.fit(pd.DataFrame(X[tr], columns=cols), yi[tr])
                pred = np.asarray(m.predict(pd.DataFrame(X[te], columns=cols))).astype(int)
                fold_ba.append(balanced_accuracy_score(yi[te], pred))
            if fold_ba:
                per_seed.append(float(np.mean(fold_ba)))
        out[key] = {
            "ba_mean": float(np.mean(per_seed)) if per_seed else np.nan,
            "ba_sd": float(np.std(per_seed)) if per_seed else np.nan,
            "per_seed": [round(b, 4) for b in per_seed],
            "n": int(mask.sum()), "n_bioprojects": n_bp, "kept": kept_counts,
        }
        print(f"  [within-pig] {key:14s} grouped 6-stage BA = "
              f"{out[key]['ba_mean']:.3f} ± {out[key]['ba_sd']:.3f}", flush=True)
    return out


def main_expB() -> None:
    print("=" * 90)
    print("Experiment B — nonlinearity sweep (within-pig BA vs cross-species transfer rho)")
    print("=" * 90)

    # ---- Axis A: within-pig Muscle grouped-CV BA --------------------------
    print("\n[Axis A] within-pig Muscle balanced accuracy (grouped CV, 6-stage, 5 seeds)")
    within = within_pig_muscle_ba()

    # ---- Axis B: cross-species transfer rho per cell ----------------------
    print("\n[Axis B] cross-species transfer rho per well-powered cell (1000x scramble)")
    pig_cache: Dict[str, pd.DataFrame] = {}

    def get_pig(t: str) -> pd.DataFrame:
        if t not in pig_cache:
            print(f"  [pig] loading {t} ...", flush=True)
            pig_cache[t] = load_pig_tissue(t)
        return pig_cache[t]

    ortho_h = build_ortholog_map("Human")
    # transfer[model_key][cell_label] = {"rho":..., "scramble_p":...}
    transfer: Dict[str, Dict[str, dict]] = {k: {} for k, _, _ in LADDER}
    cell_meta: Dict[str, dict] = {}
    for substrate, pig_tissue, spec in EXPB_CELLS:
        cell = f"{substrate}:{pig_tissue}"
        print(f"\n--- {cell} ---", flush=True)
        pig_X = get_pig(pig_tissue)
        if spec["kind"] == "dgtex":
            tgt_expr, tgt_level = load_dgtex_target(spec["smts"], spec["slug"])
        else:
            tgt_expr, tgt_level = load_cardoso_target("Human", spec["organ"])
        res = transfer_rho_all_models(pig_X, ortho_h, tgt_expr, tgt_level,
                                      do_scramble=True)
        any_key = next(iter(res))
        cell_meta[cell] = {"n_orthologs": res[any_key]["n_orthologs"],
                           "n_target_samples": res[any_key]["n_target_samples"],
                           "K": res[any_key]["K"]}
        for key in transfer:
            transfer[key][cell] = {"rho": res[key]["rho"],
                                   "scramble_p": res[key]["scramble_p"]}
            r = res[key]["rho"]; p = res[key]["scramble_p"]
            print(f"    {key:14s} rho={r:+.3f}  p={p:.3f}" if not np.isnan(r)
                  else f"    {key:14s} rho=  nan  p=  nan", flush=True)

    # ---- assemble the sweep table -----------------------------------------
    cells_all = [f"{s}:{t}" for s, t, _ in EXPB_CELLS]
    cells_exmuscle = [c for c in cells_all if not c.endswith(":Muscle")]
    rows = []
    for key, kind, _ in LADDER:
        rhos_all = np.array([transfer[key][c]["rho"] for c in cells_all], dtype=float)
        rhos_exm = np.array([transfer[key][c]["rho"] for c in cells_exmuscle], dtype=float)
        # Two honest aggregations:
        #   *_defined : mean over cells with a DEFINED rho (nan = constant
        #               score = collapse, skipped). n_cells_defined says how
        #               many cells the mean spans.
        #   *_nan0    : nan treated as 0 (a collapsed/constant score contributes
        #               zero transfer), so the mean is over ALL cells. This is
        #               the conservative reading and is the headline number.
        row = {
            "model": key,
            "kind": kind,
            "muscle_BA_mean": round(within[key]["ba_mean"], 4),
            "muscle_BA_sd": round(within[key]["ba_sd"], 4),
            "mean_rho_all5": round(float(np.nanmean(rhos_all)), 4),
            "mean_rho_exMuscle4": round(float(np.nanmean(rhos_exm)), 4),
            "n_cells_defined_all5": int(np.isfinite(rhos_all).sum()),
            "n_cells_defined_exM4": int(np.isfinite(rhos_exm).sum()),
            "mean_rho_all5_nan0": round(float(np.nan_to_num(rhos_all, nan=0.0).mean()), 4),
            "mean_rho_exMuscle4_nan0": round(float(np.nan_to_num(rhos_exm, nan=0.0).mean()), 4),
        }
        for c in cells_all:
            tag = c.replace("Cardoso-Human:", "Card_").replace("dGTEx:", "dG_")
            row[f"rho_{tag}"] = round(transfer[key][c]["rho"], 4) if not np.isnan(transfer[key][c]["rho"]) else np.nan
            row[f"p_{tag}"] = round(transfer[key][c]["scramble_p"], 4) if not np.isnan(transfer[key][c]["scramble_p"]) else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(EXPB_OUT_CSV, index=False)
    print(f"\nWrote {EXPB_OUT_CSV}  ({len(df)} models)")
    print("\ncell metadata:")
    for c, m in cell_meta.items():
        print(f"  {c:24s} n_orth={m['n_orthologs']} n_target={m['n_target_samples']} K={m['K']}")
    print("\n" + df.to_string(index=False))


# ============================================================================
# FULL 6-STAGE TRANSFER MATRIX (Fig 4 / Table S7 data layer; PLS only)
# ============================================================================
def pig_expressed_ortholog_count(pig_X: pd.DataFrame) -> int:
    """Pig per-tissue expressed-gene breadth = #genes with median TPM >= 1.

    Computed on the log2(TPM+1) feature matrix: median across samples per gene,
    expressed iff median log2(TPM+1) >= 1 (i.e. median TPM >= 1). Matches the
    committed determinants_tests.json H2_pig_breadth proxy (Liver/Testis/Kidney
    reproduce exactly).
    """
    feat = pig_X.drop(columns=["Age_Days", "BioProject"])
    med = feat.median(axis=0)
    return int((med >= 1.0).sum())


def cardoso_cell_from_matrix(
    pig_X: pd.DataFrame,
    species: str,
    organ: str,
    cardoso_df: pd.DataFrame,
    ortho: pd.DataFrame,
) -> Optional[dict]:
    """PLS 6-stage transfer for one (pig organ -> Cardoso species/organ) cell,
    using a PRE-LOADED Cardoso matrix + ortholog map (so each large species file
    is read only once across its six organs).

    Returns the per-cell record or None if the organ is absent for that species
    (e.g. Ovary in Human/Macaque) / not fittable.
    """
    organ_cols = [c for c in cardoso_df.columns if c.split(".")[0] == organ]
    if not organ_cols:
        return None  # organ absent for this species
    # build the postnatal-only (genes x samples) submatrix + ordinal level
    ranks: Dict[str, float] = {}
    keep: List[str] = []
    for c in organ_cols:
        parts = c.split(".")
        stage_token = ".".join(parts[1:-1]) if len(parts) >= 3 else parts[1]
        r, is_post = cardoso_stage_rank(stage_token, species=species)
        if r is not None and is_post:
            ranks[c] = r
            keep.append(c)
    if len(keep) < 3:
        return None
    tgt_expr = np.log2(cardoso_df[keep] + 1.0)
    tgt_level = pd.Series(ranks)
    res = run_cell_pls_only(pig_X, ortho, tgt_expr, tgt_level, "6stage",
                            do_scramble=True)
    if res is None:
        return None
    return {
        "organ": organ,
        "species": species,
        "divergence_mya": DIVERGENCE_MYA[species],
        "rho": res["rho_pls"],
        "scramble_p": res["scramble_p_pls"],
        "n_orth": res["n_orthologs"],
        "n_target": res["n_target_samples"],
        "n_pig": res["n_pig_samples"],
        "kept_pig_stages": res["kept_stages"],
        "K": res["K"],
    }


def _spearman_p(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    """Spearman r, p, n over finite pairs."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), float("nan"), int(m.sum())
    r = stats.spearmanr(x[m], y[m])
    return float(r.correlation), float(r.pvalue), int(m.sum())


def compute_determinants(matrix: pd.DataFrame, breadth: Dict[str, int]) -> dict:
    """Compute H1/H2/H3 + variance partition over the Cardoso transfer cells.

    Mirrors the committed determinants_tests.json schema and adds the variance
    partition (OLS rho ~ C(organ) + divergence -> unique R^2 of each term).
    `matrix` is the per-cell Cardoso table; only rows with a finite rho are used.
    """
    d = matrix[matrix["rho"].notna()].copy()
    out: dict = {}

    # ---- H1: rho vs divergence MYA --------------------------------------
    r, p, n = _spearman_p(d["rho"], d["divergence_mya"])
    out["H1_rho_vs_divergence_spearman"] = {"spearman_r": r, "p": p, "n": n}
    pr = stats.pearsonr(d["rho"], d["divergence_mya"])
    out["H1_rho_vs_divergence_pearson"] = {"pearson_r": float(pr[0]), "p": float(pr[1])}
    euth = d[d["species"].isin(EUTHERIAN_SPECIES)]
    opo = d[d["species"] == "Opossum"]
    chk = d[d["species"] == "Chicken"]
    out["H1_eutherian_mean_rho"] = float(euth["rho"].mean())
    out["H1_opossum_mean_rho"] = float(opo["rho"].mean())
    out["H1_chicken_mean_rho"] = float(chk["rho"].mean())
    out["H1_eutherian_n"] = int(len(euth))
    out["H1_opossum_n"] = int(len(opo))
    out["H1_chicken_n"] = int(len(chk))

    # ---- H2: per-organ mean rho + rho vs complexity (pig breadth) -------
    organ_mean = d.groupby("organ")["rho"].mean()
    out["H2_tissue_mean_rho"] = {o: float(v) for o, v in organ_mean.items()}
    out["H2_tissue_n_species"] = {o: int(c) for o, c in d.groupby("organ")["rho"].size().items()}
    out["H2_pig_breadth"] = {o: int(breadth[o]) for o in organ_mean.index}
    bx = np.array([breadth[o] for o in organ_mean.index], float)
    r2, p2, n2 = _spearman_p(organ_mean.values, bx)
    out["H2_rho_vs_complexity"] = {"spearman_r": r2, "p": p2, "n_tissues": int(len(organ_mean))}

    # ---- H3: rho vs n_target & n_orthologs (pooled + per-organ) ---------
    r, p, n = _spearman_p(d["rho"], d["n_target"])
    out["H3_pooled_rho_vs_n_target"] = {"spearman_r": r, "p": p, "n": n}
    r, p, n = _spearman_p(d["rho"], d["n_orth"])
    out["H3_pooled_rho_vs_n_orthologs"] = {"spearman_r": r, "p": p, "n": n}
    for organ in sorted(d["organ"].unique()):
        sub = d[d["organ"] == organ]
        r, p, n = _spearman_p(sub["rho"], sub["n_target"])
        out[f"H3_{organ}_rho_vs_n_target"] = {"spearman_r": r, "p": p, "n": n}
        r, p, n = _spearman_p(sub["rho"], sub["n_orth"])
        out[f"H3_{organ}_rho_vs_n_orthologs"] = {"spearman_r": r, "p": p, "n": n}

    # ---- Variance partition: OLS rho ~ C(organ) + divergence ------------
    # unique R^2(term) = R^2(full) - R^2(full minus term).
    out["variance_partition"] = _variance_partition(d)
    return out


def _ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    """R^2 of an OLS fit of y on design X (X already includes an intercept col)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _variance_partition(d: pd.DataFrame) -> dict:
    """Unique R^2 of organ and divergence from OLS rho ~ C(organ) + divergence."""
    y = d["rho"].to_numpy(float)
    n = len(y)
    organs = sorted(d["organ"].unique())
    # organ dummies (drop first for identifiability) + divergence + intercept
    org_idx = {o: i for i, o in enumerate(organs)}
    Dorg = np.zeros((n, len(organs)))
    for i, o in enumerate(d["organ"].to_numpy()):
        Dorg[i, org_idx[o]] = 1.0
    Dorg_dropped = Dorg[:, 1:]                       # drop first organ -> reference
    div = d["divergence_mya"].to_numpy(float)[:, None]
    icept = np.ones((n, 1))
    X_full = np.hstack([icept, Dorg_dropped, div])
    X_no_org = np.hstack([icept, div])
    X_no_div = np.hstack([icept, Dorg_dropped])
    r2_full = _ols_r2(y, X_full)
    r2_no_org = _ols_r2(y, X_no_org)
    r2_no_div = _ols_r2(y, X_no_div)
    return {
        "r2_full": r2_full,
        "r2_organ_only": _ols_r2(y, np.hstack([icept, Dorg_dropped])),
        "r2_divergence_only": _ols_r2(y, np.hstack([icept, div])),
        "unique_r2_organ": r2_full - r2_no_org,
        "unique_r2_divergence": r2_full - r2_no_div,
        "n": int(n),
    }


def validate_against_panel_e(matrix: pd.DataFrame) -> pd.DataFrame:
    """Join the new 6-stage matrix to the committed panel_e and compute deltas."""
    if not PANEL_E_CSV.exists():
        return pd.DataFrame()
    pe = pd.read_csv(PANEL_E_CSV)[["organ", "species", "rho"]].rename(
        columns={"rho": "rho_panel_e"})
    m = matrix.merge(pe, on=["organ", "species"], how="left")
    m["delta_vs_panel_e"] = m["rho"] - m["rho_panel_e"]
    m["moved_gt_0.15"] = m["delta_vs_panel_e"].abs() > 0.15
    return m


def dgtex_cell_6stage(pig_tissue: str, smts: str, slug: str) -> Optional[dict]:
    """PLS 6-stage dGTEx transfer cell (for the supplementary dGTEx table)."""
    pig_X = load_pig_tissue(pig_tissue)
    ortho = build_ortholog_map("Human")  # dGTEx uses human ENSG ids
    tgt_expr, tgt_level = load_dgtex_target(smts, slug)
    res = run_cell_pls_only(pig_X, ortho, tgt_expr, tgt_level, "6stage", do_scramble=True)
    if res is None:
        return None
    flag = "underpowered" if res["n_target_samples"] < 12 else np.nan
    return {
        "organ": pig_tissue,
        "species": "Human",
        "rho": round(res["rho_pls"], 4),
        "n_target": res["n_target_samples"],
        "scramble_p": round(res["scramble_p_pls"], 6),
        "flag": flag,
        "n_orth": res["n_orthologs"],
    }


def main_matrix() -> None:
    print("=" * 90)
    print("Full 6-stage PLS transfer matrix (Fig 4 / Table S7 data layer)")
    print("=" * 90)

    # ---- Cardoso matrix: 7 species x 6 organs --------------------------
    # Outer loop = species so each large '<Species>_rpkm.txt' matrix and its
    # ortholog map are loaded ONCE and sliced across all six organs.
    print("\n[Cardoso] 7 species x 6 organs (PLS 6-stage, postnatal-only, 1000x scramble)")
    rows: List[dict] = []
    pig_cache: Dict[str, pd.DataFrame] = {}
    breadth: Dict[str, int] = {}
    for organ in CARDOSO_ORGANS:
        pig_cache[organ] = load_pig_tissue(organ)   # pig source tissue == organ
        breadth[organ] = pig_expressed_ortholog_count(pig_cache[organ])

    for species in CARDOSO_SPECIES:
        print(f"  [load] {species} matrix + orthologs ...", flush=True)
        cardoso_df = _read_cardoso_matrix(species)
        ortho = build_ortholog_map(species)
        for organ in CARDOSO_ORGANS:
            rec = cardoso_cell_from_matrix(pig_cache[organ], species, organ,
                                           cardoso_df, ortho)
            if rec is None:
                print(f"  {organ:7s} {species:8s}  -- absent / not fittable")
                continue
            rows.append(rec)
            print(f"  {organ:7s} {species:8s}  rho={rec['rho']:+.3f}  "
                  f"p={rec['scramble_p']:.3f}  n_orth={rec['n_orth']:5d}  "
                  f"n_tgt={rec['n_target']:3d}  Kpig={rec['K']}", flush=True)
        del cardoso_df
    matrix = pd.DataFrame(rows)
    # round for the stored CSV
    mat_out = matrix.copy()
    mat_out["rho"] = mat_out["rho"].round(4)
    mat_out["scramble_p"] = mat_out["scramble_p"].round(6)
    mat_out.to_csv(MATRIX_OUT_CSV, index=False)
    print(f"\nWrote {MATRIX_OUT_CSV}  ({len(matrix)} cells)")

    # ---- determinant tests ---------------------------------------------
    print("\n[Determinants] H1/H2/H3 + variance partition")
    det = compute_determinants(matrix, breadth)
    DETERMINANTS_OUT_JSON.write_text(json.dumps(det, indent=1))
    print(f"Wrote {DETERMINANTS_OUT_JSON}")
    h1 = det["H1_rho_vs_divergence_spearman"]
    vp = det["variance_partition"]
    print(f"  H1 rho~divergence: Spearman={h1['spearman_r']:+.3f} (p={h1['p']:.3f}, n={h1['n']})"
          f"  | eutherian {det['H1_eutherian_mean_rho']:.3f} / opossum "
          f"{det['H1_opossum_mean_rho']:.3f} / chicken {det['H1_chicken_mean_rho']:.3f}")
    print(f"  H2 rho~complexity: Spearman={det['H2_rho_vs_complexity']['spearman_r']:+.3f} "
          f"(p={det['H2_rho_vs_complexity']['p']:.3f}, n_tissues={det['H2_rho_vs_complexity']['n_tissues']})")
    print(f"  H3 pooled rho~n_target: Spearman="
          f"{det['H3_pooled_rho_vs_n_target']['spearman_r']:+.3f} "
          f"(p={det['H3_pooled_rho_vs_n_target']['p']:.3f})")
    print(f"  Variance partition: unique R2 organ={vp['unique_r2_organ']:.3f}  "
          f"unique R2 divergence={vp['unique_r2_divergence']:.3f}  (full R2={vp['r2_full']:.3f})")

    # ---- validation vs committed panel_e -------------------------------
    print("\n[Validation] 6-stage matrix vs committed panel_e_rho_vs_divergence.csv")
    val = validate_against_panel_e(matrix)
    if not val.empty:
        movers = val[val["moved_gt_0.15"] & val["rho_panel_e"].notna()]
        absdiff = val["delta_vs_panel_e"].abs()
        print(f"  cells compared: {int(val['rho_panel_e'].notna().sum())}  | "
              f"mean |delta| = {absdiff.mean():.3f}  median = {absdiff.median():.3f}  "
              f"max = {absdiff.max():.3f}")
        print(f"  cells moved >0.15: {len(movers)}")
        for _, r in movers.iterrows():
            print(f"    {r['organ']:7s} {r['species']:8s}  6stage={r['rho']:+.3f}  "
                  f"panel_e={r['rho_panel_e']:+.3f}  delta={r['delta_vs_panel_e']:+.3f}")

    # ---- dGTEx supplementary table (6-stage) ---------------------------
    print("\n[dGTEx] supplementary table organs (PLS 6-stage)")
    drows: List[dict] = []
    for pig_tissue, smts, slug in DGTEX_TABLE_SUBSTRATES:
        rec = dgtex_cell_6stage(pig_tissue, smts, slug)
        if rec is None:
            print(f"  {pig_tissue:8s}  -- not fittable")
            continue
        drows.append(rec)
        print(f"  {pig_tissue:8s}  rho={rec['rho']:+.3f}  n_target={rec['n_target']:3d}  "
              f"scramble_p={rec['scramble_p']:.4f}  flag={rec['flag']}", flush=True)
    dgtex_df = pd.DataFrame(drows)[
        ["organ", "species", "rho", "n_target", "scramble_p", "flag", "n_orth"]]
    dgtex_df.to_csv(DGTEX_TABLE_OUT_CSV, index=False)
    print(f"\nWrote {DGTEX_TABLE_OUT_CSV}  ({len(dgtex_df)} organs)")

    print("\nDONE.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["A", "B", "matrix"], default="A",
                    help="A = PLS-vs-LightGBM gate (default); B = nonlinearity "
                         "sweep; matrix = full 6-stage transfer matrix + "
                         "determinants + dGTEx table")
    args = ap.parse_args()
    if args.exp == "B":
        main_expB()
    elif args.exp == "matrix":
        main_matrix()
    else:
        main()
