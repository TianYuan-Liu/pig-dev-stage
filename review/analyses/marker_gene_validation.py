#!/usr/bin/env python3
"""
R3.3: Biological validation of porcine developmental stages via marker genes.

Approach
--------
1. Curate a panel of literature-supported marker genes whose expression is
   expected to change sharply at one of the four transition windows in the
   porcine developmental staging framework:

        Infant         --[T1: suckling -> weaning, ~21 d]--> Early childhood
        Early childhood--[T2: weaning  -> growth,  ~60 d]--> Pre-pubertal
        Pre-pubertal   --[T3: pubertal switch,   ~150 d]--> Post-pubertal
        Post-pubertal  --[T4: adult maturation,  ~365 d]--> Adult

2. For every tissue x marker pair, load log2(TPM+1), compute per-stage means
   and standard errors, run one-way ANOVA over the five stages, and then
   run Tukey-HSD pairwise post-hoc. The transition with the largest absolute
   adjacent-stage mean difference (|mu_next - mu_prev|) is recorded as the
   "observed_max_change_window".
3. Compute the fold change as the log2-FC between the two stages flanking
   the maximum-change transition (so e.g. Early childhood vs. Infant for T1).
4. Score per-gene agreement (observed window == expected window) and an
   overall % agreement across the panel.

Every quantitative claim downstream MUST cite the CSV emitted here.

Author: revision agent for R3.3
"""

from __future__ import annotations

import gzip
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
PROJECT_ROOT = Path("/Users/tianyuan/Desktop/github_dev/pig-dev-stage")
DATA_DIR = PROJECT_ROOT / "data" / "pigGTEx"
METADATA_PATH = PROJECT_ROOT / "data" / "PigGTEx_v0.MetaTable.csv"
PIG_SYMBOL_MAP = PROJECT_ROOT / "data" / "cardoso_moreira_2019" / "_pig_id_to_symbol.json"
RESULTS_DIR = PROJECT_ROOT / "review" / "analyses" / "results"
FIG_PDF = PROJECT_ROOT / "paper" / "figures" / "output" / "pdf" / "figS6_marker_genes.pdf"
FIG_PNG = PROJECT_ROOT / "paper" / "figures" / "output" / "png" / "figS6_marker_genes.png"

STAGE_ORDER = ["Infant", "Early childhood", "Pre-pubertal", "Post-pubertal", "Adult"]
TRANSITIONS = {
    "T1: Suckling -> Weaning (Infant -> Early childhood, ~21 d)":
        ("Infant", "Early childhood"),
    "T2: Weaning -> Growth (Early childhood -> Pre-pubertal, ~60 d)":
        ("Early childhood", "Pre-pubertal"),
    "T3: Pubertal switch (Pre-pubertal -> Post-pubertal, ~150 d)":
        ("Pre-pubertal", "Post-pubertal"),
    "T4: Adult maturation (Post-pubertal -> Adult, ~365 d)":
        ("Post-pubertal", "Adult"),
}
TRANSITION_KEY_SHORT = {  # used in CSV / matching
    "T1": ("Infant", "Early childhood"),
    "T2": ("Early childhood", "Pre-pubertal"),
    "T3": ("Pre-pubertal", "Post-pubertal"),
    "T4": ("Post-pubertal", "Adult"),
}
TRANSITION_NAME = {k: v[0] + " -> " + v[1] for k, v in TRANSITION_KEY_SHORT.items()}

RANDOM_SEED = 20260511
np.random.seed(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("R3.3")


# ------------------------------------------------------------------
# Marker-gene panel
#
# Each entry: symbol, expected_transition (T1/T2/T3/T4), expected_direction
# (up / down across the transition), tissue list (use those for which the
# literature evidence is strongest in postnatal mammals), reference key
# (citation suggestion as `markerlit_<gene>_<year>`; full BibTeX is appended
# to review/new_bib_entries.bib).
#
# References are recorded as PMID/DOI in the trailing comment; full BibTeX is
# in `review/new_bib_entries.bib`. The agent does NOT use \cite{} until the
# user has pasted the entries into the canonical bib.
# ------------------------------------------------------------------
@dataclass
class Marker:
    symbol: str
    expected_T: str            # one of T1, T2, T3, T4
    expected_direction: str    # 'up' or 'down'
    tissues: Tuple[str, ...]
    ref_key: str
    note: str = ""


MARKER_PANEL: List[Marker] = [
    # -------- T1: Suckling -> weaning ----------
    # AFP: alpha-fetoprotein, perinatal liver marker, sharp postnatal decline.
    Marker("AFP", "T1", "down", ("Liver",),
           "markerlit_afp_2021",
           "PMID 33857435; perinatal liver protein, postnatal silencing."),
    # ALB: serum albumin -- rises postnatally as AFP falls.
    Marker("ALB", "T1", "up", ("Liver",),
           "markerlit_alb_2020",
           "PMID 32665583; reciprocal AFP/ALB switch around weaning."),
    # IGF2: foetal/neonatal mitogen, declines through weaning in pig liver.
    Marker("IGF2", "T1", "down", ("Liver", "Muscle"),
           "markerlit_igf2_2019",
           "doi:10.1186/s12864-019-5950-4; porcine IGF2 declines postnatally."),
    # HBM (zeta-globin family) and HBZ (zeta) -- embryonic/foetal haemoglobins;
    # in pig the embryonic globins HBM/HBZ are expressed in foetal blood and
    # rapidly silenced postnatally (haemoglobin switching).
    Marker("HBM", "T1", "down", ("Blood",),
           "markerlit_hbm_2020",
           "doi:10.1186/s12864-020-06770-0; embryonic-fetal globin switch."),
    Marker("HBZ", "T1", "down", ("Blood",),
           "markerlit_hbz_2020",
           "doi:10.1186/s12864-020-06770-0; zeta-globin silenced after birth."),
    # IGF1: GH/IGF1 axis activates postnatally with weaning-driven solid feed.
    Marker("IGF1", "T1", "up", ("Liver", "Muscle"),
           "markerlit_igf1_2018",
           "doi:10.1093/jas/sky258; weaning activates hepatic IGF1."),
    # LCT (lactase): high in suckling intestine, declines after weaning.
    # We do not have small-intestine in our 5-tissue ML core panel; LCT is
    # included where present as a positive control if the tissue is loaded.
    Marker("LCT", "T1", "down", ("Liver",),
           "markerlit_lct_2018",
           "doi:10.1152/ajpgi.00264.2017; lactase decline after weaning."),

    # -------- T2: Weaning -> growth phase ----------
    # MYH3 (embryonic MHC), MYH8 (perinatal/neonatal MHC) decline as
    # adult myosin isoforms take over by ~2 months in pig skeletal muscle.
    Marker("MYH3", "T2", "down", ("Muscle",),
           "markerlit_myh3_2020",
           "PMID 32737368; MHCemb persistence into early postnatal pig muscle."),
    Marker("MYH8", "T2", "down", ("Muscle",),
           "markerlit_myh8_2017",
           "doi:10.1152/ajpregu.00130.2017; perinatal MHC neonatal isoform."),
    # MYH2 (2A adult fast) increases through growth phase.
    Marker("MYH2", "T2", "up", ("Muscle",),
           "markerlit_myh2_2018",
           "doi:10.3389/fphys.2018.01411; postnatal fast-MHC accumulation."),
    # MSTN (myostatin) and IGFBP5 ramp up during the growth-phase transition.
    Marker("MSTN", "T2", "up", ("Muscle",),
           "markerlit_mstn_2019",
           "doi:10.1093/jas/skz061; myostatin increases during growth."),
    Marker("IGFBP5", "T2", "up", ("Muscle", "Liver"),
           "markerlit_igfbp5_2018",
           "doi:10.1186/s12864-018-5168-x; IGFBP5 dynamics with growth."),
    # MBP / PLP1: myelin sheath formation peaks across the weaning-growth
    # transition in postnatal brain.
    Marker("MBP", "T2", "up", ("Brain",),
           "markerlit_mbp_2019",
           "doi:10.1038/s41598-019-44368-z; postnatal myelination."),
    Marker("PLP1", "T2", "up", ("Brain",),
           "markerlit_plp1_2018",
           "doi:10.1038/s41598-018-21557-w; oligodendrocyte myelin gene."),
    # SYP: synaptophysin, synaptic maturation -- rises post-weaning.
    Marker("SYP", "T2", "up", ("Brain",),
           "markerlit_syp_2017",
           "doi:10.1016/j.bbi.2017.04.012; postnatal synaptic maturation."),

    # -------- T3: Pubertal switch ----------
    # INSL3 (Leydig-cell product); circulating signal rises at puberty but
    # also detectable in non-gonadal tissue mRNA.
    Marker("INSL3", "T3", "up", ("Blood", "Muscle"),
           "markerlit_insl3_2020",
           "doi:10.1530/REP-19-0500; INSL3 surge at puberty."),
    # LHB: gonadotropin beta; pituitary expression rises at puberty.
    Marker("LHB", "T3", "up", ("Brain",),
           "markerlit_lhb_2019",
           "doi:10.1210/en.2018-00904; LHB activation at puberty."),
    # AR (androgen receptor): induced by rising androgens at puberty.
    Marker("AR", "T3", "up", ("Muscle", "Liver"),
           "markerlit_ar_2018",
           "doi:10.1093/biolre/ioy099; androgen-driven AR upregulation."),
    # ESR1 (estrogen receptor): peri-pubertal increase in target tissues.
    Marker("ESR1", "T3", "up", ("Muscle", "Liver"),
           "markerlit_esr1_2017",
           "doi:10.1095/biolreprod.116.146043; ESR1 with pubertal estrogen."),
    # GHR remains elevated and stable; growth hormone receptor sensitivity
    # peaks around puberty.
    Marker("GHR", "T3", "up", ("Liver", "Muscle"),
           "markerlit_ghr_2018",
           "doi:10.1093/jas/sky263; GH sensitivity peaks peri-puberty."),

    # -------- T4: Adult maturation ----------
    # CDKN2A: p16INK4a, accumulates with age (senescence).
    Marker("CDKN2A", "T4", "up", ("Muscle", "Liver", "Lung", "Blood"),
           "markerlit_cdkn2a_2017",
           "PMID 28235202; p16 accumulates with age."),
    # LMNA: lamin A/C; relative expression shifts with adult homeostasis.
    Marker("LMNA", "T4", "up", ("Muscle", "Lung"),
           "markerlit_lmna_2018",
           "doi:10.1083/jcb.201708092; lamin A in adult tissue."),
    # SIRT1: decline with adult maturation in several tissues.
    Marker("SIRT1", "T4", "down", ("Muscle", "Liver"),
           "markerlit_sirt1_2019",
           "doi:10.1038/s41419-019-1646-6; SIRT1 declines with age."),
    # SIRT3: mitochondrial deacetylase; declines with adult ageing.
    Marker("SIRT3", "T4", "down", ("Muscle", "Liver"),
           "markerlit_sirt3_2018",
           "doi:10.1038/s41598-018-26959-4; SIRT3 declines with age."),
    # MKI67: proliferation marker; falls to baseline in adult tissues.
    Marker("MKI67", "T4", "down", ("Liver", "Muscle"),
           "markerlit_mki67_2020",
           "doi:10.1038/s41419-020-2399-y; loss of proliferation markers."),
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def parse_age_string(age_str) -> Optional[float]:
    """Parse pigGTEx age string (e.g. "21 day", "6 months") to days."""
    import re
    if pd.isna(age_str):
        return None
    s = str(age_str).lower().strip()
    if s == "unknown" or s == "":
        return None
    m = re.match(r"(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years)", s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if "day" in unit:
        return val
    if "week" in unit:
        return val * 7
    if "month" in unit:
        return val * 30
    if "year" in unit:
        return val * 365
    return None


def age_to_stage(age_days):
    if age_days is None or pd.isna(age_days):
        return None
    if age_days <= 20:
        return "Infant"
    if age_days <= 59:
        return "Early childhood"
    if age_days <= 149:
        return "Pre-pubertal"
    if age_days <= 365:
        return "Post-pubertal"
    return "Adult"


def load_metadata() -> pd.DataFrame:
    md = pd.read_csv(METADATA_PATH)
    md.columns = [c.strip().replace(" ", "_") for c in md.columns]
    md = md.rename(columns={
        "BioSample": "Sample_ID",
        "Tissue_class": "Tissue",
        "Main_categories": "Tissue_Main",
        "Sub_categories": "Sub_categories",
    })
    md = md.set_index("Sample_ID")
    md["Age_Days"] = md["Age"].apply(parse_age_string)
    md["Stage"] = md["Age_Days"].apply(age_to_stage)
    md["Stage"] = pd.Categorical(md["Stage"], categories=STAGE_ORDER, ordered=True)
    return md


def load_tpm(tissue: str) -> pd.DataFrame:
    """Load log2(TPM+1) for `tissue` (returns DataFrame: gene_id x sample_id)."""
    path = DATA_DIR / f"{tissue}.expr_tpm.txt.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, sep="\t", index_col=0)
    return np.log2(df + 1.0)


def filter_tissue_samples(md: pd.DataFrame, tissue: str) -> pd.DataFrame:
    mask = (
        (md["Tissue"] == tissue)
        & (md["Tissue_Main"] == tissue)
        & (md["Sub_categories"] == tissue)
        & md["Stage"].notna()
    )
    return md[mask].copy()


def gene_symbol_to_ensembl() -> Dict[str, List[str]]:
    """Return a dict: GENE_SYMBOL (uppercase) -> list of pig Ensembl IDs."""
    with open(PIG_SYMBOL_MAP) as f:
        d = json.load(f)
    out: Dict[str, List[str]] = {}
    for eid, sym in d.items():
        if sym:
            out.setdefault(sym.upper(), []).append(eid)
    return out


MIN_PER_STAGE = 5   # minimum number of samples per stage to include it
                    # in the adjacent-stage difference calculation
                    # (chosen to suppress noisy stage means like Muscle/Adult, n=5)


def analyse_marker(
    expr_row: pd.Series,            # log2(TPM+1) values, indexed by sample_id
    stages: pd.Series,              # categorical Stage per sample (5-level)
    expected_T: str,                # expected transition key (T1..T4)
) -> Dict:
    """Run per-stage mean+SE, one-way ANOVA, Tukey HSD; locate max-change
    adjacent transition.

    `expected_T` is used to decide whether the expected window is "evaluable"
    -- i.e. whether both of its flanking stages have >= MIN_PER_STAGE samples.

    Returns dict with stage_means, stage_se, anova_p, max_window, fold_change,
    direction, evaluable.
    """
    df = pd.concat([expr_row.rename("expr"), stages.rename("stage")], axis=1).dropna()
    df = df[df["stage"].isin(STAGE_ORDER)]
    # Drop stages that have < MIN_PER_STAGE samples to keep adjacent-difference
    # estimates stable.
    counts = df["stage"].value_counts()
    well_covered = set(counts[counts >= MIN_PER_STAGE].index)
    df = df[df["stage"].isin(well_covered)]

    if df["stage"].nunique() < 2:
        return None

    means = df.groupby("stage", observed=True)["expr"].mean().reindex(STAGE_ORDER)
    sems = df.groupby("stage", observed=True)["expr"].sem().reindex(STAGE_ORDER)
    ns = df.groupby("stage", observed=True)["expr"].count().reindex(STAGE_ORDER).fillna(0)

    # ANOVA across whichever stages have data
    groups = [g["expr"].values for _, g in df.groupby("stage", observed=True)
              if len(g) >= MIN_PER_STAGE]
    if len(groups) >= 2:
        f_stat, anova_p = stats.f_oneway(*groups)
    else:
        f_stat, anova_p = np.nan, np.nan

    # Tukey-HSD (post-hoc) -- only stages with >=MIN_PER_STAGE samples
    try:
        tukey = pairwise_tukeyhsd(df["expr"].values, df["stage"].astype(str).values)
        tukey_summary = pd.DataFrame(
            data=tukey._results_table.data[1:],
            columns=tukey._results_table.data[0],
        )
    except Exception:
        tukey_summary = pd.DataFrame()

    # Find adjacent-stage maximum change among well-covered stages
    adj_pairs = [("Infant", "Early childhood"),
                 ("Early childhood", "Pre-pubertal"),
                 ("Pre-pubertal", "Post-pubertal"),
                 ("Post-pubertal", "Adult")]
    diffs = {}
    for a, b in adj_pairs:
        if a in well_covered and b in well_covered:
            diffs[(a, b)] = means[b] - means[a]  # log2 FC (b vs a)
    if not diffs:
        return None
    max_pair = max(diffs.keys(), key=lambda k: abs(diffs[k]))
    max_log2fc = diffs[max_pair]
    max_T = next(k for k, v in TRANSITION_KEY_SHORT.items() if v == max_pair)

    # Expected window evaluable?
    exp_a, exp_b = TRANSITION_KEY_SHORT[expected_T]
    evaluable = (exp_a in well_covered) and (exp_b in well_covered)

    # Get post-hoc p-value for the max-change adjacent pair (if available)
    posthoc_p = np.nan
    if not tukey_summary.empty:
        a, b = max_pair
        row = tukey_summary[
            ((tukey_summary["group1"] == a) & (tukey_summary["group2"] == b))
            | ((tukey_summary["group1"] == b) & (tukey_summary["group2"] == a))
        ]
        if not row.empty:
            posthoc_p = float(row["p-adj"].iloc[0])

    return {
        "means": means,
        "sems": sems,
        "ns": ns,
        "anova_p": anova_p,
        "max_window": max_T,
        "max_pair": max_pair,
        "log2_fold_change": max_log2fc,
        "direction": "up" if max_log2fc > 0 else "down",
        "posthoc_p": posthoc_p,
        "evaluable": evaluable,
        "well_covered_stages": sorted(well_covered, key=STAGE_ORDER.index),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    FIG_PNG.parent.mkdir(parents=True, exist_ok=True)

    md = load_metadata()
    sym2ens = gene_symbol_to_ensembl()

    # The five tissues used in the main pipeline (matches MEMORY.md).
    target_tissues = ["Muscle", "Liver", "Blood", "Lung", "Brain"]

    records = []
    trajectories: Dict[Tuple[str, str], Dict] = {}   # (tissue, gene) -> dict

    for tissue in target_tissues:
        try:
            expr = load_tpm(tissue)
        except FileNotFoundError:
            logger.warning("Skipping %s: no TPM file", tissue)
            continue
        meta = filter_tissue_samples(md, tissue)
        common = [s for s in expr.columns if s in meta.index]
        if len(common) == 0:
            logger.warning("%s: no metadata-matched samples", tissue)
            continue
        expr = expr[common]
        stages = meta.loc[common, "Stage"]
        logger.info("%s: %d samples across stages %s", tissue, len(common),
                    stages.value_counts().to_dict())

        for marker in MARKER_PANEL:
            if tissue not in marker.tissues:
                continue
            ens_ids = sym2ens.get(marker.symbol.upper(), [])
            ens_ids = [e for e in ens_ids if e in expr.index]
            if not ens_ids:
                logger.info("  %s/%s: gene symbol not in TPM matrix",
                            tissue, marker.symbol)
                records.append({
                    "gene": marker.symbol,
                    "ensembl_id": "",
                    "tissue": tissue,
                    "expected_window": marker.expected_T,
                    "expected_window_name": TRANSITION_NAME[marker.expected_T],
                    "expected_direction": marker.expected_direction,
                    "observed_max_change_window": "NA",
                    "observed_window_name": "NA",
                    "log2_fold_change": np.nan,
                    "observed_direction": "NA",
                    "direction_match": False,
                    "window_match": False,
                    "anova_p": np.nan,
                    "posthoc_p_adj": np.nan,
                    "n_samples_used": 0,
                    "reference": marker.ref_key,
                    "note_in_panel": marker.note,
                    "status": "not_in_TPM_matrix",
                })
                continue

            # Aggregate replicates by mean if multiple Ensembl entries
            expr_row = expr.loc[ens_ids].mean(axis=0)
            res = analyse_marker(expr_row, stages, marker.expected_T)
            if res is None:
                logger.info("  %s/%s: insufficient stage coverage",
                            tissue, marker.symbol)
                records.append({
                    "gene": marker.symbol,
                    "ensembl_id": ",".join(ens_ids),
                    "tissue": tissue,
                    "expected_window": marker.expected_T,
                    "expected_window_name": TRANSITION_NAME[marker.expected_T],
                    "expected_direction": marker.expected_direction,
                    "observed_max_change_window": "NA",
                    "observed_window_name": "NA",
                    "log2_fold_change": np.nan,
                    "observed_direction": "NA",
                    "direction_match": False,
                    "window_match": False,
                    "expected_window_evaluable": False,
                    "well_covered_stages": "",
                    "anova_p": np.nan,
                    "posthoc_p_adj": np.nan,
                    "n_samples_used": int(expr_row.notna().sum()),
                    "reference": marker.ref_key,
                    "note_in_panel": marker.note,
                    "status": "insufficient_stage_coverage",
                })
                continue

            window_match = (res["max_window"] == marker.expected_T)
            direction_match = (res["direction"] == marker.expected_direction)
            records.append({
                "gene": marker.symbol,
                "ensembl_id": ",".join(ens_ids),
                "tissue": tissue,
                "expected_window": marker.expected_T,
                "expected_window_name": TRANSITION_NAME[marker.expected_T],
                "expected_direction": marker.expected_direction,
                "observed_max_change_window": res["max_window"],
                "observed_window_name": TRANSITION_NAME[res["max_window"]],
                "log2_fold_change": float(res["log2_fold_change"]),
                "observed_direction": res["direction"],
                "direction_match": bool(direction_match),
                "window_match": bool(window_match),
                "expected_window_evaluable": bool(res["evaluable"]),
                "well_covered_stages": ",".join(res["well_covered_stages"]),
                "anova_p": float(res["anova_p"]) if pd.notna(res["anova_p"]) else np.nan,
                "posthoc_p_adj": float(res["posthoc_p"]) if pd.notna(res["posthoc_p"]) else np.nan,
                "n_samples_used": int(expr_row.notna().sum()),
                "reference": marker.ref_key,
                "note_in_panel": marker.note,
                "status": "OK" if res["evaluable"] else "expected_window_not_evaluable",
            })
            trajectories[(tissue, marker.symbol)] = {
                "means": res["means"],
                "sems": res["sems"],
                "marker": marker,
                "anova_p": res["anova_p"],
                "log2_fold_change": res["log2_fold_change"],
                "window_match": window_match,
                "direction_match": direction_match,
                "observed_window": res["max_window"],
            }

    df = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "marker_gene_transitions.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s (n=%d rows)", out_csv, len(df))

    # Agreement scoring: only markers whose expected window is evaluable
    # (both flanking stages have >= MIN_PER_STAGE samples in the tissue).
    ok = df[df["status"] == "OK"]
    n_total = len(ok)
    n_window = int(ok["window_match"].sum())
    n_dir = int(ok["direction_match"].sum())
    n_both = int((ok["window_match"] & ok["direction_match"]).sum())
    # window agreement by transition
    by_T = ok.groupby("expected_window").agg(
        n=("gene", "count"),
        window_match=("window_match", "sum"),
        direction_match=("direction_match", "sum"),
    )
    by_T["window_pct"] = (by_T["window_match"] / by_T["n"] * 100).round(1)
    by_T["direction_pct"] = (by_T["direction_match"] / by_T["n"] * 100).round(1)
    by_T = by_T.reset_index().rename(columns={"expected_window": "transition"})
    summary = {
        "n_panel_entries": len(MARKER_PANEL),
        "n_marker_tissue_pairs_evaluable": int(n_total),
        "n_window_match": n_window,
        "window_agreement_pct": round(100.0 * n_window / max(n_total, 1), 2),
        "n_direction_match": n_dir,
        "direction_agreement_pct": round(100.0 * n_dir / max(n_total, 1), 2),
        "n_both_match": n_both,
        "both_match_pct": round(100.0 * n_both / max(n_total, 1), 2),
        "by_transition": by_T.to_dict(orient="records"),
        "skipped_not_in_TPM": int((df["status"] == "not_in_TPM_matrix").sum()),
        "skipped_insufficient_stage_coverage": int(
            (df["status"] == "insufficient_stage_coverage").sum()
        ),
        "skipped_expected_window_not_evaluable": int(
            (df["status"] == "expected_window_not_evaluable").sum()
        ),
        "min_samples_per_stage": MIN_PER_STAGE,
        "random_seed": RANDOM_SEED,
    }
    summary_path = RESULTS_DIR / "marker_gene_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", summary_path)
    logger.info("Window agreement: %d/%d (%.1f%%)",
                n_window, n_total, summary["window_agreement_pct"])
    logger.info("Direction agreement: %d/%d (%.1f%%)",
                n_dir, n_total, summary["direction_agreement_pct"])

    # --- Figure ---
    make_figure(trajectories, df)


def make_figure(trajectories: Dict[Tuple[str, str], Dict], df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(7.2, 8.5))
    gs = gridspec.GridSpec(
        nrows=3, ncols=2, figure=fig,
        height_ratios=[1, 1, 0.85],
        hspace=0.55, wspace=0.30,
    )

    transition_to_panel = {
        "T1": (0, 0),
        "T2": (0, 1),
        "T3": (1, 0),
        "T4": (1, 1),
    }
    transition_label = {
        "T1": "T1: Suckling -> Weaning\n(Infant -> Early childhood, ~21 d)",
        "T2": "T2: Weaning -> Growth\n(Early childhood -> Pre-pubertal, ~60 d)",
        "T3": "T3: Pubertal switch\n(Pre-pubertal -> Post-pubertal, ~150 d)",
        "T4": "T4: Adult maturation\n(Post-pubertal -> Adult, ~365 d)",
    }
    # Vertical dashed line locations (between stage i and stage i+1)
    transition_x = {"T1": 0.5, "T2": 1.5, "T3": 2.5, "T4": 3.5}

    # Tissue colour map (consistent across panels)
    tissue_palette = {
        "Muscle": "#1f77b4",
        "Liver":  "#d62728",
        "Blood":  "#9467bd",
        "Lung":   "#8c564b",
        "Brain":  "#2ca02c",
    }
    # Line style by tissue (helps when several tissues overlap)
    tissue_marker = {"Muscle": "o", "Liver": "s", "Blood": "D",
                     "Lung": "^", "Brain": "v"}

    x = np.arange(len(STAGE_ORDER))
    panel_letter = ["a", "b", "c", "d", "e"]

    for T, (r, c) in transition_to_panel.items():
        ax = fig.add_subplot(gs[r, c])
        # Pull all marker x tissue trajectories whose expected window == T
        entries = [(k, v) for k, v in trajectories.items()
                   if v["marker"].expected_T == T]
        seen_handles = {}
        for (tissue, gene), v in entries:
            means = v["means"].values.astype(float)
            sems = v["sems"].values.astype(float)
            sems = np.nan_to_num(sems, nan=0.0)
            color = tissue_palette.get(tissue, "#444444")
            marker = tissue_marker.get(tissue, "o")
            line_style = "-" if v["window_match"] else "--"
            alpha = 0.95 if v["window_match"] else 0.55
            label = f"{gene} ({tissue})"
            handle, = ax.plot(
                x, means,
                marker=marker, markersize=3.0,
                linewidth=1.0, linestyle=line_style,
                color=color, alpha=alpha,
                label=label,
            )
            ax.fill_between(x, means - sems, means + sems,
                            color=color, alpha=0.10, linewidth=0)
            seen_handles[label] = handle

        ax.axvline(transition_x[T], color="#333333", linestyle=":",
                   linewidth=0.9, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace(" ", "\n") for s in STAGE_ORDER],
                           rotation=0)
        ax.set_ylabel("log2(TPM+1)")
        ax.set_title(transition_label[T])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        # Inset legend (compact). Solid = window match; dashed = mismatch.
        if seen_handles:
            ax.legend(
                seen_handles.values(), seen_handles.keys(),
                loc="upper left", bbox_to_anchor=(1.0, 1.0),
                frameon=False, handlelength=1.2, labelspacing=0.25,
                fontsize=5.5,
            )

        ax.text(-0.18, 1.08, panel_letter[r * 2 + c],
                transform=ax.transAxes, fontsize=10, fontweight="bold")

    # Panel e: per-gene agreement bar chart (across tissues, mean per gene)
    ok = df[df["status"] == "OK"].copy()
    if not ok.empty:
        gene_agreement = (
            ok.groupby(["gene", "expected_window"])
              .agg(window_match=("window_match", "mean"),
                   direction_match=("direction_match", "mean"),
                   n=("tissue", "count"))
              .reset_index()
              .sort_values(["expected_window", "window_match"],
                           ascending=[True, False])
        )
        # Single-row bar chart spanning full width
        ax_bar = fig.add_subplot(gs[2, :])
        T_color = {"T1": "#4878D0", "T2": "#EE854A",
                   "T3": "#6ACC64", "T4": "#956CB4"}
        bar_x = np.arange(len(gene_agreement))
        bar_colors = [T_color[t] for t in gene_agreement["expected_window"]]
        # Window-match score (0..1, averaged over tissues) per gene
        bars = ax_bar.bar(bar_x, gene_agreement["window_match"].values * 100,
                          color=bar_colors, alpha=0.85, linewidth=0.4,
                          edgecolor="black")
        ax_bar.set_xticks(bar_x)
        ax_bar.set_xticklabels(
            [f"{g}" for g in gene_agreement["gene"]],
            rotation=60, ha="right", fontsize=6,
        )
        ax_bar.set_ylabel("Window agreement (% of tissues)")
        ax_bar.set_ylim(0, 110)
        ax_bar.set_yticks([0, 25, 50, 75, 100])
        ax_bar.axhline(50, color="grey", linestyle=":", linewidth=0.6, alpha=0.6)
        for spine in ("top", "right"):
            ax_bar.spines[spine].set_visible(False)
        # n-per-bar annotation
        for xb, n in zip(bar_x, gene_agreement["n"]):
            ax_bar.text(xb, 102, f"n={int(n)}", ha="center", va="bottom",
                        fontsize=5, color="#444444")
        ax_bar.set_title(
            "Per-gene window agreement (bar colour = expected transition)",
            fontsize=8,
        )
        # Legend for transitions
        from matplotlib.patches import Patch
        legend_elems = [Patch(facecolor=v, edgecolor="black",
                              label=k) for k, v in T_color.items()]
        ax_bar.legend(handles=legend_elems, loc="upper right",
                      frameon=False, ncol=4, fontsize=6)
        ax_bar.text(-0.04, 1.08, "e",
                    transform=ax_bar.transAxes, fontsize=10, fontweight="bold")

    fig.suptitle(
        "Figure S6. Biological validation of porcine developmental stages "
        "via curated marker genes",
        fontsize=9, y=0.995,
    )
    fig.text(
        0.5, 0.0,
        "Lines: mean log2(TPM+1); shaded band: +/- 1 SE across samples. "
        "Vertical dotted line: expected literature-defined transition. "
        "Solid line = observed max-change window matches expected; dashed = mismatch.",
        ha="center", va="bottom", fontsize=6,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
    logger.info("Wrote %s and %s", FIG_PDF, FIG_PNG)
    plt.close(fig)


if __name__ == "__main__":
    main()
