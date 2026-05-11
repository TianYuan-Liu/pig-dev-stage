#!/usr/bin/env python3
"""
Tissue-specific developmental features (Reviewer R3.4)
======================================================

For each of the five tissues with calibrated OrdinalLightGBM models
(Muscle, Brain, Liver, Lung, Blood), this script:

1. Reads ``machine_learning/model_outputs/{Tissue}_results.json`` and extracts
   the top 20 genes ranked by aggregate ``feature_importance``.
2. Annotates each gene with a brief functional description from external
   knowledge bases:
     * primary lookup: mygene.info (REST API)            -- https://mygene.info/
     * fallback:       UniProt REST API                  -- https://rest.uniprot.org/
   Failed lookups fall back to the gene symbol (or the Ensembl ID if no
   symbol could be resolved) and are logged for manual curation.
3. Loads the corresponding per-tissue TPM matrix from
   ``data/pigGTEx/{Tissue}.expr_tpm.txt.gz`` and the pigGTEx metadata in
   ``data/PigGTEx_v0.MetaTable.csv``, aligning sample IDs and mapping each
   sample to one of the five developmental stages (``Infant``,
   ``Early childhood``, ``Pre-pubertal``, ``Post-pubertal``, ``Adult``) using
   the conventions in ``machine_learning/data_processing/data_loader.py``.
4. For every (tissue, gene) pair computes the mean log2(TPM + 1) per stage,
   the peak stage, and the peak/trough fold change.
5. Writes three outputs:
     * ``review/analyses/results/tissue_specific_features.csv``  (long form)
     * ``paper/figures/output/pdf/figS5_tissue_features.pdf`` (and .png)
       -- heatmap of the top 10 genes per tissue across the five stages
     * ``paper/figures/output/supplementary/tableS7_tissue_features.tex``
       -- ``\\begin{longtable}...\\end{longtable}`` fragment for inclusion
       via ``\\input{}`` in ``supplementary.tex``

Run from the project root:

    python review/analyses/tissue_specific_features.py

Random seeds are not required (deterministic data processing); however,
all external API responses are cached in
``review/analyses/results/.gene_annotation_cache.json`` so that subsequent
runs are reproducible and offline.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path("/Users/tianyuan/Desktop/github_dev/pig-dev-stage")
MODEL_OUTPUT_DIR = PROJECT_ROOT / "machine_learning" / "model_outputs"
DATA_DIR = PROJECT_ROOT / "data" / "pigGTEx"
METADATA_PATH = PROJECT_ROOT / "data" / "PigGTEx_v0.MetaTable.csv"
RESULTS_DIR = PROJECT_ROOT / "review" / "analyses" / "results"
FIG_PDF_DIR = PROJECT_ROOT / "paper" / "figures" / "output" / "pdf"
FIG_PNG_DIR = PROJECT_ROOT / "paper" / "figures" / "output" / "png"
SUPP_TEX_DIR = PROJECT_ROOT / "paper" / "figures" / "output" / "supplementary"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_PDF_DIR.mkdir(parents=True, exist_ok=True)
FIG_PNG_DIR.mkdir(parents=True, exist_ok=True)
SUPP_TEX_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = RESULTS_DIR / "tissue_specific_features.log"
CACHE_PATH = RESULTS_DIR / ".gene_annotation_cache.json"
CSV_PATH = RESULTS_DIR / "tissue_specific_features.csv"
PDF_PATH = FIG_PDF_DIR / "figS5_tissue_features.pdf"
PNG_PATH = FIG_PNG_DIR / "figS5_tissue_features.png"
TEX_PATH = SUPP_TEX_DIR / "tableS7_tissue_features.tex"

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
TISSUES: List[str] = ["Muscle", "Brain", "Liver", "Lung", "Blood"]
TOP_N: int = 20
TOP_N_HEATMAP: int = 10

# Stage definitions (canonical project conventions; mirror DataLoader)
STAGE_ORDER: List[str] = [
    "Infant",
    "Early childhood",
    "Pre-pubertal",
    "Post-pubertal",
    "Adult",
]
STAGE_BOUNDS_DAYS: Dict[str, Tuple[float, float]] = {
    "Infant": (0, 20),
    "Early childhood": (21, 59),
    "Pre-pubertal": (60, 149),
    "Post-pubertal": (150, 365),
    "Adult": (366, float("inf")),
}
AGE_CONVERSION = {"day": 1, "week": 7, "month": 30, "year": 365}

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
MYGENE_URL = "https://mygene.info/v3/query"
PIG_TAXON = 9823
REQUEST_TIMEOUT = 20
REQUEST_PAUSE_S = 0.05  # mygene allows 3000 req/min; keep gentle

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("tissue_specific_features")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def parse_age_string(age_str) -> Optional[float]:
    """Convert ``"6 months"`` / ``"0 day"`` / ``"1 year"`` to days.

    Mirrors the parser in ``machine_learning/data_processing/data_loader.py``.
    """
    if pd.isna(age_str):
        return None
    s = str(age_str).strip().lower()
    if s in {"unknown", ""}:
        return None
    import re

    m = re.match(r"(\d+(?:\.\d+)?)\s*(day|week|month|year)s?", s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    return val * AGE_CONVERSION[unit]


def age_to_stage(age_days: Optional[float]) -> Optional[str]:
    if age_days is None or pd.isna(age_days):
        return None
    for stage, (lo, hi) in STAGE_BOUNDS_DAYS.items():
        if lo <= age_days <= hi:
            return stage
    return "Adult"


def load_metadata() -> pd.DataFrame:
    log.info("Reading metadata from %s", METADATA_PATH)
    meta = pd.read_csv(METADATA_PATH)
    meta.columns = [c.strip().replace(" ", "_") for c in meta.columns]
    if "BioSample" in meta.columns and "Sample_ID" not in meta.columns:
        meta["Sample_ID"] = meta["BioSample"]
    if "Tissue_class" in meta.columns and "Tissue" not in meta.columns:
        meta["Tissue"] = meta["Tissue_class"]
    if "Main_categories" in meta.columns and "Tissue_Main" not in meta.columns:
        meta["Tissue_Main"] = meta["Main_categories"]
    meta = meta.set_index("Sample_ID")
    meta["Age_Days"] = meta["Age"].apply(parse_age_string)
    meta["Stage"] = meta["Age_Days"].apply(age_to_stage)
    return meta


def filter_metadata_for_tissue(meta: pd.DataFrame, tissue: str) -> pd.DataFrame:
    """Apply 3-column tissue filter (matches DataLoader)."""
    mask = (
        (meta["Tissue"] == tissue)
        & (meta["Tissue_Main"] == tissue)
        & (meta["Sub_categories"] == tissue)
    )
    return meta[mask].copy()


def load_expression(tissue: str) -> pd.DataFrame:
    fp = DATA_DIR / f"{tissue}.expr_tpm.txt.gz"
    log.info("Loading expression matrix %s", fp)
    with gzip.open(fp, "rt") as f:
        df = pd.read_csv(f, sep="\t", index_col=0)
    log.info("  shape: %s", df.shape)
    return df


# ----------------------------------------------------------------------
# Gene annotation (mygene.info + UniProt fallback)
# ----------------------------------------------------------------------
def load_cache() -> Dict[str, dict]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            log.warning("Could not parse cache, starting fresh")
    return {}


def save_cache(cache: Dict[str, dict]) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _truncate(text: str, max_len: int = 280) -> str:
    """Trim long UniProt FUNCTION blurbs to a single readable sentence."""
    if not text:
        return ""
    # Strip leading "FUNCTION: " if present
    text = text.strip()
    if text.upper().startswith("FUNCTION:"):
        text = text[len("FUNCTION:") :].strip()
    # Remove embedded evidence tags like {ECO:0000269|PubMed:12345}
    import re

    text = re.sub(r"\{[^}]*\}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # If multiple sentences, take the first two.
    parts = [p.strip() for p in text.split(". ") if p.strip()]
    snippet = ". ".join(parts[:2])
    if not snippet.endswith("."):
        snippet += "."
    if len(snippet) > max_len:
        snippet = snippet[: max_len - 1].rsplit(" ", 1)[0] + "."
    return snippet


def query_mygene(ensembl_id: str) -> Optional[dict]:
    try:
        r = requests.get(
            MYGENE_URL,
            params={
                "q": f"ensembl.gene:{ensembl_id}",
                "fields": "symbol,name,summary,uniprot",
                "species": "pig",
                "size": 1,
            },
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        hits = r.json().get("hits", [])
        if hits:
            return hits[0]
    except Exception as exc:
        log.warning("mygene query failed for %s: %s", ensembl_id, exc)
    return None


def query_uniprot_by_ensembl(ensembl_id: str) -> Optional[dict]:
    try:
        r = requests.get(
            UNIPROT_SEARCH_URL,
            params={
                "query": f"xref:ensembl-{ensembl_id} AND organism_id:{PIG_TAXON}",
                "fields": "accession,gene_names,protein_name,cc_function",
                "format": "json",
                "size": 1,
            },
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            return results[0]
    except Exception as exc:
        log.warning("UniProt query (ensembl) failed for %s: %s", ensembl_id, exc)
    return None


def query_uniprot_by_symbol(symbol: str) -> Optional[dict]:
    try:
        r = requests.get(
            UNIPROT_SEARCH_URL,
            params={
                "query": f"gene_exact:{symbol} AND organism_id:{PIG_TAXON}",
                "fields": "accession,gene_names,protein_name,cc_function",
                "format": "json",
                "size": 1,
            },
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            return results[0]
        # Fall back to human ortholog (UniProt FUNCTION blocks are usually
        # better curated for human; gene symbol is a strong cross-species key)
        r2 = requests.get(
            UNIPROT_SEARCH_URL,
            params={
                "query": f"gene_exact:{symbol} AND organism_id:9606 AND reviewed:true",
                "fields": "accession,gene_names,protein_name,cc_function",
                "format": "json",
                "size": 1,
            },
            timeout=REQUEST_TIMEOUT,
        )
        r2.raise_for_status()
        results = r2.json().get("results", [])
        if results:
            return results[0]
    except Exception as exc:
        log.warning("UniProt query (symbol) failed for %s: %s", symbol, exc)
    return None


def extract_uniprot_function(record: dict) -> str:
    """Pull the FUNCTION comment from a UniProt JSON record, or fall back to
    the protein name.
    """
    if not record:
        return ""
    comments = record.get("comments", []) or []
    for c in comments:
        if c.get("commentType") == "FUNCTION":
            texts = c.get("texts", []) or []
            if texts:
                return _truncate(texts[0].get("value", ""))
    # No FUNCTION comment: use protein name
    pd_ = record.get("proteinDescription", {}) or {}
    rec = pd_.get("recommendedName", {}) or {}
    full = rec.get("fullName", {}) or {}
    if isinstance(full, dict) and "value" in full:
        return _truncate(full["value"])
    submission = pd_.get("submissionNames", []) or []
    if submission:
        sn = submission[0].get("fullName", {}) or {}
        if isinstance(sn, dict) and "value" in sn:
            return _truncate(sn["value"])
    return ""


def annotate_gene(ensembl_id: str, cache: Dict[str, dict]) -> dict:
    if ensembl_id in cache:
        return cache[ensembl_id]

    symbol = ""
    function = ""
    source = "none"

    # Step 1: mygene.info for the symbol
    mg = query_mygene(ensembl_id)
    if mg and mg.get("symbol"):
        symbol = mg["symbol"]
        # Try the summary first (mygene sometimes carries Entrez summary)
        summary = mg.get("summary") or ""
        if summary:
            function = _truncate(summary)
            source = "mygene_summary"

    # Step 2: UniProt by ensembl xref (best for FUNCTION text)
    if not function:
        ur = query_uniprot_by_ensembl(ensembl_id)
        if ur:
            if not symbol:
                gn = ur.get("genes", []) or []
                if gn:
                    val = (gn[0].get("geneName") or {}).get("value")
                    if val:
                        symbol = val
            function = extract_uniprot_function(ur)
            source = "uniprot_ensembl" if function else source

    # Step 3: UniProt by gene symbol (pig then human ortholog)
    if symbol and not function:
        ur = query_uniprot_by_symbol(symbol)
        if ur:
            function = extract_uniprot_function(ur)
            source = "uniprot_symbol" if function else source

    if not symbol:
        symbol = ensembl_id  # no annotation at all - placeholder
        log.warning("No symbol resolved for %s (will use Ensembl ID)", ensembl_id)
    if not function:
        function = symbol  # placeholder so manual curation is easy
        log.warning("No function resolved for %s (%s) - placeholder used", ensembl_id, symbol)

    record = {
        "ensembl": ensembl_id,
        "symbol": symbol,
        "function": function,
        "source": source,
    }
    cache[ensembl_id] = record
    save_cache(cache)
    time.sleep(REQUEST_PAUSE_S)
    return record


# ----------------------------------------------------------------------
# Per-tissue analysis
# ----------------------------------------------------------------------
def per_tissue_top_features(
    tissue: str,
    metadata: pd.DataFrame,
    cache: Dict[str, dict],
) -> pd.DataFrame:
    json_fp = MODEL_OUTPUT_DIR / f"{tissue}_results.json"
    with json_fp.open() as f:
        result = json.load(f)
    fi: Dict[str, float] = result["feature_importance"]
    # Sort by importance desc
    sorted_genes = sorted(fi.items(), key=lambda kv: kv[1], reverse=True)
    top = sorted_genes[:TOP_N]
    log.info("[%s] top %d importances range %.3f -> %.3f",
             tissue, len(top), top[0][1], top[-1][1])

    expr = load_expression(tissue)
    tmeta = filter_metadata_for_tissue(metadata, tissue)
    # Align
    common = [s for s in expr.columns if s in tmeta.index]
    if not common:
        log.error("[%s] no common samples between expression and metadata", tissue)
        return pd.DataFrame()
    expr = expr[common]
    tmeta = tmeta.loc[common]
    log.info("[%s] %d aligned samples", tissue, len(common))

    # Stage-wise mean log2(TPM+1)
    stage_labels = tmeta["Stage"].astype(str)

    rows = []
    for rank, (ensembl, imp) in enumerate(top, start=1):
        if ensembl not in expr.index:
            log.warning("[%s] %s missing from TPM matrix", tissue, ensembl)
            continue
        log_tpm = np.log2(expr.loc[ensembl] + 1.0)
        per_stage = (
            pd.DataFrame({"Stage": stage_labels.values, "log2tpm": log_tpm.values})
            .groupby("Stage")["log2tpm"]
            .mean()
            .reindex(STAGE_ORDER)
        )
        valid = per_stage.dropna()
        if valid.empty:
            log.warning("[%s] %s has no per-stage means", tissue, ensembl)
            continue
        peak_stage = valid.idxmax()
        peak_val = valid.max()
        trough_val = valid.min()
        # log2 fold change between peak and trough (mean log2(TPM+1) values)
        fc = peak_val - trough_val

        ann = annotate_gene(ensembl, cache)
        rows.append(
            {
                "tissue": tissue,
                "rank": rank,
                "ensembl": ensembl,
                "gene": ann["symbol"],
                "importance": imp,
                "function": ann["function"],
                "annotation_source": ann["source"],
                "Infant_TPM": per_stage.get("Infant", np.nan),
                "Early_childhood_TPM": per_stage.get("Early childhood", np.nan),
                "Pre-pubertal_TPM": per_stage.get("Pre-pubertal", np.nan),
                "Post-pubertal_TPM": per_stage.get("Post-pubertal", np.nan),
                "Adult_TPM": per_stage.get("Adult", np.nan),
                "peak_stage": peak_stage,
                "fold_change_peak_vs_trough": fc,
            }
        )

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
def build_heatmap(long_df: pd.DataFrame) -> None:
    log.info("Building heatmap of top %d genes per tissue", TOP_N_HEATMAP)
    blocks = []
    for tissue in TISSUES:
        sub = long_df[long_df["tissue"] == tissue].sort_values("rank").head(TOP_N_HEATMAP)
        mat = sub.set_index("gene")[[
            "Infant_TPM",
            "Early_childhood_TPM",
            "Pre-pubertal_TPM",
            "Post-pubertal_TPM",
            "Adult_TPM",
        ]]
        mat.index = [f"{tissue}: {g}" for g in mat.index]
        blocks.append(mat)
    heat = pd.concat(blocks)
    heat.columns = STAGE_ORDER

    # Row-wise z-score to highlight trajectory shape (not absolute level)
    row_mean = heat.mean(axis=1).values.reshape(-1, 1)
    row_std = heat.std(axis=1).replace(0, np.nan).values.reshape(-1, 1)
    z = (heat.values - row_mean) / row_std
    z = pd.DataFrame(z, index=heat.index, columns=heat.columns).fillna(0)

    n_rows = z.shape[0]
    fig, ax = plt.subplots(figsize=(7.2, max(8.0, n_rows * 0.22)))
    sns.heatmap(
        z,
        cmap="RdBu_r",
        center=0,
        vmin=-2,
        vmax=2,
        cbar_kws={"label": "Row z-score of mean log$_2$(TPM+1)"},
        linewidths=0.25,
        linecolor="#dddddd",
        ax=ax,
    )

    # Tissue dividers
    counts = (
        long_df[long_df["rank"] <= TOP_N_HEATMAP]
        .groupby("tissue")["rank"]
        .count()
        .reindex(TISSUES, fill_value=0)
    )
    cum = 0
    for tissue in TISSUES[:-1]:
        cum += counts[tissue]
        ax.axhline(cum, color="black", linewidth=1.0)

    ax.set_title("Top developmental-stage features per tissue", pad=12, fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Tissue: gene", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    fig.savefig(PDF_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, bbox_inches="tight", dpi=300)
    plt.close(fig)
    log.info("Saved %s and %s", PDF_PATH, PNG_PATH)


# ----------------------------------------------------------------------
# LaTeX table
# ----------------------------------------------------------------------
def _latex_escape(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def build_latex_table(long_df: pd.DataFrame) -> None:
    """Write a longtable fragment that supplementary.tex can \\input{}."""
    log.info("Building LaTeX longtable fragment")
    lines: List[str] = []
    add = lines.append

    add(r"% Auto-generated by review/analyses/tissue_specific_features.py")
    add(r"% Top-20 features per tissue (Reviewer R3.4)")
    add(r"\begingroup")
    add(r"\scriptsize")
    add(r"\setlength{\tabcolsep}{4pt}")
    add(r"\renewcommand{\arraystretch}{1.12}")
    add(r"\begin{longtable}{@{}l r l p{0.55\linewidth} l r@{}}")
    add(r"\caption{\textbf{Top 20 developmental-stage features per tissue.} "
        r"Genes ranked by aggregate LightGBM importance across the K$-$1 "
        r"binary classifiers within the calibrated ordinal model for each "
        r"tissue. Functional descriptions are drawn from mygene.info "
        r"(\url{https://mygene.info/}) and UniProt "
        r"(\url{https://rest.uniprot.org/}); placeholder entries equal the "
        r"gene symbol indicate that no curated description was returned by "
        r"either source. ``Peak stage'' is the developmental stage with the "
        r"highest mean $\log_{2}$(TPM$+$1); ``FC'' is the peak minus trough "
        r"in the same units. \label{tab:S7}} \\")
    add(r"\toprule")
    add(r"Tissue & Rank & Gene & Function & Peak stage & FC \\")
    add(r"\midrule")
    add(r"\endfirsthead")
    add(r"\multicolumn{6}{c}{\textit{Table~\ref{tab:S7} continued from previous page}} \\")
    add(r"\toprule")
    add(r"Tissue & Rank & Gene & Function & Peak stage & FC \\")
    add(r"\midrule")
    add(r"\endhead")
    add(r"\midrule")
    add(r"\multicolumn{6}{r}{\textit{continued on next page}} \\")
    add(r"\endfoot")
    add(r"\bottomrule")
    add(r"\endlastfoot")

    for tissue in TISSUES:
        sub = long_df[long_df["tissue"] == tissue].sort_values("rank")
        if sub.empty:
            continue
        for i, (_, row) in enumerate(sub.iterrows()):
            tissue_cell = _latex_escape(tissue) if i == 0 else ""
            gene_cell = r"\textit{" + _latex_escape(row["gene"]) + "}"
            fc = row["fold_change_peak_vs_trough"]
            fc_str = f"{fc:.2f}"
            add(
                f"{tissue_cell} & {int(row['rank'])} & {gene_cell} & "
                f"{_latex_escape(row['function'])} & "
                f"{_latex_escape(row['peak_stage'])} & {fc_str} \\\\"
            )
        # Inter-tissue divider (skip after the final tissue handled by lastfoot)
        if tissue != TISSUES[-1]:
            add(r"\midrule")

    add(r"\end{longtable}")
    add(r"\endgroup")

    TEX_PATH.write_text("\n".join(lines) + "\n")
    log.info("Saved %s", TEX_PATH)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Disable external API calls; rely on cache only.",
    )
    args = parser.parse_args(argv)

    if args.no_network:
        # Monkeypatch the queriers to return None so we only use the cache.
        global query_mygene, query_uniprot_by_ensembl, query_uniprot_by_symbol
        query_mygene = lambda _: None  # type: ignore
        query_uniprot_by_ensembl = lambda _: None  # type: ignore
        query_uniprot_by_symbol = lambda _: None  # type: ignore
        log.info("--no-network: API calls disabled")

    log.info("Working dir: %s", PROJECT_ROOT)
    cache = load_cache()
    log.info("Loaded annotation cache with %d entries", len(cache))

    metadata = load_metadata()
    log.info("Loaded metadata: %d rows", len(metadata))

    frames = []
    for tissue in TISSUES:
        log.info("=" * 60)
        log.info("Processing %s", tissue)
        df = per_tissue_top_features(tissue, metadata, cache)
        if df.empty:
            log.error("No rows produced for %s", tissue)
            continue
        frames.append(df)

    if not frames:
        log.error("No tissue produced any output; aborting")
        return 1

    long_df = pd.concat(frames, ignore_index=True)
    long_df.to_csv(CSV_PATH, index=False)
    log.info("Wrote %s (%d rows)", CSV_PATH, len(long_df))

    build_heatmap(long_df)
    build_latex_table(long_df)

    # QC summary
    placeholder = long_df[long_df["annotation_source"] == "none"]
    log.info("QC: %d / %d genes have no curated function (placeholder used)",
             len(placeholder), len(long_df))
    log.info("QC outputs:")
    log.info("  CSV : %s  exists=%s", CSV_PATH, CSV_PATH.exists())
    log.info("  PDF : %s  exists=%s", PDF_PATH, PDF_PATH.exists())
    log.info("  PNG : %s  exists=%s", PNG_PATH, PNG_PATH.exists())
    log.info("  TEX : %s  exists=%s", TEX_PATH, TEX_PATH.exists())
    return 0


if __name__ == "__main__":
    sys.exit(main())
