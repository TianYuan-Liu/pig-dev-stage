"""PigDevStage: postnatal developmental stage of pig tissue from bulk RNA-seq.

Web companion to
    Liu T, Lei R, Khan IM, Theobald P. Tissue identity is the dominant determinant
    of cross-species transferability of a porcine developmental programme.
    BMC Genomics 27:919 (2026). https://doi.org/10.1186/s12864-026-13103-0

One linear PLS predictor per tissue, stored in models/<Tissue>_pls.npz:
    score = ((log2(TPM + 1) - mean) / scale) @ coef + intercept
Predicted stage = nearest stage index to the score. Every number the app shows
comes from models/model_card.json or the npz files, both written by
export_pls_models.py. Runtime needs only streamlit, numpy and pandas.

Run locally:  streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
CARD_PATH = MODELS_DIR / "model_card.json"
EXAMPLE_TPM = APP_DIR / "example" / "muscle_example_tpm.csv"
EXAMPLE_INFO = APP_DIR / "example" / "muscle_example_samples.csv"
EXAMPLE_TISSUE = "Muscle"

# Display order: the cross-study validated tissue first.
TISSUES = ["Muscle", "Liver", "Brain", "Lung", "Blood"]

# Gene-coverage policy for uploads (fraction of the model's genes present).
COVERAGE_WARN = 0.80
COVERAGE_REFUSE = 0.50

# TPM is at most 1e6, so log2(TPM + 1) never exceeds this value.
LOG2_TPM_MAX = float(np.log2(1e6 + 1.0))

PIG_GENE_PREFIX = "ENSSSCG"

# Cross-study validation status per tissue, as stated in the paper.
STATUS = {
    "Muscle": "validated",
    "Liver": "caution",
    "Brain": "exploratory",
    "Lung": "exploratory",
    "Blood": "exploratory",
}
STATUS_LABEL = {
    "validated": "Cross-study validated",
    "caution": "Use with caution",
    "exploratory": "Exploratory",
}

DOI = "10.1186/s12864-026-13103-0"
DOI_URL = f"https://doi.org/{DOI}"
CITATION = (
    "Liu T, Lei R, Khan IM, Theobald P. Tissue identity is the dominant determinant "
    "of cross-species transferability of a porcine developmental programme. "
    "BMC Genomics. 2026;27:919."
)
BIBTEX = """@article{liu2026tissue,
  title   = {Tissue identity is the dominant determinant of cross-species
             transferability of a porcine developmental programme},
  author  = {Liu, Tianyuan and Lei, Rujing and Khan, Ilyas M. and Theobald, Peter},
  journal = {BMC Genomics},
  volume  = {27},
  pages   = {919},
  year    = {2026},
  doi     = {10.1186/s12864-026-13103-0}
}"""
PIGGTEX_URL = "https://piggtex.farmgtex.org/"
PIGGTEX_CITATION = (
    "Teng J, et al. A compendium of genetic regulatory effects across pig tissues. "
    "Nat Genet. 2024;56:112-123."
)
PIGGTEX_DOI_URL = "https://doi.org/10.1038/s41588-023-01585-7"
REPO_URL = "https://github.com/TianYuan-Liu/pig-dev-stage"

SYSTEM_FONT = ('-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
               '"Helvetica Neue", Arial, sans-serif')
CSS = f"""
<style>
html, body, .stApp, h1, h2, h3, h4, h5, h6, p, li, label, input, textarea, select,
button p, [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"] p, [data-testid="stTable"] td, [data-testid="stTable"] th,
[data-baseweb="tab"] p {{
    font-family: {SYSTEM_FONT} !important;
}}
h1, h2, h3, h4 {{ font-weight: 600 !important; letter-spacing: 0 !important; }}
.block-container, [data-testid="stMainBlockContainer"] {{ max-width: 1080px; }}
</style>
"""


# ---------------------------------------------------------------------------
# Input parsing (pure functions; no Streamlit calls)
# ---------------------------------------------------------------------------

def strip_ensembl_version(gene_id) -> str:
    """ENSSSCG00000037539.2 -> ENSSSCG00000037539 (only for Ensembl-style ids)."""
    s = str(gene_id).strip()
    if s.upper().startswith("ENS"):
        return re.sub(r"\.\d+$", "", s)
    return s


def looks_like_pig_gene(label) -> bool:
    return str(label).strip().upper().startswith(PIG_GENE_PREFIX)


def detect_delimiter(filename: str, raw: bytes) -> str:
    """Pick tab, comma or semicolon from the header line; extension breaks ties."""
    first = raw.split(b"\n", 1)[0].decode("utf-8", errors="replace")
    counts = {"\t": first.count("\t"), ",": first.count(","), ";": first.count(";")}
    best = max(counts, key=counts.get)
    ranked = sorted(counts.values(), reverse=True)
    if ranked[0] > 0 and ranked[0] > ranked[1]:
        return best
    name = filename.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    return "," if name.endswith(".csv") else "\t"


def parse_expression_table(raw: bytes, filename: str = "upload.csv") -> tuple[pd.DataFrame, list[str]]:
    """Parse an uploaded table into a samples x genes float DataFrame.

    Genes may be rows or columns. Values that are not numbers become NaN
    (treated later as missing genes for that sample). Ensembl version suffixes
    are stripped and duplicated gene ids averaged. Returns (table, notes);
    raises ValueError for unusable input.
    """
    notes: list[str] = []
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    if not raw.strip():
        raise ValueError("The file is empty.")
    sep = detect_delimiter(filename, raw)
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=sep, index_col=0,
                         encoding="utf-8-sig", encoding_errors="replace")
    except Exception as exc:  # pandas raises several parser error types
        raise ValueError(f"Could not read the file as a delimited table ({exc}).") from exc
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("The table needs a header row, one identifier column and at "
                         "least one column of values.")
    df.index = pd.Index([str(i).strip() for i in df.index])
    df.columns = pd.Index([str(c).strip() for c in df.columns])

    n_row_genes = sum(looks_like_pig_gene(i) for i in df.index)
    n_col_genes = sum(looks_like_pig_gene(c) for c in df.columns)
    if n_row_genes != n_col_genes:
        genes_in_rows = n_row_genes > n_col_genes
    else:
        genes_in_rows = df.shape[0] >= df.shape[1]
        notes.append("No pig Ensembl gene ids (ENSSSCG...) were recognised; the longer "
                     "side of the table was taken to be genes.")
    notes.append("Genes were read from the rows, one column per sample."
                 if genes_in_rows else "Genes were read from the columns, one row per sample.")
    sample_axis = "column" if genes_in_rows else "row"
    gene_axis = "row" if genes_in_rows else "column"

    # genes x samples; coerce one sample at a time
    G = df if genes_in_rows else df.T
    G = G.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    empty_samples = G.isna().all(axis=0).to_numpy()
    if empty_samples.any():
        dropped = [str(c) for c in G.columns[empty_samples]]
        shown = ", ".join(dropped[:5]) + (", ..." if len(dropped) > 5 else "")
        notes.append(f"Ignored {len(dropped)} {sample_axis}(s) with no numeric values: {shown}.")
        G = G.loc[:, ~empty_samples]
    if G.shape[1] == 0:
        raise ValueError("No numeric expression values were found.")

    empty_genes = G.isna().all(axis=1).to_numpy()
    if empty_genes.any():
        notes.append(f"Ignored {int(empty_genes.sum()):,} gene {gene_axis}(s) with no numeric values.")
        G = G.loc[~empty_genes]

    n_missing = int(G.isna().to_numpy().sum())
    if n_missing:
        notes.append(f"{n_missing:,} value(s) could not be read as numbers; they are treated "
                     "as missing and set to the training mean of that gene.")

    stripped = [strip_ensembl_version(g) for g in G.index]
    n_versioned = sum(a != str(b) for a, b in zip(stripped, G.index))
    if n_versioned:
        notes.append(f"Removed version suffixes from {n_versioned:,} gene id(s).")
    G.index = pd.Index(stripped)
    dup = G.index.duplicated(keep=False)
    if dup.any():
        n_dup_ids = int(pd.Index(G.index[dup]).nunique())
        G = G.groupby(level=0, sort=False).mean()
        notes.append(f"{n_dup_ids:,} gene id(s) occurred more than once; their values were averaged.")

    vals = G.to_numpy(dtype=np.float64)
    if np.any(vals[np.isfinite(vals)] < 0):
        raise ValueError("Negative values were found. The app expects TPM or log2(TPM + 1), "
                         "both of which are non-negative.")

    X = G.T.astype(np.float64)
    X.index = pd.Index([str(s) for s in X.index], name="Sample")
    X.columns = pd.Index([str(g) for g in X.columns])
    return X, notes


def looks_log_transformed(X: pd.DataFrame) -> bool:
    """True if no value exceeds the largest possible log2(TPM + 1)."""
    v = X.to_numpy(dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return False
    return bool(v.max() <= LOG2_TPM_MAX)


# ---------------------------------------------------------------------------
# Model and prediction (pure functions)
# ---------------------------------------------------------------------------

def read_model(path: Path) -> dict:
    """Read one <Tissue>_pls.npz written by export_pls_models.py."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    top = pd.DataFrame({
        "Gene id": d["top_vip_gene"].astype(str),
        "Symbol": d["top_vip_symbol"].astype(str),
        "VIP": d["top_vip_value"].astype(np.float64),
        "Coefficient": d["top_vip_coef"].astype(np.float64),
        "Sign": d["top_vip_sign"].astype(int),
    })
    return {
        "tissue": str(d["tissue"]),
        "gene_ids": d["gene_ids"].astype(str),
        "mean": d["mean"].astype(np.float64),
        "scale": d["scale"].astype(np.float64),
        "coef": d["coef"].astype(np.float64),
        "intercept": float(d["intercept"]),
        "kept_stages": [str(s) for s in d["kept_stages"]],
        "stage_n": [int(n) for n in d["stage_n"]],
        "n_train": int(d["n_train"]),
        "n_components": int(d["n_components"]),
        "proba_sigma": float(d["proba_sigma"]),
        "top_genes": top,
    }


def load_card(path: Path = CARD_PATH) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def pls_scores(model: dict, X_log) -> np.ndarray:
    """Continuous stage score; non-finite entries (missing genes) get z = 0."""
    X_log = np.asarray(X_log, dtype=np.float64)
    if X_log.ndim == 1:
        X_log = X_log[None, :]
    Z = (X_log - model["mean"]) / model["scale"]
    Z[~np.isfinite(Z)] = 0.0
    return Z @ model["coef"] + model["intercept"]


def nearest_stage_index(scores, n_stages: int) -> np.ndarray:
    """round(score) clipped to [0, K-1]; exact halves go to the lower stage."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    ks = np.arange(n_stages, dtype=np.float64)
    return np.argmin((s[:, None] - ks[None, :]) ** 2, axis=1)


def stage_probabilities(scores, n_stages: int, sigma: float) -> np.ndarray:
    """Display-only P(c) proportional to exp(-(score - c)^2 / (2 sigma^2))."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    ks = np.arange(n_stages, dtype=np.float64)
    logits = -((s[:, None] - ks[None, :]) ** 2) / (2.0 * sigma ** 2)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def align_to_model(model: dict, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Reorder columns to the model's genes (NaN where absent); return file-level coverage."""
    genes = pd.Index(model["gene_ids"])
    coverage = float(genes.isin(X.columns).mean())
    A = X.reindex(columns=genes).to_numpy(dtype=np.float64)
    return A, coverage


def predict_table(model: dict, X: pd.DataFrame, already_log: bool) -> tuple[pd.DataFrame, float]:
    """Per-sample stage call, score, display probabilities and gene coverage."""
    A, file_coverage = align_to_model(model, X)
    if not already_log:
        with np.errstate(invalid="ignore"):
            A = np.log2(A + 1.0)
    coverage = np.isfinite(A).mean(axis=1)
    scores = pls_scores(model, A)
    stages = model["kept_stages"]
    idx = nearest_stage_index(scores, len(stages))
    probs = stage_probabilities(scores, len(stages), model["proba_sigma"])
    ok = coverage >= COVERAGE_REFUSE

    res = pd.DataFrame(index=pd.Index(X.index.astype(str), name="Sample"))
    res["Predicted stage"] = np.where(ok, np.asarray(stages, dtype=object)[idx], "Not predicted")
    res["Score"] = np.where(ok, scores, np.nan)
    for k, stage in enumerate(stages):
        res[f"P({stage})"] = np.where(ok, probs[:, k], np.nan)
    res["Gene coverage (%)"] = 100.0 * coverage
    return res, file_coverage


# ---------------------------------------------------------------------------
# Wording built from the model card
# ---------------------------------------------------------------------------

def fmt(x, nd: int = 2) -> str:
    return "n/a" if x is None else f"{float(x):.{nd}f}"


def stage_range_text(stage: dict) -> str:
    if stage["max_days"] is None:
        return f"over {stage['min_days'] - 1} days"
    return f"{stage['min_days']}–{stage['max_days']} days"


def status_message(tissue: str, card: dict) -> tuple[str, str]:
    """Return (kind, markdown) describing the tissue's cross-study status."""
    t = card["tissues"][tissue]
    g = t["grouped"]
    kind = STATUS[tissue]
    if kind == "validated":
        text = (
            f"**{tissue}: cross-study validated.** In grouped cross-validation that holds out "
            f"whole studies (BioProjects), the model recovered stage with balanced accuracy "
            f"{fmt(g['BA_mean'])} ± {fmt(g['BA_sd'])}, against a chance level of "
            f"{fmt(t['chance'])} for {t['n_class']} stages, and an ordinal Spearman ρ of "
            f"{fmt(g['rho_mean'])}. Most errors fall between adjacent stages. This is the "
            f"paper's estimate for a new, unseen study."
        )
    elif kind == "caution":
        r = t.get("grouped_restricted", {})
        names = [s.lower() for s in r.get("kept_stages", [])]
        span = f"{names[0]} to {names[-1]}" if names else "the multi-study stages"
        text = (
            f"**{tissue}: use with caution.** At the full {t['n_class']}-stage resolution, the "
            f"cross-study (grouped) balanced accuracy is {fmt(g['BA_mean'])} ± {fmt(g['BA_sd'])}, "
            f"close to chance ({fmt(t['chance'])}). Restricted to the {r.get('n_class', '')} "
            f"stages that span several studies ({span}), the estimate is a modest "
            f"{fmt(r.get('BA_mean'))} ± {fmt(r.get('BA_sd'))} (chance {fmt(r.get('chance'))}; "
            f"ordinal ρ {fmt(r.get('rho_mean'))})."
        )
        share = r.get("largest_cohort_share_pct") or {}
        if "Neonatal" in share and "Adult" in share:
            text += (
                f" Neonatal and adult liver samples come largely from single studies (the "
                f"largest study supplies {share['Neonatal']:.0f}% and {share['Adult']:.0f}% of "
                f"them), so neonatal and adult calls rest on single cohorts."
            )
        else:
            text += " Neonatal and adult calls rest on single cohorts."
    else:
        text = (
            f"**{tissue}: exploratory.** In PigGTEx, {t['single_stage_pct']:.0f}% of the "
            f"retained {tissue.lower()} samples come from studies that sampled a single stage, "
            f"so stage is confounded with study and the paper reports no cross-study accuracy "
            f"for this tissue. Recalibrate on a small labelled in-house cohort before using "
            f"these predictions."
        )
        if tissue == "Blood":
            text += (
                " Blood's top-ranked genes are dominated by Y-chromosome genes (for example "
                "ZFY and EIF1AY), so part of its signal separates male from female animals "
                "(a sex confound) rather than stages."
            )
    return kind, text


def show_status(tissue: str, card: dict) -> None:
    kind, text = status_message(tissue, card)
    if kind == "validated":
        st.info(text)
    else:
        st.warning(text)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_card() -> dict | None:
    return load_card(CARD_PATH)


@st.cache_resource(show_spinner=False)
def cached_model(tissue: str) -> dict | None:
    path = MODELS_DIR / f"{tissue}_pls.npz"
    return read_model(path) if path.exists() else None


@st.cache_data(show_spinner=False, max_entries=4)
def cached_parse(raw: bytes, filename: str) -> tuple[pd.DataFrame, list[str]]:
    return parse_expression_table(raw, filename)


def available_tissues(card: dict) -> list[str]:
    return [t for t in TISSUES
            if t in card.get("tissues", {}) and (MODELS_DIR / f"{t}_pls.npz").exists()]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_predict(card: dict) -> None:
    tissues = available_tissues(card)
    tissue = st.selectbox("Tissue", tissues, key="tissue")
    model = cached_model(tissue)
    stages_by_name = {s["name"]: s for s in card["stages"]}
    kept = model["kept_stages"]
    dropped = [s["name"] for s in card["stages"] if s["name"] not in kept]

    show_status(tissue, card)
    kept_text = "; ".join(f"{s} ({stage_range_text(stages_by_name[s])})" for s in kept)
    msg = f"The {tissue.lower()} model assigns one of {len(kept)} stages: {kept_text}."
    if dropped:
        msg += (f" It was not trained on {', '.join(d.lower() for d in dropped)} samples "
                f"(too few in PigGTEx), so such samples are assigned to the nearest stage it knows.")
    st.write(msg)

    uploaded = st.file_uploader("Expression table (CSV, TSV or TXT; gzip accepted)",
                                type=["csv", "tsv", "txt", "gz"])
    use_example = False
    if EXAMPLE_TPM.exists():
        use_example = st.checkbox(
            "Use the example file: three real PigGTEx muscle samples (raw TPM, all genes)",
            key="use_example",
        )
        st.download_button("Download the example file", EXAMPLE_TPM.read_bytes(),
                           file_name=EXAMPLE_TPM.name, mime="text/csv")

    with st.expander("Input format"):
        st.markdown(
            "**Layout.** A CSV, TSV or TXT table, optionally gzip-compressed, with one header "
            "row and one identifier column. Genes may be rows (one column per sample, as in "
            "PigGTEx downloads) or columns (one row per sample); the app works out which from "
            "the gene identifiers.\n\n"
            "**Gene identifiers.** Pig Ensembl gene ids (ENSSSCG...), as used by PigGTEx. "
            "Version suffixes such as `.2` are removed and ids that occur more than once are "
            "averaged. Columns without numbers, such as a gene-name column, are ignored.\n\n"
            "**Values.** TPM from bulk RNA-seq. The model works on log2(TPM + 1); the app "
            "applies this transform unless your values are already on that scale. This is "
            "detected from the largest value and can be overridden. Negative values are "
            "rejected.\n\n"
            f"**Gene coverage.** Each model uses every gene in its tissue's PigGTEx matrix. "
            f"Genes missing from your file are set to their training mean, so they add "
            f"nothing to the score. The app warns when fewer than {COVERAGE_WARN:.0%} of the "
            f"model's genes are present and does not predict below {COVERAGE_REFUSE:.0%}."
        )
    with st.expander("Other annotations, microarrays, single-cell data and new cohorts"):
        st.markdown(
            "**Gene symbols.** Convert them to pig Ensembl gene ids first (for example with "
            "BioMart, the Ensembl REST API or mygene.info), then check the coverage reported "
            "here.\n\n"
            "**Other assemblies or RefSeq ids.** Map them to the Sscrofa11.1 Ensembl gene ids "
            "used by PigGTEx before upload.\n\n"
            "**Microarrays.** Map probe sets to Ensembl genes, average the probes of each gene "
            "and upload on the log2 scale. The model was trained on RNA-seq TPM, so the scale "
            "differs and the calls are less reliable.\n\n"
            "**Single-cell RNA-seq.** Sum or average cells to one pseudo-bulk profile per "
            "sample before upload.\n\n"
            "**Cohorts unlike PigGTEx.** Within each stage, the PigGTEx cohorts are not "
            "balanced for breed, sex or study, and accuracy varied between breeds in the "
            "paper's subgroup audit. For a cohort dominated by an under-represented breed (for "
            "example only Large White pigs), and for brain, lung and blood in general, "
            "recalibrate first: score a small set of your own samples of known age, compare "
            "the scores with their true stages, and set your own score cut-offs (or refit) "
            "before relying on the calls."
        )

    if uploaded is not None:
        raw, name = uploaded.getvalue(), uploaded.name
        if use_example:
            st.caption("A file was uploaded, so the example is not used.")
            use_example = False
    elif use_example:
        raw, name = EXAMPLE_TPM.read_bytes(), EXAMPLE_TPM.name
        if tissue != EXAMPLE_TISSUE:
            st.warning(f"The example contains {EXAMPLE_TISSUE.lower()} samples; select "
                       f"{EXAMPLE_TISSUE} to score them with the matching model.")
    else:
        return

    try:
        X, notes = cached_parse(raw, name)
    except ValueError as exc:
        st.error(str(exc))
        return
    for note in notes:
        st.caption(note)

    n_model = len(model["gene_ids"])
    n_found = int(pd.Index(model["gene_ids"]).isin(X.columns).sum())
    coverage = n_found / n_model
    st.write(
        f"{X.shape[0]:,} sample(s). Gene coverage: {n_found:,} of {n_model:,} model genes "
        f"found ({coverage:.1%}). Missing genes are set to their training mean (z = 0)."
    )
    if coverage < COVERAGE_REFUSE:
        st.error(
            f"Gene coverage is below {COVERAGE_REFUSE:.0%}, too low for a meaningful "
            f"prediction, so none is made. Check that the gene identifiers are pig Ensembl "
            f"ids (ENSSSCG...) and that the right tissue is selected."
        )
        return
    if coverage < COVERAGE_WARN:
        st.warning(
            f"Gene coverage is below {COVERAGE_WARN:.0%}. Many model genes are imputed with "
            f"their training mean, which pulls scores toward the middle of the stage range. "
            f"Treat these predictions as unreliable."
        )

    file_key = hashlib.sha1(raw).hexdigest()
    if st.session_state.get("_file_key") != file_key or "already_log" not in st.session_state:
        st.session_state["_file_key"] = file_key
        st.session_state["already_log"] = looks_log_transformed(X)
    already_log = st.checkbox(
        "Values are already log2(TPM + 1)",
        key="already_log",
        help="Ticked automatically when no value is larger than a log2(TPM + 1) value can be. "
             "Leave unticked for raw TPM.",
    )

    res, _ = predict_table(model, X, already_log)
    label = STATUS_LABEL[STATUS[tissue]]
    res.insert(1, "Cross-study status", label)
    if use_example and EXAMPLE_INFO.exists():
        info = pd.read_csv(EXAMPLE_INFO, dtype=str).set_index("sample")
        res.insert(1, "Recorded stage", info["recorded_stage"].reindex(res.index).fillna(""))

    st.subheader(f"Predictions: {tissue}")
    show_status(tissue, card)
    n_withheld = int((res["Predicted stage"] == "Not predicted").sum())
    if n_withheld:
        st.warning(f"{n_withheld} sample(s) have gene coverage below {COVERAGE_REFUSE:.0%} and "
                   f"were not predicted.")
    col_cfg = {"Score": st.column_config.NumberColumn(format="%.2f"),
               "Gene coverage (%)": st.column_config.NumberColumn(format="%.1f")}
    for stage in kept:
        col_cfg[f"P({stage})"] = st.column_config.NumberColumn(format="%.2f")
    st.dataframe(res, column_config=col_cfg)

    if use_example:
        st.caption(
            "These three samples were part of the muscle training data. Their calls are shown "
            "as the model makes them, including disagreements with the recorded stage; the "
            "example illustrates the input format and is not an accuracy estimate."
        )

    out = res.copy()
    out.insert(0, "Tissue", tissue)
    st.download_button("Download predictions (CSV)", out.to_csv().encode("utf-8"),
                       file_name=f"pigdevstage_{tissue.lower()}_predictions.csv",
                       mime="text/csv")

    sigma = fmt(model["proba_sigma"], 1)
    first, last = kept[0], kept[-1]
    with st.expander("How to read the output", expanded=False):
        st.markdown(
            f"**Predicted stage** is the stage whose index is nearest to the score.\n\n"
            f"**Score** is the PLS output on the stage-index scale, from 0 ({first}) to "
            f"{len(kept) - 1} ({last}). A score between two whole numbers lies between two "
            f"stages; scores below 0 or above {len(kept) - 1} are assigned to the first or "
            f"last stage.\n\n"
            f"**P(stage)** converts the score into a spread over stages with a Gaussian kernel "
            f"(σ = {sigma}). It is for display only: it does not change the call and is not a "
            f"calibrated probability of being correct.\n\n"
            f"**Gene coverage** is the share of the model's genes with a value for that "
            f"sample.\n\n"
            f"**Cross-study status** repeats the validation status of the {tissue.lower()} "
            f"model shown above, so that it travels with every exported prediction."
        )


def page_performance(card: dict) -> None:
    st.header("Performance")
    ev = card.get("evaluation", {})
    st.write(
        "The paper separates two questions. Cross-study accuracy asks whether a model "
        "deploys to a study it has never seen; it is estimated by grouped cross-validation "
        f"that holds out whole BioProjects ({ev.get('n_folds', '')} folds, "
        f"{ev.get('n_seeds', '')} random seeds). In-distribution separability asks only "
        "whether stage is encoded in the transcriptome; stratified cross-validation puts "
        "samples from the same study in both training and test folds, so it overstates "
        "accuracy on new data. Chance is 1/K for K stages."
    )

    st.subheader("Cross-study accuracy (deployment estimate)")
    rows = []
    for tissue in TISSUES:
        t = card["tissues"].get(tissue)
        if t is None:
            continue
        kind = STATUS[tissue]
        g = t["grouped"]
        if kind == "validated":
            ba, rho, chance = f"{fmt(g['BA_mean'])} ± {fmt(g['BA_sd'])}", fmt(g["rho_mean"]), fmt(t["chance"])
        elif kind == "caution":
            r = t.get("grouped_restricted", {})
            ba = f"{fmt(r.get('BA_mean'))} ± {fmt(r.get('BA_sd'))} ({r.get('n_class', '')} stages)"
            rho, chance = fmt(r.get("rho_mean")), fmt(r.get("chance"))
        else:
            ba, rho, chance = "Not identifiable", "-", fmt(t["chance"])
        rows.append({
            "Tissue": tissue,
            "Status": STATUS_LABEL[kind],
            "Stages": t["n_class"],
            "Samples": f"{t['n_train']:,}",
            "BioProjects": t["n_bioprojects_train"],
            "Single-stage studies": f"{t['single_stage_pct']:.0f}%",
            "Balanced accuracy": ba,
            "Ordinal ρ": rho,
            "Chance": chance,
        })
    st.table(pd.DataFrame(rows).set_index("Tissue"))

    notes = []
    liver = card["tissues"].get("Liver")
    if liver:
        g, r = liver["grouped"], liver.get("grouped_restricted", {})
        names = ", ".join(s.lower() for s in r.get("kept_stages", []))
        notes.append(
            f"Liver: the table gives the estimate over the {r.get('n_class', '')} stages that "
            f"span several studies ({names}). Over all {liver['n_class']} stages the "
            f"cross-study balanced accuracy is {fmt(g['BA_mean'])} ± {fmt(g['BA_sd'])} "
            f"(chance {fmt(liver['chance'])}), because the neonatal and adult samples each come "
            f"mainly from one study."
        )
    notes.append(
        "Not identifiable: most or all retained stages of these tissues come from studies "
        "that sampled a single stage, so a held-out study presents a stage the model has "
        "barely seen and stage cannot be separated from study. The paper reports no "
        "cross-study estimate for them."
    )
    notes.append(
        "Samples and BioProjects count the retained samples (stages with enough samples) used "
        "to fit each model. Single-stage studies is the share of those samples drawn from "
        "BioProjects that sampled one stage only."
    )
    for n in notes:
        st.caption(n)

    with st.expander("Per-stage recall under grouped cross-validation"):
        for tissue in ["Muscle", "Liver"]:
            t = card["tissues"].get(tissue)
            if not t:
                continue
            rec = t["grouped"].get("per_class_recall") or {}
            if rec:
                st.markdown(f"**{tissue}, all {t['n_class']} stages**")
                st.table(pd.DataFrame({
                    "Recall (mean)": {s: fmt(v["mean"]) for s, v in rec.items()},
                    "Recall (sd)": {s: fmt(v["sd"]) for s, v in rec.items()},
                }))
            rrec = (t.get("grouped_restricted") or {}).get("per_class_recall") or {}
            if rrec:
                st.markdown(f"**{tissue}, {t['grouped_restricted']['n_class']} multi-study stages**")
                st.table(pd.DataFrame({"Recall (mean over seeds)": {s: fmt(v) for s, v in rrec.items()}}))

    st.subheader("In-distribution separability (not a deployment estimate)")
    st.warning(
        "These numbers come from stratified cross-validation, in which samples from the same "
        "study appear in both training and test folds. They show that stage is encoded in "
        "every tissue, but they are in-distribution figures and should not be read as the "
        "accuracy to expect on a new study."
    )
    rows = []
    for tissue in TISSUES:
        t = card["tissues"].get(tissue)
        if t is None:
            continue
        s = t["stratified"]
        rows.append({
            "Tissue": tissue,
            "Stages": t["n_class"],
            "Balanced accuracy": f"{fmt(s.get('BA_mean'))} ± {fmt(s.get('BA_sd'))}",
            "Ordinal ρ": fmt(s.get("rho_mean")),
            "Chance": fmt(t["chance"]),
        })
    st.table(pd.DataFrame(rows).set_index("Tissue"))

    st.subheader("Top genes")
    tissues = available_tissues(card)
    if not tissues:
        return
    tissue = st.selectbox("Tissue", tissues, key="genes_tissue")
    model = cached_model(tissue)
    top = model["top_genes"].copy()
    top.insert(0, "Rank", np.arange(1, len(top) + 1))
    top["Direction"] = np.where(top["Sign"] > 0, "rises with age",
                                np.where(top["Sign"] < 0, "falls with age", "flat"))
    top["VIP"] = top["VIP"].map(lambda v: f"{v:.2f}")
    top["Symbol"] = top["Symbol"].replace("", "-")
    st.write(
        f"The {len(top)} genes with the highest Variable Importance in Projection (VIP) in the "
        f"{tissue.lower()} model, fitted on all retained samples. VIP is normalised so that its "
        f"mean square is 1, so values above 1 mark above-average contributors. Direction is "
        f"the sign of the gene's PLS coefficient on the stage index. Unannotated pig loci have "
        f"no symbol."
    )
    if tissue == "Blood":
        st.warning("Blood is a sex-confound negative control in the paper: its top genes are "
                   "dominated by Y-chromosome genes.")
    st.dataframe(top[["Rank", "Gene id", "Symbol", "VIP", "Direction"]].set_index("Rank"))


def page_stages(card: dict) -> None:
    st.header("Stage definitions")
    st.write(
        "Ages in days are grouped into six stages anchored on porcine physiological "
        "milestones (the perinatal transition, weaning, early juvenile growth, the gilt "
        "pubertal window and reproductive maturity). Cut points are upper-inclusive."
    )
    st.table(pd.DataFrame(
        [{"Stage": s["name"], "Age": stage_range_text(s)} for s in card["stages"]]
    ).set_index("Stage"))

    min_n = card.get("model", {}).get("min_samples_per_stage")
    st.subheader("Stages modelled per tissue")
    st.write(
        f"In each tissue, stages with fewer than {min_n} samples are dropped rather than merged "
        f"into a neighbour, and the remaining stages are numbered 0 to K - 1 in developmental "
        f"order. Each model therefore works at the resolution its data support. Cells give the "
        f"number of training samples; a dash marks a stage the model does not know."
    )
    names = [s["name"] for s in card["stages"]]
    rows = []
    for tissue in TISSUES:
        t = card["tissues"].get(tissue)
        if t is None:
            continue
        n_by = {s["name"]: s["n"] for s in t["kept_stages"]}
        row = {"Tissue": tissue}
        row.update({n: (f"{n_by[n]:,}" if n in n_by else "-") for n in names})
        row["K"] = t["n_class"]
        rows.append(row)
    st.table(pd.DataFrame(rows).set_index("Tissue"))
    brain = card["tissues"].get("Brain", {})
    if brain.get("regions"):
        st.caption("Brain is a forebrain aggregate of PigGTEx samples labelled "
                   + ", ".join(brain["regions"]) + ".")


def page_methods(card: dict) -> None:
    st.header("Methods")
    m = card.get("model", {})
    genes = sorted({t.get("n_genes") for t in card["tissues"].values() if t.get("n_genes")})
    gene_text = (f"{genes[0]:,}" if len(genes) == 1 else
                 ", ".join(f"{g:,}" for g in genes)) if genes else "all"
    ev = card.get("evaluation", {})
    st.markdown(
        f"**Data.** Bulk RNA-seq TPM matrices and sample metadata from PigGTEx, one matrix per "
        f"tissue (muscle, liver, lung, blood and brain; brain is a forebrain aggregate). Ages "
        f"are mapped to the six stages on the Stages tab, and stages with fewer than "
        f"{m.get('min_samples_per_stage', '')} samples in a tissue are dropped.\n\n"
        f"**Features.** log2(TPM + 1) of every gene in the tissue's matrix ({gene_text} pig "
        f"Ensembl genes), with no variance filter. Each gene is z-scored with the mean and "
        f"standard deviation of the training samples; genes that are constant in training "
        f"get a standard deviation of 1.\n\n"
        f"**Model.** One partial least squares (PLS) regression per tissue "
        f"({m.get('n_components', '')} latent components, scikit-learn PLSRegression with "
        f"scale=False) regresses the integer stage index 0 to K - 1 on the z-scored genes. "
        f"The fitted model reduces to one weight per gene, which is what this app stores and "
        f"applies.\n\n"
        f"**Prediction.** The predicted stage is the score rounded to the nearest stage index, "
        f"clipped to the modelled range. Display probabilities use a Gaussian-distance softmax, "
        f"P(c) proportional to exp(-(score - c)² / (2σ²)) with σ = {fmt(m.get('proba_sigma'), 1)}; "
        f"they never change the call.\n\n"
        f"**Gene importance.** Variable Importance in Projection (VIP), computed per latent "
        f"component and normalised so that the mean squared VIP is 1. The sign of a gene's "
        f"coefficient separates genes that rise with stage from genes that fall.\n\n"
        f"**Evaluation.** Stratified cross-validation (in-distribution) and grouped "
        f"cross-validation holding out whole BioProjects (deployment), {ev.get('n_folds', '')} "
        f"folds × {ev.get('n_seeds', '')} seeds; balanced accuracy and ordinal Spearman ρ. The "
        f"models served here are refitted on all retained samples of each tissue.\n\n"
        f"**Missing genes.** A gene absent from an upload is set to its training mean "
        f"(z = 0), which adds nothing to the score."
    )
    exp = card.get("export", {})
    pk = exp.get("packages", {})
    st.caption(
        f"Models exported {exp.get('timestamp_utc', 'n/a')} with "
        + ", ".join(f"{k} {v}" for k, v in pk.items())
        + (f"; repository commit {exp['git']['commit'][:10]}"
           if exp.get("git", {}).get("commit") else "")
        + "."
    )


def page_cite() -> None:
    st.header("How to cite")
    st.markdown(f"{CITATION} [https://doi.org/{DOI}]({DOI_URL})")
    st.code(BIBTEX, language="bibtex")
    st.subheader("Data source")
    st.markdown(
        f"All models are trained on public bulk RNA-seq from PigGTEx, the pig component of the "
        f"Farm Animal Genotype-Tissue Expression project ([piggtex.farmgtex.org]({PIGGTEX_URL})). "
        f"Please also cite: {PIGGTEX_CITATION} [doi.org/10.1038/s41588-023-01585-7]({PIGGTEX_DOI_URL})"
    )
    st.subheader("Code")
    st.markdown(f"Source code for the analysis and this app: [{REPO_URL}]({REPO_URL})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="PigDevStage", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.title("PigDevStage")
    st.write(
        "Estimate the postnatal developmental stage of a pig tissue sample from its bulk "
        "RNA-seq profile. Companion to Liu et al., "
        f"[BMC Genomics 27:919 (2026)]({DOI_URL})."
    )

    card = cached_card()
    if card is None or not available_tissues(card):
        st.error("Model files were not found. Run `python streamlit_app/export_pls_models.py` "
                 "from the repository root to create streamlit_app/models/.")
        return

    tab_predict, tab_perf, tab_stages, tab_methods, tab_cite = st.tabs(
        ["Predict", "Performance", "Stages", "Methods", "Cite and data"]
    )
    with tab_predict:
        page_predict(card)
    with tab_perf:
        page_performance(card)
    with tab_stages:
        page_stages(card)
    with tab_methods:
        page_methods(card)
    with tab_cite:
        page_cite()


if __name__ == "__main__":
    main()
