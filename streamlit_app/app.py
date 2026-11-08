"""
Streamlit web app for porcine developmental stage prediction.

Loads pre-trained OrdinalLightGBM models (one per tissue) and predicts
developmental stages from user-uploaded gene expression data.

Run:
    streamlit run streamlit_app/app.py
"""

import hashlib
import json
import re
import sys
from pathlib import Path

# Project root must be on sys.path so joblib can unpickle OrdinalLightGBM
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).resolve().parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
_FALLBACK_RESULTS_DIR = PROJECT_ROOT / "machine_learning" / "model_outputs"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_HIRES_PATH = ASSETS_DIR / "cardiff_logo_hires.png"
LOGO_PATH = LOGO_HIRES_PATH if LOGO_HIRES_PATH.exists() else ASSETS_DIR / "cardiff_logo.png"
STUDY_DESIGN_PATH = PROJECT_ROOT / "paper" / "figures" / "output" / "png" / "fig1_study_design.png"
TISSUES = ["Muscle", "Liver", "Blood", "Brain", "Lung"]

# Ordinal color palette (light → dark for developmental progression)
STAGE_COLORS_4 = ["#a8d8ea", "#3dc1d3", "#e77f67", "#c0392b"]
STAGE_COLORS_2 = ["#a8d8ea", "#e77f67"]

# Stage definitions with age boundaries and colors
STAGE_DEFINITIONS = [
    {"name": "Infant", "min_days": 0, "max_days": 20, "color": "#a8d8ea"},
    {"name": "Early childhood", "min_days": 21, "max_days": 59, "color": "#3dc1d3"},
    {"name": "Pre-pubertal", "min_days": 60, "max_days": 149, "color": "#e77f67"},
    {"name": "Post-pubertal", "min_days": 150, "max_days": 365, "color": "#c0392b"},
    {"name": "Adult", "min_days": 366, "max_days": 730, "color": "#7b241c"},
]


# ---------------------------------------------------------------------------
# Model & data loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    """Load all serialized model artifacts."""
    models = {}
    for tissue in TISSUES:
        path = MODELS_DIR / f"{tissue}.joblib"
        if path.exists():
            models[tissue] = joblib.load(path)
    return models


@st.cache_data
def load_ml_results():
    """Load per-tissue ML results JSONs."""
    results = {}
    for tissue in TISSUES:
        path = RESULTS_DIR / f"{tissue}_results.json"
        if not path.exists():
            path = _FALLBACK_RESULTS_DIR / f"{tissue}_results.json"
        if path.exists():
            with open(path) as f:
                results[tissue] = json.load(f)
    return results


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def inject_custom_css():
    st.markdown("""
    <style>
    /*
     * Theme-agnostic design: uses inherit / rgba overlays so custom
     * elements adapt to whichever Streamlit theme is active without
     * relying on prefers-color-scheme (which tracks the OS, not
     * Streamlit's own light/dark toggle).
     */
    :root {
        --cardiff-red: #9B1B30;
        --cardiff-red-dark: #7A1526;
        --cardiff-gold: #D4A843;
        --cardiff-gold-light: #E8C97A;
        --radius: 12px;
        --radius-lg: 16px;
        --overlay-subtle: rgba(128,128,128,0.08);
        --overlay-border: rgba(128,128,128,0.18);
    }

    /* ── Metric cards (sidebar) ────────────────────────── */
    div[data-testid="stMetric"] {
        background: var(--overlay-subtle);
        border: 1px solid var(--overlay-border);
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* Tab panel padding */
    div[data-testid="stTabPanel"] {
        padding-top: 0.5rem;
    }

    /* ── Scheme badge ──────────────────────────────────── */
    .scheme-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: 600;
        color: white;
    }
    .scheme-4class { background: #27ae60; }
    .scheme-2class { background: #2980b9; }

    /* ── Citation box ──────────────────────────────────── */
    .citation-box {
        background: var(--overlay-subtle);
        border-left: 4px solid var(--cardiff-red);
        border-radius: 0 var(--radius) var(--radius) 0;
        padding: 1.2rem 1.4rem;
        font-size: 0.85rem;
        color: inherit;
        opacity: 0.85;
        line-height: 1.65;
    }
    .citation-box .citation-header {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--cardiff-red);
        opacity: 1;
        margin-bottom: 0.5rem;
    }

    /* ── Figure caption ────────────────────────────────── */
    .figure-caption {
        text-align: center;
        font-style: italic;
        font-size: 0.85rem;
        color: inherit;
        opacity: 0.6;
        margin-top: 0.5rem;
        padding: 0 2rem;
        line-height: 1.5;
    }
    .figure-caption strong {
        font-style: normal;
        opacity: 0.8;
    }

    /* ── Sidebar panel ─────────────────────────────────── */
    .sidebar-panel {
        background: var(--overlay-subtle);
        border: 1px solid var(--overlay-border);
        border-radius: var(--radius);
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .sidebar-panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.8rem;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid var(--overlay-border);
    }
    .sidebar-tissue-name {
        font-size: 1.1rem;
        font-weight: 800;
        color: var(--cardiff-red);
    }
    .sidebar-metrics-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.6rem;
    }
    .sidebar-metric {
        text-align: center;
        padding: 0.5rem 0.3rem;
        border-radius: 8px;
        background: var(--overlay-subtle);
    }
    .sidebar-metric-val {
        font-size: 1.15rem;
        font-weight: 800;
        color: var(--cardiff-red);
    }
    .sidebar-metric-lbl {
        font-size: 0.65rem;
        font-weight: 600;
        color: inherit;
        opacity: 0.55;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .sidebar-ci {
        text-align: center;
        font-size: 0.72rem;
        color: inherit;
        opacity: 0.55;
        margin-top: 0.5rem;
    }
    .sidebar-section {
        margin-top: 0.5rem;
    }
    .sidebar-section-title {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: inherit;
        opacity: 0.55;
        margin-bottom: 0.5rem;
    }
    .sidebar-stage {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0;
        font-size: 0.85rem;
        color: inherit;
    }
    .sidebar-stage-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .sidebar-feature-count {
        margin-top: 0.7rem;
        padding-top: 0.6rem;
        border-top: 1px solid currentColor;
        border-top-color: rgba(128,128,128,0.25);
        font-size: 0.8rem;
        color: inherit;
        opacity: 0.55;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* ── Sidebar app header ───────────────────────────── */
    .sidebar-app-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--cardiff-red);
        padding: 0.2rem 0 0.6rem 0;
        letter-spacing: 0.02em;
        line-height: 1.3;
    }
    .sidebar-app-header span {
        display: block;
        font-size: 0.75rem;
        font-weight: 400;
        color: inherit;
        opacity: 0.55;
        letter-spacing: 0;
        margin-top: 0.15rem;
    }

    /* ── Tissue context banner ────────────────────────── */
    .tissue-context-banner {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.6rem 1rem;
        margin-bottom: 1rem;
        background: var(--overlay-subtle);
        border-left: 4px solid var(--cardiff-red);
        border-radius: 0 6px 6px 0;
    }
    .tissue-context-banner .tcb-tissue {
        font-weight: 700;
        font-size: 1.05rem;
        color: var(--cardiff-red);
    }
    .tissue-context-banner .tcb-scheme {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.15rem 0.5rem;
        border-radius: 10px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .tissue-context-banner .tcb-scheme.scheme-4class {
        background: #e8d5f5;
        color: #6c3d8f;
    }
    .tissue-context-banner .tcb-scheme.scheme-2class {
        background: #d5eaff;
        color: #2a6cb6;
    }
    .tissue-context-banner .tcb-stat {
        font-size: 0.85rem;
        color: inherit;
        opacity: 0.7;
    }
    .tissue-context-banner .tcb-stat strong {
        color: inherit;
        opacity: 1;
    }

    /* ── Section header ────────────────────────────────── */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid var(--border);
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _tissue_banner_html(tissue: str, scheme: str, ba: float) -> str:
    """Return HTML for the tissue-context banner shown on Predict / Performance tabs."""
    badge_cls = "scheme-4class" if scheme == "4-class" else "scheme-2class"
    return (
        '<div class="tissue-context-banner">'
        f'<span class="tcb-tissue">{tissue}</span>'
        f'<span class="tcb-scheme {badge_cls}">{scheme}</span>'
        f'<span class="tcb-stat">BA: <strong>{ba:.3f}</strong></span>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Data parsing helpers
# ---------------------------------------------------------------------------

def strip_ensembl_version(gene_id: str) -> str:
    """ENSSSCG00000037539.2 → ENSSSCG00000037539"""
    return re.sub(r"\.\d+$", "", str(gene_id))


def detect_delimiter(filename: str, raw_bytes: bytes) -> str:
    name = filename.removesuffix(".gz")
    if name.endswith((".tsv", ".txt")):
        return "\t"
    first_line = raw_bytes.split(b"\n")[0].decode("utf-8", errors="replace")
    if first_line.count("\t") > first_line.count(","):
        return "\t"
    return ","


def parse_upload(uploaded_file) -> pd.DataFrame:
    """Parse uploaded CSV/TSV into a samples x genes DataFrame with Ensembl IDs."""
    raw = uploaded_file.getvalue()

    # Decompress gzip if needed
    if raw[:2] == b"\x1f\x8b":
        import gzip
        raw = gzip.decompress(raw)

    sep = detect_delimiter(uploaded_file.name, raw)

    from io import BytesIO
    df = pd.read_csv(BytesIO(raw), sep=sep, index_col=0)

    # Force numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    n_nan = df.isna().sum().sum()
    if n_nan > 0:
        st.warning(f"{n_nan} non-numeric values were coerced to NaN and filled with 0.")
        df = df.fillna(0.0)

    # Detect orientation: genes as rows vs samples as rows
    index_has_ensembl = sum(1 for g in df.index if str(g).startswith("ENSSSCG")) > len(df) * 0.5
    cols_have_ensembl = sum(1 for g in df.columns if str(g).startswith("ENSSSCG")) > len(df.columns) * 0.5

    if index_has_ensembl and not cols_have_ensembl:
        df = df.T
    elif not cols_have_ensembl and not index_has_ensembl:
        if df.shape[0] > df.shape[1] * 5:
            df = df.T

    # Strip version suffixes from gene columns
    df.columns = [strip_ensembl_version(c) for c in df.columns]

    # Handle duplicate gene IDs (average)
    if df.columns.duplicated().any():
        n_dups = df.columns.duplicated().sum()
        st.warning(f"{n_dups} duplicate gene IDs found — averaging their values.")
        df = df.T.groupby(level=0).mean().T

    # Check for negative values
    if (df < 0).any().any():
        st.error("Negative values detected. TPM expression values cannot be negative.")
        st.stop()

    return df


# ---------------------------------------------------------------------------
# Gene alignment
# ---------------------------------------------------------------------------

def align_genes(user_df: pd.DataFrame, expected_genes: list) -> tuple:
    """Align user genes to the model's expected feature set."""
    user_genes_stripped = {strip_ensembl_version(g): g for g in user_df.columns}

    matched = {}
    for exp_gene in expected_genes:
        if exp_gene in user_genes_stripped:
            matched[exp_gene] = user_genes_stripped[exp_gene]

    n_matched = len(matched)
    n_expected = len(expected_genes)

    aligned = pd.DataFrame(0.0, index=user_df.index, columns=expected_genes)
    for exp_gene, user_gene in matched.items():
        aligned[exp_gene] = user_df[user_gene].values

    return aligned, n_matched, n_expected


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def detect_log_transformed(df: pd.DataFrame) -> bool:
    """Heuristic: check distribution shape to detect log2-transformed data."""
    max_val = df.max().max()
    median_val = df.median().median()
    if max_val > 30:
        return False
    if max_val < 18 and median_val < 10:
        return True
    return False


def predict_stages(model_artifact: dict, aligned_df: pd.DataFrame, is_log: bool):
    """Run prediction pipeline. Returns results DataFrame."""
    model = model_artifact["model"]
    class_labels = model_artifact["class_labels"]

    if is_log:
        X = aligned_df.values
    else:
        X = np.log2(aligned_df.values + 1)

    X_df = pd.DataFrame(X, index=aligned_df.index, columns=aligned_df.columns)

    proba = model.predict_proba(X_df)
    preds = model.predict(X_df)

    pred_labels = [class_labels[int(p)] for p in preds]
    confidence = proba.max(axis=1)

    results = pd.DataFrame({
        "Sample": aligned_df.index,
        "Predicted Stage": pred_labels,
        "Confidence": confidence,
    })

    for i, label in enumerate(class_labels):
        results[f"P({label})"] = proba[:, i]

    results = results.set_index("Sample")
    return results


# ---------------------------------------------------------------------------
# Visualization — existing
# ---------------------------------------------------------------------------

def plot_probabilities(results: pd.DataFrame, class_labels: list):
    """Horizontal stacked bar chart of per-class probabilities."""
    prob_cols = [f"P({label})" for label in class_labels]
    prob_data = results[prob_cols]

    n_classes = len(class_labels)
    colors = STAGE_COLORS_4[:n_classes] if n_classes <= 4 else STAGE_COLORS_4
    if n_classes == 2:
        colors = STAGE_COLORS_2

    n_samples = len(results)
    fig_height = max(2.5, 0.4 * n_samples + 1)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = np.arange(n_samples)
    left = np.zeros(n_samples)

    for i, col in enumerate(prob_cols):
        widths = prob_data[col].values
        ax.barh(y_pos, widths, left=left, color=colors[i], label=class_labels[i],
                edgecolor="white", linewidth=0.5)
        left += widths

    ax.set_yticks(y_pos)
    ax.set_yticklabels(results.index, fontsize=8)
    ax.set_xlabel("Probability")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Predicted Stage Probabilities")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Visualization — ML performance
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, class_labels):
    """Annotated heatmap of the confusion matrix with counts and row percentages."""
    cm = np.array(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = np.where(row_sums > 0, cm / row_sums * 100, 0)

    n = len(class_labels)
    fig, ax = plt.subplots(figsize=(max(4, n * 1.2), max(3.5, n * 1.0)))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            pct = row_pct[i, j]
            color = "white" if val > cm.max() * 0.6 else "black"
            ax.text(j, i, f"{val}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_labels, fontsize=8, rotation=30, ha="right")
    ax.set_yticklabels(class_labels, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title("Confusion Matrix", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig


def plot_per_class_metrics(precision, recall, f1, class_labels):
    """Grouped bar chart of precision, recall, and F1 per class."""
    n = len(class_labels)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(5, n * 1.5), 4))
    ax.bar(x - width, precision, width, label="Precision", color="#3498db", edgecolor="white")
    ax.bar(x, recall, width, label="Recall", color="#e74c3c", edgecolor="white")
    ax.bar(x + width, f1, width, label="F1", color="#2ecc71", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-Class Metrics", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # Annotate values
    for bars in ax.containers:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=6.5)

    plt.tight_layout()
    return fig


def plot_cv_folds(cv_data):
    """Bar chart of per-fold balanced accuracy with mean line and SD band."""
    agg = cv_data["aggregated_metrics"]["balanced_accuracy"]
    fold_scores = agg["per_fold"]
    mean_ba = agg["mean"]
    std_ba = agg["std"]
    n_folds = len(fold_scores)

    fig, ax = plt.subplots(figsize=(max(5, n_folds * 1.2), 4))
    x = np.arange(n_folds)
    bars = ax.bar(x, fold_scores, color="#3498db", edgecolor="white", width=0.6)

    # Mean line
    ax.axhline(mean_ba, color="#e74c3c", linewidth=2, linestyle="--",
               label=f"Mean = {mean_ba:.3f}")
    # SD band
    ax.axhspan(mean_ba - std_ba, mean_ba + std_ba, alpha=0.15, color="#e74c3c",
               label=f"\u00b11 SD = {std_ba:.3f}")

    # Annotate bars
    for bar, score in zip(bars, fold_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{score:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i + 1}" for i in range(n_folds)], fontsize=9)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_ylim(0, min(1.15, max(fold_scores) + 0.1))
    ax.set_title("Cross-Validation Performance", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    return fig


def plot_feature_importance(feat_imp, top_n=20):
    """Horizontal bar chart of top N feature importances."""
    # feat_imp is dict {gene_id: importance}
    sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
    genes = [g for g, _ in reversed(sorted_feats)]
    importances = [v for _, v in reversed(sorted_feats)]

    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.3)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(genes)))
    ax.barh(range(len(genes)), importances, color=colors, edgecolor="white")
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=7)
    ax.set_xlabel("Importance (split gain)")
    ax.set_title(f"Top {top_n} Features", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_stage_timeline():
    """Horizontal bar chart showing age ranges for each developmental stage."""
    fig, ax = plt.subplots(figsize=(9, 3))

    for i, stage in enumerate(STAGE_DEFINITIONS):
        width = stage["max_days"] - stage["min_days"]
        ax.barh(i, width, left=stage["min_days"], color=stage["color"],
                edgecolor="white", height=0.6)
        # Center label
        cx = stage["min_days"] + width / 2
        ax.text(cx, i, f"{stage['name']}\n({stage['min_days']}-{stage['max_days']}d)",
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="white" if i >= 2 else "black")

    ax.set_yticks(range(len(STAGE_DEFINITIONS)))
    ax.set_yticklabels([s["name"] for s in STAGE_DEFINITIONS], fontsize=9)
    ax.set_xlabel("Age (days)")
    ax.set_title("Developmental Stage Timeline", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def plot_tissue_comparison(ml_results):
    """Grouped bar chart comparing balanced accuracy across all tissues."""
    tissues = []
    ba_values = []
    sample_counts = []
    schemes = []

    for tissue in TISSUES:
        if tissue in ml_results:
            r = ml_results[tissue]
            tissues.append(tissue)
            ba_values.append(r["metrics"]["balanced_accuracy"])
            sample_counts.append(r["n_samples"])
            schemes.append(r["scheme"])

    n = len(tissues)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.5), 4.5))
    x = np.arange(n)

    colors = ["#27ae60" if s == "4-class" else "#2980b9" for s in schemes]
    bars = ax.bar(x, ba_values, color=colors, edgecolor="white", width=0.6)

    # Annotate with BA and sample count
    for bar, ba, ns in zip(bars, ba_values, sample_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"BA={ba:.3f}\nn={ns}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(tissues, fontsize=10)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_ylim(0, min(1.15, max(ba_values) + 0.12))
    ax.set_title("Cross-Tissue Model Comparison", fontsize=12, fontweight="bold")

    # Legend for scheme colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#27ae60", label="4-class"),
        Patch(facecolor="#2980b9", label="2-class"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Porcine Developmental Stage Predictor",
        page_icon="\U0001f437",
        layout="wide",
    )

    inject_custom_css()

    models = load_models()
    ml_results = load_ml_results()

    if not models:
        st.error(
            "No models found. Run `python streamlit_app/export_models.py` first "
            "to export trained models."
        )
        return

    # --- Sidebar ---
    st.sidebar.markdown(
        '<div class="sidebar-app-header">'
        'PigDevStage'
        '<span>Porcine Developmental Stage Predictor</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    tissue = st.sidebar.selectbox("Select Tissue", list(models.keys()))
    artifact = models[tissue]
    tissue_results = ml_results.get(tissue, {})
    metrics = tissue_results.get("metrics", {})
    scheme = artifact["scheme"]
    n_classes = len(artifact["class_labels"])

    # Build a single styled HTML panel for the selected tissue
    if metrics:
        ba = metrics.get("balanced_accuracy", 0)
        f1_val = metrics.get("f1_macro", 0)
        n_samples = tissue_results.get("n_samples", 0)
        bootstrap = metrics.get("bootstrap", {})
        ba_boot = bootstrap.get("balanced_accuracy", {})
        ci_str = ""
        if ba_boot:
            ci_str = f"95% CI [{ba_boot.get('ci_lower', 0):.3f}, {ba_boot.get('ci_upper', 0):.3f}]"

        badge_cls = "scheme-4class" if scheme == "4-class" else "scheme-2class"

        st.sidebar.markdown(
            f'<div class="sidebar-panel">'
            f'<div class="sidebar-panel-header">'
            f'  <span class="sidebar-tissue-name">{tissue}</span>'
            f'  <span class="scheme-badge {badge_cls}">{scheme}</span>'
            f'</div>'
            f'<div class="sidebar-metrics-grid">'
            f'  <div class="sidebar-metric">'
            f'    <div class="sidebar-metric-val">{ba:.3f}</div>'
            f'    <div class="sidebar-metric-lbl">Balanced Acc</div>'
            f'  </div>'
            f'  <div class="sidebar-metric">'
            f'    <div class="sidebar-metric-val">{f1_val:.3f}</div>'
            f'    <div class="sidebar-metric-lbl">F1 Macro</div>'
            f'  </div>'
            f'  <div class="sidebar-metric">'
            f'    <div class="sidebar-metric-val">{n_samples:,}</div>'
            f'    <div class="sidebar-metric-lbl">Samples</div>'
            f'  </div>'
            f'  <div class="sidebar-metric">'
            f'    <div class="sidebar-metric-val">{n_classes}</div>'
            f'    <div class="sidebar-metric-lbl">Classes</div>'
            f'  </div>'
            f'</div>'
            + (f'<div class="sidebar-ci">{ci_str}</div>' if ci_str else "")
            + '</div>',
            unsafe_allow_html=True,
        )

    # Stage labels with colored dots
    stage_html_items = []
    for label in artifact["class_labels"]:
        # Match stage color from STAGE_DEFINITIONS
        color = "#9B1B30"  # default fallback
        for sd in STAGE_DEFINITIONS:
            if sd["name"] == label or label.startswith(sd["name"]):
                color = sd["color"]
                break
        stage_html_items.append(
            f'<div class="sidebar-stage">'
            f'<span class="sidebar-stage-dot" style="background:{color}"></span>'
            f'{label}</div>'
        )

    st.sidebar.markdown(
        '<div class="sidebar-section">'
        '<div class="sidebar-section-title">Developmental Stages</div>'
        + "".join(stage_html_items)
        + f'<div class="sidebar-feature-count">'
        f'&#x2630; {len(artifact["feature_names"]):,} genes'
        f'</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Main area: tabs ---
    ba_val = metrics.get("balanced_accuracy", 0) if metrics else 0

    tab_predict, tab_performance, tab_about = st.tabs(
        ["Predict", "Model Performance", "About"]
    )

    # ===================== Predict tab =====================
    with tab_predict:
        st.markdown(
            _tissue_banner_html(tissue, scheme, ba_val),
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload expression data (CSV or TSV)",
            type=["csv", "tsv", "txt", "gz"],
        )

        with st.expander("Expected input format"):
            st.markdown("""
**Format:** CSV or TSV with one header row and one index column.

**Gene IDs:** Ensembl porcine IDs (`ENSSSCG` format). Version suffixes (e.g., `.2`) are stripped automatically.

**Values:** Raw TPM expression values (the app applies log2(x+1) transformation).
If your data is already log2-transformed, check the box below.

**Orientation:** Either:
- **Samples x Genes** (rows = samples, columns = genes) — *preferred*
- **Genes x Samples** (rows = genes, columns = samples) — auto-detected and transposed

**Gene coverage:** The model uses ~31,908 genes. Missing genes are filled with 0 (unexpressed).
At least 1,000 matching genes are recommended for reliable predictions.
            """)

        with st.expander("Using data with different annotations or microarray data"):
            st.markdown("""
The classifier was trained on the PigGTEx v0 Ensembl porcine annotation
(31,908 genes). For datasets that use different annotations or platforms,
follow the guidance below.

**Gene symbols instead of Ensembl IDs.** Convert gene symbols to
`ENSSSCG` identifiers before upload (e.g. via `mygene.info`,
BioMart, or the Ensembl REST homology endpoint). Verify that at least
80% of the model's expected genes are matched in the resulting file
(the percentage is shown above the prediction results); below that
threshold, predictions should be treated as exploratory.

**RefSeq or alternative Ensembl assemblies.** Lift identifiers to the
PigGTEx Ensembl IDs (`Sscrofa11.1`-style `ENSSSCG`) before upload. Use
the latest Ensembl REST `xref` endpoint or the BioMart cross-reference
tables. Cross-assembly differences mainly affect novel/long-noncoding
loci; protein-coding overlap is typically high.

**Microarray data (e.g. Affymetrix Porcine Genome arrays).** Convert
probe-set IDs to Ensembl gene IDs via the platform's annotation file;
aggregate multiple probes per gene by mean or median signal; upload
the resulting matrix as a "log2-transformed" dataset (microarray
intensities are typically log2-distributed). Note that the model was
trained on RNA-seq TPM and its dynamic range differs from microarrays;
prediction confidence should be interpreted accordingly.

**Out-of-distribution cohorts.** The classifier is calibrated to the
breed/sex composition of PigGTEx, in which no stage is a balanced
mixture of breeds. For cohorts dominated by under-represented breeds
(notably Large white for muscle, where balanced accuracy drops to
0.61), or for pooled-sex tissue, we recommend re-calibration on a
small in-house cohort before treating the prediction as authoritative.

**Single-cell RNA-seq.** The classifier was trained on bulk samples
and expects pseudo-bulk (per-sample summed or averaged) expression
values. Apply per-cell-type or whole-sample pseudo-bulking before
upload.
            """)

        if uploaded is not None:
            user_df = parse_upload(uploaded)

            aligned_df, n_matched, n_expected = align_genes(user_df, artifact["feature_names"])
            match_pct = n_matched / n_expected * 100

            if n_matched == 0:
                st.error(
                    "No genes matched the model's expected features. "
                    "Ensure your data uses Ensembl porcine gene IDs (ENSSSCG format)."
                )
            else:
                # Gene match and sample info
                st.caption(
                    f"{user_df.shape[0]} samples | "
                    f"{n_matched:,}/{n_expected:,} genes matched ({match_pct:.1f}%)"
                )
                if n_matched < 1000:
                    st.warning(
                        "Fewer than 1,000 genes matched — predictions may be unreliable."
                    )

                # Log-transform detection
                is_log = detect_log_transformed(user_df)
                file_hash = hashlib.md5(uploaded.getvalue()).hexdigest()
                if st.session_state.get("_upload_hash") != file_hash:
                    st.session_state["_upload_hash"] = file_hash
                    st.session_state["log_override"] = is_log

                log_override = st.checkbox(
                    "My data is already log2-transformed",
                    key="log_override",
                    help="Check this if your expression values are already log2(TPM+1). "
                         "Leave unchecked for raw TPM values.",
                )

                # Auto-predict
                results = predict_stages(artifact, aligned_df, log_override)

                st.subheader("Results")
                st.dataframe(
                    results.style.format({
                        "Confidence": "{:.3f}",
                        **{f"P({l})": "{:.3f}" for l in artifact["class_labels"]}
                    }),
                    use_container_width=True,
                )

                fig = plot_probabilities(results, artifact["class_labels"])
                st.pyplot(fig)
                plt.close(fig)

                csv = results.to_csv()
                st.download_button(
                    "Download results as CSV",
                    csv,
                    file_name=f"{tissue}_predictions.csv",
                    mime="text/csv",
                )

    # ===================== Model Performance tab =====================
    with tab_performance:
        st.markdown(
            _tissue_banner_html(tissue, scheme, ba_val),
            unsafe_allow_html=True,
        )
        if not tissue_results:
            st.warning("No results data available for this tissue.")
        else:
            class_labels = artifact["class_labels"]
            n_classes = len(class_labels)

            # Row 1: Confusion matrix | Per-class metrics
            row1_left, row1_right = st.columns(2)

            with row1_left:
                cm = metrics.get("confusion_matrix", [])
                if cm:
                    fig = plot_confusion_matrix(cm, class_labels)
                    st.pyplot(fig)
                    plt.close(fig)

            with row1_right:
                prec = metrics.get("per_class_precision", {})
                rec = metrics.get("per_class_recall", {})
                f1_vals = metrics.get("per_class_f1", {})
                if prec and rec and f1_vals:
                    prec_list = [prec[str(i)] for i in range(n_classes)]
                    rec_list = [rec[str(i)] for i in range(n_classes)]
                    f1_list = [f1_vals[str(i)] for i in range(n_classes)]
                    fig = plot_per_class_metrics(prec_list, rec_list, f1_list, class_labels)
                    st.pyplot(fig)
                    plt.close(fig)

            # Row 2: CV folds | Feature importance
            row2_left, row2_right = st.columns(2)

            with row2_left:
                cv_data = tissue_results.get("cross_validation", {})
                if cv_data and "aggregated_metrics" in cv_data:
                    fig = plot_cv_folds(cv_data)
                    st.pyplot(fig)
                    plt.close(fig)

            with row2_right:
                feat_imp = tissue_results.get("feature_importance", {})
                if feat_imp:
                    fig = plot_feature_importance(feat_imp, top_n=20)
                    st.pyplot(fig)
                    plt.close(fig)


    # ===================== About tab =====================
    with tab_about:
        st.subheader("Methodology")
        st.markdown("""
This application uses **OrdinalLightGBM** classifiers trained on the PigGTEx
transcriptomic atlas to predict porcine developmental stages from gene expression
profiles.

**Key details:**
- **Algorithm:** OrdinalLightGBM (ordinal classification preserving stage ordering)
- **Validation:** 5-fold nested cross-validation with Optuna hyperparameter tuning
- **Input features:** 31,908 genes (full transcriptome, no pre-selection)
- **Expression units:** log2(TPM + 1) normalized
- **Tissues:** 5 tissue-specific models (Muscle, Liver, Blood, Brain, Lung)
- **Classification schemes:** 4-class (Muscle, Liver) or 2-class (Blood, Brain, Lung)
  based on sample availability across developmental stages
        """)

        # ── Study design figure ───────────────────────────
        if STUDY_DESIGN_PATH.exists():
            st.subheader("Study Design")
            st.image(str(STUDY_DESIGN_PATH), width="stretch")
            st.markdown(
                '<p class="figure-caption">'
                "<strong>Figure 1.</strong> Overview of the porcine developmental "
                "transcriptomic atlas study design, including tissue sampling, "
                "stage classification scheme, and machine-learning pipeline."
                "</p>",
                unsafe_allow_html=True,
            )

        st.subheader("Developmental Stages")
        fig = plot_stage_timeline()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Cross-Tissue Comparison")
        if ml_results:
            fig = plot_tissue_comparison(ml_results)
            st.pyplot(fig)
            plt.close(fig)

        # ── Citation ──────────────────────────────────────
        st.markdown(
            '<div class="citation-box">'
            '<div class="citation-header">How to Cite</div>'
            "Liu T, Lei R, Khan IM, Theobald P. (2026). "
            "A multi-tissue transcriptomic atlas of porcine development identifies "
            "conserved molecular programs with humans. <em>In preparation.</em>"
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
