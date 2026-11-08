"""Shared helpers for variant evaluation."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent  # cross_species_joint_ml/
PROJECT_ROOT = ANALYSIS.parent.parent.parent
ORTHO_CSV = (PROJECT_ROOT / "review" / "analyses" / "results"
             / "pig_human_one_to_one_orthologs.csv")
CACHE = ANALYSIS / "cache"
ANNOT_CACHE = CACHE / "mygene_annotations.json"
LOG_FILE = HERE / "log.tsv"

TISSUES = ["Muscle", "Brain", "Liver", "Lung", "Blood"]
HUMAN_TISSUES = ["Muscle", "Brain", "Liver"]  # tissues with human matrices

PIG_YOUNG = {"Infant", "Early childhood"}
PIG_OLD = {"Post-pubertal", "Adult"}

PSEUDO = 0.01
N_BOOTSTRAP = 1000
STAB_THRESHOLD = 8  # >= 8/10 seeds


def log_row(variant: str, tissue: str, step: str,
            metric_name: str, metric_value, note: str = "") -> None:
    aid = f"{int(time.time())}_{variant}_{tissue}_{step}"
    line = (f"{aid}\t{variant}\t{tissue}\t{step}\t{metric_name}\t"
            f"{metric_value}\t{note}\n")
    with LOG_FILE.open("a") as fh:
        fh.write(line)


def load_ortho() -> pd.DataFrame:
    df = pd.read_csv(ORTHO_CSV)
    df = df[df["is_one_to_one"]] if "is_one_to_one" in df.columns else df
    df["symbol"] = df["symbol"].str.upper()
    return df[["symbol", "pig_gene_id", "human_gene_id"]].drop_duplicates()


def load_pig_cached(tissue: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X = pd.read_parquet(CACHE / f"pig_{tissue}.parquet")
    y = pd.read_csv(CACHE / f"pig_{tissue}_y.csv", index_col=0).iloc[:, 0]
    raw = pd.read_parquet(CACHE / f"pig_{tissue}_rawTPM.parquet")
    return X, y, raw


def load_human_cached(tissue: str
                      ) -> Optional[Tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    if tissue == "Muscle":
        Xp = CACHE / "human_muscle.parquet"
        yp = CACHE / "human_muscle_y.csv"
        rp = CACHE / "human_muscle_raw.parquet"
    elif tissue in ("Brain", "Liver"):
        Xp = CACHE / f"human_cardoso_{tissue}.parquet"
        yp = CACHE / f"human_cardoso_{tissue}_y.csv"
        rp = CACHE / f"human_cardoso_{tissue}_raw.parquet"
    else:
        return None
    if not Xp.exists():
        return None
    X = pd.read_parquet(Xp)
    y = pd.read_csv(yp, index_col=0).iloc[:, 0]
    raw = pd.read_parquet(rp)
    return X, y, raw


def fold_change(raw: pd.DataFrame, samples_young: List[str],
                samples_old: List[str]) -> pd.Series:
    yvals = raw.loc[samples_young].mean(axis=0)
    ovals = raw.loc[samples_old].mean(axis=0)
    return np.log2((ovals + PSEUDO) / (yvals + PSEUDO))


def pig_fold_change(tissue: str) -> Optional[pd.Series]:
    _, y, raw = load_pig_cached(tissue)
    young = y[y.isin(PIG_YOUNG)].index.tolist()
    old = y[y.isin(PIG_OLD)].index.tolist()
    if not young or not old:
        return None
    return fold_change(raw, young, old)


def human_fold_change(tissue: str) -> Optional[pd.Series]:
    data = load_human_cached(tissue)
    if data is None:
        return None
    _, y, raw = data
    young = y[y == 0].index.tolist()
    old = y[y == 1].index.tolist()
    if not young or not old:
        return None
    return fold_change(raw, young, old)


def bootstrap_pearson(x: np.ndarray, y: np.ndarray, n: int = N_BOOTSTRAP,
                      seed: int = 42) -> Tuple[float, float, float]:
    if len(x) < 5:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        xs, ys = x[idx], y[idx]
        if np.std(xs) > 0 and np.std(ys) > 0:
            r, _ = stats.pearsonr(xs, ys)
            if not np.isnan(r):
                rs.append(r)
    rs = np.array(rs)
    return (float(np.mean(rs)),
            float(np.percentile(rs, 2.5)),
            float(np.percentile(rs, 97.5)))


def stable_set_from_seeds(seed_imp: pd.DataFrame, k: int,
                          threshold: int = STAB_THRESHOLD
                          ) -> Tuple[List[str], pd.DataFrame]:
    """Bug-fixed: exclude zero-gain features, use nlargest."""
    counter: Dict[str, int] = defaultdict(int)
    rank_records: Dict[str, List[int]] = defaultdict(list)
    gain_records: Dict[str, List[float]] = defaultdict(list)
    for col in seed_imp.columns:
        s = seed_imp[col]
        nonzero = s[s > 0]
        top = nonzero.nlargest(k)
        for rank, (g, v) in enumerate(top.items(), start=1):
            counter[g] += 1
            rank_records[g].append(rank)
            gain_records[g].append(float(v))
    rows = []
    for g, c in counter.items():
        rows.append({
            "gene_id": g,
            "n_seeds_top": int(c),
            "mean_rank": float(np.mean(rank_records[g])),
            "median_rank": float(np.median(rank_records[g])),
            "mean_gain": float(np.mean(gain_records[g])),
        })
    tab = (pd.DataFrame(rows)
           .sort_values(["n_seeds_top", "mean_gain"],
                        ascending=[False, False])
           .reset_index(drop=True))
    stable = tab[tab["n_seeds_top"] >= threshold]["gene_id"].tolist()
    return stable, tab


def per_seed_rank(seed_imp: pd.DataFrame) -> pd.DataFrame:
    """Per-seed rank: nonzero features get rank by gain descending (1=best);
    zero-gain features get sentinel rank n_features + 1."""
    n_feat = seed_imp.shape[0]
    out = pd.DataFrame(index=seed_imp.index, columns=seed_imp.columns,
                       dtype=float)
    for col in seed_imp.columns:
        s = seed_imp[col]
        # Build rank Series indexed by gene
        nz = s[s > 0].sort_values(ascending=False)
        ranks_nz = pd.Series(np.arange(1, len(nz) + 1, dtype=float),
                             index=nz.index)
        # zero-gain features get sentinel
        out[col] = ranks_nz.reindex(seed_imp.index).fillna(n_feat + 1)
    return out


def load_pig_seed_imp(tissue: str, variants_dir: Optional[Path] = None
                      ) -> Optional[pd.DataFrame]:
    """Load pig seed importance — prefer variants_dir override if given."""
    if variants_dir is not None:
        p = variants_dir / f"seed_importance_pig_{tissue}.parquet"
        if p.exists():
            return pd.read_parquet(p)
    p = ANALYSIS / f"seed_importance_pig_{tissue}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


def load_human_seed_imp(tissue: str, variants_dir: Optional[Path] = None
                        ) -> Optional[pd.DataFrame]:
    if variants_dir is not None:
        p = variants_dir / f"seed_importance_human_{tissue}.parquet"
        if p.exists():
            return pd.read_parquet(p)
    p = ANALYSIS / f"seed_importance_human_{tissue}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


def load_annot_cache() -> dict:
    if ANNOT_CACHE.exists():
        return json.loads(ANNOT_CACHE.read_text())
    return {}


BIO_KEYWORDS: Dict[str, List[str]] = {
    "muscle_contraction": [
        "muscle contraction", "myofibril", "sarcomere", "actin filament",
        "myosin", "troponin", "contractile fiber", "skeletal muscle",
    ],
    "neuronal_synaptic": [
        "synap", "neuron projection", "axon", "dendrit", "myelin",
        "neurotransmit", "glutamatergic", "GABA", "nervous system development",
    ],
    "liver_metabolic": [
        "lipid metabolism", "bile", "lipoprotein", "cholesterol",
        "cytochrome P450", "drug metabolism", "fatty acid metabolism",
        "gluconeogenesis", "xenobiotic",
    ],
    "immune_response": [
        "immune", "lymphocyte", "interleukin", "interferon",
        "complement", "chemokine", "antigen", "B cell", "T cell",
    ],
    "cell_cycle_proliferation": [
        "cell cycle", "mitosis", "cyclin", "kinetochore",
        "DNA replication", "G1/S transition", "M phase",
    ],
    "extracellular_matrix": [
        "extracellular matrix", "collagen", "laminin", "elastin",
        "basement membrane", "fibrillar",
    ],
    "general_metabolism": [
        "biosynthetic process", "catabolic process", "glycolysis",
        "oxidative phosphorylation", "TCA cycle", "amino acid metabolism",
    ],
    "transcription_regulation": [
        "transcription factor", "DNA binding", "homeobox",
        "zinc finger", "chromatin",
    ],
}


def classify_symbol(symbol: str, annot: dict) -> str:
    rec = annot.get(symbol, {})
    text = " | ".join(rec.get("go_bp", []) + [rec.get("name", "")]).lower()
    best = "other"
    best_hits = 0
    for cat, kws in BIO_KEYWORDS.items():
        hits = sum(kw.lower() in text for kw in kws)
        if hits > best_hits:
            best = cat
            best_hits = hits
    return best


def annotation_summary(symbols: List[str], annot: dict) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for s in symbols:
        counts[classify_symbol(s, annot)] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def compute_concordance(joint_pairs: List[Tuple[str, str]],
                        pig_fc: Optional[pd.Series],
                        human_fc: Optional[pd.Series]) -> dict:
    out = {
        "n_joint_for_fc": 0, "pearson_r": None, "ci_low": None,
        "ci_high": None, "directional_concordance_pct": None,
        "n_concordant_direction": 0,
    }
    if not joint_pairs or pig_fc is None or human_fc is None:
        return out
    pids = [p for p, _ in joint_pairs]
    hids = [h for _, h in joint_pairs]
    x = pig_fc.reindex(pids).values
    y_v = human_fc.reindex(hids).values
    ok = np.isfinite(x) & np.isfinite(y_v)
    x, y_v = x[ok], y_v[ok]
    if len(x) >= 5 and np.std(x) > 0 and np.std(y_v) > 0:
        r, _ = stats.pearsonr(x, y_v)
        r_m, ci_lo, ci_hi = bootstrap_pearson(x, y_v)
        concordant = int(np.sum(np.sign(x) == np.sign(y_v)))
        out.update({
            "n_joint_for_fc": int(len(x)),
            "pearson_r": float(r),
            "ci_low": float(ci_lo),
            "ci_high": float(ci_hi),
            "directional_concordance_pct": float(100.0 * concordant / len(x)),
            "n_concordant_direction": concordant,
        })
    return out


def expected_n_by_chance(k_pig: int, k_human: int, n_orthologs: int = 7485
                         ) -> float:
    """Under uniform random selection, expected overlap of two top-K sets
    drawn from N items = (k_pig * k_human) / N."""
    if n_orthologs <= 0:
        return float("nan")
    return float(k_pig * k_human / n_orthologs)
