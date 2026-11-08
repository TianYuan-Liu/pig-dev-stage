#!/usr/bin/env python3
"""Export the per-tissue 6-stage PLS models served by the PigDevStage web app.

Run from the repository root, on a machine that has the PigGTEx inputs
(data/pigGTEx/<Tissue>.expr_tpm.txt.gz and data/PigGTEx_v0.MetaTable.csv):

    python streamlit_app/export_pls_models.py            # all five tissues
    python streamlit_app/export_pls_models.py Muscle     # one tissue (card is merged)

For each tissue this script

  1. selects samples exactly as paper/figures/provenance/within_pig_6stage_figdata.py
     does (DataLoader for Muscle/Liver/Lung/Blood; the verify_brain_6stage forebrain
     helpers for Brain), maps ages to the six stages and drops stages with fewer
     than 10 samples (dropped, not merged);
  2. fits machine_learning.model_training.models.OrdinalPLS (PLSRegression,
     10 components, scale=False, per-gene z-scoring inside) on ALL retained samples;
  3. writes streamlit_app/models/<Tissue>_pls.npz holding the linear predictor
         score = ((log2(TPM+1) - mean) / scale) @ coef + intercept
     plus the stage list, counts and the top-50 genes by VIP;
  4. asserts that the npz formula, re-read from disk, reproduces the fitted
     sklearn model's training predictions (max |diff| < 1e-8) and its class calls.

It also writes streamlit_app/models/model_card.json with the performance numbers
copied verbatim from paper/figures/data/within_pig_6stage.json and
review/analyses/results/liver_loco_stage_restricted.json, and (unless
--no-example) streamlit_app/example/ with three real PigGTEx muscle samples.

The npz files contain only plain numeric and fixed-width string arrays, so the
app reads them with numpy.load(allow_pickle=False).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROV_DIR = ROOT / "paper" / "figures" / "provenance"
APP_DIR = ROOT / "streamlit_app"
MODELS_DIR = APP_DIR / "models"
EXAMPLE_DIR = APP_DIR / "example"

DATA_DIR = ROOT / "data" / "pigGTEx"
META_PATH = ROOT / "data" / "PigGTEx_v0.MetaTable.csv"
SYMBOL_PATH = ROOT / "data" / "cardoso_moreira_2019" / "_pig_id_to_symbol.json"
PERF_JSON = ROOT / "paper" / "figures" / "data" / "within_pig_6stage.json"
LIVER_RESTRICTED_JSON = ROOT / "review" / "analyses" / "results" / "liver_loco_stage_restricted.json"
LIVER_RESTRICTED_SCENARIO = "drop Neonatal + Adult"
PAPER_VIP_CSV = ROOT / "review" / "analyses" / "results" / "pls_vip_features.csv"

# Import the repository's own machine_learning package FIRST, so that it is the
# cached module when within_pig_6stage_figdata (which prepends a hard-coded
# author-laptop path to sys.path) is imported below.
sys.path.insert(0, str(ROOT))
from machine_learning.data_processing.data_loader import DataLoader  # noqa: E402
from machine_learning.model_training.models import OrdinalPLS  # noqa: E402

sys.path.insert(1, str(PROV_DIR))
import within_pig_6stage_figdata as figdata  # noqa: E402

# Drop the hard-coded path the provenance module inserted (if it is not ours).
if Path(str(figdata.ROOT)).resolve() != ROOT:
    sys.path[:] = [p for p in sys.path if p != str(figdata.ROOT)]

TISSUES = ["Muscle", "Liver", "Lung", "Blood", "Brain"]
SCHEMA_VERSION = 1
TOP_VIP = 50
PARITY_TOL = 1e-8
SEED = 42  # OrdinalPLS seed (PLS itself is deterministic)

# Upper-inclusive day cut points of the six stages; checked against figdata.stage6.
STAGE_CUTS = [3, 28, 90, 165, 540]

# Retained n and BioProjects per tissue as printed in Supplementary Table S1.
# Used only for an informational printout, never written to the card.
TABLE_S1_CHECK = {"Muscle": (757, 54), "Lung": (119, 17), "Liver": (309, 34),
                  "Blood": (74, 7), "Brain": (170, 17)}

EXAMPLE_TISSUE = "Muscle"
EXAMPLE_STAGES = ["Neonatal", "Juvenile", "Post-pubertal"]


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

def stage_table() -> list[dict]:
    """Six stages with day ranges, verified against figdata.stage6."""
    order = list(figdata.ORDER6)
    assert len(order) == len(STAGE_CUTS) + 1
    rows = []
    lo = 0
    for i, name in enumerate(order):
        hi = STAGE_CUTS[i] if i < len(STAGE_CUTS) else None
        rows.append({"name": name, "min_days": lo, "max_days": hi})
        # cut point c belongs to stage i, c + 1 to stage i + 1
        if hi is not None:
            assert figdata.stage6(float(hi)) == name, (hi, name)
            assert figdata.stage6(float(hi) + 1.0) == order[i + 1], (hi + 1, order[i + 1])
            lo = hi + 1
    assert figdata.stage6(0.0) == order[0]
    return rows


# ---------------------------------------------------------------------------
# Sample selection (mirrors within_pig_6stage_figdata.main, per tissue)
# ---------------------------------------------------------------------------

def select_samples(tissue: str, dl: DataLoader) -> dict:
    """Return the figdata selection for one tissue plus the gene ids.

    The body below is a line-for-line copy of the selection block in
    within_pig_6stage_figdata.main(); only the bookkeeping of sample ids,
    gene ids and raw TPM (for the example file) is added.
    """
    extra_meta = None
    if tissue == "Brain":
        import verify_brain_6stage as vb6
        meta = vb6._load_meta()
        expr = vb6._load_expr()
        fb_idx = vb6.select_forebrain(meta)
        samples = [s for s in fb_idx if s in expr.columns]
        expr_sub = expr[samples].T
        X = np.log2(expr_sub.values + 1.0)
        age = meta.loc[samples, "Age_Days"]
        bp = meta.loc[samples, "BioProject"].astype(str).reset_index(drop=True)
        y6 = age.map(figdata.stage6).reset_index(drop=True)
        sample_ids = list(samples)
        gene_ids = [str(g) for g in expr.index]
        regions = list(vb6.FOREBRAIN)
    else:
        expr, meta = dl.load_expression(tissue)
        X = np.log2(expr.T.values + 1.0)
        age = meta["Age_Days"] if "Age_Days" in meta else meta["Age"]
        age = pd.to_numeric(age, errors="coerce")
        bp = meta["BioProject"].astype(str).reset_index(drop=True)
        y6 = age.map(figdata.stage6).reset_index(drop=True)
        sample_ids = [str(s) for s in expr.columns]
        gene_ids = [str(g) for g in expr.index]
        regions = None
        extra_meta = meta

    keep_age = age.notna().values
    X = X[keep_age]
    y6 = y6[keep_age].reset_index(drop=True)
    bp = bp[keep_age].reset_index(drop=True)
    sample_ids = [s for s, k in zip(sample_ids, keep_age) if k]

    kept6 = figdata.drop_small(y6, figdata.ORDER6)
    code = {s: i for i, s in enumerate(kept6)}
    mask = y6.isin(kept6).values
    Xk = X[mask]
    yk = y6[mask].values
    gk = bp[mask].values
    yi = np.array([code[v] for v in yk])

    return {
        "X": Xk,
        "y": yi,
        "groups": gk,
        "kept": kept6,
        "gene_ids": gene_ids,
        "sample_ids": [s for s, m in zip(sample_ids, mask) if m],
        "stage_labels": list(yk),
        # figdata summary fields (definitions identical to within_pig_6stage.json)
        "n_samples": int(keep_age.sum()),
        "n_bioprojects": int(bp.nunique()),
        "kept6": {s: int((y6 == s).sum()) for s in kept6},
        "single_stage_pct": figdata.single_stage_pct(y6, bp, kept6),
        "regions": regions,
        "expr": expr if tissue == EXAMPLE_TISSUE else None,
        "meta": extra_meta if tissue == EXAMPLE_TISSUE else None,
    }


# ---------------------------------------------------------------------------
# Model fit and linear extraction
# ---------------------------------------------------------------------------

def linear_form(model: OrdinalPLS, p: int) -> tuple[np.ndarray, float]:
    """Return (coef, intercept) such that score = Xz @ coef + intercept.

    Xz is the z-scored matrix ((X - model._mean) / model._scale). The inner
    PLSRegression uses scale=False, so its own x scaling is the identity and
    coef_ applies directly to Xz after its internal centring; the intercept is
    read off as the model's prediction at Xz = 0, which is exact for an affine
    predictor and independent of the sklearn version's attribute layout.
    """
    pls = model.model_
    x_std = np.asarray(getattr(pls, "_x_std", np.ones(p)), dtype=np.float64)
    assert np.allclose(x_std, 1.0), "expected PLSRegression(scale=False)"
    coef = np.asarray(pls.coef_, dtype=np.float64)
    assert coef.size == p, f"coef_ has {coef.size} entries, expected {p}"
    coef = coef.reshape(-1).copy()
    intercept = float(np.asarray(pls.predict(np.zeros((1, p)))).ravel()[0])
    return coef, intercept


def npz_scores(npz_path: Path, X_log: np.ndarray) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as z:
        mean, scale, coef = z["mean"], z["scale"], z["coef"]
        intercept = float(z["intercept"])
    return ((X_log - mean) / scale) @ coef + intercept


def nearest_stage(scores: np.ndarray, k: int) -> np.ndarray:
    ks = np.arange(k)
    return np.argmin((scores[:, None] - ks[None, :]) ** 2, axis=1)


def fixed_str(values) -> np.ndarray:
    """Fixed-width unicode array (loadable without pickle)."""
    values = [str(v) for v in values]
    width = max([1] + [len(v) for v in values])
    return np.array(values, dtype=f"<U{width}")


def export_tissue(tissue: str, dl: DataLoader, perf: dict, symbols: dict,
                  write_example: bool) -> dict:
    print(f"\n=== {tissue} ===")
    sel = select_samples(tissue, dl)
    X, y, kept = sel["X"], sel["y"], sel["kept"]
    gene_ids = [g.strip() for g in sel["gene_ids"]]
    p = X.shape[1]
    assert len(gene_ids) == p
    assert len(set(gene_ids)) == p, "duplicate gene ids in the TPM matrix"

    # --- parity of the selection with the paper's figure data ---------------
    ref = perf[tissue]
    got = {"n_samples": sel["n_samples"], "n_bioprojects": sel["n_bioprojects"],
           "kept6": sel["kept6"], "single_stage_pct": sel["single_stage_pct"]}
    for key, val in got.items():
        print(f"  {key:<17} export={val}  within_pig_6stage.json={ref[key]}")
        assert val == ref[key], f"{tissue}: {key} differs from within_pig_6stage.json"
    n_train = int(len(y))
    n_bp_train = int(pd.Series(sel["groups"]).nunique())
    s1 = TABLE_S1_CHECK.get(tissue)
    if s1 is not None:
        flag = "ok" if (n_train, n_bp_train) == s1 else "DIFFERS (check)"
        print(f"  retained n={n_train}, BioProjects={n_bp_train}; "
              f"Supplementary Table S1 lists n={s1[0]}, BioProjects={s1[1]}: {flag}")

    # --- fit on all retained samples ----------------------------------------
    model = OrdinalPLS(n_components=figdata.N_COMPONENTS, seed=SEED)
    model.fit(pd.DataFrame(X, columns=gene_ids), y)
    n_comp = int(model.model_.n_components)
    mean = np.asarray(model._mean, dtype=np.float64)
    scale = np.asarray(model._scale, dtype=np.float64)
    coef, intercept = linear_form(model, p)
    vip = np.asarray(model._vip, dtype=np.float64)
    sigma = float(model.proba_sigma)

    order = np.argsort(-vip, kind="stable")[:TOP_VIP]
    top_genes = [gene_ids[i] for i in order]
    stage_n = np.array([int((y == k).sum()) for k in range(len(kept))], dtype=np.int64)
    assert int(stage_n.sum()) == n_train
    assert [sel["kept6"][s] for s in kept] == stage_n.tolist()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / f"{tissue}_pls.npz"
    np.savez_compressed(
        out,
        schema_version=np.int64(SCHEMA_VERSION),
        tissue=np.array(tissue),
        gene_ids=fixed_str(gene_ids),
        mean=mean,
        scale=scale,
        coef=coef,
        intercept=np.float64(intercept),
        kept_stages=fixed_str(kept),
        stage_n=stage_n,
        n_train=np.int64(n_train),
        n_bioprojects_train=np.int64(n_bp_train),
        n_samples=np.int64(sel["n_samples"]),
        n_bioprojects=np.int64(sel["n_bioprojects"]),
        n_components=np.int64(n_comp),
        proba_sigma=np.float64(sigma),
        top_vip_gene=fixed_str(top_genes),
        top_vip_symbol=fixed_str([symbols.get(g, "") for g in top_genes]),
        top_vip_value=vip[order],
        top_vip_coef=coef[order],
        top_vip_sign=np.sign(coef[order]).astype(np.int8),
    )

    # --- parity of the saved linear formula with the fitted model -----------
    Xz = (X - mean) / scale
    y_sklearn = np.asarray(model.model_.predict(Xz), dtype=np.float64).ravel()
    y_npz = npz_scores(out, X)
    max_diff = float(np.max(np.abs(y_npz - y_sklearn)))
    print(f"  npz formula vs sklearn training predictions: max |diff| = {max_diff:.3e}")
    assert max_diff < PARITY_TOL, f"{tissue}: npz formula differs by {max_diff}"
    cls_ref = np.asarray(model.predict(pd.DataFrame(X, columns=gene_ids)), dtype=int)
    cls_npz = nearest_stage(y_npz, len(kept))
    n_mismatch = int((cls_ref != cls_npz).sum())
    print(f"  class calls: npz nearest-stage vs OrdinalPLS.predict mismatches = {n_mismatch}")
    assert n_mismatch == 0
    print(f"  kept stages: {dict(zip(kept, stage_n.tolist()))}; genes={p}; components={n_comp}")
    print(f"  wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")

    # informational: overlap with the paper's committed top-20 VIP list
    if PAPER_VIP_CSV.exists():
        try:
            paper_vip = pd.read_csv(PAPER_VIP_CSV)
            ref20 = set(paper_vip.loc[paper_vip["tissue"] == tissue, "ensembl"].astype(str))
            if ref20:
                ov = len(ref20 & set(top_genes[:20]))
                print(f"  top-20 VIP overlap with review/analyses/results/pls_vip_features.csv: "
                      f"{ov}/{len(ref20)}")
        except Exception as exc:  # informational only
            print(f"  (VIP cross-check skipped: {exc})")

    if write_example and tissue == EXAMPLE_TISSUE:
        write_example_files(sel)

    return {
        "kept": kept,
        "stage_n": stage_n.tolist(),
        "n_train": n_train,
        "n_bioprojects_train": n_bp_train,
        "n_genes": p,
        "n_components": n_comp,
        "proba_sigma": sigma,
        "regions": sel["regions"],
    }


def write_example_files(sel: dict) -> None:
    """Three real PigGTEx muscle samples (raw TPM, all genes), one per stage."""
    expr, meta = sel["expr"], sel["meta"]
    if expr is None or meta is None:
        return
    chosen = []
    for stage in EXAMPLE_STAGES:
        hits = [s for s, lab in zip(sel["sample_ids"], sel["stage_labels"]) if lab == stage]
        if hits:
            chosen.append((hits[0], stage))
    if not chosen:
        print("  example: no samples found, skipped")
        return
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    ids = [s for s, _ in chosen]
    tpm = expr[ids].copy()
    tpm.index.name = "gene_id"
    tpm_path = EXAMPLE_DIR / "muscle_example_tpm.csv"
    tpm.to_csv(tpm_path)
    info = pd.DataFrame({
        "sample": ids,
        "BioProject": [str(meta.loc[s, "BioProject"]) for s in ids],
        "Breed": [str(meta.loc[s, "Breed"]) if "Breed" in meta else "" for s in ids],
        "Sex": [str(meta.loc[s, "Sex"]) if "Sex" in meta else "" for s in ids],
        "Age": [str(meta.loc[s, "Age"]) for s in ids],
        "Age_Days": [float(meta.loc[s, "Age_Days"]) for s in ids],
        "recorded_stage": [stage_name for _, stage_name in chosen],
    })
    info_path = EXAMPLE_DIR / "muscle_example_samples.csv"
    info.to_csv(info_path, index=False)
    print(f"  example: wrote {tpm_path.relative_to(ROOT)} ({tpm.shape[0]} genes x "
          f"{tpm.shape[1]} samples) and {info_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------

def _round(v, nd=6):
    return None if v is None else round(float(v), nd)


def perf_block(block: dict, with_recall: bool) -> dict:
    if block.get("BA_mean") is None:
        return {"BA_mean": None, "note": block.get("note")}
    out = {
        "BA_mean": _round(block["BA_mean"]),
        "BA_sd": _round(block["BA_sd"]),
        "rho_mean": _round(block.get("rho_mean")),
        "rho_sd": _round(block.get("rho_sd")),
        "n_seeds": block.get("n_seeds"),
    }
    if with_recall and "per_class_recall" in block:
        out["per_class_recall"] = block["per_class_recall"]
    return out


def package_versions() -> dict:
    vers = {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__}
    try:
        import sklearn
        vers["scikit-learn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import scipy
        vers["scipy"] = scipy.__version__
    except Exception:
        pass
    return vers


def git_state() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=ROOT, capture_output=True, text=True,
                                    check=True).stdout.strip())
        return {"commit": commit, "uncommitted_changes": dirty}
    except Exception:
        return {"commit": None, "uncommitted_changes": None}


def build_card(exported: dict, perf: dict, liver_restricted: dict, previous: dict | None) -> dict:
    tissues = dict(previous.get("tissues", {})) if previous else {}
    for tissue, ex in exported.items():
        ref = perf[tissue]
        entry = {
            "kept_stages": [{"name": s, "n": int(n)} for s, n in zip(ex["kept"], ex["stage_n"])],
            "n_class": int(ref["n_class"]),
            "chance": ref["chance"],
            "n_samples": int(ref["n_samples"]),
            "n_bioprojects": int(ref["n_bioprojects"]),
            "single_stage_pct": ref["single_stage_pct"],
            "n_train": ex["n_train"],
            "n_bioprojects_train": ex["n_bioprojects_train"],
            "n_genes": ex["n_genes"],
            "stratified": perf_block(ref["stratified_6stage"], with_recall=False),
            "grouped": perf_block(ref["grouped_6stage"], with_recall=True),
        }
        if ex["regions"]:
            entry["regions"] = ex["regions"]
        if tissue == "Liver":
            sc = liver_restricted[LIVER_RESTRICTED_SCENARIO]
            # The source stores BA_mean/BA_sd already rounded to 3 decimals
            # (0.395), which would print as 0.40 at 2 decimals although the
            # paper reports 0.39. Recompute both from the stored per-seed values
            # (np.std, as in the source script) and keep the rounded originals.
            seeds = np.asarray(sc.get("per_seed_ba") or [], dtype=float)
            ba_mean = float(seeds.mean()) if seeds.size else sc["BA_mean"]
            ba_sd = float(seeds.std()) if seeds.size else sc["BA_sd"]
            entry["grouped_restricted"] = {
                "scenario": LIVER_RESTRICTED_SCENARIO,
                "kept_stages": sc["kept"],
                "n": sc["n"],
                "n_class": sc["K"],
                "chance": sc["chance"],
                "BA_mean": _round(ba_mean),
                "BA_sd": _round(ba_sd),
                "BA_mean_source_rounded": sc["BA_mean"],
                "BA_sd_source_rounded": sc["BA_sd"],
                "per_seed_ba": sc.get("per_seed_ba"),
                "rho_mean": sc["rho"],
                "per_class_recall": sc.get("recall"),
                "largest_cohort_share_pct": liver_restricted.get("_largest_cohort_share_pct"),
                "criterion": liver_restricted.get("_criterion"),
            }
        tissues[tissue] = entry

    first = next(iter(exported.values()))
    return {
        "schema_version": SCHEMA_VERSION,
        "stages": stage_table(),
        "model": {
            "estimator": "PLSRegression (scikit-learn), scale=False, on per-gene z-scored "
                         "log2(TPM+1); integer stage target 0..K-1",
            "n_components": first["n_components"],
            "proba_sigma": first["proba_sigma"],
            "min_samples_per_stage": int(figdata.MIN_PER_STAGE),
        },
        "evaluation": {
            "n_folds": int(figdata.N_FOLDS),
            "n_seeds": len(figdata.SEEDS),
            "stratified": "StratifiedKFold (in-distribution, batch-inclusive)",
            "grouped": "StratifiedGroupKFold over BioProject (whole studies held out)",
        },
        "tissues": tissues,
        "sources": {
            "performance": str(PERF_JSON.relative_to(ROOT)),
            "liver_stage_restricted": str(LIVER_RESTRICTED_JSON.relative_to(ROOT)),
            "sample_selection": "paper/figures/provenance/within_pig_6stage_figdata.py",
            "estimator": "machine_learning/model_training/models.py::OrdinalPLS",
            "expression": "data/pigGTEx/<Tissue>.expr_tpm.txt.gz",
            "metadata": "data/PigGTEx_v0.MetaTable.csv",
        },
        "export": {
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "packages": package_versions(),
            "git": git_state(),
            "tissues_exported_this_run": sorted(exported),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tissues", nargs="*", default=TISSUES, help="subset of " + ", ".join(TISSUES))
    ap.add_argument("--no-example", action="store_true", help="do not write streamlit_app/example/")
    args = ap.parse_args()
    bad = [t for t in args.tissues if t not in TISSUES]
    if bad:
        ap.error(f"unknown tissue(s): {bad}")

    perf = json.loads(PERF_JSON.read_text())
    liver_restricted = json.loads(LIVER_RESTRICTED_JSON.read_text())
    symbols = json.loads(SYMBOL_PATH.read_text()) if SYMBOL_PATH.exists() else {}
    card_path = MODELS_DIR / "model_card.json"
    previous = None
    if set(args.tissues) != set(TISSUES) and card_path.exists():
        previous = json.loads(card_path.read_text())

    dl = DataLoader(data_dir=DATA_DIR, metadata_path=META_PATH)
    dl.load_metadata()
    stage_table()  # fail early if the stage cut points disagree with figdata

    exported = {}
    for tissue in args.tissues:
        exported[tissue] = export_tissue(tissue, dl, perf, symbols,
                                         write_example=not args.no_example)
        sys.stdout.flush()

    card = build_card(exported, perf, liver_restricted, previous)
    missing = [t for t in TISSUES if t not in card["tissues"]]
    if missing:
        print(f"\nWARNING: model card has no entry for {missing}; export them too.")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    card_path.write_text(json.dumps(card, indent=2) + "\n")
    print(f"\nWrote {card_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
