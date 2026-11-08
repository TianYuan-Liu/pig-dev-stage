"""Tests for the PigDevStage web app.

Run from the repository root:

    pytest streamlit_app/tests -q

Model-dependent tests skip until streamlit_app/export_pls_models.py has written
streamlit_app/models/<Tissue>_pls.npz and streamlit_app/models/model_card.json.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")

APP_DIR = Path(__file__).resolve().parents[1]
APP_PATH = APP_DIR / "app.py"
MODELS_DIR = APP_DIR / "models"
CARD_PATH = MODELS_DIR / "model_card.json"
EXAMPLE_TPM = APP_DIR / "example" / "muscle_example_tpm.csv"
EXAMPLE_INFO = APP_DIR / "example" / "muscle_example_samples.csv"
TISSUES = ["Muscle", "Liver", "Lung", "Blood", "Brain"]

# Stages retained per tissue, as stated in the paper.
PAPER_STAGES = {
    "Liver": ["Neonatal", "Suckling", "Juvenile", "Pre-pubertal", "Post-pubertal", "Adult"],
    "Muscle": ["Neonatal", "Suckling", "Juvenile", "Pre-pubertal", "Post-pubertal"],
    "Lung": ["Neonatal", "Juvenile", "Post-pubertal"],
    "Blood": ["Juvenile", "Post-pubertal"],
    "Brain": ["Neonatal", "Suckling", "Juvenile", "Post-pubertal"],
}


def _load_app():
    spec = importlib.util.spec_from_file_location("pigdevstage_app", APP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


app = _load_app()


def _npz(tissue: str) -> Path:
    return MODELS_DIR / f"{tissue}_pls.npz"


@pytest.fixture(scope="module")
def card():
    if not CARD_PATH.exists():
        pytest.skip("model_card.json not exported yet")
    return json.loads(CARD_PATH.read_text())


@pytest.fixture(scope="module", params=TISSUES)
def model(request):
    path = _npz(request.param)
    if not path.exists():
        pytest.skip(f"{path.name} not exported yet")
    return request.param, app.read_model(path)


def _tpm_at_mean(m: dict) -> np.ndarray:
    """Raw TPM whose log2(TPM + 1) equals the model's training mean."""
    return np.exp2(m["mean"]) - 1.0


# ---------------------------------------------------------------------------
# (a) npz contents against the model card
# ---------------------------------------------------------------------------

def test_npz_shapes_and_stages(model, card):
    tissue, m = model
    p = len(m["gene_ids"])
    assert p > 1000
    assert len(set(m["gene_ids"])) == p
    assert all(str(g).startswith("ENSSSCG") for g in m["gene_ids"])
    for key in ("mean", "scale", "coef"):
        assert m[key].shape == (p,)
        assert np.all(np.isfinite(m[key]))
    assert np.all(m["scale"] > 0)
    assert np.isfinite(m["intercept"])
    assert m["proba_sigma"] > 0

    t = card["tissues"][tissue]
    assert m["kept_stages"] == [s["name"] for s in t["kept_stages"]]
    assert m["stage_n"] == [s["n"] for s in t["kept_stages"]]
    assert m["kept_stages"] == PAPER_STAGES[tissue]
    assert len(m["kept_stages"]) == t["n_class"]
    order = [s["name"] for s in card["stages"]]
    assert m["kept_stages"] == [s for s in order if s in m["kept_stages"]]
    assert sum(m["stage_n"]) == m["n_train"] == t["n_train"]
    assert min(m["stage_n"]) >= card["model"]["min_samples_per_stage"]
    assert t["n_genes"] == p
    assert m["n_components"] == card["model"]["n_components"]
    assert m["proba_sigma"] == pytest.approx(card["model"]["proba_sigma"])

    with np.load(_npz(tissue), allow_pickle=False) as z:
        assert int(z["n_samples"]) == t["n_samples"]
        assert int(z["n_bioprojects"]) == t["n_bioprojects"]
        assert int(z["n_bioprojects_train"]) == t["n_bioprojects_train"]

    top = m["top_genes"]
    assert len(top) == 50
    assert set(top["Gene id"]) <= set(m["gene_ids"])
    assert np.all(np.diff(top["VIP"].to_numpy()) <= 0)
    pos = pd.Index(m["gene_ids"]).get_indexer(top["Gene id"])
    assert np.array_equal(np.sign(m["coef"][pos]).astype(int), top["Sign"].to_numpy())


def test_card_performance_fields(card):
    assert [s["name"] for s in card["stages"]] == [
        "Neonatal", "Suckling", "Juvenile", "Pre-pubertal", "Post-pubertal", "Adult"]
    for tissue in TISSUES:
        t = card["tissues"][tissue]
        for regime in ("stratified", "grouped"):
            assert t[regime]["BA_mean"] is not None
            assert t[regime]["BA_sd"] is not None
        assert 0 < t["chance"] < 1
        assert 0 <= t["single_stage_pct"] <= 100
    r = card["tissues"]["Liver"]["grouped_restricted"]
    assert r["kept_stages"] == ["Suckling", "Juvenile", "Pre-pubertal", "Post-pubertal"]
    for key in ("BA_mean", "BA_sd", "rho_mean", "chance"):
        assert r[key] is not None


# ---------------------------------------------------------------------------
# (b) predictor on inputs built from the npz itself
# ---------------------------------------------------------------------------

def test_score_at_training_mean_is_intercept(model):
    _, m = model
    s = app.pls_scores(m, m["mean"])
    assert s.shape == (1,)
    assert s[0] == pytest.approx(m["intercept"], abs=1e-9)
    # every gene missing -> z = 0 everywhere -> intercept
    s_nan = app.pls_scores(m, np.full(m["mean"].shape, np.nan))
    assert s_nan[0] == pytest.approx(m["intercept"], abs=1e-12)


def test_unit_step_adds_gene_coefficient(model):
    _, m = model
    j = int(np.argmax(np.abs(m["coef"])))
    x = m["mean"].copy()
    x[j] += m["scale"][j]
    s = app.pls_scores(m, x)[0]
    assert s - m["intercept"] == pytest.approx(m["coef"][j], rel=1e-9, abs=1e-12)


def test_predict_table_from_npz(model):
    _, m = model
    X = pd.DataFrame([_tpm_at_mean(m)], columns=m["gene_ids"], index=["at_mean"])
    res, coverage = app.predict_table(m, X, already_log=False)
    assert coverage == 1.0
    assert res.loc["at_mean", "Score"] == pytest.approx(m["intercept"], abs=1e-6)
    k = int(app.nearest_stage_index([m["intercept"]], len(m["kept_stages"]))[0])
    assert res.loc["at_mean", "Predicted stage"] == m["kept_stages"][k]
    pcols = [f"P({s})" for s in m["kept_stages"]]
    assert res.loc["at_mean", pcols].sum() == pytest.approx(1.0)
    assert res.loc["at_mean", "Gene coverage (%)"] == pytest.approx(100.0)

    # same profile already on the log scale
    Xlog = pd.DataFrame([m["mean"]], columns=m["gene_ids"], index=["at_mean"])
    res_log, _ = app.predict_table(m, Xlog, already_log=True)
    assert res_log.loc["at_mean", "Score"] == pytest.approx(m["intercept"], abs=1e-9)


def test_coverage_gate(model):
    _, m = model
    p = len(m["gene_ids"])
    tpm = _tpm_at_mean(m)
    keep_hi = int(np.ceil(0.6 * p))
    keep_lo = int(np.floor(0.4 * p))
    X_hi = pd.DataFrame([tpm[:keep_hi]], columns=m["gene_ids"][:keep_hi], index=["hi"])
    X_lo = pd.DataFrame([tpm[:keep_lo]], columns=m["gene_ids"][:keep_lo], index=["lo"])
    res_hi, cov_hi = app.predict_table(m, X_hi, already_log=False)
    res_lo, cov_lo = app.predict_table(m, X_lo, already_log=False)
    assert cov_hi == pytest.approx(keep_hi / p)
    assert cov_lo == pytest.approx(keep_lo / p)
    assert res_hi.loc["hi", "Predicted stage"] in m["kept_stages"]
    assert res_lo.loc["lo", "Predicted stage"] == "Not predicted"
    assert np.isnan(res_lo.loc["lo", "Score"])


def test_nearest_stage_and_probabilities():
    s = np.array([-1.3, 0.2, 0.5, 0.51, 1.49, 2.7, 9.0])
    idx = app.nearest_stage_index(s, 3)
    assert idx.tolist() == [0, 0, 0, 1, 1, 2, 2]
    P = app.stage_probabilities(s, 3, 0.8)
    assert P.shape == (7, 3)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.array_equal(P.argmax(axis=1), idx)


# ---------------------------------------------------------------------------
# (c) input parser on small in-memory tables
# ---------------------------------------------------------------------------

G1, G2, G3 = "ENSSSCG00000000001", "ENSSSCG00000000002", "ENSSSCG00000000003"


def test_parse_genes_as_rows_versions_and_duplicates():
    raw = (f"gene_id,S1,S2\n{G1}.2,1,2\n{G2},3,4\n{G2}.1,5,6\n{G3}.7,0,0\n").encode()
    X, notes = app.parse_expression_table(raw, "table.csv")
    assert list(X.index) == ["S1", "S2"]
    assert list(X.columns) == [G1, G2, G3]
    assert X.loc["S1", G1] == 1.0
    assert X.loc["S1", G2] == 4.0  # mean of 3 and 5
    assert X.loc["S2", G2] == 5.0  # mean of 4 and 6
    assert any("averaged" in n for n in notes)
    assert any("version" in n for n in notes)


def test_parse_genes_as_columns_tab_separated():
    raw = (f"sample\t{G1}\t{G2}.3\nA\t0\t10\nB\t1.5\t2\n").encode()
    X, _ = app.parse_expression_table(raw, "table.tsv")
    assert list(X.index) == ["A", "B"]
    assert list(X.columns) == [G1, G2]
    assert X.loc["B", G1] == 1.5
    assert X.loc["A", G2] == 10.0


def test_parse_orientation_ignores_extension():
    raw = (f"gene_id\tS1\n{G1}\t1\n{G2}\t2\n").encode()  # tab-separated, named .csv
    X, _ = app.parse_expression_table(raw, "mislabelled.csv")
    assert list(X.index) == ["S1"]
    assert list(X.columns) == [G1, G2]


def test_parse_non_numeric_values_and_annotation_column():
    raw = (f"gene_id,gene_name,S1,S2\n{G1},ABC,1,2\n{G2},DEF,oops,3\n").encode()
    X, notes = app.parse_expression_table(raw, "table.csv")
    assert list(X.index) == ["S1", "S2"]  # gene_name column ignored
    assert np.isnan(X.loc["S1", G2])
    assert X.loc["S2", G2] == 3.0
    assert any("gene_name" in n for n in notes)
    assert any("could not be read as numbers" in n for n in notes)


def test_parse_gzip():
    raw = (f"gene_id,S1\n{G1},1\n{G2},2\n").encode()
    X, _ = app.parse_expression_table(gzip.compress(raw), "table.csv.gz")
    assert X.loc["S1", G2] == 2.0


def test_parse_rejects_negative_and_empty():
    with pytest.raises(ValueError):
        app.parse_expression_table((f"gene_id,S1\n{G1},-1\n{G2},2\n").encode(), "t.csv")
    with pytest.raises(ValueError):
        app.parse_expression_table(b"", "t.csv")
    with pytest.raises(ValueError):
        app.parse_expression_table((f"gene_id,S1\n{G1},a\n{G2},b\n").encode(), "t.csv")


def test_log_scale_detection():
    tpm = pd.DataFrame({G1: [1200.0], G2: [3.0]}, index=["S1"])
    logged = np.log2(tpm + 1.0)
    assert app.looks_log_transformed(tpm) is False
    assert app.looks_log_transformed(logged) is True


def test_strip_ensembl_version():
    assert app.strip_ensembl_version("ENSSSCG00000037539.2") == "ENSSSCG00000037539"
    assert app.strip_ensembl_version("ENSSSCG00000037539") == "ENSSSCG00000037539"
    assert app.strip_ensembl_version("sample.1") == "sample.1"


# ---------------------------------------------------------------------------
# Example upload (optional)
# ---------------------------------------------------------------------------

def test_example_matches_muscle_model():
    if not EXAMPLE_TPM.exists() or not _npz("Muscle").exists():
        pytest.skip("example or muscle model not exported yet")
    m = app.read_model(_npz("Muscle"))
    X, _ = app.parse_expression_table(EXAMPLE_TPM.read_bytes(), EXAMPLE_TPM.name)
    assert X.shape[0] == 3
    assert set(m["gene_ids"]) <= set(X.columns)
    assert np.all(X.to_numpy() >= 0)
    if EXAMPLE_INFO.exists():
        info = pd.read_csv(EXAMPLE_INFO, dtype=str)
        assert list(info["sample"]) == list(X.index)


# ---------------------------------------------------------------------------
# (d) smoke test of the whole app
# ---------------------------------------------------------------------------

def test_app_smoke():
    if not CARD_PATH.exists() or not all(_npz(t).exists() for t in TISSUES):
        pytest.skip("models not exported yet")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 5
    assert any("cross-study validated" in el.value for el in at.info)

    # the prediction tab never shows the in-distribution (stratified) accuracy
    predict_tab = at.tabs[0]
    texts = [el.value for el in predict_tab.info] + [el.value for el in predict_tab.warning]
    texts += [el.value for el in predict_tab.markdown]
    assert not any("stratified" in t.lower() or "in-distribution" in t.lower() for t in texts)

    if EXAMPLE_TPM.exists():
        at.checkbox(key="use_example").check().run()
        assert not at.exception
        frames = [d.value for d in at.dataframe]
        preds = [f for f in frames if "Predicted stage" in f.columns]
        assert len(preds) == 1
        res = preds[0]
        assert len(res) == 3
        assert set(res["Cross-study status"]) == {"Cross-study validated"}
        assert set(res["Predicted stage"]) <= set(PAPER_STAGES["Muscle"])
