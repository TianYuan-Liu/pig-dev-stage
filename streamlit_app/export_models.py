"""
Export trained OrdinalLightGBM models for the Streamlit app.

Retrains each tissue's final model using saved hyperparameters from
model_outputs/{Tissue}_results.json, then serializes the fitted model
along with metadata needed for inference.

Run from project root:
    python streamlit_app/export_models.py
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.data_processing.preprocessing import ExpressionPreprocessor
from machine_learning.data_processing.stage_selection import StageGranularitySelector
from machine_learning.model_training.hyperparameter_tuning import FIXED_PARAMS
from machine_learning.model_training.models import OrdinalLightGBM
from machine_learning.utils.helpers import load_and_prepare_tissue, to_samples_x_genes_df

TISSUES = ["Muscle", "Liver", "Blood", "Brain", "Lung"]
RESULTS_DIR = PROJECT_ROOT / "machine_learning" / "model_outputs"
OUTPUT_DIR = Path(__file__).resolve().parent / "models"


def export_tissue(tissue_name: str) -> None:
    """Retrain and export a single tissue model."""
    print(f"\n{'='*60}")
    print(f"Exporting {tissue_name}")
    print(f"{'='*60}")

    # Load results JSON for hyperparameters
    results_path = RESULTS_DIR / f"{tissue_name}_results.json"
    with open(results_path) as f:
        results = json.load(f)

    final_model_params = results["final_model_params"]
    scheme_name = results["scheme"]
    stored_train_ba = results["train_metrics"]["balanced_accuracy"]

    # Get class labels from scheme
    selector = StageGranularitySelector()
    scheme_info = selector.schemes[scheme_name]
    class_labels = scheme_info["labels"]

    print(f"  Scheme: {scheme_name} ({len(class_labels)} classes)")
    print(f"  Labels: {class_labels}")
    print(f"  Stored train BA: {stored_train_ba:.4f}")

    # Load and prepare data
    result = load_and_prepare_tissue(tissue_name)
    if result is None:
        print(f"  ERROR: Failed to load data for {tissue_name}")
        return
    X_raw, y, sample_ids, gene_names, loaded_scheme = result

    print(f"  Data: {X_raw.shape[0]} genes x {X_raw.shape[1]} samples")
    print(f"  Loaded scheme: {loaded_scheme}")

    # Preprocess
    preprocessor = ExpressionPreprocessor()
    X_processed = preprocessor.fit_transform(X_raw)
    X_df = to_samples_x_genes_df(X_processed, sample_ids, preprocessor, gene_names)

    print(f"  After preprocessing: {X_df.shape[0]} samples x {X_df.shape[1]} genes")

    # Train final model
    model = OrdinalLightGBM(**final_model_params, **FIXED_PARAMS, seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_df, y)

    # Verify against stored metrics
    y_pred = model.predict(X_df)
    retrained_ba = balanced_accuracy_score(y, y_pred)
    print(f"  Retrained train BA: {retrained_ba:.4f}")

    diff = abs(retrained_ba - stored_train_ba)
    if diff > 0.02:
        print(f"  WARNING: BA differs by {diff:.4f} from stored value!")
    else:
        print(f"  OK: BA matches stored value (diff={diff:.4f})")

    # Serialize
    artifact = {
        "model": model,
        "feature_names": X_df.columns.tolist(),
        "scheme": scheme_name,
        "class_labels": class_labels,
        "tissue": tissue_name,
        "balanced_accuracy": stored_train_ba,
    }

    output_path = OUTPUT_DIR / f"{tissue_name}.joblib"
    joblib.dump(artifact, output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Exporting OrdinalLightGBM models for Streamlit app")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results dir:  {RESULTS_DIR}")
    print(f"Output dir:   {OUTPUT_DIR}")

    for tissue in TISSUES:
        export_tissue(tissue)

    print(f"\n{'='*60}")
    print("Export complete. Models saved to:")
    for tissue in TISSUES:
        p = OUTPUT_DIR / f"{tissue}.joblib"
        if p.exists():
            print(f"  {p}")
        else:
            print(f"  {tissue}: MISSING")


if __name__ == "__main__":
    main()
