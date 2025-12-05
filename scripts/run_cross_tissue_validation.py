#!/usr/bin/env python3
"""
Run comprehensive cross-tissue validation analysis.
"""

import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DataLoader
from src.preprocessing import ExpressionPreprocessor
from src.stage_selection import StageGranularitySelector
from src.models import StageClassifier
from src.cross_tissue_validation import CrossTissueValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Tissue name mappings
TISSUE_MAPPING = {
    'Small_intestine': 'Small intestine',
    'Brain': 'Brain',
    'Frontal_cortex': 'Brain',
    'Hypothalamus': 'Brain',
    'Pituitary': 'Brain',
    'Ileum': 'Small intestine',
    'Jejunum': 'Small intestine',
    'Duodenum': 'Small intestine',
    'Macrophage': 'Other'
}


def load_tissue_data(tissue_file: str, data_loader: DataLoader, max_samples: int = None):
    """Load and prepare tissue data."""
    tissue_name = Path(tissue_file).stem.replace('.expr_tpm.txt', '')
    metadata_name = TISSUE_MAPPING.get(tissue_name, tissue_name)

    try:
        # Load expression data
        expr, metadata = data_loader.load_expression(
            tissue_name,
            min_tpm=0.1,
            min_detection_rate=0.1
        )

        # Limit samples if specified
        if max_samples is not None and expr.shape[1] > max_samples:
            sample_idx = np.random.choice(expr.shape[1], max_samples, replace=False)
            expr = expr.iloc[:, sample_idx]
            metadata = metadata.iloc[sample_idx]

        # No gene limiting for production runs

        return tissue_name, (expr, metadata)

    except Exception as e:
        logger.error(f"Failed to load {tissue_name}: {e}")
        return None, None


def run_cross_validation():
    """Run full cross-tissue validation pipeline."""

    print("=" * 60)
    print("CROSS-TISSUE VALIDATION PIPELINE")
    print("=" * 60)

    # Load analysis results
    analysis_path = PROJECT_ROOT / 'artifacts/results/cross_tissue/cross_tissue_analysis.json'
    if not analysis_path.exists():
        raise FileNotFoundError("Run scripts/analyze_all_tissues.py to generate cross-tissue analysis results.")

    with open(analysis_path) as f:
        analysis = json.load(f)

    # Load eligible tissues from the analysis
    eligible_tissues_path = PROJECT_ROOT / 'artifacts/results/eligible_tissues.json'
    if eligible_tissues_path.exists():
        with open(eligible_tissues_path) as f:
            eligible_data = json.load(f)
            # Use all available tissues for comprehensive cross-tissue validation
            # Use all unique eligible tissues
            all_tissues = eligible_data.get('all_eligible', eligible_data['available'])
            # Remove duplicates (e.g., Small_intestine appears twice)
            selected_tissues = list(dict.fromkeys(all_tissues))
    else:
        # Fallback to core tissues if eligible tissues file doesn't exist
        selected_tissues = ['Muscle', 'Liver', 'Brain', 'Blood', 'Adipose']

    print(f"Selected tissues for cross-validation: {selected_tissues}")

    # Initialize data loader
    data_loader = DataLoader(
        data_dir=PROJECT_ROOT / "data/pigGTEx",
        metadata_path=PROJECT_ROOT / "data/full_metadata.csv"
    )
    data_loader.load_metadata()

    # Load tissue data
    print("\n[1] Loading tissue data...")
    tissue_data = {}

    for tissue in tqdm(selected_tissues):
        tissue_file = PROJECT_ROOT / f"data/pigGTEx/{tissue}.expr_tpm.txt.gz"
        if tissue_file.exists():
            name, data = load_tissue_data(tissue_file, data_loader)
            if data:
                tissue_data[tissue] = data
                logger.info(f"Loaded {tissue}: {data[0].shape}")

    print(f"\n✓ Loaded {len(tissue_data)} tissues")

    # Initialize models and preprocessor
    preprocessor = ExpressionPreprocessor(
        log_transform=True,
        standardize=True,
        min_variance_percentile=20
    )

    # Use 2-class classification for cross-tissue (most compatible)
    model = StageClassifier(
        n_classes=2,
        model_type='binary_lr',
        C=1.0,
        random_state=42
    )

    # Initialize cross-tissue validator
    validator = CrossTissueValidator(
        model=model,
        preprocessor=preprocessor,
        min_common_genes=500
    )

    # Convert all tissues to 2-class for compatibility
    print("\n[2] Converting to 2-class scheme...")
    stage_selector = StageGranularitySelector()

    for tissue_name in tissue_data:
        expr, metadata = tissue_data[tissue_name]

        # Apply 2-class mapping
        if 'Stage' in metadata.columns:
            metadata['Stage'] = stage_selector.prepare_labels(
                metadata['Stage'], '2-class'
            )

        tissue_data[tissue_name] = (expr, metadata)

    # Run within-tissue baselines
    print("\n[3] Computing within-tissue baselines...")
    within_results = {}

    for tissue_name in tqdm(tissue_data):
        try:
            result = validator.within_tissue_baseline(
                tissue_data[tissue_name],
                tissue_name,
                test_size=0.3
            )
            within_results[tissue_name] = result
        except Exception as e:
            logger.error(f"Within-tissue failed for {tissue_name}: {e}")

    # Run leave-one-tissue-out validation
    print("\n[4] Running leave-one-tissue-out validation...")
    loto_results = {}

    for target_tissue in tqdm(tissue_data):
        try:
            result = validator.leave_one_tissue_out(
                tissue_data,
                target_tissue
            )
            loto_results[target_tissue] = result
        except Exception as e:
            logger.error(f"LOTO failed for {target_tissue}: {e}")

    # Run pairwise transfer learning
    print("\n[5] Running pairwise transfer learning...")
    transfer_results = []

    tissue_pairs = [(s, t) for s in tissue_data for t in tissue_data if s != t]

    for source, target in tqdm(tissue_pairs[:20]):  # Limit pairs for speed
        try:
            result = validator.cross_tissue_transfer(
                tissue_data[source],
                tissue_data[target],
                source,
                target
            )
            transfer_results.append(result)
        except Exception as e:
            logger.error(f"Transfer {source}->{target} failed: {e}")

    # Compute tissue similarity
    print("\n[6] Computing tissue similarity...")
    try:
        similarity_matrix = validator.compute_tissue_similarity(
            tissue_data,
            method='correlation'
        )
    except Exception as e:
        logger.error(f"Similarity computation failed: {e}")
        similarity_matrix = None

    # Create tissue embedding
    print("\n[7] Creating tissue embedding...")
    try:
        embedding = validator.create_tissue_embedding(
            tissue_data,
            method='pca',  # Use PCA instead of UMAP for speed
            n_components=2,
            max_samples_per_tissue=50
        )
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        embedding = None

    # Generate transfer matrix
    print("\n[8] Generating transfer matrix...")
    transfer_matrix = validator.generate_transfer_matrix(list(tissue_data.keys()))

    # Calculate generalization gaps
    gaps = validator.calculate_generalization_gap()

    # Save all results
    print("\n[9] Saving results...")
    results = {
        'within_tissue': within_results,
        'leave_one_out': loto_results,
        'transfers': transfer_results,
        'similarity_matrix': similarity_matrix.to_dict() if similarity_matrix is not None else None,
        'transfer_matrix': transfer_matrix.to_dict() if not transfer_matrix.empty else None,
        'generalization_gaps': gaps.to_dict() if not gaps.empty else None,
        'embedding': embedding.to_dict() if embedding is not None else None
    }

    output_path = PROJECT_ROOT / 'artifacts/results/cross_tissue/cross_tissue_validation_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    print("\nWithin-Tissue Performance:")
    for tissue, result in within_results.items():
        print(f"  {tissue:15s}: {result['balanced_accuracy']:.3f}")

    if loto_results:
        print("\nLeave-One-Out Performance:")
        for tissue, result in loto_results.items():
            print(f"  {tissue:15s}: {result['balanced_accuracy']:.3f}")

    if not gaps.empty:
        print("\nGeneralization Gaps:")
        print(f"  Mean gap: {gaps['generalization_gap'].mean():.3f}")
        print(f"  Max gap:  {gaps['generalization_gap'].max():.3f}")
        print(f"  Min gap:  {gaps['generalization_gap'].min():.3f}")

    print(f"\n✓ Results saved to {output_path.relative_to(PROJECT_ROOT)}")

    return results


if __name__ == '__main__':
    results = run_cross_validation()
