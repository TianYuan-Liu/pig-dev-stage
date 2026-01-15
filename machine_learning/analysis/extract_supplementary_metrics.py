#!/usr/bin/env python3
"""
Extract supplementary metrics for Tables S1 and S2.

Extracts:
- Per-class performance metrics (precision, recall, F1)
- Train vs. test performance (if available)
"""

import json
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_ml_results(tissue):
    """Load ML results for a tissue."""
    results_file = PROJECT_ROOT / f"machine_learning/model_outputs/{tissue}_results.json"
    if not results_file.exists():
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)

def extract_per_class_metrics():
    """Extract per-class metrics for Table S2."""
    tissues = ["Muscle", "Brain", "Liver", "Blood", "Lung"]
    all_metrics = []
    
    for tissue in tissues:
        results = load_ml_results(tissue)
        if results is None:
            continue
        
        metrics = results.get('metrics', {})
        scheme = results.get('scheme', 'unknown')
        n_samples = results.get('n_samples', 0)
        
        # Get per-class metrics
        per_class_precision = metrics.get('per_class_precision', {})
        per_class_recall = metrics.get('per_class_recall', {})
        per_class_f1 = metrics.get('per_class_f1', {})
        
        # Get class labels (stages)
        class_dist = metrics.get('class_distribution', {})
        
        # Map numeric labels to stage names
        stage_map = {
            '0': 'Infant',
            '1': 'Early childhood',
            '2': 'Pre-pubertal',
            '3': 'Post-pubertal',
            '4': 'Adult'
        }
        
        for class_label, n_class_samples in class_dist.items():
            stage_name = stage_map.get(class_label, f'Stage_{class_label}')
            
            all_metrics.append({
                'Tissue': tissue,
                'Scheme': scheme,
                'Stage': stage_name,
                'N_samples': n_class_samples,
                'Precision': per_class_precision.get(class_label, None),
                'Recall': per_class_recall.get(class_label, None),
                'F1': per_class_f1.get(class_label, None)
            })
    
    df = pd.DataFrame(all_metrics)
    return df

def extract_train_test_metrics():
    """
    Extract train vs. test metrics for Table S1.
    
    Note: The current pipeline only saves test set metrics.
    Train performance would need to be calculated from saved models.
    For now, we report test performance only.
    """
    tissues = ["Muscle", "Brain", "Liver", "Blood", "Lung"]
    all_metrics = []
    
    for tissue in tissues:
        results = load_ml_results(tissue)
        if results is None:
            continue
        
        metrics = results.get('metrics', {})
        scheme = results.get('scheme', 'unknown')
        n_samples = results.get('n_samples', 0)
        
        # Test set metrics (available)
        all_metrics.append({
            'Tissue': tissue,
            'Scheme': scheme,
            'N_samples': n_samples,
            'Set': 'Test',
            'Balanced_Accuracy': metrics.get('balanced_accuracy', None),
            'F1_macro': metrics.get('f1_macro', None),
            'F1_weighted': metrics.get('f1_weighted', None),
            'Precision_macro': metrics.get('precision_macro', None),
            'Recall_macro': metrics.get('recall_macro', None),
            'MAE': metrics.get('mae', None),
            'Spearman_r': metrics.get('spearman_r', None)
        })
        
        # Train set metrics (not available in current results)
        # Note: Would need to load saved model and predict on train set
        all_metrics.append({
            'Tissue': tissue,
            'Scheme': scheme,
            'N_samples': n_samples,
            'Set': 'Train',
            'Balanced_Accuracy': None,  # Not available
            'F1_macro': None,
            'F1_weighted': None,
            'Precision_macro': None,
            'Recall_macro': None,
            'MAE': None,
            'Spearman_r': None
        })
    
    df = pd.DataFrame(all_metrics)
    return df

def main():
    """Extract and save supplementary metrics."""
    print("=" * 70)
    print("Extracting Supplementary Metrics")
    print("=" * 70)
    
    output_dir = PROJECT_ROOT / "paper/figures/output/stats"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract per-class metrics
    print("\n1. Extracting per-class metrics...")
    per_class_df = extract_per_class_metrics()
    per_class_file = output_dir / "supplementary_per_class_metrics.csv"
    per_class_df.to_csv(per_class_file, index=False)
    print(f"   Saved to {per_class_file}")
    print(f"   {len(per_class_df)} rows")
    
    # Extract train/test metrics
    print("\n2. Extracting train/test metrics...")
    train_test_df = extract_train_test_metrics()
    train_test_file = output_dir / "supplementary_train_test_metrics.csv"
    train_test_df.to_csv(train_test_file, index=False)
    print(f"   Saved to {train_test_file}")
    print(f"   Note: Train metrics not available in current results")
    print(f"   Only test set performance is reported (standard practice)")
    
    print("\n" + "=" * 70)
    print("✓ Extraction complete")

if __name__ == "__main__":
    main()
