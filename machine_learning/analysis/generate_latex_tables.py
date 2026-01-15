#!/usr/bin/env python3
"""
Generate LaTeX table files from CSV data for supplementary materials.
"""

import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def csv_to_latex_table(csv_file, output_file, format_func=None):
    """Convert CSV to LaTeX table format."""
    df = pd.read_csv(csv_file)
    
    lines = []
    for _, row in df.iterrows():
        if format_func:
            formatted = format_func(row)
        else:
            formatted = " & ".join([str(val) if pd.notna(val) else "-" for val in row.values])
        lines.append(f"    {formatted} \\\\")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"Generated: {output_file}")

def format_train_test(row):
    """Format train/test table row."""
    tissue = row['Tissue']
    scheme = row['Scheme']
    n = int(row['N_samples']) if pd.notna(row['N_samples']) else "-"
    set_name = row['Set']
    ba = f"{row['Balanced_Accuracy']:.3f}" if pd.notna(row['Balanced_Accuracy']) else "-"
    f1m = f"{row['F1_macro']:.3f}" if pd.notna(row['F1_macro']) else "-"
    f1w = f"{row['F1_weighted']:.3f}" if pd.notna(row['F1_weighted']) else "-"
    mae = f"{row['MAE']:.2f}" if pd.notna(row['MAE']) else "-"
    
    return f"{tissue} & {scheme} & {n} & {set_name} & {ba} & {f1m} & {f1w} & {mae}"

def format_per_class(row):
    """Format per-class table row."""
    tissue = row['Tissue']
    scheme = row['Scheme']
    stage = row['Stage']
    n = int(row['N_samples']) if pd.notna(row['N_samples']) else "-"
    prec = f"{row['Precision']:.3f}" if pd.notna(row['Precision']) else "-"
    rec = f"{row['Recall']:.3f}" if pd.notna(row['Recall']) else "-"
    f1 = f"{row['F1']:.3f}" if pd.notna(row['F1']) else "-"
    
    return f"{tissue} & {scheme} & {stage} & {n} & {prec} & {rec} & {f1}"

def main():
    """Generate all LaTeX tables."""
    print("=" * 70)
    print("Generating LaTeX Tables for Supplementary Materials")
    print("=" * 70)
    
    output_dir = PROJECT_ROOT / "paper/figures/output/stats"
    
    # Table S1: Train/Test
    train_test_csv = output_dir / "supplementary_train_test_metrics.csv"
    train_test_tex = output_dir / "supplementary_train_test_table.tex"
    if train_test_csv.exists():
        csv_to_latex_table(train_test_csv, train_test_tex, format_train_test)
    else:
        print(f"Warning: {train_test_csv} not found")
    
    # Table S2: Per-class
    per_class_csv = output_dir / "supplementary_per_class_metrics.csv"
    per_class_tex = output_dir / "supplementary_per_class_table.tex"
    if per_class_csv.exists():
        csv_to_latex_table(per_class_csv, per_class_tex, format_per_class)
    else:
        print(f"Warning: {per_class_csv} not found")
    
    print("\n" + "=" * 70)
    print("✓ LaTeX table generation complete")

if __name__ == "__main__":
    main()
