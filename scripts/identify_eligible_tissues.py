#!/usr/bin/env python3
"""
Identify all eligible tissues based on sample count thresholds.
"""

import json
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DataLoader
from src.stage_selection import StageGranularitySelector

# Load metadata
data_loader = DataLoader(
    data_dir=PROJECT_ROOT / "data/pigGTEx",
    metadata_path=PROJECT_ROOT / "data/full_metadata.csv"
)
data_loader.load_metadata()

# Get tissue stage counts
tissue_stage_counts = data_loader.get_tissue_stage_counts()

# Determine eligible tissues
stage_selector = StageGranularitySelector()
eligible_info = stage_selector.evaluate_all_tissues(tissue_stage_counts)

# Categorize by scheme
four_class = []
three_class = []
two_class = []
ineligible = []

for tissue, info in eligible_info.items():
    if info['scheme'] == '4-class':
        four_class.append(tissue)
    elif info['scheme'] == '3-class':
        three_class.append(tissue)
    elif info['scheme'] == '2-class':
        two_class.append(tissue)
    else:
        ineligible.append(tissue)

print("=" * 60)
print("ELIGIBLE TISSUES BY CLASSIFICATION SCHEME")
print("=" * 60)

print(f"\n4-CLASS ELIGIBLE ({len(four_class)} tissues):")
for tissue in sorted(four_class):
    counts = eligible_info[tissue]['original_counts']
    total = sum(counts.values())
    print(f"  - {tissue}: {total} samples")

print(f"\n3-CLASS ELIGIBLE ({len(three_class)} tissues):")
for tissue in sorted(three_class):
    counts = eligible_info[tissue]['original_counts']
    total = sum(counts.values())
    print(f"  - {tissue}: {total} samples")

print(f"\n2-CLASS ELIGIBLE ({len(two_class)} tissues):")
for tissue in sorted(two_class):
    counts = eligible_info[tissue]['original_counts']
    total = sum(counts.values())
    print(f"  - {tissue}: {total} samples")

print(f"\nINELIGIBLE ({len(ineligible)} tissues):")
for tissue in sorted(ineligible)[:10]:  # Show first 10
    counts = eligible_info[tissue]['original_counts']
    total = sum(counts.values())
    print(f"  - {tissue}: {total} samples (insufficient)")

# Get all eligible tissues
all_eligible = four_class + three_class + two_class

print("\n" + "=" * 60)
print(f"TOTAL ELIGIBLE TISSUES: {len(all_eligible)}")
print("=" * 60)

# Check which expression files exist
available = []
missing = []

data_dir = PROJECT_ROOT / "data/pigGTEx"

for tissue in all_eligible:
    expr_file = data_dir / f"{tissue}.expr_tpm.txt.gz"
    if expr_file.exists():
        available.append(tissue)
    else:
        # Try alternative naming
        alt_names = [
            tissue.replace(' ', '_'),
            tissue.replace('_', ' '),
            tissue.lower(),
            tissue.upper()
        ]
        found = False
        for alt in alt_names:
            alt_file = data_dir / f"{alt}.expr_tpm.txt.gz"
            if alt_file.exists():
                available.append(tissue)
                found = True
                break
        if not found:
            missing.append(tissue)

print(f"\nEXPRESSION FILES AVAILABLE: {len(available)}/{len(all_eligible)}")
print(f"Available tissues: {available}")

if missing:
    print(f"\nMissing expression files for: {missing}")

# Save eligible tissue list
eligible_path = PROJECT_ROOT / 'artifacts/results/eligible_tissues.json'
eligible_path.parent.mkdir(parents=True, exist_ok=True)
with open(eligible_path, 'w') as f:
    json.dump({
        'four_class': four_class,
        'three_class': three_class,
        'two_class': two_class,
        'all_eligible': all_eligible,
        'available': available
    }, f, indent=2)

print(f"\n✓ Saved eligible tissue list to {eligible_path.relative_to(PROJECT_ROOT)}")
