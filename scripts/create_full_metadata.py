#!/usr/bin/env python3
"""
Create metadata file from the original PigGTEx metadata source.
"""

from pathlib import Path
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "PigGTEx_v0.MetaTable.xlsx"


def parse_age_days(age_value):
    if pd.isna(age_value):
        return None

    if isinstance(age_value, (int, float)):
        return float(age_value)

    age_str = str(age_value).strip().lower()
    if not age_str or "unknown" in age_str:
        return None

    match = re.search(r"([0-9]+(?:\\.[0-9]+)?)", age_str)
    if not match:
        return None

    value = float(match.group(1))
    if "day" in age_str:
        return value
    if "week" in age_str:
        return value * 7
    if "month" in age_str:
        return value * 30
    if "year" in age_str:
        return value * 365

    return None


def assign_stage(age_days):
    if age_days is None:
        return None
    if age_days <= 20:
        return "Infant_0_20d"
    if age_days <= 59:
        return "Early childhood_21_59d"
    if age_days <= 149:
        return "Pre_pubertal_60_149d"
    if age_days <= 365:
        return "Post_pubertal_150_365d"
    return "Adult_>365d"


if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

metadata = pd.read_excel(METADATA_PATH)
metadata = metadata.rename(
    columns={
        "BioSample": "Sample_ID",
        "Main categories": "Tissue",
        "Sub categories": "Tissue_detail",
        "Age": "Age_raw",
    }
)

metadata["Age"] = metadata["Age_raw"].apply(parse_age_days)
metadata["Stage"] = metadata["Age"].apply(assign_stage)

output = metadata[["Sample_ID", "Tissue", "Tissue_detail", "Age", "Stage", "Sex"]]
output.to_csv(PROJECT_ROOT / "data" / "full_metadata.csv", index=False)

print(f"Created metadata with {len(output)} samples")
print("\nTissue distribution:")
print(output["Tissue_detail"].value_counts(dropna=False).head(10))
print("\nStage distribution:")
print(output["Stage"].value_counts(dropna=False))
