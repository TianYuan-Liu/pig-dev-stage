"""
Construct the TechBatch composite batch variable for ML pipeline diagnostics.

TechBatch groups samples by sequencer generation x library layout, capturing
genuine technical variation while ensuring each batch contains multiple
developmental stages (unlike BioProject, which is heavily confounded with age).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ---- Sequencer model -> generation mapping --------------------------------
# Keys are substrings matched case-insensitively against the metadata Model column.
_SEQUENCER_GENERATION = {
    # Illumina Early (oldest chemistry, small N)
    "Genome Analyzer IIx": "Illumina_Early",
    "Genome Analyzer II": "Illumina_Early",
    "Genome Analyzer": "Illumina_Early",
    "HiScanSQ": "Illumina_Early",
    "HiSeq 1000": "Illumina_Early",
    "HiSeq 1500": "Illumina_Early",
    # Illumina HiSeq 2k era (random-cluster SBS v3/v4)
    "HiSeq 2000": "Illumina_HiSeq2k",
    "HiSeq 2500": "Illumina_HiSeq2k",
    "MiSeq": "Illumina_HiSeq2k",
    # Illumina HiSeq 3k era (patterned flowcell)
    "HiSeq 3000": "Illumina_HiSeq3k",
    "HiSeq 4000": "Illumina_HiSeq3k",
    "HiSeq X Ten": "Illumina_HiSeq3k",
    "NextSeq 500": "Illumina_HiSeq3k",
    "NextSeq 550": "Illumina_HiSeq3k",
    # Illumina NovaSeq (2-dye SBB chemistry)
    "NovaSeq 6000": "Illumina_NovaSeq",
}

# Only these large generations are further split by library layout.
_SPLIT_BY_LAYOUT = {"Illumina_HiSeq2k", "Illumina_HiSeq3k"}


def create_tech_batch(metadata: pd.DataFrame) -> pd.Series:
    """Derive a TechBatch Series from metadata columns ``Model`` and ``LibraryLayout``.

    Parameters
    ----------
    metadata : pd.DataFrame
        Must contain a ``Model`` column (sequencer model string).
        ``LibraryLayout`` is optional; if absent, no layout split is applied.

    Returns
    -------
    pd.Series
        TechBatch labels indexed like *metadata*.
    """
    if "Model" not in metadata.columns:
        logger.warning("'Model' column not found in metadata; TechBatch set to 'Unknown'")
        return pd.Series("Unknown", index=metadata.index, name="TechBatch")

    def _map_model(model_str: str) -> str:
        if pd.isna(model_str):
            return "Other_Platform"
        model_str = str(model_str).strip()
        # Match longest key first to avoid e.g. "Genome Analyzer" matching before
        # "Genome Analyzer II".
        for key in sorted(_SEQUENCER_GENERATION, key=len, reverse=True):
            if key.lower() in model_str.lower():
                return _SEQUENCER_GENERATION[key]
        return "Other_Platform"

    generation = metadata["Model"].apply(_map_model)

    has_layout = "LibraryLayout" in metadata.columns
    if has_layout:
        layout = metadata["LibraryLayout"].fillna("UNKNOWN").str.upper()
    else:
        layout = pd.Series("UNKNOWN", index=metadata.index)

    def _combine(row_gen: str, row_layout: str) -> str:
        if row_gen in _SPLIT_BY_LAYOUT:
            return f"{row_gen}_{row_layout}"
        return row_gen

    tech_batch = pd.Series(
        [_combine(g, l) for g, l in zip(generation, layout)],
        index=metadata.index,
        name="TechBatch",
    )

    counts = tech_batch.value_counts()
    logger.info(f"TechBatch distribution ({len(counts)} levels):\n{counts.to_string()}")

    return tech_batch
