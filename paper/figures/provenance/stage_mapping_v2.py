"""Uniform pig↔human stage mapping under dGTEx-only cross-species protocol.

Per Stage A discovery (2026-05-25, see review/analyses/results/dgtex_discovery_report.md):
dGTEx encodes age as AGECOHORT integer 1-4 with the EXACT SAME stage names as our pig staging.

dGTEx Subject_Phenotypes_Public_DD.xlsx data-dictionary entries (verbatim):
    AGECOHORT = 1 = Infant (0 to 2 years)
    AGECOHORT = 2 = Early Childhood (2 to 8 years)
    AGECOHORT = 3 = Pre-pubertal (8 to approximately 13)
    AGECOHORT = 4 = Post-pubertal (approximately 13 to 18)
"""

from __future__ import annotations

PIG_YOUNG_STAGES = ("Infant", "Early childhood")
PIG_OLD_STAGES = ("Post-pubertal", "Adult")

HUMAN_YOUNG_COHORTS = (1,)
HUMAN_OLD_COHORTS = (4,)

N_MIN_SAMPLES_PER_BIN = 3


def pig_bin(stage: str) -> str | None:
    if stage in PIG_YOUNG_STAGES:
        return "young"
    if stage in PIG_OLD_STAGES:
        return "old"
    return None


def human_bin(agecohort: int) -> str | None:
    if agecohort in HUMAN_YOUNG_COHORTS:
        return "young"
    if agecohort in HUMAN_OLD_COHORTS:
        return "old"
    return None


INCLUDED_TISSUES: tuple[tuple[str, str, str], ...] = (
    ("Muscle",          "Muscle",          "muscle"),
    ("Lung",            "Lung",            "lung"),
    ("Small Intestine", "Small_intestine", "small_intestine"),
    ("Liver",           "Liver",           "liver"),
    ("Adipose Tissue",  "Adipose",         "adipose_tissue"),
    ("Testis",          "Testis",          "testis"),
    ("Spleen",          "Spleen",          "spleen"),
    ("Heart",           "Heart",           "heart"),
)
