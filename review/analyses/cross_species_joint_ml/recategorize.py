#!/usr/bin/env python3
"""Re-categorize joint genes with refined biology categories.
Reads the mygene cache and updates joint_features_*.csv and results_*.json
without re-running the ML training.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
ANNOT_CACHE = OUT_DIR / "cache" / "mygene_annotations.json"
TISSUES = ["Muscle", "Brain", "Liver", "Lung", "Blood"]

# Refined categories. Order matters - first match wins.
REFINED_CATEGORIES = [
    ("synaptic_neuronal", [
        "synap", "neuron", "axon", "dendrit", "myelin",
        "neurotransmit", "glutamatergic", "GABA receptor",
        "acetylcholine",
    ]),
    ("muscle_contraction", [
        "muscle contraction", "myofibril", "sarcomere",
        "actin filament", "myosin", "troponin",
        "skeletal muscle", "contractile fiber",
    ]),
    ("lipid_fatty_acid_metabolism", [
        "lipid metabol", "fatty acid", "cholesterol",
        "lipoprotein", "phospholipid", "sphingolipid",
        "lipid transport", "triglyceride",
    ]),
    ("xenobiotic_drug_metabolism", [
        "cytochrome P450", "xenobiotic", "drug metabolism",
        "monooxygenase", "glutathione",
    ]),
    ("hepatocyte_specific", [
        "bile acid", "bile", "gluconeogenesis", "urea cycle",
        "hepatocyte differentiation",
    ]),
    ("immune_inflammation", [
        "immune response", "inflammat", "interleukin",
        "interferon", "complement activation", "chemokine",
        "lymphocyte", "antigen", "B cell", "T cell",
    ]),
    ("cell_cycle_proliferation", [
        "cell cycle", "mitotic", "cyclin", "DNA replication",
        "kinetochore", "G1/S", "G2/M", "proliferation",
    ]),
    ("dna_chromatin", [
        "DNA repair", "chromatin", "histone modification",
        "DNA binding", "transcription factor activity",
        "homeobox", "zinc finger",
    ]),
    ("extracellular_matrix", [
        "extracellular matrix", "collagen", "laminin",
        "elastin", "basement membrane", "fibrillar collagen",
        "proteoglycan",
    ]),
    ("mitochondrial_energy", [
        "mitochondri", "oxidative phosphorylation", "TCA cycle",
        "electron transport chain", "respiratory chain",
        "ATP synthesis",
    ]),
    ("erythroid_blood", [
        "erythrocyte", "hemoglobin", "heme biosynth",
        "platelet", "blood coagulation", "fibrin",
    ]),
    ("general_metabolism", [
        "metabolic process", "biosynth", "catabolic",
        "glycolysis", "amino acid metabolism",
    ]),
]


def classify(text: str) -> str:
    text_lc = text.lower()
    for cat, kws in REFINED_CATEGORIES:
        if any(kw.lower() in text_lc for kw in kws):
            return cat
    return "other"


def main() -> None:
    if not ANNOT_CACHE.exists():
        print("No mygene cache; nothing to re-categorize")
        return
    cache = json.loads(ANNOT_CACHE.read_text())
    print(f"  loaded {len(cache)} annotated symbols")

    # Reclassify cache entries
    for sym, rec in cache.items():
        text = " | ".join((rec.get("go_bp") or []) + [rec.get("name", ""),
                                                       rec.get("summary", "")])
        rec["category"] = classify(text)
    ANNOT_CACHE.write_text(json.dumps(cache, indent=1))

    # Now update per-tissue joint_features CSV and results JSON
    for t in TISSUES:
        rj = OUT_DIR / f"results_{t}.json"
        jcsv = OUT_DIR / f"joint_features_{t}.csv"
        if not rj.exists():
            continue
        d = json.loads(rj.read_text())
        feats = d.get("joint_features", [])
        for f in feats:
            sym = f.get("symbol", "")
            f["category"] = cache.get(sym, {}).get("category", "other")
        d["joint_features"] = feats

        # rebuild annotation summary
        counts = defaultdict(int)
        for f in feats:
            counts[f["category"]] += 1
        d["annotation_summary"] = dict(
            sorted(counts.items(), key=lambda kv: -kv[1])
        )
        rj.write_text(json.dumps(d, indent=2, default=str))

        # update CSV
        if jcsv.exists():
            df = pd.read_csv(jcsv)
            if len(df) > 0 and "symbol" in df.columns:
                df["category"] = df["symbol"].map(
                    lambda s: cache.get(s, {}).get("category", "other"))
                df.to_csv(jcsv, index=False)
        print(f"  {t}: {d['annotation_summary']}")


if __name__ == "__main__":
    main()
