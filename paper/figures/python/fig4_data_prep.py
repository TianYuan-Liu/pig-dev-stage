#!/usr/bin/env python3
"""
Figure 4 Data Preparation: Cross-Species Analysis
==================================================
Prepares data for cross-species (pig-human) muscle aging comparison.

Outputs:
    - fig4_orthology_mapping.csv: Gene ID to symbol mapping
    - fig4_expression_stats.csv: Fold changes and p-values for both species
"""

import json
import time
import warnings
from pathlib import Path

import gzip
import numpy as np
import pandas as pd
import requests
from scipy import stats

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures" / "output" / "stats"
ML_RESULTS = PROJECT_ROOT / "machine_learning" / "model_outputs" / "Muscle_results.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Targeted biomarkers for special labeling
BIOMARKERS = {
    "MYH3": "MHCdev",
    "MYH8": "MHCneonatal",
    "XIRP1": "XIRP1",
    "XIRP2": "XIRP2",
    "TNXB": "TNXB",
    "S100A1": "S100A1",
    "MAP4": "MAP4",
    "MLIP": "MLIP",
    "KLHL40": "KLHL40",
    "MSN": "MSN",
    "FHOD1": "FHOD1",
    "HSPA5": "HSPA5"
}

# ==============================================================================
# API FUNCTIONS
# ==============================================================================

def batch_lookup_ids(ensembl_ids: list, max_retries: int = 3) -> dict:
    """Batch lookup Ensembl IDs to get gene symbols."""
    url = "https://rest.ensembl.org/lookup/id"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    
    results = {}
    chunks = [ensembl_ids[i:i + 50] for i in range(0, len(ensembl_ids), 50)]
    
    print(f"  Querying {len(chunks)} batches for {len(ensembl_ids)} IDs...")
    
    for chunk in chunks:
        payload = json.dumps({"ids": chunk})
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, data=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    for old_id, info in data.items():
                        if info:
                            results[old_id] = {
                                "pig_gene_id": old_id,
                                "gene_symbol": info.get("display_name", ""),
                                "description": info.get("description", ""),
                                "success": True
                            }
                        else:
                            results[old_id] = {"pig_gene_id": old_id, "success": False}
                    break
                elif response.status_code == 429:
                    time.sleep(int(response.headers.get("Retry-After", 2)))
            except Exception as e:
                print(f"    Retry {attempt + 1}: {e}")
                time.sleep(1)
        time.sleep(0.1)
    
    return results


def batch_lookup_symbols(symbols: list, species: str = "homo_sapiens") -> dict:
    """Batch lookup gene symbols to get lengths for TPM calculation."""
    url = f"https://rest.ensembl.org/lookup/symbol/{species}"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    
    results = {}
    valid_symbols = [s for s in symbols if s]
    if not valid_symbols:
        return {}
    
    chunks = [valid_symbols[i:i + 50] for i in range(0, len(valid_symbols), 50)]
    print(f"  Querying {len(chunks)} batches for {len(valid_symbols)} symbols...")
    
    for chunk in chunks:
        payload = json.dumps({"symbols": chunk})
        for attempt in range(3):
            try:
                response = requests.post(url, headers=headers, data=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    for sym, info in data.items():
                        if info:
                            results[sym] = info["end"] - info["start"]
                    break
                elif response.status_code == 429:
                    time.sleep(int(response.headers.get("Retry-After", 2)))
            except Exception:
                time.sleep(1)
        time.sleep(0.1)
    
    return results


def get_pig_id_from_symbol_batch(symbols: list) -> dict:
    """Batch lookup pig gene IDs from symbols."""
    url = "https://rest.ensembl.org/lookup/symbol/sus_scrofa"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    
    results = {}
    chunks = [symbols[i:i + 50] for i in range(0, len(symbols), 50)]
    
    for chunk in chunks:
        payload = json.dumps({"symbols": chunk})
        for attempt in range(3):
            try:
                response = requests.post(url, headers=headers, data=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    for sym, info in data.items():
                        if info:
                            results[sym] = info["id"]
                    break
                elif response.status_code == 429:
                    time.sleep(int(response.headers.get("Retry-After", 2)))
            except Exception:
                time.sleep(1)
    
    return results


# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

def load_ml_results() -> tuple:
    """Load ML results and extract top genes with importance."""
    with open(ML_RESULTS) as f:
        data = json.load(f)
    
    top_genes = data.get("top_genes", [])[:1000]
    importance_map = data.get("feature_importance", {})
    
    return top_genes, importance_map


def build_gene_metadata(top_genes: list, biomarkers: dict) -> pd.DataFrame:
    """Build unified gene list with metadata."""
    combined_genes = []
    seen = set()
    
    # Map biomarkers to pig IDs
    print("Mapping biomarkers to pig IDs...")
    biomarker_symbols = list(biomarkers.keys())
    pig_id_map = get_pig_id_from_symbol_batch(biomarker_symbols)
    
    biomarker_map = {}
    for sym, label in biomarkers.items():
        pig_id = pig_id_map.get(sym)
        if pig_id:
            biomarker_map[pig_id] = label
            if pig_id not in seen:
                combined_genes.append(pig_id)
                seen.add(pig_id)
            print(f"  {sym} -> {pig_id}")
    
    # Add top genes
    print(f"Adding top {len(top_genes)} genes...")
    for gene in top_genes:
        if gene not in seen:
            combined_genes.append(gene)
            seen.add(gene)
    
    # Batch lookup metadata
    print(f"Resolving metadata for {len(combined_genes)} genes...")
    pig_info_map = batch_lookup_ids(combined_genes)
    
    # Prepare for human lookup
    pig_symbols = []
    final_list = []
    
    for pig_id in combined_genes:
        info = pig_info_map.get(pig_id, {"pig_gene_id": pig_id, "success": False})
        sym = info.get("gene_symbol")
        if sym:
            pig_symbols.extend([sym, sym.upper()])
        info["biomarker_label"] = biomarker_map.get(pig_id, None)
        final_list.append(info)
    
    # Lookup human gene lengths
    print(f"Resolving human gene lengths...")
    human_len_map = batch_lookup_symbols(list(set(pig_symbols)))
    
    # Merge lengths
    for item in final_list:
        sym = item.get("gene_symbol")
        length = None
        if sym:
            length = human_len_map.get(sym) or human_len_map.get(sym.upper())
        item["human_length"] = length
    
    return pd.DataFrame(final_list)


def load_pig_expression() -> tuple:
    """Load pig muscle expression data."""
    print("\nLoading pig expression...")
    
    meta = pd.read_csv(DATA_DIR / "full_metadata.csv")
    muscle_meta = meta[meta["Tissue"] == "Muscle"]
    
    young_ids = muscle_meta[muscle_meta["Stage"] == "Infant_0_20d"]["Sample_ID"].tolist()
    adult_ids = muscle_meta[muscle_meta["Stage"] == "Adult_>365d"]["Sample_ID"].tolist()
    
    expr_path = DATA_DIR / "pigGTEx" / "Muscle.expr_tpm.txt.gz"
    with gzip.open(expr_path, 'rt') as f:
        expr = pd.read_csv(f, sep='\t', index_col=0)
    
    y_av = [x for x in young_ids if x in expr.columns]
    a_av = [x for x in adult_ids if x in expr.columns]
    
    print(f"  Pig samples: {len(y_av)} young, {len(a_av)} adult")
    return expr, y_av, a_av


def load_human_expression(len_map: dict) -> tuple:
    """Load human muscle expression data."""
    print("\nLoading human expression...")
    
    path = DATA_DIR / "human_muscle" / "GSE257558_read_counts_healty.csv"
    expr = pd.read_csv(path, index_col=0)
    
    infant_cols = [c for c in expr.columns if c.startswith("I")]
    adult_cols = [c for c in expr.columns if c.startswith("A")]
    
    # CPM normalization
    cpm = (expr / expr.sum()) * 1e6
    tpm = pd.DataFrame(index=expr.index, columns=expr.columns)
    
    symbol_upper_map = {str(k).upper(): v for k, v in len_map.items() if v and isinstance(k, str)}
    
    for gene in expr.index:
        g = gene.upper()
        if g in symbol_upper_map:
            l_kb = symbol_upper_map[g] / 1000.0
            tpm.loc[gene] = cpm.loc[gene] / l_kb
        else:
            tpm.loc[gene] = cpm.loc[gene]
    
    print(f"  Human samples: {len(infant_cols)} infant, {len(adult_cols)} adult")
    return tpm, infant_cols, adult_cols


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("Figure 4 Data Preparation: Cross-Species Analysis")
    print("=" * 70)
    
    # Load ML results
    top_genes, importance_map = load_ml_results()
    print(f"Loaded {len(top_genes)} top genes from ML results")
    
    # Build gene metadata
    meta_df = build_gene_metadata(top_genes, BIOMARKERS)
    meta_df["importance"] = meta_df["pig_gene_id"].map(importance_map).fillna(0)
    
    # Save orthology mapping
    meta_df.to_csv(OUTPUT_DIR / "fig4_orthology_mapping.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'fig4_orthology_mapping.csv'}")
    
    # Load expression data
    pig_expr, p_young, p_adult = load_pig_expression()
    
    len_map = dict(zip(meta_df.gene_symbol, meta_df.human_length))
    human_expr, h_infant, h_adult = load_human_expression(len_map)
    
    # Calculate statistics
    print("\nCalculating expression statistics...")
    stats_rows = []
    
    for _, row in meta_df.iterrows():
        pig_id = row["pig_gene_id"]
        sym = row["gene_symbol"]
        label = row["biomarker_label"]
        imp = row["importance"]
        
        # Pig stats
        fc_pig, p_pig = np.nan, np.nan
        if pig_id in pig_expr.index:
            v1 = pig_expr.loc[pig_id, p_young].astype(float)
            v2 = pig_expr.loc[pig_id, p_adult].astype(float)
            fc_pig = np.log2((v1.mean() + 0.01) / (v2.mean() + 0.01))
            try:
                p_pig = stats.mannwhitneyu(v1, v2)[1]
            except:
                p_pig = 1.0
        
        # Human stats
        fc_human, p_human = np.nan, np.nan
        found_sym = None
        
        if sym and isinstance(sym, str):
            for c in [sym, sym.upper()]:
                if c in human_expr.index:
                    found_sym = c
                    break
            
            if found_sym:
                h1 = human_expr.loc[found_sym, h_infant].astype(float)
                h2 = human_expr.loc[found_sym, h_adult].astype(float)
                fc_human = np.log2((h1.mean() + 0.01) / (h2.mean() + 0.01))
                try:
                    p_human = stats.mannwhitneyu(h1, h2)[1]
                except:
                    p_human = 1.0
        
        stats_rows.append({
            "pig_gene_id": pig_id,
            "gene_symbol": sym,
            "biomarker_label": label,
            "importance": imp,
            "log2fc_pig": fc_pig,
            "p_pig": p_pig,
            "log2fc_human": fc_human,
            "p_human": p_human
        })
    
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUTPUT_DIR / "fig4_expression_stats.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'fig4_expression_stats.csv'}")
    
    # Summary
    valid = stats_df.dropna(subset=["log2fc_pig", "log2fc_human"])
    sig = valid[(valid["p_pig"] < 0.05) & (valid["p_human"] < 0.05)]
    conserved = sig[np.sign(sig["log2fc_pig"]) == np.sign(sig["log2fc_human"])]
    
    print(f"\nSummary:")
    print(f"  Total genes: {len(stats_df)}")
    print(f"  With both FC values: {len(valid)}")
    print(f"  Significant in both: {len(sig)}")
    print(f"  Conserved direction: {len(conserved)}")
    
    print("\n✓ Data preparation completed!")


if __name__ == "__main__":
    main()
