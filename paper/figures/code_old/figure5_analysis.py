#!/usr/bin/env python3
"""
Figure 5 Analysis: Cross-Species Muscle Tissue Age-Related Gene Expression
Updated to include Targeted Biomarker Analysis.
"""

import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import gzip
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures" / "out"
ML_RESULTS = PROJECT_ROOT / "machine_learning" / "model_outputs" / "Muscle_results.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define Targeted Biomarkers
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

def load_ml_results(results_path: Path):
    """Load top genes and feature importance."""
    with open(results_path) as f:
        data = json.load(f)
    top_genes = data.get("top_genes", [])[:1000]
    importance_map = data.get("feature_importance", {})
    return top_genes, importance_map

def batch_lookup_ids(ensembl_ids, max_retries=3):
    """Batch lookup Ensembl IDs."""
    url = "https://rest.ensembl.org/lookup/id"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    
    # Chunk into 50s
    results = {}
    chunks = [ensembl_ids[i:i + 50] for i in range(0, len(ensembl_ids), 50)]
    
    print(f"  Performing {len(chunks)} batch requests for {len(ensembl_ids)} IDs...")
    
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
                    continue
            except Exception as e:
                print(f"    Batch request failed: {e}")
                time.sleep(1)
        time.sleep(0.1)
        
    return results

def batch_lookup_symbols(symbols, species="homo_sapiens", max_retries=3):
    """Batch lookup Symbols."""
    url = f"https://rest.ensembl.org/lookup/symbol/{species}"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    
    results = {}
    valid_symbols = [s for s in symbols if s]
    if not valid_symbols:
        return {}
        
    chunks = [valid_symbols[i:i + 50] for i in range(0, len(valid_symbols), 50)]
    print(f"  Performing {len(chunks)} batch requests for {len(valid_symbols)} Symbols...")
    
    for chunk in chunks:
        payload = json.dumps({"symbols": chunk})
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, data=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    for sym, info in data.items():
                        if info:
                            # Calculate length
                            length = info["end"] - info["start"]
                            results[sym] = length
                        else:
                            results[sym] = None
                    break
                elif response.status_code == 429:
                    time.sleep(int(response.headers.get("Retry-After", 2)))
                    continue
            except Exception as e:
                print(f"    Batch request failed: {e}")
                time.sleep(1)
        time.sleep(0.1)
        
    return results

def get_pig_id_from_symbol_batch(symbols, max_retries=3):
    """Batch lookup Pig IDs from symbols."""
    url = "https://rest.ensembl.org/lookup/symbol/sus_scrofa"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    
    results = {}
    chunks = [symbols[i:i + 50] for i in range(0, len(symbols), 50)]
    
    for chunk in chunks:
        payload = json.dumps({"symbols": chunk})
        for attempt in range(max_retries):
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
                    continue
            except:
                time.sleep(1)
    return results

def build_gene_metadata(top_genes: list, biomarkers: dict):
    """Build unified list of genes with metadata using Batch APIs."""
    combined_genes = []
    seen = set()
    
    # 1. Process Biomarkers (Reverse Lookup)
    print("Mapping biomarkers to Pig IDs (Batch)...")
    biomarker_symbols = list(biomarkers.keys())
    pig_id_map = get_pig_id_from_symbol_batch(biomarker_symbols)
    
    biomarker_map = {} # PigID -> Biomarker Label
    for sym, label in biomarkers.items():
        pig_id = pig_id_map.get(sym)
        if pig_id:
            biomarker_map[pig_id] = label
            if pig_id not in seen:
                combined_genes.append(pig_id)
                seen.add(pig_id)
            print(f"  {sym} -> {pig_id}")
        else:
            print(f"  {sym} -> Not found")

    # 2. Add Top Genes
    print(f"Adding top genes (total {len(top_genes)})...")
    for gene in top_genes:
        if gene not in seen:
            combined_genes.append(gene)
            seen.add(gene)
            
    # 3. Batch Lookup Pig Metadata
    print(f"Resolving metadata for {len(combined_genes)} genes...")
    pig_info_map = batch_lookup_ids(combined_genes)
    
    # 4. Prepare for Human Lookup
    pig_symbols = []
    final_list = []
    
    for pig_id in combined_genes:
        info = pig_info_map.get(pig_id, {"pig_gene_id": pig_id, "success": False})
        sym = info.get("gene_symbol")
        if sym:
            pig_symbols.append(sym)
            # Also try UPPER for human
            pig_symbols.append(sym.upper())
            
        info["biomarker_label"] = biomarker_map.get(pig_id, None)
        final_list.append(info)
        
    # 5. Batch Lookup Human Lengths
    print(f"Resolving human lengths for potential {len(pig_symbols)} symbols...")
    human_len_map = batch_lookup_symbols(list(set(pig_symbols)))
    
    # 6. Merge Lengths
    for item in final_list:
        sym = item.get("gene_symbol")
        length = None
        if sym:
            length = human_len_map.get(sym)
            if not length:
                length = human_len_map.get(sym.upper())
        item["human_length"] = length
        
    return pd.DataFrame(final_list)

def load_pig_expr(data_dir: Path, metadata_path: Path):
    print("\nLoading pig expression...")
    meta = pd.read_csv(metadata_path)
    muscle_meta = meta[meta["Tissue"] == "Muscle"]
    
    # Using Infant (0-20d) vs Adult (>365d)
    young_ids = muscle_meta[muscle_meta["Stage"] == "Infant_0_20d"]["Sample_ID"].tolist()
    adult_ids = muscle_meta[muscle_meta["Stage"] == "Adult_>365d"]["Sample_ID"].tolist()
    
    expr_path = data_dir / "pigGTEx" / "Muscle.expr_tpm.txt.gz"
    with gzip.open(expr_path, 'rt') as f:
        expr = pd.read_csv(f, sep='\t', index_col=0)
        
    y_av = [x for x in young_ids if x in expr.columns]
    a_av = [x for x in adult_ids if x in expr.columns]
    print(f"  Pig samples: {len(y_av)} Young, {len(a_av)} Adult")
    return expr, y_av, a_av

def load_human_expr(data_dir: Path, len_map: dict):
    print("\nLoading human expression...")
    path = data_dir / "human_muscle" / "GSE257558_read_counts_healty.csv"
    expr = pd.read_csv(path, index_col=0)
    
    inf = [c for c in expr.columns if c.startswith("I")]
    ad = [c for c in expr.columns if c.startswith("A")]
    
    # TPM calculation
    cpm = (expr / expr.sum()) * 1e6
    tpm = pd.DataFrame(index=expr.index, columns=expr.columns)
    
    symbol_upper_map = {str(k).upper(): v for k, v in len_map.items() if v and isinstance(k, str)}
    
    for gene in expr.index:
        g = gene.upper()
        if g in symbol_upper_map:
            l_kb = symbol_upper_map[g] / 1000.0
            tpm.loc[gene] = cpm.loc[gene] / l_kb
        else:
            tpm.loc[gene] = cpm.loc[gene] # Fallback
            
    return tpm, inf, ad

def main():
    print("Starting Biormarker Analysis...")
    
    # 1. Data Setup
    top_genes, importance_map = load_ml_results(ML_RESULTS)
    meta_df = build_gene_metadata(top_genes, BIOMARKERS)
    
    # Add importance to meta
    meta_df["importance"] = meta_df["pig_gene_id"].map(importance_map).fillna(0)
    
    # Save meta
    meta_df.to_csv(OUTPUT_DIR / "figure5_orthology_mapping.csv", index=False)
    
    # 2. Expression Data
    pig_expr, p_y, p_a = load_pig_expr(DATA_DIR, DATA_DIR / "full_metadata.csv")
    
    len_map = dict(zip(meta_df.gene_symbol, meta_df.human_length))
    hu_expr, h_i, h_a = load_human_expr(DATA_DIR, len_map)
    
    # 3. Stats Calculation
    stats_rows = []
    
    # Map pig IDs to Human Symbols for join
    pig_to_human = dict(zip(meta_df.pig_gene_id, meta_df.gene_symbol))
    
    for _, row in meta_df.iterrows():
        pig_id = row["pig_gene_id"]
        sym = row["gene_symbol"]
        label = row["biomarker_label"]
        imp = row["importance"]
        
        # Pig Stats
        if pig_id in pig_expr.index:
            v1 = pig_expr.loc[pig_id, p_y].astype(float)
            v2 = pig_expr.loc[pig_id, p_a].astype(float)
            fc_pig = np.log2((v1.mean()+0.01)/(v2.mean()+0.01))
            p_pig = stats.mannwhitneyu(v1, v2)[1]
        else:
            fc_pig, p_pig = np.nan, np.nan
            
        # Human Stats
        # Human dataframe index is Symbol
        # Try exact, then upper
        fc_hu, p_hu = np.nan, np.nan
        found_sym = None
        
        if sym and isinstance(sym, str):
            candidates = [sym, sym.upper()]
            for c in candidates:
                if c in hu_expr.index:
                    found_sym = c
                    break
            
            if found_sym:
                h1 = hu_expr.loc[found_sym, h_i].astype(float)
                h2 = hu_expr.loc[found_sym, h_a].astype(float)
                fc_hu = np.log2((h1.mean()+0.01)/(h2.mean()+0.01))
                try:
                    p_hu = stats.mannwhitneyu(h1, h2)[1]
                except:
                    p_hu = 1.0
                    
        stats_rows.append({
            "pig_gene_id": pig_id,
            "gene_symbol": sym,
            "biomarker_label": label,
            "importance": imp,
            "log2fc_pig": fc_pig,
            "p_pig": p_pig,
            "log2fc_human": fc_hu,
            "p_human": p_hu
        })
        
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUTPUT_DIR / "figure5_expression_stats.csv", index=False)
    print(f"Saved stats to {OUTPUT_DIR / 'figure5_expression_stats.csv'}")

if __name__ == "__main__":
    main()
