#!/usr/bin/env python3
"""
Pathway Analysis for Top Feature Genes

Uses gProfiler API to perform KEGG, Reactome, and WikiPathways enrichment
analysis for the top genes selected by the LightGBM model for each tissue.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import time

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

import pandas as pd


def load_top_genes(ml_dir: Path, top_n: int = 50) -> Dict[str, List[str]]:
    """Load top N genes per tissue from model results."""
    tissue_genes = {}
    for json_file in ml_dir.glob("*_results.json"):
        tissue = json_file.stem.replace("_results", "")
        with open(json_file, 'r') as f:
            data = json.load(f)
            genes = data.get('top_genes', [])[:top_n]
            tissue_genes[tissue] = genes
    return tissue_genes


def run_gprofiler_enrichment(gene_list: List[str], 
                              organism: str = "sscrofa",
                              sources: List[str] = None) -> Dict:
    """
    Run pathway enrichment using gProfiler API.
    
    Args:
        gene_list: List of gene IDs (Ensembl)
        organism: Organism code (sscrofa for pig)
        sources: Pathway databases to query
    
    Returns:
        Dictionary with enrichment results
    """
    if not HAS_REQUESTS:
        raise ImportError("requests library required for gProfiler API")
    
    if sources is None:
        sources = ["KEGG", "REAC", "WP", "GO:BP", "GO:MF", "GO:CC"]
    
    # gProfiler API endpoint
    url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
    
    payload = {
        "organism": organism,
        "query": gene_list,
        "sources": sources,
        "user_threshold": 0.05,
        "significance_threshold_method": "fdr",
        "domain_scope": "annotated",
        "no_iea": False,
        "ordered": True,  # Genes are ordered by importance
        "measure_underrepresentation": False,
        "no_evidences": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"  API request failed: {e}")
        return None


def parse_gprofiler_results(results: Dict) -> pd.DataFrame:
    """Parse gProfiler results into a DataFrame."""
    if not results or 'result' not in results:
        return pd.DataFrame()
    
    data = []
    for item in results['result']:
        data.append({
            'source': item.get('source', ''),
            'term_id': item.get('native', ''),
            'term_name': item.get('name', ''),
            'p_value': item.get('p_value', 1.0),
            'term_size': item.get('term_size', 0),
            'query_size': item.get('query_size', 0),
            'intersection_size': item.get('intersection_size', 0),
            'precision': item.get('precision', 0),
            'recall': item.get('recall', 0),
            'intersecting_genes': ','.join(item.get('intersections', []) or [])
        })
    
    df = pd.DataFrame(data)
    if not df.empty:
        df['fold_enrichment'] = (df['intersection_size'] / df['query_size']) / (df['term_size'] / 20000)
        df = df.sort_values('p_value')
    
    return df


def run_analysis_for_all_tissues(tissue_genes: Dict[str, List[str]], 
                                   output_dir: Path) -> Dict[str, pd.DataFrame]:
    """Run pathway analysis for all tissues."""
    all_results = {}
    
    # Pathway sources to query
    sources = ["KEGG", "REAC", "WP"]  # KEGG, Reactome, WikiPathways
    
    for tissue, genes in tissue_genes.items():
        print(f"\n  Analyzing {tissue} ({len(genes)} genes)...")
        
        results = run_gprofiler_enrichment(genes, sources=sources)
        
        if results:
            df = parse_gprofiler_results(results)
            if not df.empty:
                # Save tissue-specific results
                df['tissue'] = tissue
                df.to_csv(output_dir / f"{tissue}_pathway_enrichment.csv", index=False)
                all_results[tissue] = df
                
                print(f"    Found {len(df)} significant pathways")
                
                # Print top 5 pathways
                print(f"    Top pathways:")
                for _, row in df.head(5).iterrows():
                    print(f"      - {row['term_name'][:50]} ({row['source']}, p={row['p_value']:.2e})")
            else:
                print(f"    No significant pathways found")
        else:
            print(f"    API query failed")
        
        time.sleep(1)  # Rate limiting
    
    return all_results


def run_combined_analysis(tissue_genes: Dict[str, List[str]], 
                           output_dir: Path) -> pd.DataFrame:
    """Run pathway analysis on all genes combined."""
    print("\n  Analyzing all tissues combined...")
    
    # Combine all unique genes
    all_genes = list(set(g for genes in tissue_genes.values() for g in genes))
    print(f"    Total unique genes: {len(all_genes)}")
    
    sources = ["KEGG", "REAC", "WP", "GO:BP"]
    results = run_gprofiler_enrichment(all_genes, sources=sources)
    
    if results:
        df = parse_gprofiler_results(results)
        if not df.empty:
            df['tissue'] = 'Combined'
            df.to_csv(output_dir / "combined_pathway_enrichment.csv", index=False)
            print(f"    Found {len(df)} significant pathways")
            return df
    
    print("    No significant pathways found")
    return pd.DataFrame()


def create_pathway_summary(all_results: Dict[str, pd.DataFrame], 
                            output_dir: Path) -> str:
    """Create a summary report of pathway analysis results."""
    
    report = ["# Pathway Enrichment Analysis Results\n"]
    report.append("Analysis of top 50 feature genes per tissue using KEGG, Reactome, and WikiPathways.\n\n")
    
    for tissue, df in all_results.items():
        if df.empty:
            continue
            
        report.append(f"\n## {tissue}\n\n")
        
        # Separate by source
        for source in ["KEGG", "REAC", "WP", "GO:BP"]:
            source_df = df[df['source'] == source]
            if source_df.empty:
                continue
            
            source_name = {
                "KEGG": "KEGG Pathways",
                "REAC": "Reactome Pathways", 
                "WP": "WikiPathways",
                "GO:BP": "GO Biological Process"
            }.get(source, source)
            
            report.append(f"\n### {source_name}\n\n")
            report.append("| Pathway | p-value | Genes |\n")
            report.append("|---------|---------|-------|\n")
            
            for _, row in source_df.head(10).iterrows():
                name = row['term_name'][:45] + "..." if len(row['term_name']) > 45 else row['term_name']
                p_val = f"{row['p_value']:.2e}"
                n_genes = row['intersection_size']
                report.append(f"| {name} | {p_val} | {n_genes} |\n")
    
    report_text = "".join(report)
    
    with open(output_dir / "pathway_analysis_summary.md", 'w') as f:
        f.write(report_text)
    
    return report_text


def main():
    """Main analysis function."""
    script_dir = Path(__file__).parent
    ml_dir = script_dir.parent / "model_outputs"
    output_dir = script_dir / "pathway_analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Pathway Enrichment Analysis")
    print("=" * 60)
    
    if not HAS_REQUESTS:
        print("\nERROR: requests library required. Install with: pip install requests")
        return
    
    # Load genes
    print("\n1. Loading top 50 genes per tissue...")
    tissue_genes = load_top_genes(ml_dir, top_n=50)
    print(f"   Loaded genes for {len(tissue_genes)} tissues")
    
    # Run per-tissue analysis
    print("\n2. Running pathway analysis per tissue...")
    all_results = run_analysis_for_all_tissues(tissue_genes, output_dir)
    
    # Run combined analysis
    print("\n3. Running combined pathway analysis...")
    combined_df = run_combined_analysis(tissue_genes, output_dir)
    if not combined_df.empty:
        all_results['Combined'] = combined_df
    
    # Create summary report
    print("\n4. Creating summary report...")
    summary = create_pathway_summary(all_results, output_dir)
    
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    
    # Print summary
    print("\n" + summary)
    
    return all_results


if __name__ == "__main__":
    main()
