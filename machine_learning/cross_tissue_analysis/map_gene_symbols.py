#!/usr/bin/env python3
"""
Map Ensembl Gene IDs to Gene Symbols for Biological Interpretation

This script maps the top feature genes from LightGBM model outputs to gene symbols
using the BioMart Ensembl API, enabling biological interpretation.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set
import time

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


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


def query_biomart(ensembl_ids: List[str], chunk_size: int = 200) -> Dict[str, dict]:
    """
    Query Ensembl BioMart for gene information.
    
    Falls back to a manual lookup if BioMart is not accessible.
    """
    if not HAS_REQUESTS:
        print("  Warning: requests library not available, using offline mode")
        return {}
    
    all_results = {}
    
    # Process in chunks to avoid timeout
    for i in range(0, len(ensembl_ids), chunk_size):
        chunk = ensembl_ids[i:i+chunk_size]
        
        # BioMart XML query for pig (Sus scrofa)
        gene_list = ",".join(f'"{g}"' for g in chunk)
        xml_query = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
    <Dataset name="sscrofa_gene_ensembl" interface="default">
        <Filter name="ensembl_gene_id" value="{','.join(chunk)}"/>
        <Attribute name="ensembl_gene_id"/>
        <Attribute name="external_gene_name"/>
        <Attribute name="description"/>
        <Attribute name="gene_biotype"/>
        <Attribute name="chromosome_name"/>
    </Dataset>
</Query>'''
        
        try:
            url = "http://www.ensembl.org/biomart/martservice"
            response = requests.post(url, data={'query': xml_query}, timeout=30)
            
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                if len(lines) > 1:
                    for line in lines[1:]:  # Skip header
                        parts = line.split('\t')
                        if len(parts) >= 4:
                            ensembl_id = parts[0]
                            all_results[ensembl_id] = {
                                'symbol': parts[1] if parts[1] else ensembl_id,
                                'description': parts[2] if len(parts) > 2 else '',
                                'biotype': parts[3] if len(parts) > 3 else '',
                                'chromosome': parts[4] if len(parts) > 4 else ''
                            }
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"  Warning: BioMart query failed for chunk {i//chunk_size + 1}: {e}")
    
    return all_results


def create_gene_symbol_report(tissue_genes: Dict[str, List[str]], 
                               gene_info: Dict[str, dict],
                               output_dir: Path) -> str:
    """Create a detailed report with gene symbols and annotations."""
    
    report_lines = ["# Top Feature Genes by Tissue\n"]
    report_lines.append("Gene ID to Symbol Mapping for LightGBM Developmental Stage Model\n\n")
    
    all_genes_df_lines = ["tissue,rank,ensembl_id,symbol,description,biotype\n"]
    
    for tissue, genes in sorted(tissue_genes.items()):
        report_lines.append(f"\n## {tissue}\n\n")
        report_lines.append("| Rank | Ensembl ID | Symbol | Description |\n")
        report_lines.append("|------|------------|--------|-------------|\n")
        
        for rank, gene_id in enumerate(genes, 1):
            info = gene_info.get(gene_id, {})
            symbol = info.get('symbol', gene_id)
            desc = info.get('description', 'N/A')
            biotype = info.get('biotype', 'N/A')
            
            # Truncate description for table
            desc_short = desc[:50] + "..." if len(desc) > 50 else desc
            
            report_lines.append(f"| {rank} | {gene_id} | {symbol} | {desc_short} |\n")
            all_genes_df_lines.append(f"{tissue},{rank},{gene_id},{symbol},\"{desc}\",{biotype}\n")
    
    # Save CSV
    with open(output_dir / "top_genes_with_symbols.csv", 'w') as f:
        f.writelines(all_genes_df_lines)
    
    report = "".join(report_lines)
    with open(output_dir / "gene_symbol_report.md", 'w') as f:
        f.write(report)
    
    return report


def identify_known_developmental_markers(gene_info: Dict[str, dict], 
                                          tissue_genes: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Identify known developmental/aging markers among the top genes.
    Based on literature-known markers.
    """
    # Known developmental marker gene symbols/patterns
    known_markers = {
        'muscle_development': ['MYH', 'MYL', 'ACTN', 'TTN', 'MYOG', 'MYOD', 'PAX3', 'PAX7', 
                               'DMD', 'DES', 'TNNT', 'TNNI', 'TNNC', 'MEF2'],
        'liver_development': ['ALB', 'AFP', 'CYP', 'HNF', 'SERPINA', 'APOB', 'APOA', 
                              'TTR', 'FABP1', 'ADH', 'ALDH'],
        'brain_development': ['MBP', 'GFAP', 'SYP', 'SYN', 'SNAP', 'DLG', 'NEFH', 'NEFL', 
                              'TUBB', 'MAP2', 'SOX', 'OLIG', 'NES'],
        'blood_development': ['HBB', 'HBA', 'CD', 'PTPRC', 'SPN', 'IL', 'CCL', 'CXCL'],
        'lung_development': ['SFTPA', 'SFTPB', 'SFTPC', 'SFTPD', 'NKX2-1', 'FOXA', 'SCGB'],
        'general_aging': ['CDKN', 'TP53', 'TERT', 'LMNA', 'FOXO', 'SIRT', 'IGF', 'GH'],
        'cell_cycle': ['CCND', 'CCNE', 'CDK', 'RB1', 'E2F', 'MCM', 'PCNA'],
        'apoptosis': ['BCL', 'BAX', 'BAK', 'CASP', 'APAF', 'BID', 'BAD'],
        'growth_factors': ['IGF', 'EGF', 'FGF', 'TGF', 'VEGF', 'PDGF', 'HGF']
    }
    
    found_markers = {}
    
    for tissue, genes in tissue_genes.items():
        tissue_markers = []
        for gene_id in genes:
            info = gene_info.get(gene_id, {})
            symbol = info.get('symbol', '')
            
            for category, patterns in known_markers.items():
                for pattern in patterns:
                    if pattern.upper() in symbol.upper():
                        tissue_markers.append((symbol, category))
                        break
        
        if tissue_markers:
            found_markers[tissue] = tissue_markers
    
    return found_markers


def main():
    """Main function."""
    script_dir = Path(__file__).parent
    ml_dir = script_dir.parent / "model_outputs"
    output_dir = script_dir / "overlap_analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Gene Symbol Mapping for Top Features")
    print("=" * 60)
    
    # Load genes
    print("\n1. Loading top 50 genes per tissue...")
    tissue_genes = load_top_genes(ml_dir, top_n=50)
    total_genes = sum(len(g) for g in tissue_genes.values())
    unique_genes = list(set(g for genes in tissue_genes.values() for g in genes))
    print(f"   Loaded {total_genes} gene IDs ({len(unique_genes)} unique)")
    
    # Query BioMart
    print("\n2. Querying Ensembl BioMart for gene annotations...")
    print("   This may take a moment...")
    gene_info = query_biomart(unique_genes)
    print(f"   Retrieved annotations for {len(gene_info)} genes")
    
    # If BioMart failed, create basic info
    if not gene_info:
        print("   BioMart query failed, using gene IDs as symbols")
        gene_info = {g: {'symbol': g, 'description': '', 'biotype': ''} for g in unique_genes}
    
    # Create report
    print("\n3. Creating gene symbol report...")
    report = create_gene_symbol_report(tissue_genes, gene_info, output_dir)
    
    # Identify known markers
    print("\n4. Identifying known developmental markers...")
    known_markers = identify_known_developmental_markers(gene_info, tissue_genes)
    
    if known_markers:
        print("\n   Found known developmental markers:")
        for tissue, markers in known_markers.items():
            print(f"\n   {tissue}:")
            for symbol, category in markers:
                print(f"     - {symbol} ({category})")
    else:
        print("   No exact matches to pre-defined developmental marker patterns")
        print("   (This is common with pig Ensembl IDs - manual review recommended)")
    
    # Save marker analysis
    with open(output_dir / "known_markers_found.json", 'w') as f:
        json.dump(known_markers, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    
    # Print list of all unique genes for manual lookup
    print("\n\nTop genes per tissue (Ensembl IDs for manual lookup):")
    print("-" * 60)
    for tissue, genes in tissue_genes.items():
        print(f"\n{tissue} (top 10):")
        for g in genes[:10]:
            info = gene_info.get(g, {})
            symbol = info.get('symbol', 'N/A')
            print(f"  {g} -> {symbol}")


if __name__ == "__main__":
    main()
