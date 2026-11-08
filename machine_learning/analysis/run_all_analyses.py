#!/usr/bin/env python3
"""
Master Analysis Script for Feature Selection Validation

This script runs all validation analyses to determine whether low cross-tissue
feature overlap is due to overfitting or biological tissue-specificity.

Analyses performed:
1. Feature Stability - Do same genes get selected across random seeds?
2. Expression Correlation - Do top genes correlate with developmental stage?

The combination of these analyses provides strong evidence for or against
the biological interpretation of tissue-specific feature selection.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.analysis.feature_stability import run_stability_analysis
from machine_learning.analysis.expression_correlation import run_correlation_analysis


def generate_final_report(
    stability_results: Dict,
    correlation_results: Dict,
    output_dir: Path
) -> str:
    """Generate comprehensive analysis report."""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("COMPREHENSIVE FEATURE SELECTION VALIDATION REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)

    # ============== SUMMARY TABLE ==============
    report_lines.append("\n## SUMMARY TABLE\n")

    summary_data = []
    tissues = set(list(stability_results.keys()) +
                  list(correlation_results.keys()))

    for tissue in sorted(tissues):
        row = {'Tissue': tissue}

        # Stability metrics
        if tissue in stability_results:
            sr = stability_results[tissue]
            row['Stability (Jaccard)'] = f"{sr['jaccard_similarity']['mean']:.2f}"
            row['Stable Genes'] = sr['n_stable_genes']
        else:
            row['Stability (Jaccard)'] = 'N/A'
            row['Stable Genes'] = 'N/A'

        # Correlation metrics
        if tissue in correlation_results:
            cr = correlation_results[tissue]
            row['Correlated %'] = f"{cr['pct_significant']:.0f}%"
            row['Mean |r|'] = f"{cr['mean_abs_correlation']:.2f}"
        else:
            row['Correlated %'] = 'N/A'
            row['Mean |r|'] = 'N/A'

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    report_lines.append(summary_df.to_string(index=False))

    # ============== ANALYSIS 1: FEATURE STABILITY ==============
    report_lines.append("\n\n" + "=" * 80)
    report_lines.append("ANALYSIS 1: FEATURE STABILITY ACROSS RANDOM SEEDS")
    report_lines.append("=" * 80)

    if stability_results:
        mean_jaccard = np.mean([r['jaccard_similarity']['mean'] for r in stability_results.values()])
        mean_stable_ratio = np.mean([r['n_stable_genes']/r['n_features'] for r in stability_results.values()])

        report_lines.append(f"\nOverall Results:")
        report_lines.append(f"  Average Jaccard similarity: {mean_jaccard:.3f}")
        report_lines.append(f"  Average stable gene ratio: {mean_stable_ratio:.1%}")

        if mean_jaccard >= 0.5:
            report_lines.append(f"\n  VERDICT: PASS - Features are stable across random seeds")
            report_lines.append(f"  This suggests real biological signal, NOT overfitting to noise")
            stability_verdict = "PASS"
        elif mean_jaccard >= 0.3:
            report_lines.append(f"\n  VERDICT: MODERATE - Some feature stability")
            stability_verdict = "MODERATE"
        else:
            report_lines.append(f"\n  VERDICT: FAIL - Features are unstable")
            report_lines.append(f"  This suggests potential overfitting")
            stability_verdict = "FAIL"

        report_lines.append("\n  Per-tissue details:")
        for tissue, result in stability_results.items():
            js = result['jaccard_similarity']
            report_lines.append(
                f"    {tissue:15}: Jaccard={js['mean']:.3f} (+/-{js['std']:.3f}), "
                f"Stable genes={result['n_stable_genes']}/{result['n_features']}"
            )
    else:
        stability_verdict = "NOT_RUN"
        report_lines.append("\n  Analysis not run or no results available")

    # ============== ANALYSIS 2: EXPRESSION CORRELATION ==============
    report_lines.append("\n\n" + "=" * 80)
    report_lines.append("ANALYSIS 2: EXPRESSION-STAGE CORRELATION")
    report_lines.append("=" * 80)

    if correlation_results:
        mean_pct_significant = np.mean([r['pct_significant'] for r in correlation_results.values()])
        mean_abs_corr = np.mean([r['mean_abs_correlation'] for r in correlation_results.values()])

        report_lines.append(f"\nOverall Results:")
        report_lines.append(f"  Average % significant correlations: {mean_pct_significant:.1f}%")
        report_lines.append(f"  Average |correlation|: {mean_abs_corr:.3f}")

        if mean_pct_significant >= 70:
            report_lines.append(f"\n  VERDICT: PASS - Top genes show strong stage correlations")
            report_lines.append(f"  This confirms real developmental expression changes")
            correlation_verdict = "PASS"
        elif mean_pct_significant >= 50:
            report_lines.append(f"\n  VERDICT: MODERATE - Most genes correlate with stage")
            correlation_verdict = "MODERATE"
        else:
            report_lines.append(f"\n  VERDICT: FAIL - Many genes don't correlate with stage")
            report_lines.append(f"  This raises overfitting concerns")
            correlation_verdict = "FAIL"

        report_lines.append("\n  Per-tissue details:")
        for tissue, result in correlation_results.items():
            report_lines.append(
                f"    {tissue:15}: {result['pct_significant']:.1f}% significant, "
                f"mean |r|={result['mean_abs_correlation']:.3f}"
            )
    else:
        correlation_verdict = "NOT_RUN"
        report_lines.append("\n  Analysis not run or no results available")

    # ============== FINAL CONCLUSION ==============
    report_lines.append("\n\n" + "=" * 80)
    report_lines.append("FINAL CONCLUSION")
    report_lines.append("=" * 80)

    verdicts = [stability_verdict, correlation_verdict]
    pass_count = verdicts.count("PASS")
    moderate_count = verdicts.count("MODERATE")

    if pass_count >= 2:
        report_lines.append("\n  OVERALL VERDICT: FEATURES ARE CAPTURING REAL BIOLOGY")
        report_lines.append("")
        report_lines.append("  Evidence:")
        if stability_verdict == "PASS":
            report_lines.append("    - Features are STABLE across random seeds")
        if correlation_verdict == "PASS":
            report_lines.append("    - Top genes CORRELATE with developmental stage")

        report_lines.append("")
        report_lines.append("  CONCLUSION: Low cross-tissue feature overlap is a BIOLOGICAL phenomenon,")
        report_lines.append("  NOT a sign of overfitting. Each tissue has genuinely different")
        report_lines.append("  developmental gene expression programs.")

    elif pass_count + moderate_count >= 2:
        report_lines.append("\n  OVERALL VERDICT: MODERATE EVIDENCE FOR BIOLOGICAL SIGNAL")
        report_lines.append("")
        report_lines.append("  The model likely captures real biology, but some features may be noise.")
        report_lines.append("  Consider using more stringent feature selection criteria.")

    else:
        report_lines.append("\n  OVERALL VERDICT: POTENTIAL OVERFITTING CONCERNS")
        report_lines.append("")
        report_lines.append("  The evidence does not strongly support biological interpretation.")
        report_lines.append("  Consider:")
        report_lines.append("    - Using more regularization")
        report_lines.append("    - Reducing number of features")
        report_lines.append("    - Using cross-validation for feature selection")

    report_lines.append("\n" + "=" * 80)

    # Join and save report
    report_text = "\n".join(report_lines)

    report_file = output_dir / "comprehensive_validation_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)

    # Also save as markdown
    md_file = output_dir / "comprehensive_validation_report.md"
    with open(md_file, 'w') as f:
        f.write(report_text.replace("=" * 80, "---"))

    print(f"\nReport saved to: {report_file}")
    print(f"Markdown report: {md_file}")

    return report_text


def run_all_analyses(
    tissues: Optional[List[str]] = None,
    n_features: int = 50,
    stability_seeds: Optional[List[int]] = None,
    skip_stability: bool = False,
    skip_correlation: bool = False,
):
    """Run all validation analyses."""

    if tissues is None:
        tissues = ['Muscle', 'Liver', 'Brain', 'Blood', 'Lung']

    if stability_seeds is None:
        stability_seeds = [42, 123, 456, 789, 1024]

    output_dir = PROJECT_ROOT / "machine_learning/analysis/results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE FEATURE SELECTION VALIDATION")
    print("=" * 80)
    print(f"Tissues: {tissues}")
    print(f"Features: {n_features}")
    print(f"Output: {output_dir}")

    results = {}

    # Analysis 1: Feature Stability
    if not skip_stability:
        print("\n" + "=" * 80)
        print("Running FEATURE STABILITY analysis...")
        print("=" * 80)
        try:
            results['stability'] = run_stability_analysis(
                tissues=tissues,
                seeds=stability_seeds,
                n_features=n_features,
                output_dir=output_dir
            )
        except Exception as e:
            print(f"Stability analysis failed: {e}")
            results['stability'] = {}
    else:
        results['stability'] = {}

    # Analysis 2: Expression Correlation
    if not skip_correlation:
        print("\n" + "=" * 80)
        print("Running EXPRESSION CORRELATION analysis...")
        print("=" * 80)
        try:
            results['correlation'] = run_correlation_analysis(
                tissues=tissues,
                n_genes=n_features,
                output_dir=output_dir
            )
        except Exception as e:
            print(f"Correlation analysis failed: {e}")
            results['correlation'] = {}
    else:
        results['correlation'] = {}

    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("Generating COMPREHENSIVE REPORT...")
    print("=" * 80)

    report = generate_final_report(
        results.get('stability', {}),
        results.get('correlation', {}),
        output_dir
    )

    print("\n" + report)

    # Save all results
    all_results_file = output_dir / "all_analysis_results.json"
    with open(all_results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved to: {all_results_file}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run all feature selection validation analyses")
    parser.add_argument(
        '--tissues', nargs='+',
        default=['Muscle', 'Liver', 'Brain', 'Blood', 'Lung'],
        help='Tissues to analyze'
    )
    parser.add_argument(
        '--n-features', type=int, default=50,
        help='Number of top features to analyze'
    )
    parser.add_argument(
        '--seeds', nargs='+', type=int,
        default=[42, 123, 456, 789, 1024],
        help='Random seeds for stability analysis'
    )
    parser.add_argument(
        '--skip-stability', action='store_true',
        help='Skip stability analysis (slowest)'
    )
    parser.add_argument(
        '--skip-correlation', action='store_true',
        help='Skip correlation analysis'
    )
    args = parser.parse_args()

    run_all_analyses(
        tissues=args.tissues,
        n_features=args.n_features,
        stability_seeds=args.seeds,
        skip_stability=args.skip_stability,
        skip_correlation=args.skip_correlation,
    )


if __name__ == "__main__":
    main()
