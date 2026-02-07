#!/usr/bin/env python3
"""
Feature Stability Analysis — thin wrapper around seed_robustness.

Kept for backward compatibility with run_all_analyses.py imports.
The main implementation lives in seed_robustness.py.
"""

from machine_learning.analysis.seed_robustness import run_seed_robustness


def run_stability_analysis(tissues=None, seeds=None, n_features=50, output_dir=None):
    """Run feature stability analysis (delegates to seed_robustness)."""
    return run_seed_robustness(
        tissues=tissues,
        seeds=seeds,
        n_features=n_features,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    from machine_learning.analysis.seed_robustness import main
    main()
