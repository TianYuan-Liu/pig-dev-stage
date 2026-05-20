#!/usr/bin/env python3
"""Driver for the rvalue-loop autoresearch experiments.

Bypasses the choices= validation on --model-type in run_pipeline.py so that
new model names registered in model_factory.py can be used.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from machine_learning.run_pipeline import main


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tissues", nargs="+", required=True)
    p.add_argument("--model-type", dest="model_type", required=True)
    p.add_argument("--n-cv-folds", dest="n_cv_folds", type=int, default=5)
    p.add_argument("--n-trials", dest="n_trials", type=int, default=10)
    p.add_argument("--n-inner-folds", dest="n_inner_folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--log-level", dest="log_level", default="WARNING")
    p.add_argument("--output-suffix", dest="output_suffix", type=str, default="")
    p.add_argument("--all-tissues", dest="all_tissues", action="store_true")
    args = p.parse_args()
    main(args)
