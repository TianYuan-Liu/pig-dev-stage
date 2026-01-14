#!/usr/bin/env python3
"""
Deprecated script.

This file previously simulated developmental expression and enrichment results.
Use the R workflows in slides/R/run_go_enrichment.R or the visualization scripts
that consume real enrichment outputs instead.
"""

import sys


def main() -> None:
    print(
        "ERROR: scripts/enrichment_analysis_revised.py is deprecated because it "
        "generated simulated enrichment data. Use the R enrichment workflow "
        "instead."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
