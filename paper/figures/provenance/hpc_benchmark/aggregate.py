#!/usr/bin/env python3
"""Aggregate per-model benchmark JSONs into one comparison table.

Outputs (to results/):
  comparison.csv   one row per model: grouped/stratified macro BA, Muscle BA,
                   median fit time, peak RSS
  comparison.md    human-readable markdown table (sorted by Muscle grouped BA)

Honesty note baked into the output: grouped macro-BA averages over all 5
tissues, 4 of which are study-confounded (near chance); the Muscle column is
the only tissue where grouped CV is a valid deployment estimate.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

DEFAULT_WS = Path("/storage/gge/Tian/pig_ml_benchmark")
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WS / "results"
TISSUES = ["Muscle", "Brain", "Liver", "Blood", "Lung"]

rows = []
for jf in sorted(RES.glob("*.json")):
    d = json.loads(jf.read_text())
    name = d["model"]
    g_ba, s_ba, fts = [], [], []
    muscle_g = muscle_s = None
    for t in TISSUES:
        ti = d["tissues"].get(t, {})
        g = ti.get("grouped", {}) or {}
        s = ti.get("stratified", {}) or {}
        if g.get("BA_mean") is not None:
            g_ba.append(g["BA_mean"])
            if g.get("fit_time_s") is not None:
                fts.append(g["fit_time_s"])
        if s.get("BA_mean") is not None:
            s_ba.append(s["BA_mean"])
        if t == "Muscle":
            muscle_g = g.get("BA_mean")
            muscle_s = s.get("BA_mean")
    rows.append({
        "model": name,
        "muscle_grouped_BA": muscle_g,
        "macro_grouped_BA": round(float(np.mean(g_ba)), 4) if g_ba else None,
        "macro_grouped_SD": round(float(np.std(g_ba)), 4) if g_ba else None,
        "muscle_stratified_BA": muscle_s,
        "macro_stratified_BA": round(float(np.mean(s_ba)), 4) if s_ba else None,
        "median_fit_s": round(float(np.median(fts)), 4) if fts else None,
        "peak_rss_mb": d.get("peak_rss_mb"),
        "n_tissues_grouped": len(g_ba),
    })

# sort by muscle grouped BA (the identifiable deployment estimate), desc
rows.sort(key=lambda r: (r["muscle_grouped_BA"] is None, -(r["muscle_grouped_BA"] or 0)))

# CSV
import csv
cols = ["model", "muscle_grouped_BA", "macro_grouped_BA", "macro_grouped_SD",
        "muscle_stratified_BA", "macro_stratified_BA", "median_fit_s",
        "peak_rss_mb", "n_tissues_grouped"]
with open(RES / "comparison.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

# Markdown
def cell(x):
    return "--" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))

lines = []
lines.append("| Model | Muscle grouped BA | Macro grouped BA (±SD) | Muscle strat. BA | Macro strat. BA | Fit s | Peak MB |")
lines.append("|---|---|---|---|---|---|---|")
for r in rows:
    mg = cell(r["macro_grouped_BA"])
    sd = cell(r["macro_grouped_SD"])
    lines.append(f"| {r['model']} | {cell(r['muscle_grouped_BA'])} | {mg} ± {sd} | "
                 f"{cell(r['muscle_stratified_BA'])} | {cell(r['macro_stratified_BA'])} | "
                 f"{cell(r['median_fit_s'])} | {cell(r['peak_rss_mb'])} |")
(RES / "comparison.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nWrote {RES/'comparison.csv'} and {RES/'comparison.md'}")
