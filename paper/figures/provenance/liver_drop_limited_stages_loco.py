#!/usr/bin/env python3
"""Liver leave-one-BioProject-out (grouped) BA when single-cohort-dominated
stages are dropped. Reuses the canonical within-pig pipeline verbatim
(OrdinalPLS, StratifiedGroupKFold, 5 folds x 5 seeds). Real data only.
"""
import sys, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path("/Users/tianyuan/Desktop/github_dev/pig-dev-stage")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # provenance dir: sibling modules
from within_pig_6stage_figdata import stage6, eval_regime, summarize, ORDER6
from machine_learning.data_processing.data_loader import DataLoader

dl = DataLoader(data_dir=ROOT / "data" / "pigGTEx",
                metadata_path=ROOT / "data" / "PigGTEx_v0.MetaTable.csv")
dl.load_metadata()
expr, meta = dl.load_expression("Liver")
X = np.log2(expr.T.values + 1.0)
age = pd.to_numeric(meta["Age_Days"] if "Age_Days" in meta else meta["Age"], errors="coerce")
bp = meta["BioProject"].astype(str).reset_index(drop=True)
y6 = age.map(stage6).reset_index(drop=True)
ka = age.notna().values
X = X[ka]; y6 = y6[ka].reset_index(drop=True); bp = bp[ka].reset_index(drop=True)


def run(kept):
    kept = [s for s in ORDER6 if s in kept]            # keep ordinal order
    code = {s: i for i, s in enumerate(kept)}
    mask = y6.isin(kept).values
    Xk = X[mask]; yk = y6[mask].values; gk = bp[mask].values
    yi = np.array([code[v] for v in yk])
    reg = eval_regime(Xk, yi, gk, kept, "grouped")
    n = int(mask.sum())
    if reg is None:
        return dict(kept=kept, n=n, K=len(kept), note="not identifiable")
    s = summarize(reg, kept, with_recall=True)
    return dict(kept=kept, n=n, K=len(kept), chance=round(1/len(kept), 3),
                BA_mean=round(s["BA_mean"], 3), BA_sd=round(s["BA_sd"], 3),
                rho=s["rho_mean"], per_seed_ba=s["per_seed_ba"],
                recall={k: v["mean"] for k, v in s["per_class_recall"].items()})


SCEN = {
    "all 6 stages (baseline)": ORDER6,
    "drop Neonatal (94% 1 cohort)": ["Suckling", "Juvenile", "Pre-pubertal", "Post-pubertal", "Adult"],
    "drop Neonatal + Adult": ["Suckling", "Juvenile", "Pre-pubertal", "Post-pubertal"],
    "keep multi-cohort only (Juv/PrePub/PostPub)": ["Juvenile", "Pre-pubertal", "Post-pubertal"],
}
out = {name: run(kept) for name, kept in SCEN.items()}
# largest-cohort share per stage (the a priori, design-based drop criterion)
import collections
df = pd.DataFrame({"stage": y6, "bp": bp})
share = {}
for s in ORDER6:
    sub = df[df.stage == s]
    if len(sub):
        share[s] = round(100 * sub.bp.value_counts().iloc[0] / len(sub), 1)
out["_largest_cohort_share_pct"] = share
out["_criterion"] = "drop stages whose largest single BioProject contributes >50% of the stage's samples (Neonatal 94%, Adult 62%); the other four are <=40%"
res_path = ROOT / "review" / "analyses" / "results" / "liver_loco_stage_restricted.json"
res_path.parent.mkdir(parents=True, exist_ok=True)
res_path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print("\nWrote", res_path)
