#!/usr/bin/env python3
"""Score the joint metric (BA + cross-species r) for a set of pipeline
output JSONs. Used to evaluate each rvalue-loop experiment.

Usage:
  python3 autoresearch/score_rvalue.py --suffix _rvalue_xxx \
      --tissues Muscle Brain
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autoresearch.score_ml_cross_species import cross_species_r_for_pig_genes


def score(suffix: str, tissues, top_k: int = 200) -> dict:
    out = {}
    for T in tissues:
        fp = ROOT / "machine_learning" / "model_outputs" / f"{T}{suffix}_results.json"
        if not fp.exists():
            out[T] = {"status": "MISSING", "BA": None, "r": None}
            continue
        r = json.load(open(fp))
        BA = r["metrics"]["balanced_accuracy"]
        top200 = r["top_genes"][:top_k]
        if T == "Blood":
            xs = {"pearson_r": None, "n_genes_used": 0}
        else:
            xs = cross_species_r_for_pig_genes(top200, T, top_k=top_k)
        out[T] = {
            "BA": float(BA),
            "r": xs.get("pearson_r"),
            "n_used": xs.get("n_genes_used"),
        }
    return out


def fmt(scored: dict) -> str:
    lines = []
    BAs = []
    rs = []
    joints = []
    for T, s in scored.items():
        if s.get("BA") is None:
            lines.append(f"  {T}: MISSING")
            continue
        BA = s["BA"]
        r = s["r"]
        n = s.get("n_used", 0)
        BAs.append(BA)
        if r is not None:
            rs.append(r)
        joint = BA + (r if r is not None else 0.0)
        joints.append(joint)
        r_str = f"{r:+.3f} (n={n})" if r is not None else "n/a"
        lines.append(f"  {T}: BA={BA:.3f}  r={r_str}  joint={joint:.3f}")
    if BAs:
        mean_BA = sum(BAs) / len(BAs)
        mean_r = (sum(rs) / len(rs)) if rs else None
        mean_joint = sum(joints) / len(joints)
        mean_r_str = f"{mean_r:+.3f}" if mean_r is not None else "n/a"
        lines.append(
            f"  mean: BA={mean_BA:.3f}  r={mean_r_str}  joint={mean_joint:.3f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", required=True)
    p.add_argument("--tissues", nargs="+", default=["Muscle", "Brain"])
    p.add_argument("--top-k", dest="top_k", type=int, default=200)
    args = p.parse_args()
    scored = score(args.suffix, args.tissues, top_k=args.top_k)
    print(fmt(scored))
    print("---TSV---")
    BAs = []
    rs = []
    joints = []
    for T in args.tissues:
        s = scored[T]
        BA = s.get("BA")
        r = s.get("r")
        if BA is not None:
            BAs.append(BA)
            if r is not None:
                rs.append(r)
            joints.append(BA + (r if r is not None else 0.0))
        BA_s = f"{BA:.4f}" if BA is not None else "NA"
        r_s = f"{r:.4f}" if r is not None else "NA"
        print(f"{T}\t{BA_s}\t{r_s}")
    if BAs:
        mean_BA = sum(BAs) / len(BAs)
        mean_r = (sum(rs) / len(rs)) if rs else 0.0
        mean_joint = sum(joints) / len(joints)
        print(f"MEAN\t{mean_BA:.4f}\t{mean_r:.4f}\t{mean_joint:.4f}")
