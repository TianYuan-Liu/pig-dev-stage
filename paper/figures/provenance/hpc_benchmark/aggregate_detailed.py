#!/usr/bin/env python3
"""Detailed 6-stage benchmark table: BA (both CV regimes) + full timing/memory.

Timing source per model:
  - 18 monolith models: SLURM sacct of the monolith array (2852773) + multiclass
    rerun (2853049): total wall, total CPU-time, peak RSS.
  - 3 fold-split models (gradient_boosting/lda/soft_vote): the merged JSON's
    `timing` block (total_fit_cpu_s, single/Muscle fit, all at 1 thread) plus peak
    RSS taken as the max across that model's split tasks in job 2856395.

Outputs results_6stage/detailed.md, resources.md, detailed.csv.
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import numpy as np

WS = Path("/storage/gge/Tian/pig_ml_benchmark")
RES = WS / "results_6stage"
TISSUES = ["Muscle", "Brain", "Liver", "Blood", "Lung"]
MODELS = (WS / "code" / "MODELS.txt").read_text().split()
MONO_JOBS = "2852773,2853049"
SPLIT_JOB = "2856395"
STUCK = ["gradient_boosting", "lda", "soft_vote"]   # tasklist order: 250 tasks each


def parse_hms(s):
    s = (s or "").strip()
    if not s or s in ("INVALID",):
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1); days = int(d)
    p = [float(x) for x in s.split(":")]
    while len(p) < 3:
        p.insert(0, 0.0)
    return days * 86400 + p[0] * 3600 + p[1] * 60 + p[2]


def mem_mb(s):
    s = (s or "").strip().upper()
    num = "".join(c for c in s if c.isdigit() or c == ".")
    if not num:
        return None
    v = float(num)
    return v * 1024 if "G" in s else (v / 1024 if "K" in s else v)


def sacct_mono():
    out = subprocess.run(["sacct", "-j", MONO_JOBS, "-P", "--units=M",
                          "--format=JobID,State,Elapsed,TotalCPU,AllocCPUS,ReqMem,MaxRSS"],
                         capture_output=True, text=True).stdout
    main, batch = {}, {}
    for line in out.splitlines()[1:]:
        f = line.split("|")
        if len(f) < 7:
            continue
        jid, state, el, tcpu, alloc, req, rss = f[:7]
        if jid.endswith(".batch"):
            batch[jid[:-6]] = mem_mb(rss)
        elif "." not in jid:
            main[jid] = {"state": state, "elapsed_s": parse_hms(el), "cpu_s": parse_hms(tcpu),
                         "alloc": int(alloc) if alloc.isdigit() else None, "reqmem_mb": mem_mb(req)}
    for b in main:
        main[b]["maxrss_mb"] = batch.get(b)
    by = {}
    for jid, row in main.items():
        if jid.startswith("2852773_"):
            i = int(jid.split("_")[1]); m = MODELS[i] if i < len(MODELS) else None
        elif jid.startswith("2853049_"):
            m = "multiclass_lgb"
        else:
            m = None
        if m and not (m == "multiclass_lgb" and jid.startswith("2852773_")):
            by[m] = row
    return by


def split_peak_rss():
    """max MaxRSS (MB) per stuck model across its 250 tasks in SPLIT_JOB."""
    out = subprocess.run(["sacct", "-j", SPLIT_JOB, "-P", "--units=M",
                          "--format=JobID,MaxRSS"], capture_output=True, text=True).stdout
    peak = {m: 0.0 for m in STUCK}
    for line in out.splitlines()[1:]:
        f = line.split("|")
        if len(f) < 2 or not f[0].endswith(".batch"):
            continue
        base = f[0][:-6]
        if "_" not in base:
            continue
        try:
            idx = int(base.split("_")[1])
        except ValueError:
            continue
        m = STUCK[idx // 250] if idx // 250 < len(STUCK) else None
        r = mem_mb(f[1])
        if m and r:
            peak[m] = max(peak[m], r)
    return {m: (v or None) for m, v in peak.items()}


def main():
    mono = sacct_mono()
    srss = split_peak_rss()
    rows = []
    for m in MODELS:
        jf = RES / f"{m}.json"
        d = json.loads(jf.read_text()) if jf.exists() else {"tissues": {}}
        if d.get("did_not_complete"):
            rows.append({"model": m, "state": "DNF", "tissues_done": 0, "K_muscle": None,
                         "g_muscle": None, "s_muscle": None, "g_macro": None, "g_macro_sd": None,
                         "s_macro": None, "fit_med_s": None, "fit_muscle_s": None, "wall_s": None,
                         "cpu_s": None, "cores": None, "reqmem_mb": None, "maxrss_mb": None,
                         "dnf_reason": d.get("reason", "")})
            continue
        tis = d.get("tissues", {})
        mus = tis.get("Muscle", {})
        g_mus = (mus.get("grouped") or {}).get("BA_mean")
        s_mus = (mus.get("stratified") or {}).get("BA_mean")
        nK = mus.get("n_classes")
        gs = [(tis.get(t, {}).get("grouped") or {}).get("BA_mean") for t in TISSUES]
        ss = [(tis.get(t, {}).get("stratified") or {}).get("BA_mean") for t in TISSUES]
        gs = [x for x in gs if x is not None]; ss = [x for x in ss if x is not None]
        split = d.get("merged_from_split")
        if split:                                   # fold-split model: timing from JSON
            tm = d.get("timing", {})
            row = {"model": m, "state": "split(1thr)", "tissues_done": len(gs),
                   "K_muscle": nK, "g_muscle": g_mus, "s_muscle": s_mus,
                   "g_macro": round(np.mean(gs), 3) if gs else None,
                   "g_macro_sd": round(np.std(gs), 3) if gs else None,
                   "s_macro": round(np.mean(ss), 3) if ss else None,
                   "fit_med_s": tm.get("single_fit_s_median"),
                   "fit_muscle_s": tm.get("muscle_fit_s_median"),
                   "wall_s": None, "cpu_s": tm.get("total_fit_cpu_s"),
                   "cores": 1, "reqmem_mb": 6144, "maxrss_mb": srss.get(m)}
        else:                                       # monolith: timing from sacct
            fits = [(tis.get(t, {}).get("grouped") or {}).get("fit_time_s") for t in TISSUES]
            fits = [x for x in fits if x is not None]
            mf = (mus.get("grouped") or {}).get("fit_time_s")
            sa = mono.get(m, {})
            row = {"model": m, "state": sa.get("state", "?"), "tissues_done": len(gs),
                   "K_muscle": nK, "g_muscle": g_mus, "s_muscle": s_mus,
                   "g_macro": round(np.mean(gs), 3) if gs else None,
                   "g_macro_sd": round(np.std(gs), 3) if gs else None,
                   "s_macro": round(np.mean(ss), 3) if ss else None,
                   "fit_med_s": round(np.median(fits), 2) if fits else None,
                   "fit_muscle_s": round(mf, 1) if mf else None,
                   "wall_s": sa.get("elapsed_s"), "cpu_s": sa.get("cpu_s"),
                   "cores": sa.get("alloc"), "reqmem_mb": sa.get("reqmem_mb"),
                   "maxrss_mb": sa.get("maxrss_mb")}
        rows.append(row)
    rows.sort(key=lambda r: (r["g_muscle"] is None, -(r["g_muscle"] or -1)))

    def c(x, f="{:.3f}"):
        return "--" if x is None else f.format(x)
    def hm(s):
        if s is None:
            return "--"
        h = int(s // 3600); m2 = int((s % 3600) // 60)
        return f"{h}h{m2:02d}m" if h else f"{m2}m{int(s%60):02d}s"

    # detailed table
    L = ["| Model | State | K | Grouped Muscle BA | Grouped macro ±SD | Strat Muscle | Strat macro | Inflation | Single-fit s | Muscle fit s | CPU-time | Peak RSS (GB) |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        infl = (r["s_muscle"] - r["g_muscle"]) if (r["s_muscle"] is not None and r["g_muscle"] is not None) else None
        L.append("| {mo} | {st} | {K} | {gm} | {gma} ± {gsd} | {sm} | {sma} | {inf} | {fm} | {fmus} | {cpu} | {rss} |".format(
            mo=r["model"], st=r["state"], K=r["K_muscle"] or "--",
            gm=c(r["g_muscle"]), gma=c(r["g_macro"]), gsd=c(r["g_macro_sd"]),
            sm=c(r["s_muscle"]), sma=c(r["s_macro"]), inf=c(infl, "{:+.3f}"),
            fm=c(r["fit_med_s"], "{:.2f}"), fmus=c(r["fit_muscle_s"], "{:.1f}"),
            cpu=hm(r["cpu_s"]), rss=c(r["maxrss_mb"]/1024.0 if r["maxrss_mb"] else None, "{:.2f}")))
    (RES / "detailed.md").write_text("\n".join(L) + "\n")

    # resources table
    def pct(a, b):
        return (100.0 * a / b) if (a and b) else None
    R = ["| Model | State | Cores | CPU-time | Peak RSS (GB) | Single-fit s | Muscle fit s |",
         "|---|---|--:|--:|--:|--:|--:|"]
    for r in rows:
        R.append("| {mo} | {st} | {co} | {cpu} | {rss} | {fm} | {fmus} |".format(
            mo=r["model"], st=r["state"], co=r["cores"] or "--", cpu=hm(r["cpu_s"]),
            rss=c(r["maxrss_mb"]/1024.0 if r["maxrss_mb"] else None, "{:.2f}"),
            fm=c(r["fit_med_s"], "{:.2f}"), fmus=c(r["fit_muscle_s"], "{:.1f}")))
    (RES / "resources.md").write_text("\n".join(R) + "\n")

    import csv
    with open(RES / "detailed.csv", "w", newline="") as f:
        cols = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print("\n".join(L))
    print(f"\nWrote {RES/'detailed.md'}, {RES/'resources.md'}, {RES/'detailed.csv'}")


if __name__ == "__main__":
    main()
