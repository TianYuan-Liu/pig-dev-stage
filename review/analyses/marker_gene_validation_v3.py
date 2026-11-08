#!/usr/bin/env python3
"""
R3.3 (v3): Marker-gene concordance under a refined, conserved, bulk-detectable
marker panel, evaluated against BOTH the original day-cut staging and a
literature-grounded revised staging.

WHAT CHANGED FROM v2 (and why)
------------------------------
Two independent, citation-verified literature reviews (see review notes) showed:

 1. Marker definition must respect what bulk RNA-seq can actually see. The
    transitions that have SHARP, CONSERVED, BULK-DETECTABLE mRNA markers are the
    EARLY ones (perinatal liver AFP/ALB; the embryonic->adult myosin switch in
    muscle; intestinal weaning enzymes).  PUBERTY has no sharp bulk marker (it is
    driven by pulsatile hypothalamic-pituitary signalling in rare cells; the only
    bulk readout is a sex-specific gonadal ramp), and AGEING has no robust sharp
    conserved bulk marker (CDKN2A is low-abundance/noisy and needs aged animals,
    which PigGTEx lacks).  The v2 panel's failure at T3/T4 was therefore intrinsic
    to the biology+assay, not a marker-selection error.

 2. The original day-cuts (28/90/165/540) only partly match pig biology:
      - 28 d  = commercial weaning (EU Directive 2008/120/EC) -- a husbandry cut,
                not a developmental transition; the pig is still suckling.
      - 90 d  = onset of prepubertal ovarian follicular activation in gilts (Graves 2019,
                PND ~75-95); a sex-specific gonadal landmark, not a body-wide transcriptional one.
      - 165 d = genuine median female puberty (Patterson 2010; Graves 2019). OK.
      - 540 d = genuine skeletal/physiological maturity (Reiland 1978). OK.
      - The single largest molecular transition -- BIRTH/perinatal -- is buried
        inside "Infant 0-28" with no boundary, despite being well sampled
        (e.g. 184 muscle samples at 0-3 d).

REVISED STAGING (literature + data grounded)
  Neonatal 0-3 | Suckling 4-28 | Juvenile 29-90 | Pre-pubertal 91-165 |
  Post-pubertal 166-540 | Adult >540   (boundaries B1..B5 below)
  - B1 (perinatal ~3 d): NEW -- captures the largest transition; literature
    neonatal window (0-3 d) + dense data.
  - B2 (weaning ~28 d): kept, relabelled (suckling->juvenile).
  - B3 (~90 d): kept (juvenile->pre-pubertal); a 115-d alternative (ovarian
    follicular activation, Graves 2019) is noted in the text.
  - B4 (puberty ~165 d) and B5 (maturity ~540 d): kept -- both literature-anchored,
    but flagged as PHYSIOLOGICAL anchors that bulk markers cannot validate.

Refs (PMIDs/DOIs all re-verified against NCBI/CrossRef on 2026-05-29):
  AFP Spear2006 PMID17041810 | ALB Suzuki2020 PMID33297405 |
  MYH3/8/2 Fazarinc2017 PMID27345286 (+Lefaucheur2004 doi10.2527/2004.8271931x) |
  LCT Hampson1986 PMID3085180 | SI Marion2005 PMID16205506 |
  IGF1 Burrin1997 doi10.2527/1997.75102739x |
  INSL3/AMH Sansone2019 doi10.1111/andr.12597 |
  CDKN2A Krishnamurthy2004 PMID15520862 |
  staging: weaning EU-Dir-2008/120/EC; puberty Patterson2010 doi10.2527/jas.2008-1756,
  Graves2019 doi10.1093/tas/txz173; maturity Reiland1978 PMID233594.
"""
from __future__ import annotations
import gzip, json, logging, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "pigGTEx"
META = ROOT / "data" / "PigGTEx_v0.MetaTable.csv"
SYMMAP = ROOT / "data" / "cardoso_moreira_2019" / "_pig_id_to_symbol.json"
RES = ROOT / "review" / "analyses" / "results"
MIN_PER_STAGE, ALPHA = 5, 0.05
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("v3")

# ----- staging systems -----
STAGE_SYSTEMS = {
    "original": {
        "labels": ["Infant", "Early childhood", "Pre-pubertal", "Post-pubertal", "Adult"],
        "cuts": [28, 90, 165, 540],
    },
    "revised": {
        "labels": ["Neonatal", "Suckling", "Juvenile", "Pre-pubertal", "Post-pubertal", "Adult"],
        "cuts": [3, 28, 90, 165, 540],
    },
}

def age_days(s):
    if pd.isna(s): return None
    s = str(s).lower().strip()
    m = re.match(r"(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years)", s)
    if not m: return None
    v = float(m.group(1)); u = m.group(2)
    return v * (1 if "day" in u else 7 if "week" in u else 30 if "month" in u else 365)

def to_stage(d, system):
    if d is None or pd.isna(d): return None
    labels, cuts = system["labels"], system["cuts"]
    for i, c in enumerate(cuts):
        if d <= c: return labels[i]
    return labels[-1]

# ----- refined marker panel -----
# boundary keys are 1-based transition indices into the REVISED staging:
#   B1 Neonatal->Suckling (perinatal ~3d) ; B2 Suckling->Juvenile (weaning ~28d) ;
#   B3 Juvenile->Pre-pubertal (~90d) ; B4 Pre->Post-pubertal (puberty ~165d) ;
#   B5 Post-pubertal->Adult (maturity ~540d)
@dataclass
class M:
    sym: str; boundary: int; direction: str; tissue: str; ref: str
    bulk: str  # 'sharp' | 'partial' | 'ramp' | 'weak'
    species: str; note: str = ""; ens: Optional[str] = None

PANEL: List[M] = [
    # B1 perinatal (markable)
    M("AFP",  1, "down", "Liver",  "PMID17041810", "sharp", "mammal", "perinatal hepatic silencing"),
    M("ALB",  1, "up",   "Liver",  "PMID33297405", "sharp", "mammal", "mature-hepatocyte, reciprocal to AFP"),
    M("IGF2", 1, "down", "Liver",  "PMID18448610", "sharp", "mammal", "imprinted fetal growth factor; postnatal decline (Lui 2008 network)"),
    M("PEG3", 1, "down", "Liver",  "PMID18448610", "sharp", "mammal", "imprinted gene; down in the postnatal-growth network (Lui 2008)"),
    M("MYH3", 1, "down", "Muscle", "PMID27345286", "sharp", "pig",    "embryonic MyHC; falls in first weeks (perinatal switch)"),
    M("MYH8", 1, "down", "Muscle", "PMID27345286", "sharp", "pig",    "perinatal/neonatal MyHC; falls in first weeks (perinatal switch)"),
    M("MYL4", 1, "down", "Muscle", "DOI10.1152/physrev.00031.2010", "sharp", "mammal", "fetal/atrial myosin light chain; perinatal myofibrillar switch"),
    M("MYH2", 1, "up",   "Muscle", "PMID27345286", "sharp", "pig",    "adult fast 2A MyHC; surges in first weeks (perinatal switch)"),
    M("TNNT3",1, "up",   "Muscle", "DOI10.1152/physrev.00031.2010", "sharp", "mammal", "fast skeletal troponin T; adult myofibrillar program"),
    # B2 weaning (markable) -- intestinal lactase decline.
    # NOTE: sucrase/maltase (SI, MGAM) were excluded: Marion 2005 (PMID16205506) shows their
    # weaning rise is POST-TRANSCRIPTIONAL (only maltase activity tracked mRNA), so they are
    # not valid bulk-mRNA markers. LCT (lactase) decline is transcriptional and is retained.
    M("LCT",  2, "down", "Ileum",  "PMID3085180",  "sharp", "pig",    "lactase; transcriptional decline after weaning"),
    # B3 juvenile growth (ramp, no sharp marker)
    M("IGF1", 3, "up",   "Liver",  "DOI10.2527/1997.75102739x", "ramp", "pig", "GH/IGF1 growth axis -- a ramp, not a step"),
    # B4 puberty (NOT bulk-markable; gonadal ramp only)
    M("INSL3",4, "up",   "Testis", "DOI10.1111/andr.12597", "ramp", "mammal", "Leydig maturation; sex-specific gonadal ramp"),
    M("AMH",  4, "down", "Testis", "DOI10.1111/andr.12597", "ramp", "mammal", "Sertoli maturation; falls at puberty"),
    # B5 adult/ageing (NOT bulk-markable; needs aged animals)
    M("CDKN2A",5,"up",   "Liver",  "PMID15520862", "weak", "mammal", "p16 senescence; low-abundance, needs aged animals"),
]

def load_meta(system):
    md = pd.read_csv(META); md.columns = [c.strip().replace(" ", "_") for c in md.columns]
    md = md.rename(columns={"BioSample": "Sample_ID", "Tissue_class": "Tissue"}).set_index("Sample_ID")
    md["d"] = md["Age"].apply(age_days)
    md["Stage"] = md["d"].apply(lambda x: to_stage(x, system))
    return md

def load_tpm(t):
    p = DATA / f"{t}.expr_tpm.txt.gz"
    if not p.exists(): return None
    with gzip.open(p, "rt") as f:
        return np.log2(pd.read_csv(f, sep="\t", index_col=0) + 1.0)

def sym2ens():
    out = {}
    for e, s in json.load(open(SYMMAP)).items():
        if s: out.setdefault(s.upper(), []).append(e)
    return out

def analyse(expr_row, stages, labels, a, b):
    """Per-stage means; test the a->b boundary (direction + Tukey p)."""
    df = pd.concat([expr_row.rename("e"), stages.rename("s")], axis=1).dropna()
    df = df[df["s"].isin(labels)]
    means = df.groupby("s", observed=True)["e"].mean().reindex(labels)
    ns = df.groupby("s", observed=True)["e"].count().reindex(labels).fillna(0)
    well = set(ns[ns >= MIN_PER_STAGE].index)
    evaluable = a in well and b in well
    delta = (means[b] - means[a]) if evaluable else np.nan
    p = np.nan
    if evaluable:
        dfw = df[df["s"].isin(well)]
        if dfw["s"].nunique() >= 2:
            try:
                tk = pairwise_tukeyhsd(dfw["e"].values, dfw["s"].astype(str).values)
                tt = pd.DataFrame(tk._results_table.data[1:], columns=tk._results_table.data[0])
                for _, r in tt.iterrows():
                    if {r["group1"], r["group2"]} == {a, b}: p = float(r["p-adj"])
            except Exception: pass
    # global largest adjacent change among well-covered stages
    diffs = {}
    for i in range(len(labels) - 1):
        x, y = labels[i], labels[i + 1]
        if x in well and y in well: diffs[i + 1] = means[y] - means[x]
    maxb = max(diffs, key=lambda k: abs(diffs[k])) if diffs else None
    return dict(means=means, ns=ns, evaluable=evaluable, delta=delta, p=p,
                obs_dir=("up" if delta > 0 else "down") if evaluable else "NA",
                maxb=maxb)

def run(system_name):
    system = STAGE_SYSTEMS[system_name]; labels = system["labels"]
    md = load_meta(system); s2e = sym2ens()
    tcache = {}
    recs, traj = [], {}
    for mk in PANEL:
        ens = [mk.ens] if mk.ens else s2e.get(mk.sym.upper(), [])
        if mk.tissue not in tcache:
            e = load_tpm(mk.tissue)
            if e is not None:
                common = [c for c in e.columns if c in md.index]
                tcache[mk.tissue] = (e[common], md.loc[common, "Stage"])
            else: tcache[mk.tissue] = (None, None)
        expr, stages = tcache[mk.tissue]
        base = dict(gene=mk.sym, tissue=mk.tissue, boundary=f"B{mk.boundary}",
                    direction=mk.direction, bulk=mk.bulk, species=mk.species,
                    ref=mk.ref, note=mk.note)
        if expr is None: recs.append({**base, "status": "no_tissue"}); continue
        ens = [x for x in ens if x in expr.index]
        if not ens: recs.append({**base, "status": "gene_absent"}); continue
        row = expr.loc[ens].mean(0)
        # boundary a->b in this staging
        bi = mk.boundary  # 1-based; boundary bi separates labels[bi-1] and labels[bi]
        if bi >= len(labels): recs.append({**base, "status": "boundary_absent"}); continue
        a, b = labels[bi - 1], labels[bi]
        r = analyse(row, stages, labels, a, b)
        dir_match = r["evaluable"] and r["obs_dir"] == mk.direction
        sig = r["evaluable"] and pd.notna(r["p"]) and r["p"] < ALPHA
        recs.append({**base, "stage_before": a, "stage_after": b,
                     "n_before": int(r["ns"][a]), "n_after": int(r["ns"][b]),
                     "evaluable": bool(r["evaluable"]),
                     "log2fc": float(r["delta"]) if pd.notna(r["delta"]) else np.nan,
                     "obs_dir": r["obs_dir"],
                     "tukey_p": float(r["p"]) if pd.notna(r["p"]) else np.nan,
                     "dir_match": bool(dir_match), "sig": bool(sig),
                     "corroborated": bool(dir_match and sig),
                     "obs_max_boundary": f"B{r['maxb']}" if r["maxb"] else "NA",
                     "status": "OK" if r["evaluable"] else "not_evaluable"})
        traj[(mk.tissue, mk.sym)] = dict(means=r["means"], ns=r["ns"], mk=mk,
                                         dir_match=dir_match, sig=sig, maxb=r["maxb"],
                                         labels=labels, boundary=mk.boundary)
    df = pd.DataFrame(recs)
    return df, traj, labels

def compute_velocity(tissues, binsize=24):
    """Genome-wide transcriptome change-rate vs age: RMS per-gene z-shift between
    consecutive equal-count age bins (placed at the geometric-mean age of the pair).
    Identifies where the transcriptome moves fastest = where boundaries belong."""
    md = load_meta(STAGE_SYSTEMS["revised"])
    out = {}
    for t in tissues:
        e = load_tpm(t)
        if e is None: continue
        sub = md[(md.index.isin(e.columns)) & md["d"].notna()]
        common = sub.index.tolist()
        if len(common) < 3 * 10: continue
        e = e[common]; ages = sub["d"]
        order = ages.sort_values().index.tolist()
        expr = e[(e > 1).mean(axis=1) > 0.2]
        Z = expr.sub(expr.mean(1), axis=0).div(expr.std(1).replace(0, np.nan), axis=0).dropna()
        bins = [order[i:i + binsize] for i in range(0, len(order), binsize)]
        bins = [b for b in bins if len(b) >= 10]
        cents = [Z[b].mean(1) for b in bins]
        mids = [float(ages[b].median()) for b in bins]
        ax, ay = [], []
        for k in range(len(bins) - 1):
            ax.append(float(np.sqrt(max(mids[k], 0.5) * max(mids[k + 1], 0.5))))
            ay.append(float(np.sqrt(((cents[k + 1] - cents[k]) ** 2).mean())))
        if ax: out[t] = dict(age=ax, rate=ay)
    return out

def make_figure(results, vel):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle, FancyBboxPatch
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 8,
                         "axes.labelsize": 7, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
                         "legend.fontsize": 5.6, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    FIGP = ROOT / "paper" / "figures" / "output"
    tissue_color = {"Muscle": "#4C72B0", "Liver": "#C44E52", "Ileum": "#DD8452",
                    "Lung": "#55A868", "Testis": "#8172B3", "Ovary": "#937860"}
    BULK = {"sharp": "#2A9D8F", "partial": "#9CCB3B", "ramp": "#E9A33A", "weak": "#D1495B"}
    df_r, traj_r, labels_r = results["revised"]

    fig = plt.figure(figsize=(7.6, 12.3))
    gs = gridspec.GridSpec(5, 1, figure=fig, height_ratios=[2.2, 5.0, 0.95, 1.6, 1.9],
                           hspace=0.5, left=0.13, right=0.94, top=0.965, bottom=0.045)

    # ===== Panel a: genome-wide developmental velocity vs age =====
    axV = fig.add_subplot(gs[0])
    for t in ["Muscle", "Liver"]:
        if t in vel:
            d = vel[t]
            axV.plot(d["age"], d["rate"], "-o", ms=2.8, lw=1.3, color=tissue_color[t],
                     label=t, alpha=0.92)
    axV.set_xscale("log"); axV.set_xlim(0.4, 750)
    # headroom for a two-row milestone-label band that sits clear of the data (data tops ~1.18)
    ylo, yhi = axV.get_ylim()
    axV.set_ylim(ylo, yhi + 0.95 * (yhi - ylo))
    # milestone guide lines: perinatal solid (key boundary), day-cuts dashed; neutral colours (no red highlight)
    GREY, DARK = "#9a9a9a", "#444444"
    mil = {3:   ("perinatal", DARK, "-",  1.4),
           28:  ("weaning",   GREY, "--", 0.8),
           90:  ("90 d",      GREY, "--", 0.8),
           165: ("puberty",   GREY, "--", 0.8),
           540: ("maturity",  GREY, "--", 0.8)}
    for c, (lab, col, ls, lw) in mil.items():
        axV.axvline(c, color=col, ls=ls, lw=lw, zorder=0)
    # labels: horizontal, on a two-row stagger so the close 90/165/540 cuts never collide; a
    # white halo masks the guide line so each label reads as a tag pinned to its own line.
    trans = axV.get_xaxis_transform()          # x in data units, y in axes fraction
    ROW = {3: 0.70, 90: 0.70, 540: 0.70,       # lower row
           28: 0.90, 165: 0.90}                # upper row
    for c, (lab, col, ls, lw) in mil.items():
        axV.text(c, ROW[c], lab, transform=trans, ha="center", va="center",
                 fontsize=5.8, color="#555", fontweight="normal", zorder=5,
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92))
    axV.set_xlabel("age (days, log scale)")
    axV.set_ylabel("transcriptome change rate\n(RMS z-shift, age bins)")
    axV.legend(loc="upper left", bbox_to_anchor=(0.01, 0.72), frameon=False,
               fontsize=5.8, handlelength=1.3)
    for sp in ("top", "right"): axV.spines[sp].set_visible(False)
    axV.set_title("Transcriptome change concentrates in early postnatal life",
                  fontsize=7.2, loc="left")
    axV.text(-0.10, 1.10, "a", transform=axV.transAxes, fontsize=11, fontweight="bold")

    # ===== Panel b: refined-marker heatmap under the REVISED staging =====
    axH = fig.add_subplot(gs[1])
    SH = ["Neonatal\n0–3 d", "Suckling\n4–28 d", "Juvenile\n29–90 d", "Pre-pub.\n91–165 d",
          "Post-pub.\n166–540 d", "Adult\n>540 d"]
    BCOL = {1: 0.5, 2: 1.5, 3: 2.5, 4: 3.5, 5: 4.5}
    rows = []
    for mk in PANEL:
        key = (mk.tissue, mk.sym)
        if key not in traj_r: continue
        v = traj_r[key]
        means = v["means"].reindex(labels_r).values.astype(float)
        ns = v["ns"].reindex(labels_r).values.astype(float)
        means = np.where(ns >= 3, means, np.nan)
        sd = np.nanstd(means)
        z = (means - np.nanmean(means)) / sd if sd > 0 else means * 0
        rec = df_r[(df_r.gene == mk.sym) & (df_r.tissue == mk.tissue)].iloc[0]
        st = "corr" if rec["corroborated"] else ("dir" if rec["dir_match"] else "miss")
        rows.append(dict(mk=mk, z=z, status=st, maxb=v["maxb"]))
    nrow = len(rows)
    Z = np.ma.masked_invalid(np.vstack([r["z"] for r in rows]))
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#e8e8e8")
    axH.imshow(Z, aspect="auto", cmap=cmap, vmin=-2, vmax=2, origin="upper",
               interpolation="nearest", extent=[-0.5, 5.5, nrow - 0.5, -0.5])
    axH.set_xticks(range(6)); axH.set_xticklabels(SH, fontsize=6.0)
    axH.set_yticks(range(nrow)); axH.set_yticklabels([r["mk"].sym for r in rows], fontsize=6.4)
    for tick, r in zip(axH.get_yticklabels(), rows):
        tick.set_color(tissue_color.get(r["mk"].tissue, "#333")); tick.set_fontweight("bold")
    axH.set_xticks(np.arange(-0.5, 6, 1), minor=True); axH.set_yticks(np.arange(-0.5, nrow, 1), minor=True)
    axH.grid(which="minor", color="white", lw=0.8); axH.tick_params(which="both", length=0)
    # boundary blocks + labels
    BLAB = {1: "B1 perinatal", 2: "B2 weaning", 3: "B3 growth", 4: "B4 puberty", 5: "B5 adult"}
    i = 0
    while i < nrow:
        bnd = rows[i]["mk"].boundary; j = i
        while j < nrow and rows[j]["mk"].boundary == bnd: j += 1
        a, b = i - 0.5, j - 0.5
        if i > 0: axH.plot([-0.5, 5.5], [a, a], color="#3a3a3a", lw=1.0, clip_on=False)
        axH.plot([BCOL[bnd]] * 2, [a, b], color="#111", lw=2.6, solid_capstyle="butt")
        markable = bnd <= 3
        axH.text(-1.75, (a + b) / 2, BLAB[bnd], ha="center", va="center", fontsize=6.0,
                 fontweight="bold", color="#1a7f6b" if markable else "#b06a00")
        i = j
    # right columns: exp dir, corroboration glyph, observed-max diamond, bulk flag
    glyph = {"corr": "●", "dir": "◐", "miss": "✕"}; GC = {"corr": "#2A9D8F", "dir": "#E9A33A", "miss": "#D1495B"}
    xdir, xglyph, xbulk = 5.95, 6.4, 6.95
    for k, r in enumerate(rows):
        if r["maxb"] in BCOL:
            axH.plot(BCOL[r["maxb"]], k, "D", ms=3.3, mfc="white", mec="#111", mew=0.9, zorder=6, clip_on=False)
        axH.text(xdir, k, "↑" if r["mk"].direction == "up" else "↓", ha="center", va="center", fontsize=8, color="#333")
        axH.text(xglyph, k, glyph[r["status"]], ha="center", va="center", fontsize=8.5,
                 color=GC[r["status"]], fontweight="bold")
        axH.text(xbulk, k, r["mk"].bulk, ha="left", va="center", fontsize=5.4,
                 color=BULK[r["mk"].bulk], fontweight="bold")
    axH.text(xdir, -1.25, "exp.\ndir.", ha="center", va="center", fontsize=5.2, fontweight="bold", color="#333")
    axH.text(xglyph, -1.25, "at\nbnd.", ha="center", va="center", fontsize=5.2, fontweight="bold", color="#333")
    axH.text(xbulk, -1.25, "bulk-\nmarkable?", ha="left", va="center", fontsize=5.2, fontweight="bold", color="#333")
    axH.set_xlim(-2.4, 8.0); axH.set_ylim(nrow - 0.5, -1.8)
    for sp in axH.spines.values(): sp.set_visible(False)
    axH.set_title("Refined conserved markers, correct tissue, tested at each boundary "
                  "(revised staging)", fontsize=7.6, loc="left")
    axH.text(-0.10, 1.035, "b", transform=axH.transAxes, fontsize=11, fontweight="bold")
    # colourbar + legends
    # ---- legend strip DIRECTLY UNDER panel b: four labelled sections, boxed ----
    axL = fig.add_subplot(gs[2]); axL.axis("off")
    axL.add_patch(FancyBboxPatch((0.012, 0.08), 0.976, 0.82, boxstyle="round,pad=0.006",
                  transform=axL.transAxes, fill=False, ec="#b5b5b5", lw=0.8, clip_on=False))
    for xs in (0.215, 0.45, 0.69):                         # section separators
        axL.plot([xs, xs], [0.14, 0.74], transform=axL.transAxes, color="#dddddd", lw=0.7, clip_on=False)
    for cx, h in zip([0.113, 0.332, 0.570, 0.840],         # aligned section headers
                     ["heat-map colour", "boundary marks", "at boundary", "tissue (gene labels)"]):
        axL.text(cx, 0.83, h, transform=axL.transAxes, ha="center", va="top",
                 fontsize=6.1, fontweight="bold", color="#333")
    cbax = axL.inset_axes([0.045, 0.40, 0.135, 0.11])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-2, 2), cmap="RdBu_r"),
                      cax=cbax, orientation="horizontal")
    cb.set_label("row z-score", fontsize=5.7, labelpad=2)
    cb.ax.xaxis.set_label_position("top")          # label ABOVE the bar -> stays inside the box
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.tick_params(labelsize=5.3, length=2); cb.set_ticks([-2, 0, 2])
    leg_bound = [Line2D([0], [0], color="#111", lw=2.4, label="predicted boundary"),
                 Line2D([0], [0], marker="D", ls="", mfc="white", mec="#111", mew=0.9, ms=5.5, label="strongest observed shift")]
    leg_atb = [Line2D([0], [0], marker="o", ls="", mfc=GC["corr"], mec=GC["corr"], ms=6, label="corroborated"),
               Line2D([0], [0], marker="o", ls="", mfc="none", mec=GC["dir"], mew=1.4, ms=6, label="direction only"),
               Line2D([0], [0], marker="x", ls="", mec=GC["miss"], mew=1.8, ms=6, label="mismatch")]
    leg_tis = [Line2D([0], [0], marker="s", ls="", mfc=tissue_color[t], mec=tissue_color[t], ms=6, label=t)
               for t in ["Muscle", "Liver", "Ileum", "Testis"]]
    lA = axL.legend(handles=leg_bound, loc="center left", bbox_to_anchor=(0.235, 0.40), ncol=1,
                    frameon=False, fontsize=5.9, handletextpad=0.5, labelspacing=0.6)
    axL.add_artist(lA)
    lB = axL.legend(handles=leg_atb, loc="center left", bbox_to_anchor=(0.475, 0.38), ncol=1,
                    frameon=False, fontsize=5.9, handletextpad=0.5, labelspacing=0.5)
    axL.add_artist(lB)
    axL.legend(handles=leg_tis, loc="center left", bbox_to_anchor=(0.715, 0.40), ncol=2,
               frameon=False, fontsize=5.9, handletextpad=0.4, labelspacing=0.45, columnspacing=0.9)

    # ===== Panel c: per-boundary verdict, revised staging =====
    axS = fig.add_subplot(gs[3])
    bsum = {}
    ok = df_r[df_r["status"] == "OK"]
    for bi in range(1, 6):
        sub = ok[ok["boundary"] == f"B{bi}"]
        bsum[bi] = (int(sub["corroborated"].sum()), len(sub))
    order = [1, 2, 3, 4, 5]; yv = list(range(5))[::-1]
    NAMES = {1: "B1 perinatal (~3 d)", 2: "B2 weaning (~28 d)", 3: "B3 growth (~90 d)",
             4: "B4 puberty (~165 d)", 5: "B5 adult (~540 d)"}
    for y, bi in zip(yv, order):
        corr, n = bsum[bi]
        markable = bi <= 3
        axS.barh(y, max(n, 1), color="#f0f0f0", edgecolor="#cfcfcf", height=0.62, lw=0.6)
        axS.barh(y, corr, color="#2A9D8F" if markable else "#bdbdbd", height=0.62)
        tag = f"{corr}/{n} corroborated" if n else "not evaluable"
        axS.text(max(n, 1) + 0.12, y, tag, va="center", fontsize=6.2,
                 color="#1a7f6b" if (markable and corr) else "#666", fontweight="bold")
        axS.text(-0.15, y, NAMES[bi], va="center", ha="right", fontsize=6.3, fontweight="bold",
                 color="#1a7f6b" if markable else "#b06a00")
    axS.axhspan(-0.5, 1.5, color="#fff4e6", zorder=-2)   # B4/B5 band (literature-anchored)
    axS.axhspan(1.5, 4.5, color="#e9f6f2", zorder=-2)     # B1-B3 band (transcriptomically validated)
    axS.text(5.6, 2.5, "transcriptomically\nvalidated", fontsize=6.2, color="#1a7f6b", style="italic", ha="left", va="center")
    axS.text(5.6, 0.5, "literature-anchored only\n(no bulk mRNA marker exists)", fontsize=6.2,
             color="#b06a00", style="italic", ha="left", va="center")
    axS.set_xlim(0, 11.5); axS.set_ylim(-0.5, 4.5); axS.set_yticks([]); axS.set_xticks([0, 2, 4, 6, 8])
    axS.set_xlabel("markers corroborated", fontsize=6.5); axS.tick_params(axis="x", labelsize=6)
    for sp in ("top", "right", "left"): axS.spines[sp].set_visible(False)
    axS.set_title("Per-boundary verdict — early boundaries validated; puberty/adult are "
                  "physiological anchors, not transcriptomic", fontsize=7.0, loc="left", pad=4)
    axS.text(-0.10, 1.12, "c", transform=axS.transAxes, fontsize=11, fontweight="bold")

    # ===== Panel d: marker-free leave-one-cohort-out AUC vs candidate day-cut =====
    # Source: review/analyses/results/boundary_auc_loco.json (paper/figures/provenance/boundary_auc_loco.py).
    # Held-out AUC of a genome-wide young-vs-old contrast fit on training BioProjects
    # (leave-one-cohort-out), swept over candidate young/old day-cuts. Muscle (the only
    # cross-study-identifiable tissue). Cuts where either class has <20 samples are dropped
    # (older class too sparse beyond ~165 d), so the curve is not extrapolated into noise.
    axA = fig.add_subplot(gs[4])
    auc_path = RES / "boundary_auc_loco.json"
    if auc_path.exists():
        ad = json.load(open(auc_path))
        if "Muscle" in ad:
            rows = [r for r in ad["Muscle"]["sweep"]
                    if r["auc"] is not None and r["n_young"] >= 20 and r["n_old"] >= 20]
            xs = [r["cut_d"] for r in rows]; ys = [r["auc"] for r in rows]
            axA.plot(xs, ys, "-o", ms=3.0, lw=1.4, color=tissue_color["Muscle"],
                     label="Muscle", alpha=0.95)
        axA.axhline(0.5, color="#9a9a9a", ls=":", lw=0.9)
        axA.text(0.6, 0.515, "chance", fontsize=5.4, color="#777", va="bottom")
        for c, (lab, col, ls, lw) in mil.items():
            axA.axvline(c, color=col, ls=ls, lw=lw, zorder=0)
        axA.set_xscale("log"); axA.set_xlim(0.4, 750); axA.set_ylim(0.45, 1.0)
        axA.set_xlabel("young/old day-cut (days, log scale)")
        axA.set_ylabel("held-out AUC\n(leave-one-cohort-out)")
        axA.legend(loc="upper right", frameon=False, fontsize=5.8, handlelength=1.3)
        for sp in ("top", "right"): axA.spines[sp].set_visible(False)
        axA.set_title("Marker-free young-vs-old separability (muscle) peaks at the "
                      "perinatal cut, broad not sharp", fontsize=7.0, loc="left")
    else:
        axA.axis("off")
        axA.text(0.5, 0.5, "boundary_auc_loco.json not found\n(run paper/figures/provenance/boundary_auc_loco.py)",
                 ha="center", va="center", fontsize=7, color="#b06a00", transform=axA.transAxes)
    axA.text(-0.10, 1.10, "d", transform=axA.transAxes, fontsize=11, fontweight="bold")

    fig.savefig(FIGP / "pdf" / "figS6_marker_genes_v3.pdf", dpi=300, bbox_inches="tight")
    if (FIGP / "png").is_dir():   # png exports are optional (removed in manuscript-only cleanup)
        fig.savefig(FIGP / "png" / "figS6_marker_genes_v3.png", dpi=300, bbox_inches="tight")
    log.info("Wrote figS6_marker_genes_v3.pdf")
    plt.close(fig)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    summary, results = {}, {}
    for sysname in ["original", "revised"]:
        df, traj, labels = run(sysname)
        df.to_csv(RES / f"marker_v3_{sysname}.csv", index=False)
        results[sysname] = (df, traj, labels)
        ok = df[df["status"] == "OK"]
        nb = len(STAGE_SYSTEMS[sysname]["cuts"])
        by_b = {f"B{bi}": dict(n=len(ok[ok["boundary"] == f"B{bi}"]),
                               corroborated=int(ok[ok["boundary"] == f"B{bi}"]["corroborated"].sum()),
                               dir_match=int(ok[ok["boundary"] == f"B{bi}"]["dir_match"].sum()))
                for bi in range(1, nb + 1)}
        summary[sysname] = dict(n_markers=len(df), n_evaluable=len(ok),
                                n_corroborated=int(ok["corroborated"].sum()), by_boundary=by_b)
        log.info("%s: corroborated %d/%d  by-boundary %s", sysname,
                 summary[sysname]["n_corroborated"], len(ok),
                 {k: f"{v['corroborated']}/{v['n']}" for k, v in by_b.items()})
    json.dump(summary, open(RES / "marker_v3_summary.json", "w"), indent=2)
    vel = compute_velocity(["Muscle", "Liver", "Lung", "Testis"])
    json.dump(vel, open(RES / "marker_v3_velocity.json", "w"), indent=2)
    make_figure(results, vel)
    return summary

if __name__ == "__main__":
    main()
