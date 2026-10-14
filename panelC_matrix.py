"""
Panel C - ProtDBench generative benchmark = the ZERO-LABEL-NOISE control.
7 generators x 2 target sets x every published success definition present in the files.

Two questions:
  (1) Which success definitions exist, and for WHICH generators?  (schema coverage)
  (2) Does "which designer wins" depend on the definition and the aggregation protocol?
      Every success flag is provably a deterministic function of the design's own predicted
      structure (shallow-tree accuracy = 1.000), so label noise = 0 by construction.

Outputs -> out/panelC_matrix.csv, panelC_rank_ci.csv, panelC_agreement_*.csv,
           panelC_top1.csv, panelC_defs_agreement.csv, panelC_determinism.csv,
           panelC_coverage.csv, panelC_boot.npz, work/panelC_designs.parquet
"""
import os
import pathlib
import numpy as np, pandas as pd, itertools, glob, os, time
from sklearn.tree import DecisionTreeClassifier

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = str(ROOT)
OUT = f"{TASK}/out"
PB = os.environ.get("PROTDBENCH_ROOT", "")
if not PB:
    raise SystemExit("[panelC] set PROTDBENCH_ROOT to the protdbench_meta "
                     "directory -- see DATA_SOURCES.md.")
BASE = f"{PB}/data/generative_benchmark"
rng = np.random.default_rng(20261014)
B = 300
TARGETSETS = ["10_targets", "cao5_targets"]
# Harmonised interface-confidence score.  Every generative-benchmark file scores its designs
# with the *same* Protenix interface-pTM predictor, but the column is named `ptx_mini_iptm`
# in every file except BindCraft/10_targets, where that same predictor appears as `ptx_iptm`
# (that one file additionally carries a newer `ptx_large_*` run).  We coalesce to one
# canonical column so enrichment is comparable across generators and target sets.
# `i_pTM` exists in every file but is 100% NaN -> unusable.
SCORE_CANDS = ["ptx_mini_iptm", "ptx_iptm"]
SCORE_COL = "score_iptm"

def sf(v):
    """success flag -> float ndarray, NaN preserved (some designs are unlabelled)."""
    return pd.to_numeric(v, errors="coerce").values.astype(float)

# ---------------- load ----------------
frames, cov, det = [], [], []
for ts in TARGETSETS:
    for f in sorted(glob.glob(f"{BASE}/{ts}/*.csv.gz")):
        gen = os.path.basename(f).split("_full")[0]
        x = pd.read_csv(f)
        for c in x.columns:
            if x[c].dtype == object and str(x[c].iloc[0]).startswith("["):
                x[c] = x[c].astype(str).str.strip("[]").astype(float)
        defs = [c for c in x.columns if c.endswith("_success")]
        num = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])
               and not c.endswith("_success") and c != "seq_idx"]
        for d in defs:
            yv0 = sf(x[d]); fin = np.isfinite(yv0)
            if fin.sum() < 10: continue
            xs, yv = x[num].fillna(0)[fin], yv0[fin].astype(int)
            cov.append(dict(targetset=ts, generator=gen, definition=d,
                            available=True, n=len(yv), base_rate=yv.mean()))
            t = DecisionTreeClassifier(max_depth=4, min_samples_leaf=1, random_state=0)
            t.fit(xs, yv)
            det.append(dict(targetset=ts, generator=gen, definition=d, n=len(yv),
                            base_rate=yv.mean(), tree_acc=t.score(xs, yv)))
        sc = next((s for s in SCORE_CANDS if s in x.columns
                   and pd.to_numeric(x[s], errors="coerce").notna().mean() > .9), None)
        assert sc is not None, f"no populated interface score in {f}"
        keep = list(dict.fromkeys(["Target", "name", sc] + defs + [c for c in ["pLDDT", "i_pTM",
                "i_pAE", "pAE", "ptx_iptm_binder", "pLDDT_MONOMER", "Rg", "alpha", "beta",
                "loop", "ref_ratio"] if c in x.columns]))
        y = x[keep].copy()
        y[SCORE_COL] = pd.to_numeric(x[sc], errors="coerce").values
        y["generator"] = gen; y["targetset"] = ts; y["score_src"] = sc
        frames.append(y)
D = pd.concat(frames, ignore_index=True)
D.to_parquet(f"{TASK}/work/panelC_designs.parquet", index=False)
CV = pd.DataFrame(cov); CV.to_csv(f"{OUT}/panelC_coverage.csv", index=False)
DT = pd.DataFrame(det); DT.to_csv(f"{OUT}/panelC_determinism.csv", index=False)
D.groupby(["targetset", "generator"]).agg(score_src=("score_src", "first"),
                                          n=("score_iptm", "size")).reset_index()\
 .to_csv(f"{OUT}/panelC_score_source.csv", index=False)

DEFS = sorted(DT.definition.unique())
gens = sorted(D.generator.unique())
print(f"designs={len(D):,}  generators={len(gens)}  definitions={len(DEFS)}")
print(f"determinism: min tree accuracy over all {len(DT)} (file,definition) pairs = {DT.tree_acc.min():.6f}")
print("\n=== availability of each success definition per generator ===")
AV = CV.pivot_table(index="generator", columns="definition", values="available",
                    aggfunc="max").fillna(False).astype(int)
print(AV.to_string())
print("\nDefinitions available for ALL 7 generators:",
      [d for d in DEFS if AV[d].sum() == len(gens)])
print("Definitions available for exactly 1 generator:",
      [d for d in DEFS if AV[d].sum() == 1])
print("\nraw score column source per file:")
print(D.groupby(["targetset", "generator"])["score_src"].first().to_string())
assert D[SCORE_COL].notna().mean() > .99, "harmonised interface score has gaps"
print(f"harmonised `{SCORE_COL}` populated for {D[SCORE_COL].notna().mean():.4%} of designs")

# ---------------- protocols ----------------
AGG = ["SR_pooled", "SR_macro", "SR_median", "EF10_macro"]
PROTOS = [f"{a}[{d}]" for d in DEFS for a in AGG]
print(f"\nprotocols: {len(PROTOS)}  ({len(DEFS)} definitions x {len(AGG)} aggregations)")

def _nm(v):
    a = np.asarray(v, dtype=float); a = a[np.isfinite(a)]
    return a.mean() if len(a) else np.nan

def proto_values(frame, draw=None):
    """frame = one generator's designs inside one target set.
    draw = None -> all rows; else {target: positional index array into frame}.
    Targets absent from `draw` were not sampled in that replicate -> contribute nothing."""
    tg = frame["Target"].values
    pos = ({t: np.where(tg == t)[0] for t in np.unique(tg)} if draw is None
           else {t: v for t, v in draw.items() if len(v)})
    have = [d for d in DEFS if d in frame.columns]
    Y = {d: sf(frame[d]) for d in have}
    S = pd.to_numeric(frame[SCORE_COL], errors="coerce").values
    out = {f"{a}[{d}]": [] for d in have for a in AGG}
    med = {d: [] for d in have}
    for ix in pos.values():
        for d in have:
            yv0, ss = Y[d][ix], S[ix]
            fin = np.isfinite(yv0)
            if not fin.any(): continue
            yv, sv = yv0[fin], ss[fin]
            ok = np.isfinite(sv)
            out[f"SR_macro[{d}]"].append(yv.mean()); med[d].append(yv.mean())
            if ok.sum() >= 10 and yv[ok].sum() > 0:
                k = max(1, int(round(.10 * ok.sum())))
                o = np.argsort(-sv[ok])[:k]
                out[f"EF10_macro[{d}]"].append(yv[ok][o].mean() / yv[ok].mean())
    allp = np.concatenate(list(pos.values())) if pos else np.array([], dtype=int)
    for d in have:
        yv = Y[d][allp]; yv = yv[np.isfinite(yv)]
        out[f"SR_pooled[{d}]"] = [yv.mean() if len(yv) else np.nan]
        out[f"SR_median[{d}]"] = [np.median(med[d]) if len(med[d]) else np.nan]
    return {p: _nm(v) for p, v in out.items()}

# pre-slice each (targetset, generator) frame once
GF, TGPOS = {}, {}
for ts in TARGETSETS:
    for gen in gens:
        g = D[(D.targetset == ts) & (D.generator == gen)].reset_index(drop=True)
        GF[(ts, gen)] = g
        tg = g["Target"].values
        TGPOS[(ts, gen)] = {t: np.where(tg == t)[0] for t in np.unique(tg)}

t0 = time.time()
rows = []
for ts in TARGETSETS:
    for gen in gens:
        r = {"targetset": ts, "generator": gen}; r.update(proto_values(GF[(ts, gen)])); rows.append(r)
M = pd.DataFrame(rows).reindex(columns=["targetset", "generator"] + PROTOS)
M.to_csv(f"{OUT}/panelC_matrix.csv", index=False)
print(f"\npoint estimates done ({time.time()-t0:.0f}s)")

# ---------------- two-level bootstrap ----------------
t0 = time.time()
boot = {f"{ts}|{p}": np.full((len(gens), B), np.nan) for ts in TARGETSETS for p in PROTOS}
for ts in TARGETSETS:
    for b in range(B):
        for gi, gen in enumerate(gens):
            P = TGPOS[(ts, gen)]; keys = list(P)
            draw = {}
            for j in rng.integers(0, len(keys), len(keys)):
                loc = P[keys[j]]
                draw.setdefault(keys[j], []).append(loc[rng.integers(0, len(loc), len(loc))])
            vals = proto_values(GF[(ts, gen)], draw={k: np.concatenate(v) for k, v in draw.items()})
            for p in PROTOS: boot[f"{ts}|{p}"][gi, b] = vals.get(p, np.nan)
        if (b + 1) % 100 == 0: print(f"  {ts} boot {b+1}/{B}  {time.time()-t0:.0f}s")
np.savez_compressed(f"{OUT}/panelC_boot.npz", generators=np.array(gens, dtype=object),
                    protos=np.array(PROTOS, dtype=object),
                    targetsets=np.array(TARGETSETS, dtype=object), **boot)

ci = []
for ts in TARGETSETS:
    for gi, gen in enumerate(gens):
        est = M[(M.targetset == ts) & (M.generator == gen)].iloc[0]
        for p in PROTOS:
            v = boot[f"{ts}|{p}"][gi]; v = v[np.isfinite(v)]
            if len(v) < 30 or not np.isfinite(est[p]):
                ci.append(dict(targetset=ts, generator=gen, protocol=p, est=est[p], lo=np.nan, hi=np.nan, width=np.nan))
            else:
                lo, hi = np.percentile(v, [2.5, 97.5])
                ci.append(dict(targetset=ts, generator=gen, protocol=p, est=est[p], lo=lo, hi=hi, width=hi - lo))
pd.DataFrame(ci).to_csv(f"{OUT}/panelC_rank_ci.csv", index=False)
print(f"bootstrap done ({time.time()-t0:.0f}s)")

# ---------------- generator ranking, restricted to COMPARABLE protocols ----------------
# A protocol is comparable only if it is finite for >= MINCOV generators.  bindcraft_success /
# ptx_large_success exist in exactly ONE cell (BindCraft / 10_targets) and are therefore NOT
# leaderboard-comparable; they are analysed separately as a within-generator definition axis.
MINCOV = 6
cov_rows, T1 = [], {}
for ts in TARGETSETS:
    sub = M[M.targetset == ts].set_index("generator")[PROTOS]
    nfin = sub.notna().sum()
    CMP = [p for p in PROTOS if nfin[p] >= MINCOV]
    for p in PROTOS:
        cov_rows.append(dict(targetset=ts, protocol=p, n_generators=int(nfin[p]),
                             comparable=bool(nfin[p] >= MINCOV)))
    subc = sub[CMP]
    AG = subc.rank(ascending=False).corr(method="spearman")
    # invalidate pairs whose generator overlap is too small to carry a Spearman
    OV = pd.DataFrame(0, index=CMP, columns=CMP)
    for a, b in itertools.product(CMP, CMP):
        OV.loc[a, b] = int((subc[a].notna() & subc[b].notna()).sum())
    AG = AG.where(OV >= MINCOV)
    AG.to_csv(f"{OUT}/panelC_agreement_{ts}.csv")
    OV.to_csv(f"{OUT}/panelC_overlap_{ts}.csv")
    T1[ts] = {p: (subc[p].idxmax() if subc[p].notna().any() else None) for p in CMP}
    pairs = [AG.loc[a, b] for a, b in itertools.combinations(CMP, 2) if np.isfinite(AG.loc[a, b])]
    ws = sorted(set(v for v in T1[ts].values() if v))
    print(f"\n=== {ts}: {len(CMP)}/{len(PROTOS)} comparable protocols, "
          f"{len(pairs)}/{len(list(itertools.combinations(CMP,2)))} valid pairs")
    print(f"    distinct winners = {len(ws)}  ->  {ws}")
    print(f"    pairwise Spearman: min={min(pairs):.3f}  median={np.median(pairs):.3f}")
pd.DataFrame(cov_rows).to_csv(f"{OUT}/panelC_protocol_coverage.csv", index=False)
pd.DataFrame(T1).to_csv(f"{OUT}/panelC_top1.csv")

print("\n=== winner by definition (SR_pooled), both target sets ===")
w = pd.DataFrame({f"{ts}|{d}": {g: M[(M.targetset == ts) & (M.generator == g)][f"SR_pooled[{d}]"].values[0]
                                for g in gens} for ts in TARGETSETS for d in DEFS})
print((100 * w).round(2).to_string())

# ---------------- design-level agreement between definitions (co-present pairs only) ----
rows = []
for ts in TARGETSETS:
    for gen in gens:
        s = D[(D.targetset == ts) & (D.generator == gen)]
        have = [d for d in DEFS if d in s.columns]
        for a, b in itertools.combinations(have, 2):
            va, vb = sf(s[a]), sf(s[b]); fin = np.isfinite(va) & np.isfinite(vb)
            if fin.sum() == 0: continue          # never co-present -> drop, do not score as 0
            A, Bv = va[fin] > .5, vb[fin] > .5
            rows.append(dict(targetset=ts, generator=gen, def_a=a, def_b=b, n=int(fin.sum()),
                             jaccard=(A & Bv).sum() / max((A | Bv).sum(), 1),
                             a_only=int((A & ~Bv).sum()), b_only=int((Bv & ~A).sum()),
                             both=int((A & Bv).sum())))
DA = pd.DataFrame(rows); DA.to_csv(f"{OUT}/panelC_defs_agreement.csv", index=False)
print("\n=== design-level Jaccard between success definitions (co-present pairs only) ===")
print(DA.groupby(["def_a", "def_b"]).agg(n=("n", "sum"), jaccard=("jaccard", "mean"),
                                         a_only=("a_only", "sum"), b_only=("b_only", "sum")).round(3).to_string())

# ---------------- within-generator definition axis: BindCraft / 10_targets ----------
# BindCraft is the only cell carrying 4 definitions at once -> the definition axis can be
# tested design-by-design with the generator held completely fixed.
bc = D[(D.generator == "BindCraft") & (D.targetset == "10_targets")]
bcdefs = [d for d in DEFS if d in bc.columns]
if len(bcdefs) >= 2:
    xt = pd.DataFrame({d: sf(bc[d]) for d in bcdefs}).fillna(-1).astype(int)
    cross = xt.groupby(bcdefs).size().reset_index(name="n").sort_values("n", ascending=False)
    cross.to_csv(f"{OUT}/panelC_bindcraft_definition_crosstab.csv", index=False)
    print(f"\n=== BindCraft / 10_targets: {len(bc):,} designs x {len(bcdefs)} definitions ===")
    print("    per-definition success rate (%):",
          {d: round(100 * float((sf(bc[d]) > .5).mean()), 2) for d in bcdefs})
    print("    distinct verdict patterns:", len(cross))
    print(cross.head(8).to_string(index=False))
    per = {}
    for d in bcdefs:
        v = (sf(bc[d]) > .5)
        for t in np.unique(bc["Target"].values):
            m = bc["Target"].values == t
            per.setdefault(t, {})[d] = 100 * v[m].mean()
    PT = pd.DataFrame(per).T
    PT.to_csv(f"{OUT}/panelC_bindcraft_defaxis_bytarget.csv")
    print("\n    per-target success rate (%) by definition:")
    print(PT.round(2).to_string())
    rk = PT.rank(ascending=False)
    print("\n    per-target difficulty rank correlation (Spearman) between definitions:")
    print(rk.corr(method="spearman").round(3).to_string())
print("\nDONE")
