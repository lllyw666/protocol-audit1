"""Panel A figures F1-F6."""
import pathlib
import numpy as np, pandas as pd, itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = str(ROOT)
OUT, FIG = f"{TASK}/out", f"{TASK}/out/figs"
import os; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.linewidth": .7, "figure.dpi": 150,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "axes.titlesize": 9})


# ---------------- vector backend: quantised heatmaps + multi-format export ----------------
def vecmesh(ax, C, cmap=None, vmin=None, vmax=None, aspect="auto"):
    """imshow-equivalent heatmap drawn as vector quads.

    Reproduces the geometry the old `imshow(..., origin='upper')` gave: cell
    (i, j) is centred on data coordinates (j, i), so the existing
    ax.text(j, i, ...) annotations and Rectangle((j-.5, i-.5), 1, 1) overlays
    keep landing on the correct cell.
    """
    C = np.asarray(C, dtype=float)
    ny, nx = C.shape
    qm = ax.pcolormesh(np.arange(nx + 1) - .5, np.arange(ny + 1) - .5, C,
                       cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
                       antialiased=False, edgecolors="none", linewidth=0)
    ax.invert_yaxis()          # restore upper origin (row 0 at the top)
    ax.set_aspect(aspect)
    return qm


def save(fig, stem, formats=("png", "pdf", "svg")):
    """Write the figure in raster (png) and vector (pdf, svg) form."""
    for ext in formats:
        fig.savefig(f"{FIG}/{stem}.{ext}")


def vbar(fig, mappable, **kw):
    """fig.colorbar(), but keep the bar itself vector.

    matplotlib marks a colorbar's `solids` as rasterized, so the bar lands in
    the pdf/svg as an embedded bitmap even when the heatmap body is vector.
    Clearing that flag emits it as quads instead.
    """
    cb = fig.colorbar(mappable, **kw)
    if getattr(cb, "solids", None) is not None:
        cb.solids.set_rasterized(False)
    return cb

M = pd.read_csv(f"{OUT}/panelA_matrix.csv")
CI = pd.read_csv(f"{OUT}/panelA_rank_ci.csv")
AG = pd.read_csv(f"{OUT}/panelA_agreement.csv", index_col=0)
JC = pd.read_csv(f"{OUT}/panelA_top3_jaccard.csv", index_col=0)
IN = pd.read_csv(f"{OUT}/panelA_instability.csv")
HET = pd.read_csv(f"{OUT}/panelA_heterogeneity.csv")
FEAT = os.environ.get("OVERATH_FEATURES", "")
if not FEAT:
    raise SystemExit(
        "[panelA] the Overath per-design feature table is not redistributed with "
        "this repository.\n"
        "         Set OVERATH_FEATURES to overath_rankfeatures.parquet "
        "(see DATA_SOURCES.md).")
PM = pd.read_parquet(FEAT)
y = PM["label"].astype(int).values; tg = PM["target"].values; tnames = np.unique(tg)

PROTOS = list(AG.columns)
rk = M[PROTOS].rank(ascending=False)
order = rk.mean(axis=1).sort_values().index

# ============ F1: main matrix heatmap ============
fig, ax = plt.subplots(figsize=(7.6, 11))
R = rk.loc[order].values
im = vecmesh(ax, R, cmap="RdBu_r", vmin=1, vmax=243, aspect="auto")
ax.set_xticks(range(len(PROTOS))); ax.set_xticklabels(PROTOS, rotation=90, fontsize=7.5)
# 243 rows cannot all be labelled at a legible size; label every 8th row, evenly spread,
# and put ticks only there (243 ticks would render as a solid raster down the axis)
_lab_idx = list(range(0, len(order), 8))
ax.set_yticks(_lab_idx)
ax.set_yticklabels([M.loc[order[i], "method"] for i in _lab_idx], fontsize=6.0)
ax.set_title(f"F1  Method rank across {len(PROTOS)} evaluation protocols\n"
             f"(Overath, {len(M)} methods x {len(tnames)} targets x {int(y.sum())} positives; "
             f"blue = better rank\n"
             f"y labels: every 8th of {len(M)})")
cb = vbar(fig, im, ax=ax, fraction=.03, pad=.01); cb.set_label("rank (1 = best)", fontsize=8)
# separators between protocol families
for s in [3.5, 13.5]:
    ax.axvline(s, color="k", lw=.8)
save(fig, "F1_protocol_matrix"); plt.close(fig)

# ============ F2: agreement (Spearman + top-3 Jaccard) ============
fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6), gridspec_kw={"wspace": 0.42})
for ax, Mx, ttl, cm in [(axes[0], AG.loc[PROTOS, PROTOS], "F2a  Leaderboard Spearman", "viridis"),
                        (axes[1], JC.loc[PROTOS, PROTOS].astype(float), "F2b  top-3 set Jaccard", "magma")]:
    im = vecmesh(ax, Mx.values, cmap=cm, vmin=0 if ax is axes[1] else 0.5, vmax=1, aspect="equal")
    ax.set_xticks(range(len(PROTOS))); ax.set_xticklabels(PROTOS, rotation=90, fontsize=7)
    ax.set_yticks(range(len(PROTOS))); ax.set_yticklabels(PROTOS, fontsize=7)
    ax.set_title(ttl)
    for i, j in itertools.product(range(len(PROTOS)), repeat=2):
        v = Mx.values[i, j]
        ax.text(j, i, f"{v:.1f}"[1:] if v < 1 else "1", ha="center", va="center",
                fontsize=6.0, color="white" if v < (.65 if ax is axes[1] else .8) else "black")
    vbar(fig, im, ax=ax, fraction=.046, pad=.02)
fig.suptitle("Two protocols, two leaderboards: which pairs actually disagree", y=1.0, fontsize=10)
save(fig, "F2_agreement"); plt.close(fig)

# ============ F3: cutoff slopegraph ============
fig, ax = plt.subplots(figsize=(6.4, 6.0))
CUT = ["macro_ef1", "macro_ef5", "macro_ef10", "macro_ef20"]
sel = list(rk.mean(axis=1).sort_values().index[:14])
pal = plt.cm.tab20(np.linspace(0, 1, len(sel)))
for c, i in zip(pal, sel):
    ys = [rk.loc[i, p] for p in CUT]
    ax.plot(range(4), ys, "-o", ms=4, lw=1.2, color=c, alpha=.85)
    ax.text(-0.08, ys[0], M.loc[i, "method"], ha="right", va="center", fontsize=7.0, color=c)
    ax.text(3.08, ys[-1], f"{M.loc[i,'method']} ({ys[-1]:.0f})", ha="left", va="center", fontsize=7.0, color=c)
ax.set_xticks(range(4)); ax.set_xticklabels(CUT)
ax.invert_yaxis(); ax.set_ylabel(f"rank among {len(M)} methods")
ax.set_xlim(-2.4, 5.4)
ax.set_title("F3  Same methods, same data: rank migration along the enrichment cutoff")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "F3_cutoff_slopegraph"); plt.close(fig)

# ============ F4: instability vs target heterogeneity ============
# per method: cross-target dispersion of its macro metric (target heterogeneity it experiences)
disp = []
for i, m in enumerate(M["method"]):
    vals = []
    for j, t in enumerate(tnames):
        ix = np.where(tg == t)[0]
        s = pd.to_numeric(PM[m], errors="coerce").values[ix] * M.loc[i, "sign"]
        yy = y[ix]; ok = np.isfinite(s)
        if ok.sum() < 10 or yy[ok].sum() < 2: continue
        k = max(1, int(round(.10 * ok.sum())))
        o = np.argsort(-s[ok])[:k]
        vals.append(yy[ok][o].mean() / yy[ok].mean())
    disp.append(np.nanstd(vals) / max(np.nanmean(vals), 1e-9) if len(vals) > 3 else np.nan)
D = pd.DataFrame({"method": M["method"], "cross_target_cv": disp}).merge(IN, on="method")
D = D[np.isfinite(D["cross_target_cv"])]
rho, p = spearmanr(D["cross_target_cv"], D["rank_sd"])
fig, ax = plt.subplots(figsize=(5.4, 4.4))
ax.scatter(D["cross_target_cv"], D["rank_sd"], s=14, alpha=.45, color="#3b6")
ax.set_xlabel("cross-target dispersion of EF@10%  (CV over the 15 targets)")
ax.set_ylabel(r"rank instability across the 14 protocols ($\sigma$ of rank)")
ax.set_title(f"F4  Attribution: protocol instability vs target heterogeneity\nSpearman rho={rho:.2f} (p={p:.1e}), n={len(D)} methods")
save(fig, "F4_instability_vs_heterogeneity"); plt.close(fig)

# ============ F5: noise band ============
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6),
                         gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
CIM = CI.pivot(index="method", columns="protocol", values="est")
CIL = CI.pivot(index="method", columns="protocol", values="lo")
CIH = CI.pivot(index="method", columns="protocol", values="hi")
top = M.loc[rk["macro_ef10"].sort_values().index[:20], "method"].tolist()
ys = np.arange(len(top))
est = [CIM.loc[m, "macro_ef10"] for m in top]
lo = [CIL.loc[m, "macro_ef10"] for m in top]
hi = [CIH.loc[m, "macro_ef10"] for m in top]
ax.errorbar(est, ys, xerr=[np.array(est) - lo, np.array(hi) - est], fmt="o", ms=4,
            color="#c33", ecolor="#999", elinewidth=1.2, capsize=2)
r1 = CIM.loc[top[0], "macro_ef10"]; r10 = CIM.loc[top[9], "macro_ef10"]
band = CIH.loc[top[0], "macro_ef10"]
ax.axvspan(lo[0], band, color="orange", alpha=.15)
ax.set_yticks(ys); ax.set_yticklabels(top, fontsize=6.5); ax.invert_yaxis()
ax.set_xlabel("macro EF@10%  (95% two-level bootstrap CI)")
ax.set_title("F5a  Top-20 under EF@10%: how many are statistically separable?")
ax.spines[["top", "right"]].set_visible(False)

# how many methods in the top-40 are *statistically tied* with #1 (paired bootstrap)
# marginal-CI overlap is the wrong test for correlated methods; test the DIFFERENCE instead.
BO = np.load(f"{OUT}/panelA_boot.npz", allow_pickle=True)
NTIE = {}
ax = axes[1]
frac = []
for p in PROTOS:
    bm = BO[p]                                   # (n_methods, B)
    est = M[p].values
    o = np.argsort(-est)
    b1 = bm[o[0]]
    n = 1
    for k in range(1, min(40, len(o))):
        d = bm[o[k]] - b1
        d = d[np.isfinite(d)]
        if len(d) < 30: break
        lo, hi = np.percentile(d, [2.5, 97.5])
        if lo <= 0 <= hi: n += 1                 # difference not significant -> tied
        else: break                              # top-40 is rank-ordered, so stop at first loss
    NTIE[p] = n; frac.append(n)
ax.barh(range(len(PROTOS)), frac, color="#48a")
ax.set_yticks(range(len(PROTOS))); ax.set_yticklabels(PROTOS, fontsize=6.5)
ax.axvline(1, color="k", lw=.7, ls=":")
ax.set_xlabel("# methods with 95% paired-bootstrap tie with #1 (of top-40)")
ax.set_title("F5b  Statistically tied with the winner")
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("The noise band: enrichment metrics at this scale cannot separate most of the top of the leaderboard", fontsize=9.5, y=1.02)
save(fig, "F5_noise_band"); plt.close(fig)

# ============ F6: backbone x protocol ============
def fam(m):
    for f in ["af2", "colab", "boltz1", "af3", "rosetta"]:
        if m.startswith(f): return f
    return "other"
M["family"] = M["method"].map(fam)
FAM = ["af2", "colab", "boltz1", "af3", "rosetta", "other"]
mat = np.full((len(FAM), len(PROTOS)), np.nan); best = np.full_like(mat, np.nan)
for fi, f in enumerate(FAM):
    idx = M[M.family == f].index
    if not len(idx): continue
    for pi, p in enumerate(PROTOS):
        v = rk.loc[idx, p]
        mat[fi, pi] = v.min()                       # best rank achieved by the family
        best[fi, pi] = v.rank(pct=True).mean() * 100  # family median percentile
fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.4))
for ax, Mx, ttl in [(axes[0], mat, "F6a  best rank achieved by family"),
                    (axes[1], best, "F6b  family median percentile (100 = worst)")]:
    im = vecmesh(ax, Mx, cmap="RdBu_r" if ax is axes[0] else "RdBu", aspect="auto")
    ax.set_xticks(range(len(PROTOS))); ax.set_xticklabels(PROTOS, rotation=90, fontsize=6.8)
    ax.set_yticks(range(len(FAM))); ax.set_yticklabels(FAM, fontsize=7.5)
    for i, j in itertools.product(range(len(FAM)), range(len(PROTOS))):
        if np.isfinite(Mx[i, j]):
            ax.text(j, i, f"{Mx[i,j]:.0f}", ha="center", va="center", fontsize=6.2)
    ax.set_title(ttl, fontsize=9); vbar(fig, im, ax=ax, fraction=.03, pad=.01)
fig.suptitle("Backbone x protocol: is a predictor family's standing protocol-dependent?", fontsize=9.5)
save(fig, "F6_backbone_protocol"); plt.close(fig)

print("figures written to", FIG)
for f in sorted(os.listdir(FIG)): print("  ", f)
