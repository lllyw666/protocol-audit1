"""Panel B figures F7 (cao full matrix) and F8 (controlled scale experiment)."""
import os
import pathlib
import numpy as np, pandas as pd, itertools, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = str(ROOT)
OUT, FIG = f"{TASK}/out", f"{TASK}/out/figs"
os.makedirs(FIG, exist_ok=True)
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

Mf = pd.read_csv(f"{OUT}/panelB_matrix.csv")
AGf = pd.read_csv(f"{OUT}/panelB_agreement.csv", index_col=0)
JCf = pd.read_csv(f"{OUT}/panelB_top3_jaccard.csv", index_col=0)
PROTOS = list(AGf.columns)
rk = Mf[PROTOS].rank(ascending=False); order = rk.mean(axis=1).sort_values().index

# ================= F7: cao 236k main matrix =================
fig, ax = plt.subplots(figsize=(7.4, 5.4))
R = rk.loc[order].values
im = vecmesh(ax, R, cmap="RdBu_r", vmin=1, vmax=len(Mf), aspect="auto")
ax.set_xticks(range(len(PROTOS))); ax.set_xticklabels(PROTOS, rotation=90, fontsize=7.5)
ax.set_yticks(range(len(order))); ax.set_yticklabels([Mf.loc[i, "method"] for i in order], fontsize=6.2)
ax.set_title(f"F7  cao panel: {len(Mf)} scoring methods x {len(PROTOS)} protocols\n"
             f"(236,246 designs / 13 targets / 1,485 positives = 0.63% ; blue = better rank)")
for s in [3.5]:
    ax.axvline(s, color="k", lw=.8)
vbar(fig, im, ax=ax, fraction=.04, pad=.02).set_label("rank", fontsize=7)
save(fig, "F7_cao_matrix"); plt.close(fig)

# ================= F8: controlled scale experiment =================
COMP = pd.read_csv(f"{OUT}/panelB_scale_compare.csv")
AGs = pd.read_csv(f"{OUT}/panelB_scale_small_agreement.csv", index_col=0)
AGa = pd.read_csv(f"{OUT}/panelA_agreement.csv", index_col=0)

shared = [p for p in PROTOS if p in AGa.columns and p in AGs.columns]
prA = [AGa.loc[a, b] for a, b in itertools.combinations(shared, 2)]
prS = [AGs.loc[a, b] for a, b in itertools.combinations(shared, 2)]
prF = [AGf.loc[a, b] for a, b in itertools.combinations(shared, 2)]

fig = plt.figure(figsize=(12.6, 4.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.35, 1.05], wspace=.32)

# F8a pairwise agreement distributions
ax = fig.add_subplot(gs[0])
bars = [prA, prS, prF]
labs = [f"Overath\nn=3,669", f"cao subsampled\nn=3,185", f"cao full\nn=236,246"]
cols = ["#888", "#c96", "#36a"]
for i, (v, c) in enumerate(zip(bars, cols)):
    x = np.full(len(v), i) + np.linspace(-.16, .16, len(v))
    ax.scatter(x, v, s=16, color=c, alpha=.8)
    ax.hlines(np.median(v), i - .28, i + .28, color="k", lw=1.4)
    ax.text(i + .3, np.median(v), f"{np.median(v):.2f}", va="center", fontsize=7)
    ax.text(i, 1.03, f"min {min(v):.2f}", ha="center", fontsize=6.8, color=c)
ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=7.2)
ax.set_ylabel("pairwise leaderboard Spearman"); ax.set_ylim(-.05, 1.09)
ax.set_title("F8a  Protocol-pair agreement\n(same 11 protocols)")
ax.spines[["top", "right"]].set_visible(False)

# F8b tie counts grouped bars
ax = fig.add_subplot(gs[1])
x = np.arange(len(COMP)); w = .27
ax.bar(x - w, COMP.tie_A, w, label="Overath n=3,669", color="#888")
ax.bar(x, COMP.tie_B_small, w, label="cao subsampled n=3,185", color="#c96")
ax.bar(x + w, COMP.tie_B_full, w, label="cao full n=236,246", color="#36a")
ax.set_xticks(x); ax.set_xticklabels(COMP.protocol, rotation=90, fontsize=6.8)
ax.axhline(40, color="k", lw=.6, ls=":")
ax.set_ylabel("# methods statistically tied with #1 (of top-40)")
ax.set_title("F8b  Who is statistically distinguishable\nfrom the winner?")
ax.legend(fontsize=6.8, loc="upper left", frameon=False)
ax.spines[["top", "right"]].set_visible(False)

# F8c noise vs bias decomposition
ax = fig.add_subplot(gs[2])
noise = COMP.tie_B_small - COMP.tie_B_full          # removed by 74x more data
bias = COMP.tie_B_full                              # residual
o = np.argsort(-bias.values)
ax.barh(np.arange(len(COMP)), bias.values[o], color="#36a", label="bias: persists at n=236k")
ax.barh(np.arange(len(COMP)), noise.values[o], left=bias.values[o], color="#c96",
        alpha=.75, label="noise: removed by scale")
ax.set_yticks(np.arange(len(COMP))); ax.set_yticklabels(COMP.protocol.values[o], fontsize=6.8)
ax.set_xlabel("# methods tied with #1")
ax.set_title("F8c  Decomposition\n(tie count at n=3,185 split into noise vs bias)")
ax.legend(fontsize=6.6, frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("F8  Controlled scale experiment: same cao targets, same labels, only n changes",
             fontsize=10, y=1.03)
save(fig, "F8_scale_effect"); plt.close(fig)
print("wrote F7, F8 to", FIG)
print("\nF8a mins/medians:",
      f"A {min(prA):.3f}/{np.median(prA):.3f}",
      f"B' {min(prS):.3f}/{np.median(prS):.3f}",
      f"B {min(prF):.3f}/{np.median(prF):.3f}")
print("\nresidual (bias) tie counts at n=236k, worst first:")
for i in o:
    print(f"  {COMP.protocol.values[i]:16s} small={COMP.tie_B_small.values[i]:2d} -> full={COMP.tie_B_full.values[i]:2d}  (noise removed={noise.values[i]})")
