"""
Panel D figures.
  F12 prospective arbitration: (a) method x protocol rank matrix, (b) separability
  F13 (a) hypothesis-H scatter, (b) round-1 vs round-2 replication,
      (c) the provenance confound
"""
import os
import pathlib
import numpy as np, pandas as pd, itertools, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = str(ROOT)
OUT, FIG = f"{TASK}/out", f"{TASK}/out/figs"
B = os.environ.get("ADAPTYV_ROOT", "")
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

M = pd.read_csv(f"{OUT}/panelD_matrix.csv")
I = pd.read_csv(f"{OUT}/panelD_instability.csv")
T1 = pd.read_csv(f"{OUT}/panelD_top1.csv")
BO = np.load(f"{OUT}/panelD_boot.npz", allow_pickle=True)
PROTOS = list(T1.protocol)

# ================= F12 =================
fig = plt.figure(figsize=(14.2, 5.4))
gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.0], wspace=.20)

ax = fig.add_subplot(gs[0])
sub = M.set_index("method")[PROTOS]
rk = sub.rank(ascending=False)
order = rk.mean(axis=1).sort_values().index
im = vecmesh(ax, rk.loc[order].values, cmap="RdBu_r", vmin=1, vmax=len(sub), aspect="auto")
ax.set_xticks(range(len(PROTOS)))
ax.set_xticklabels([p.replace("[", " [") for p in PROTOS], rotation=90, fontsize=6.0)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6.3)
win = sub.idxmax()
for j, p in enumerate(PROTOS):
    if pd.notna(win[p]):
        ax.add_patch(plt.Rectangle((j - .5, list(order).index(win[p]) - .5), 1, 1,
                                   fill=False, ec="k", lw=1.3))
for j in range(6, len(PROTOS), 6):
    ax.axvline(j - .5, color="k", lw=.5, alpha=.4)
ax.set_title(f"F12a  Adaptyv EGFR round 2 (prospective, real SPR): 35 in-silico metrics x 48 protocols\n"
             f"378 designs / 53 measured binders.  Black box = winner.  "
             f"{len(set(win.values))} distinct winners", fontsize=9)
vbar(fig, im, ax=ax, fraction=.03, pad=.01).set_label("rank", fontsize=7)

ax = fig.add_subplot(gs[1])
def tiecount(p):
    bm = BO[p]; est = M[p].values; o = np.argsort(-est); b1 = bm[o[0]]; n = 1
    for k in range(1, len(o)):
        d = bm[o[k]] - b1; d = d[np.isfinite(d)]
        if len(d) < 30: break
        lo, hi = np.percentile(d, [2.5, 97.5])
        if lo <= 0 <= hi: n += 1
        else: break
    return n
sel = [p for p in PROTOS if p.startswith("pooled")][:12]
tc = [tiecount(p) for p in sel]
ypos = np.arange(len(sel))
ax.barh(ypos, tc, color="#36a")
ax.axvline(len(M), color="k", lw=.8, ls=":")
ax.text(len(M), -.9, f"all {len(M)}", fontsize=6.5, ha="right")
ax.set_yticks(ypos); ax.set_yticklabels([s.replace("pooled_", "").replace("_", " ") for s in sel], fontsize=6.6)
ax.invert_yaxis()
ax.set_xlabel("# metrics statistically tied with the winner\n(paired bootstrap on the difference)")
ax.set_title("F12b  With 53 positives nothing is\nseparable from the winner")
ax.spines[["top", "right"]].set_visible(False)

save(fig, "F12_adaptyv_prospective"); plt.close(fig)

# ================= F13 =================
fig = plt.figure(figsize=(13.2, 4.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=.34)

ax = fig.add_subplot(gs[0])
ax.scatter(I.rank_sd, I.pooled_auc_L_bind, s=22, color="#36a", alpha=.85)
for _, r in I.nlargest(2, "rank_sd").iterrows():
    ax.annotate(r.method[:14], (r.rank_sd, r.pooled_auc_L_bind), fontsize=6.5)
ax.set_xlabel("protocol instability (rank SD over 48 protocols)")
ax.set_ylabel("prospective pooled AUC (measured binding)")
ax.set_title("F13a  Hypothesis H → NOT supported\nρ = 0.078, p = 0.658, n = 35")
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[1])
RD = pd.read_csv(f"{OUT}/panelD_rounds.csv")
x = np.arange(2); w = .35
for k, (rnd, c) in enumerate([("round1", "#c96"), ("round2", "#36a")]):
    d = RD[RD["round"] == rnd].set_index("metric")
    ax.bar(x + (k - .5) * w, [d.loc["plddt", "auc_oriented"], d.loc["pae_interaction", "auc_oriented"]],
           w, label=f"{rnd} (n_pos={int(d['n_pos'].iloc[0])})", color=c)
    ax.text(x[0] + (k - .5) * w, d.loc["plddt", "auc_oriented"] + .01, f"{d.loc['plddt','auc']:.2f}",
            ha="center", fontsize=6.8)
    ax.text(x[1] + (k - .5) * w, d.loc["pae_interaction", "auc_oriented"] + .01,
            f"{d.loc['pae_interaction','auc']:.2f}", ha="center", fontsize=6.8)
ax.axhline(.5, color="k", lw=.6, ls=":")
ax.set_xticks(x); ax.set_xticklabels(["pLDDT", "pae_interaction"])
ax.set_ylim(0, .95); ax.set_ylabel("oriented AUC (label above bar = raw AUC)")
ax.set_title("F13b  Two prospective rounds, same target:\nthe metric ordering flips")
ax.legend(fontsize=6.8, frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[2])
if not B:
    raise SystemExit("[panelD F13b] set ADAPTYV_ROOT to the Adaptyv EGFR "
                     "dataset directory -- see DATA_SOURCES.md.")
r2 = pd.read_csv(f"{B}/round2/result_summary.csv").rename(columns={"name": "design_name"})
ds = pd.read_csv(f"{B}/round2/destress_binder_with_egfr.csv")
D = r2.merge(ds, on="design_name", how="left", suffixes=("", "_ds"))
D["y"] = D["binding"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
need = ["iptm", "plddt", "pae_interaction", "esm_pll", "num_residues"]
K = D[D.y.notna() & np.isfinite(D[need].astype(float)).all(axis=1)].reset_index(drop=True)
K["g"] = K["design_models"].fillna("(unreported)")
tab = K.groupby("g").agg(n=("y", "size"), rate=("y", "mean")).sort_values("rate", ascending=False)
ax.scatter(tab.n, 100 * tab.rate, s=26, color="#c33")
for g, r in tab.head(3).iterrows():
    ax.annotate(g[:22], (r.n, 100 * r.rate), fontsize=6.3, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("designs contributed by the provenance group")
ax.set_ylabel("% of them that really bind")
ax.set_title("F13c  Provenance is as predictive as any score\n"
             "group-alone AUC 0.642  vs  best score 0.667")
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("F13  Adaptyv EGFR: prospective check of the paper's claims", fontsize=10, y=1.04)
save(fig, "F13_adaptyv_checks"); plt.close(fig)
print("wrote F12, F13 to", FIG)
print("distinct winners:", len(set(win.values)))
print("tie counts on pooled protocols:", dict(zip([s.replace('pooled_','') for s in sel], tc)))
