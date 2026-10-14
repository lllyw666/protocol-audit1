"""
Panel C figures.
  F9  generator x protocol rank matrix (both target sets) - who wins flips
  F10 the label-definition axis: (a) design-level Jaccard between definitions,
      (b) BindCraft per-target success rate by definition, (c) monotone nesting
  F11 zero-label-noise calibration: protocol-pair agreement when labels are
      wet-lab (Panels A/B) vs provably deterministic (Panel C)
"""
import os
import pathlib
import numpy as np, pandas as pd, itertools, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

M = pd.read_csv(f"{OUT}/panelC_matrix.csv")
COV = pd.read_csv(f"{OUT}/panelC_protocol_coverage.csv")
TS = ["10_targets", "cao5_targets"]

# ================= F9: generator x protocol rank matrix =================
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.0), gridspec_kw={"wspace": .30})
for ax, ts in zip(axes, TS):
    CMP = COV[(COV.targetset == ts) & COV.comparable].protocol.tolist()
    sub = M[M.targetset == ts].set_index("generator")[CMP]
    rk = sub.rank(ascending=False)
    order = rk.mean(axis=1).sort_values().index
    win = sub.idxmax()
    im = vecmesh(ax, rk.loc[order].values, cmap="RdBu_r", vmin=1, vmax=len(sub), aspect="auto")
    ax.set_xticks(range(len(CMP)))
    ax.set_xticklabels([c.replace("_success", "") for c in CMP], rotation=90, fontsize=6.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7.2)
    for j, c in enumerate(CMP):
        if pd.notna(win[c]):
            i = list(order).index(win[c])
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="k", lw=1.6))
    ax.set_title(f"{ts}  —  {len(set(win.dropna()))} distinct winners / {len(CMP)} protocols")
vbar(fig, im, ax=axes, fraction=.02, pad=.015).set_label("rank (1 = best)", fontsize=7)
fig.suptitle("F9  ProtDBench generative benchmark: same 7 designers, 12 comparable protocols,\n"
             "black box = the winner.  Labels are provably deterministic functions of the "
             "design's own predicted structure (tree accuracy = 1.000000)", fontsize=9.5, y=1.10)
save(fig, "F9_generator_protocol_matrix"); plt.close(fig)

# ================= F10: label-definition axis =================
DA = pd.read_csv(f"{OUT}/panelC_defs_agreement.csv")
PT = pd.read_csv(f"{OUT}/panelC_bindcraft_defaxis_bytarget.csv", index_col=0)
DEFS = sorted(set(DA.def_a) | set(DA.def_b))
short = {d: d.replace("_success", "") for d in DEFS}

J = pd.DataFrame(np.nan, index=DEFS, columns=DEFS)
for _, r in DA.iterrows():
    J.loc[r.def_a, r.def_b] = r.jaccard; J.loc[r.def_b, r.def_a] = r.jaccard
np.fill_diagonal(J.values, 1.0)

fig = plt.figure(figsize=(13.4, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.55, .95], wspace=.34)

# F10a  Jaccard heatmap
ax = fig.add_subplot(gs[0])
im = vecmesh(ax, J.values, cmap="viridis", vmin=0, vmax=1, aspect="equal")
ax.set_xticks(range(len(DEFS))); ax.set_xticklabels([short[d] for d in DEFS], rotation=45,
                                                    ha="right", fontsize=6.8)
ax.set_yticks(range(len(DEFS))); ax.set_yticklabels([short[d] for d in DEFS], fontsize=6.8)
for i in range(len(DEFS)):
    for j in range(len(DEFS)):
        v = J.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="w" if v < .55 else "k")
        else:
            ax.text(j, i, "—", ha="center", va="center", fontsize=6.5, color="#bbb")
ax.set_title("F10a  Design-level Jaccard between\nthe 5 published success definitions")
vbar(fig, im, ax=ax, fraction=.046, pad=.03).set_label("Jaccard", fontsize=7)

# F10b  BindCraft per-target success rate by definition
ax = fig.add_subplot(gs[1])
dorder = [d for d in ["af2_opt_success", "bindcraft_success", "ptx_large_success",
                      "ptx_mini_success", "af2_easy_success"] if d in PT.columns]
x = np.arange(len(PT)); w = .8 / len(dorder)
cmap = plt.get_cmap("tab10")
for k, d in enumerate(dorder):
    ax.bar(x + (k - (len(dorder) - 1) / 2) * w, PT[d].values, w, label=short[d],
           color=cmap(k), edgecolor="none")
ax.set_xticks(x); ax.set_xticklabels(PT.index, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("% of BindCraft designs flagged 'success'")
ax.set_title("F10b  One generator (BindCraft), one target set, 5 definitions:\n"
             "the target-difficulty ranking itself changes")
ax.legend(fontsize=6.8, frameon=False, ncol=2)
ax.spines[["top", "right"]].set_visible(False)

# F10c  nesting: which definitions subsume which
ax = fig.add_subplot(gs[2]); ax.axis("off")
lines = []
for _, r in DA.iterrows():
    if r.def_a == "af2_opt_success" and r.def_b == "bindcraft_success":
        lines.append(f"af2_opt_success ⊂ bindcraft_success\n"
                     f"   ({int(r.both):,} shared; 0 in af2_opt but not bindcraft;\n"
                     f"    {int(r.b_only):,} in bindcraft only)")
    if r.def_a == "bindcraft_success" and r.def_b == "ptx_mini_success":
        lines.append(f"ptx_mini_success ⊄ bindcraft_success\n"
                     f"   ({int(r.both):,} shared; {int(r.a_only):,} bindcraft-only;\n"
                     f"    {int(r.b_only):,} ptx_mini-only)")
ax.text(0, 1, "F10c  How the definitions nest\n(BindCraft / 10_targets, n=30,992)",
        fontsize=9, va="top")
ax.text(0, .80, "\n\n".join(lines), fontsize=7.2, va="top", family="monospace")
ax.text(0, .34, "bindcraft_success ⊇ af2_opt_success\n"
                "→ the loosest definition contains every\n"
                "   design the stricter one accepts.\n\n"
                "12 distinct 5-bit verdict patterns exist:\n"
                "the definitions are not nested in general.",
        fontsize=7.2, va="top")
fig.suptitle("F10  The label-definition protocol axis (all labels deterministic → zero label noise)",
             fontsize=10, y=1.04)
save(fig, "F10_definition_axis"); plt.close(fig)

# ================= F11: zero-noise calibration =================
AgA = pd.read_csv(f"{OUT}/panelA_agreement.csv", index_col=0)
AgB = pd.read_csv(f"{OUT}/panelB_agreement.csv", index_col=0)
pairsC = []
for ts in TS:
    AG = pd.read_csv(f"{OUT}/panelC_agreement_{ts}.csv", index_col=0)
    pairsC += [AG.loc[a, b] for a, b in itertools.combinations(AG.columns, 2)
               if np.isfinite(AG.loc[a, b])]
prA = [AgA.loc[a, b] for a, b in itertools.combinations(AgA.columns, 2) if np.isfinite(AgA.loc[a, b])]
prB = [AgB.loc[a, b] for a, b in itertools.combinations(AgB.columns, 2) if np.isfinite(AgB.loc[a, b])]

fig = plt.figure(figsize=(12.6, 4.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=.30)

ax = fig.add_subplot(gs[0])
# method counts taken from the matrices themselves, never hardcoded (a stale "40" lived here once)
NA = len(pd.read_csv(f"{OUT}/panelA_matrix.csv"))
NB = len(pd.read_csv(f"{OUT}/panelB_matrix.csv"))
groups = [(f"Panel A\nOverath 3,669\nwet-lab labels\n{NA} methods", prA, "#888"),
          (f"Panel B\ncao 236,246\nwet-lab labels\n{NB} methods", prB, "#36a"),
          ("Panel C\nProtDBench 245,168\nDETERMINISTIC labels\n7 generators", pairsC, "#c33")]
for i, (lab, v, c) in enumerate(groups):
    xx = np.full(len(v), i) + np.linspace(-.18, .18, len(v))
    ax.scatter(xx, v, s=15, color=c, alpha=.8)
    ax.hlines(np.median(v), i - .30, i + .30, color="k", lw=1.6)
    ax.text(i + .32, np.median(v), f"med {np.median(v):.2f}", va="center", fontsize=7)
    ax.text(i, -1.02, f"min {min(v):.2f}", ha="center", fontsize=7, color=c)
ax.axhline(0, color="k", lw=.5, ls=":")
ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups], fontsize=6.9)
ax.set_ylabel("pairwise leaderboard Spearman between protocols")
ax.set_ylim(-1.08, 1.06)
ax.set_title("F11a  Protocol-pair agreement does NOT improve when labels are noise-free")
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[1])
DA2 = DA.groupby(["def_a", "def_b"]).agg(n=("n", "sum"), jaccard=("jaccard", "mean")).reset_index()
DA2["disc"] = 1 - DA2.jaccard
DA2 = DA2.sort_values("disc", ascending=False)
lab = [f"{a.replace('_success','')} vs {b.replace('_success','')}" for a, b in zip(DA2.def_a, DA2.def_b)]
ax.barh(np.arange(len(DA2)), 100 * DA2.disc, color="#c33", alpha=.85)
ax.set_yticks(np.arange(len(DA2))); ax.set_yticklabels(lab, fontsize=6.6)
ax.invert_yaxis()
ax.set_xlabel("% of positive designs on which the two definitions disagree\n(1 − Jaccard, design level, noise-free)")
ax.set_title("F11b  Two definitions of 'a binder'\nare different experiments")
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("F11  Zero-label-noise calibration: protocol-induced disagreement is a property "
             "of the protocol, not of noisy labels", fontsize=10, y=1.04)
save(fig, "F11_noise_floor"); plt.close(fig)

print("wrote F9, F10, F11 to", FIG)
print(f"\nF11a min/median  A {min(prA):.3f}/{np.median(prA):.3f}   "
      f"B {min(prB):.3f}/{np.median(prB):.3f}   C {min(pairsC):.3f}/{np.median(pairsC):.3f}")
print("\nF11b disagreement (1-Jaccard), worst first:")
for _, r in DA2.iterrows():
    print(f"  {r.def_a:19s} vs {r.def_b:19s}  n={int(r.n):7d}  disagree={100*r.disc:5.1f}%")
