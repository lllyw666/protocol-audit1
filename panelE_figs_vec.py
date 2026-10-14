"""
Panel E figures -- the protocol-audit toolkit.

  F14  what the toolkit does: (a) pipeline, (b) the orientation audit,
       (c) the separability audit, both on the Adaptyv table
  F15  the four-panel audit certificate (one code path, four datasets)
"""
import pathlib
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = str(ROOT)
FIG = f"{TASK}/out/figs"
TK = f"{TASK}/out/protocol_audit"
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

# ============================================================ F14
fig = plt.figure(figsize=(14.2, 4.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 1.0, 1.0], wspace=.46)

# ---- F14a pipeline ----
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
BOX = [
    (0.4, 8.3, 9.2, 1.25, "#eef3fb", "INPUT   designs x candidate scores\nscore cols + label cols + group col, one row per design"),
    (0.4, 6.6, 9.2, 1.25, "#e8f5ec", "CHECK 1   column audit\ncoverage -> drop <50% | orientation -> flip so AUC>=0.5"),
    (0.4, 4.9, 9.2, 1.25, "#fdf3e3", "CHECK 2   protocol grid\naggregation x label def x metric  (20-100 protocols)"),
    (0.4, 3.2, 9.2, 1.25, "#fbe9ec", "CHECK 3   two-level paired bootstrap\nresample groups, then items; count ties with the winner"),
    (0.4, 1.5, 9.2, 1.25, "#efe9fb", "OUTPUT   audit card + tables\nverdict, winner set, agreement, instability, separability"),
]
for x, y, w, h, c, t in BOX:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.25",
                                fc=c, ec="#666", lw=.8))
    ax.text(x + .35, y + h / 2, t, fontsize=6.9, va="center")
for y0, y1 in [(8.3, 7.85), (6.6, 6.15), (4.9, 4.45), (3.2, 2.75)]:
    ax.add_patch(FancyArrowPatch((5, y0), (5, y1), arrowstyle="-|>", mutation_scale=9,
                                 lw=.9, color="#444"))
ax.set_title("F14a  protocol_audit: three checks that leaderboards routinely skip", fontsize=9)
ax.text(.4, .55, "PoseBusters checks the prediction;  protocol_audit checks the evaluation.\n"
                 "Neither predicts anything; both refuse to return a number without a check.",
        fontsize=6.6, style="italic", color="#444")

# ---- F14b orientation audit (Adaptyv) ----
OR = pd.read_csv(f"{TK}/reports/adaptyv/orientation.csv").sort_values("auc_raw_primary_label")
ax = fig.add_subplot(gs[1])
col = ["#c33" if f else "#36a" for f in OR.flipped]
ax.barh(np.arange(len(OR)), OR.auc_raw_primary_label, color=col, alpha=.9)
ax.axvline(.5, color="k", lw=.8, ls=":")
ax.set_yticks(np.arange(len(OR)))
ax.set_yticklabels([s[:24] for s in OR.score], fontsize=6.5)
ax.set_xlabel("raw pooled AUC vs measured binding")
ax.set_xlim(0, .74)
ax.set_ylim(-1.6, len(OR) - .4)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc="#36a", label=f"already right way up ({(~OR.flipped).sum()})"),
                   Patch(fc="#c33", label=f"WRONG way up, flipped ({OR.flipped.sum()})")],
          fontsize=6.6, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.30), ncol=2)
ax.set_title("F14b  Check 1: orientation\nAdaptyv EGFR r2, 16 descriptors", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

# ---- F14c separability audit (Adaptyv) ----
SE = pd.read_csv(f"{TK}/reports/adaptyv/separability.csv")
sel = SE[SE.protocol.str.startswith("pooled")
         & SE.protocol.str.contains(r"\[L_bind\]|\[L_medium_plus\]", regex=True)].copy()
sel["lab"] = sel.protocol.str.replace("pooled_", "", regex=False)
sel = sel.sort_values("n_tied_with_winner", ascending=False)
ax = fig.add_subplot(gs[2])
ax.barh(np.arange(len(sel)), sel.n_tied_with_winner, color="#36a", alpha=.85)
ax.axvline(16, color="k", lw=.8, ls=":")
ax.text(15.8, -0.95, "all 16 methods", fontsize=6.6, ha="right")
ax.set_yticks(np.arange(len(sel)))
ax.set_yticklabels(sel.lab, fontsize=6.5)
ax.invert_yaxis()
ax.set_xlim(0, 17); ax.set_ylim(len(sel) - .4, -1.2)
ax.set_xlabel("# methods statistically tied with the winner\n(paired two-level bootstrap, 400 draws)")
ax.set_title("F14c  Check 3: separability\n378 designs / 53 positives -> nothing ranks", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("F14  The protocol-audit toolkit, shown on the prospective EGFR table", fontsize=10, y=1.03)
save(fig, "F14_protocol_audit_toolkit"); plt.close(fig)

# ============================================================ F15
CARD = pd.read_csv(f"{TK}/cross_panel_card.csv")
NICE = {"A": "A  Overath\nwet-lab", "B": "B  cao\nwet-lab",
        "C1": "C1  ProtDBench\n10_targets", "C2": "C2  ProtDBench\ncao5",
        "D": "D  Adaptyv\nprospective"}
fig = plt.figure(figsize=(13.6, 4.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=.22)

# F15a dumbbell: rho_min .. rho_med
ax = fig.add_subplot(gs[0])
y = np.arange(len(CARD))[::-1]
for i, (_, r) in enumerate(CARD.iterrows()):
    yy = y[i]
    ax.plot([r.rho_min, r.rho_med], [yy, yy], color="#999", lw=2.2, zorder=1)
    ax.scatter([r.rho_min], [yy], s=48, color="#c33", zorder=3,
               label="min (worst protocol pair)" if i == 0 else None)
    ax.scatter([r.rho_med], [yy], s=48, color="#36a", zorder=3,
               label="median" if i == 0 else None)
    ax.annotate(f"{r.n_winners} winners", (max(r.rho_min, r.rho_med), yy),
                xytext=(6, 0), textcoords="offset points", fontsize=6.7, va="center")
ax.axvline(0, color="k", lw=.7, ls=":")
ax.set_yticks(y); ax.set_yticklabels([NICE[p] for p in CARD.panel], fontsize=6.8)
ax.set_xlim(-1.15, 1.15)
ax.set_xlabel("pairwise leaderboard Spearman over the protocol grid")
ax.set_title("F15a  No panel is protocol-stable\n"
             "left dot = the two protocols that disagree most", fontsize=9)
ax.legend(fontsize=6.8, frameon=False, loc="upper center",
          bbox_to_anchor=(.5, -0.20), ncol=2)
ax.spines[["top", "right"]].set_visible(False)

# F15b certificate table
ax = fig.add_subplot(gs[1]); ax.axis("off")
cols = ["panel", "n obj", "protocols\n(comparable)", "distinct\nwinners",
        "rho min", "rho med", "sign-\ndisagree", "verdict"]
rows = []
for _, r in CARD.iterrows():
    rows.append([r.panel, f"{int(r.n_methods)}",
                 f"{int(r.n_comparable)}/{int(r.n_protocols)}",
                 str(int(r.n_winners)),
                 f"{r.rho_min:+.3f}", f"{r.rho_med:.3f}",
                 f"{r.pct_sign_disagree:.1f}%" if pd.notna(r.pct_sign_disagree) else "—",
                 r.verdict.replace("+SIGN-DISAGREEMENT", "\n+SIGN-DISAGREE")])
tab = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
tab.auto_set_font_size(False); tab.set_fontsize(7.3); tab.scale(1, 1.8)
for (i, j), c in tab.get_celld().items():
    c.set_edgecolor("#bbb"); c.set_linewidth(.5)
    if i == 0:
        c.set_facecolor("#e9e9e9"); c.set_text_props(weight="bold")
    else:
        v = rows[i - 1][7]
        c.set_facecolor("#fbe9ec" if "FRAGILE" in v else "#e8f5ec")
        if j == 7:
            c.set_text_props(color="#c33" if "FRAGILE" in v else "#2a7")
    tab[(i, 0)].set_text_props(weight="bold")
ax.set_title(f"F15b  The audit certificate: one code path, four source datasets,\n"
             f"{len(CARD)} analysis rows — and not one of them is STABLE", fontsize=9)

fig.suptitle("F15  Cross-panel audit card (protocol_audit.audit_from_matrix)", fontsize=10, y=1.03)
save(fig, "F15_audit_certificate"); plt.close(fig)

print("wrote F14, F15 to", FIG)
