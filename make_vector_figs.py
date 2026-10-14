"""
Generate "fully vector" copies work/panelX_figs_vec.py from the original figure
scripts.

Two changes:
  A. every ax.imshow(...) becomes vecmesh(ax, ...) -- heatmaps drawn as vector
     quads with pcolormesh.
     Geometric equivalence: under imshow(origin='upper') the centre of cell (i, j)
     sits at data coordinates (j, i); with pcolormesh and edges=arange(n+1)-0.5 the
     centre is again (j, i), and invert_yaxis() restores the upper origin. The
     existing ax.text(j, i, ...) annotations and Rectangle((j-.5, i-.5), 1, 1)
     overlays therefore need no change.
  B. every fig.savefig(f"{FIG}/X.png") becomes save(fig, "X") -- writing png, pdf
     and svg at once.

Guard: every anchor must match exactly once; on any mismatch nothing is written and
the script exits non-zero. The original scripts are read-only.
"""
import re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
W = ROOT / "work"

RC_ANCHOR = '                     "savefig.dpi": 300, "savefig.bbox": "tight", "axes.titlesize": 9})\n'

HELPERS = '''

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
'''

FIG_SAVE = re.compile(r'fig\.savefig\(f"\{FIG\}/([A-Za-z0-9_]+)\.png"\)')

# per-script replacement table: (label, old text, new text)
PANELS = {
    "panelA_figs.py": [
        ('F1 mesh',
         'im = ax.imshow(R, aspect="auto", cmap="RdBu_r", vmin=1, vmax=243, interpolation="nearest")',
         'im = vecmesh(ax, R, cmap="RdBu_r", vmin=1, vmax=243, aspect="auto")'),
        ('F2 mesh',
         'im = ax.imshow(Mx.values, vmin=0 if ax is axes[1] else 0.5, vmax=1, cmap=cm)',
         'im = vecmesh(ax, Mx.values, cmap=cm, vmin=0 if ax is axes[1] else 0.5, vmax=1, aspect="equal")'),
        ('F6 mesh',
         'im = ax.imshow(Mx, aspect="auto", cmap="RdBu_r" if ax is axes[0] else "RdBu")',
         'im = vecmesh(ax, Mx, cmap="RdBu_r" if ax is axes[0] else "RdBu", aspect="auto")'),
    ],
    "panelB_figs.py": [
        ('F7 mesh',
         'im = ax.imshow(R, aspect="auto", cmap="RdBu_r", vmin=1, vmax=len(Mf))',
         'im = vecmesh(ax, R, cmap="RdBu_r", vmin=1, vmax=len(Mf), aspect="auto")'),
    ],
    "panelC_figs.py": [
        ('F9 mesh',
         'im = ax.imshow(rk.loc[order].values, aspect="auto", cmap="RdBu_r", vmin=1, vmax=len(sub))',
         'im = vecmesh(ax, rk.loc[order].values, cmap="RdBu_r", vmin=1, vmax=len(sub), aspect="auto")'),
        ('F10a mesh',
         'im = ax.imshow(J.values, cmap="viridis", vmin=0, vmax=1)',
         'im = vecmesh(ax, J.values, cmap="viridis", vmin=0, vmax=1, aspect="equal")'),
    ],
    "panelD_figs.py": [
        ('F12a mesh',
         'im = ax.imshow(rk.loc[order].values, aspect="auto", cmap="RdBu_r", vmin=1, vmax=len(sub))',
         'im = vecmesh(ax, rk.loc[order].values, cmap="RdBu_r", vmin=1, vmax=len(sub), aspect="auto")'),
    ],
    "panelE_figs.py": [],
}

fails, out = [], {}

for name, reps in PANELS.items():
    src = (W / name).read_text(encoding="utf-8")

    if src.count(RC_ANCHOR) != 1:
        fails.append(f"{name}: rcParams anchor matched {src.count(RC_ANCHOR)}x")
        continue

    # ORDER MATTERS: rewrite fig.colorbar( BEFORE the HELPERS block is inserted,
    # otherwise the blanket replace also rewrites the call inside vbar() itself
    # and the helper recurses into itself.
    nb = src.count("fig.colorbar(")
    src = src.replace("fig.colorbar(", "vbar(fig, ")

    for tag, old, new in reps:
        c = src.count(old)
        if c != 1:
            fails.append(f"{name}: [{tag}] matched {c}x (expected 1)")
        else:
            src = src.replace(old, new)

    # savefig -> save():  rewrite every .png save point
    stems = FIG_SAVE.findall(src)
    if not stems:
        fails.append(f"{name}: no fig.savefig(...png) call found")
    new_src, k = FIG_SAVE.subn(lambda m: f'save(fig, "{m.group(1)}")', src)
    src = new_src

    # helpers last, so nothing above can touch their bodies
    src = src.replace(RC_ANCHOR, RC_ANCHOR + HELPERS)
    print(f"  {name:20s} colorbars -> vbar: {nb}")

    # NOTE: scan for call forms only. The HELPERS block legitimately contains the
    # strings "fig.savefig(f\"{FIG}/{stem}.{ext}\")" and the word "imshow" (docstring).
    if FIG_SAVE.search(src):
        fails.append(f"{name}: leftover fig.savefig(...png) call: {FIG_SAVE.findall(src)}")

    # the helper body must still call the real fig.colorbar, exactly once
    if src.count("fig.colorbar(mappable, **kw)") != 1:
        fails.append(f"{name}: vbar helper body corrupted")

    leftover = [l.strip() for l in src.split("\n") if ".imshow(" in l]
    if leftover:
        fails.append(f"{name}: leftover imshow call: {leftover}")
    if 'vecmesh(' not in src:
        fails.append(f"{name}: vecmesh not wired in")

    out[name] = (src, stems)
    print(f"  {name:20s} mesh replacements {len(reps)}  savefig->save {k}  stems={stems}")

if fails:
    print("\nABORT - no copies written:")
    print("\n".join("  " + f for f in fails))
    sys.exit(1)

for name, (src, stems) in out.items():
    dst = W / name.replace("_figs.py", "_figs_vec.py")
    dst.write_text(src, encoding="utf-8")
    print(f"wrote {dst.name}")

print("\nDONE")
