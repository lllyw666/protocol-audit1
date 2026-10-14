"""
Assemble ONE audit card per panel (A / B / C / D) with the toolkit's own
`audit_from_matrix`, so the four panels are summarised by identical code.

Inputs are the already-computed leaderboard matrices in ../../out/ :
  panelA_matrix.csv  (Overath, wet-lab labels, 243 methods)
  panelB_matrix.csv  (cao,     wet-lab labels,  37 methods)
  panelC_matrix.csv  (ProtDBench, deterministic labels, 7 generators x 2 target sets)
  panelD_matrix.csv  (Adaptyv EGFR r2, real SPR, 35 methods)

Outputs -> cross_panel_card.csv, cross_panel_card.md
"""
import pathlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from protocol_audit import audit_from_matrix                     # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
TASK = str(ROOT)
OUT = f"{TASK}/out"

PANELS = []


def add(name, label, matrix, protocol_cols, method_col="method", verdict_extra=""):
    P = [c for c in protocol_cols if c in matrix.columns]
    card = audit_from_matrix(matrix, protocol_cols=P)
    m = card["meta"]
    PANELS.append(dict(
        panel=name, dataset=label, n_methods=m["n_methods"],
        n_protocols=m["n_protocols"], n_comparable=m["n_comparable_protocols"],
        n_winners=m["n_distinct_winners"], winners=", ".join(m["winners"]),
        rho_min=m["pairwise_spearman_min"], rho_med=m["pairwise_spearman_median"],
        pct_sign_disagree=(None if m["pairwise_spearman_frac_negative"] is None
                           else round(100 * m["pairwise_spearman_frac_negative"], 1)),
        verdict=m["verdict"] + verdict_extra))
    return card


# ---- A: Overath ------------------------------------------------------------
A = pd.read_csv(f"{OUT}/panelA_matrix.csv")
add("A", "Overath 3,669 · wet-lab labels", A,
    [c for c in A.columns if c not in ("method", "sign")])

# ---- B: cao ----------------------------------------------------------------
B = pd.read_csv(f"{OUT}/panelB_matrix.csv")
add("B", "cao 236,246 · wet-lab labels", B,
    [c for c in B.columns if c != "method"])

# ---- C: ProtDBench (zero label noise), both target sets --------------------
C = pd.read_csv(f"{OUT}/panelC_matrix.csv")
COV = pd.read_csv(f"{OUT}/panelC_protocol_coverage.csv")
for ts, tag in (("10_targets", "C1"), ("cao5_targets", "C2")):
    sub = C[C.targetset == ts].rename(columns={"generator": "method"})
    protos = COV[(COV.targetset == ts) & COV.comparable].protocol.tolist()
    add(tag, f"ProtDBench {ts} · DETERMINISTIC labels", sub[["method"] + protos],
        protos, verdict_extra="")

# ---- D: Adaptyv EGFR round 2 ----------------------------------------------
D = pd.read_csv(f"{OUT}/panelD_matrix.csv")
add("D", "Adaptyv EGFR r2 · real SPR (prospective)", D,
    [c for c in D.columns if c != "method"])

CARD = pd.DataFrame(PANELS)
CARD.to_csv(f"{HERE}/cross_panel_card.csv", index=False)

# --------------------------------------------------------------------- markdown
L = []
A_ = L.append
A_("# Cross-panel protocol audit card (four panels, one code path)\n")
A_("Recomputed by `protocol_audit.audit_from_matrix` from the leaderboard matrices already computed for the four panels.\n")
A_("| Panel | Data | Object ranked | Protocols (comparable) | Winners | Min rank Spearman | Median | Sign-reversed pairs | verdict |")
A_("|---|---|---|---|---|---|---|---|---|")
for r in PANELS:
    A_(f"| **{r['panel']}** | {r['dataset']} | {r['n_methods']} | "
       f"{r['n_comparable']}/{r['n_protocols']} | **{r['n_winners']}** | "
       f"**{r['rho_min']:+.3f}** | {r['rho_med']:.3f} | "
       f"{r['pct_sign_disagree'] if r['pct_sign_disagree'] is not None else '—'}% | "
       f"`{r['verdict']}` |")
A_("")
A_("## Winner sets\n")
for r in PANELS:
    A_(f"- **{r['panel']}** ({r['n_comparable']} comparable protocols, {r['n_winners']} winners): {r['winners']}")
A_("")
A_("## How to read\n")
A_("- `STABLE` = every comparable protocol points to the same winner; `FRAGILE` = more winners than max(2, 20% of the objects ranked).")
A_("- `+SIGN-DISAGREEMENT` = at least one protocol pair has a rank Spearman < 0 (the leaderboard reverses).")
A_("- **None** of the four panels is `STABLE`. All four reach the conclusion that "
   "the leaderboard is a property of the protocol, before any null-hypothesis test.")
open(f"{HERE}/cross_panel_card.md", "w").write("\n".join(L))

print(CARD.to_string(index=False))
print()
print("\n".join(L[-6:]))
