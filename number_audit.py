"""
"number_audit.py -- manuscript number audit"
""
"Recomputes every headline number the manuscript takes from panels A-E, straight"
"back onto the delivered CSVs."
""
"The expected values below were copied **verbatim from the manuscript text** and were"
"never back-filled from the computed results -- otherwise the audit would be circular"
"and would prove nothing."
""
"Usage: python3 work/number_audit.py   (from the repository root)"
"Output: PASS/FAIL list; the deposit ships it as work/number_audit.txt"
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

O = "out"
N_TOP = 40          # tie statistics only look at the top 40 of the leaderboard
TIE_BOOT_MIN = 30   # minimum number of valid bootstrap draws

rows = []


def chk(name, got, claim, tol=0.0):
    if isinstance(claim, (int, float)) and not isinstance(claim, bool):
        ok = abs(float(got) - float(claim)) <= tol
    else:
        ok = got == claim
    rows.append((ok, name, got, claim))
    print(f"{'PASS' if ok else 'FAIL'}  {name:52s} got={got}  claim={claim}")


def protocols_agree(M):
    cols = list(M.columns)
    off = [M.loc[a, b] for a, b in itertools.combinations(cols, 2)
           if np.isfinite(M.loc[a, b])]
    return cols, off


def tie_count(boot, est, n_top=N_TOP):
    """Methods tied with #1 under a paired bootstrap (tests the difference, not marginal CI overlap)."""
    o = np.argsort(-est)
    b1 = boot[o[0]]
    n = 1
    for k in range(1, min(n_top, len(o))):
        d = boot[o[k]] - b1
        d = d[np.isfinite(d)]
        if len(d) < TIE_BOOT_MIN:
            break
        lo, hi = np.percentile(d, [2.5, 97.5])
        if lo <= 0 <= hi:
            n += 1
        else:
            break
    return n


# ============================================================ A
AG = pd.read_csv(f"{O}/panelA_agreement.csv", index_col=0)
JC = pd.read_csv(f"{O}/panelA_top3_jaccard.csv", index_col=0)
M = pd.read_csv(f"{O}/panelA_matrix.csv")
P, off = protocols_agree(AG)
chk("A protocols", len(P), 14)
chk("A methods", len(M), 243)
chk("A min Spearman", round(min(off), 3), 0.639, 0.001)          # <- manuscript
chk("A median Spearman", round(np.median(off), 3), 0.859, 0.001)  # <- manuscript

jz = sum(1 for a, b in itertools.combinations(P, 2) if abs(JC.loc[a, b]) < 1e-9)
chk("A pairs with top-3 Jaccard=0", jz, 7)   # <- manuscript (wrote 8, corrected)

winners = {p: M.set_index("method")[p].idxmax() for p in P}
chk("A distinct winners", len(set(winners.values())), 7)

BO = np.load(f"{O}/panelA_boot.npz", allow_pickle=True)
ties = {p: tie_count(BO[p], M[p].values) for p in P}
chk("A protocols tying 40/40", sum(1 for v in ties.values() if v == N_TOP), 5)

# heterogeneity attribution across targets
_FEAT = os.environ.get("OVERATH_FEATURES", "work/overath_rankfeatures.parquet")
if os.path.isfile(_FEAT):
    PM = pd.read_parquet(_FEAT)
else:
    PM = pd.DataFrame(columns=["label", "target"] + list(M["method"]))
    print("SKIP  the Overath per-design table is not redistributed here; set "
          "OVERATH_FEATURES to score the heterogeneity attribution below "
          "(see DATA_SOURCES.md)")
y = PM["label"].astype(int).values
tg = PM["target"].values
IN = pd.read_csv(f"{O}/panelA_instability.csv")
disp = []
for i in range(len(M)):
    vals = []
    for t in np.unique(tg):
        ix = np.where(tg == t)[0]
        s = pd.to_numeric(PM[M.loc[i, "method"]], errors="coerce").values[ix] * M.loc[i, "sign"]
        yy = y[ix]
        f = np.isfinite(s)
        if f.sum() < 10 or yy[f].sum() < 2:
            continue
        k = max(1, int(round(.10 * f.sum())))
        o = np.argsort(-s[f])[:k]
        vals.append(yy[f][o].mean() / yy[f].mean())
    disp.append(np.nanstd(vals) / max(np.nanmean(vals), 1e-9) if len(vals) > 3 else np.nan)
D = pd.DataFrame({"method": M["method"], "cv": disp}).merge(IN, on="method")
D = D[np.isfinite(D["cv"])]
if len(D) > 3:
    rho, p = spearmanr(D["cv"], D["rank_sd"])
    chk("A heterogeneity attribution rho", round(rho, 3), 0.509, 0.002)
else:
    print("SKIP  heterogeneity attribution: no per-design table available")

# ============================================================ B
AGB = pd.read_csv(f"{O}/panelB_agreement.csv", index_col=0)
MB = pd.read_csv(f"{O}/panelB_matrix.csv")
chk("B methods", len(MB), 37)
chk("B protocols", len(AGB.columns), 14)

# scale comparison: must use the **common protocol set** (55 pairs), not each side's full set
AGs = pd.read_csv(f"{O}/panelB_scale_small_agreement.csv", index_col=0)
common = [c for c in AGB.columns if c in AGs.columns]
chk("B comparable protocols (common set)", len(common), 11)
_, offA = protocols_agree(pd.read_csv(f"{O}/panelA_agreement.csv", index_col=0)[common])
_, offS = protocols_agree(AGs[common])
_, offF = protocols_agree(AGB[common])
chk("B full-set min Spearman", round(min(offF), 3), 0.226, 0.001)      # <- manuscript
chk("B full-set median Spearman", round(np.median(offF), 3), 0.701, 0.001)  # <- manuscript
chk("B downsampled min Spearman", round(min(offS), 3), 0.047, 0.001)
chk("B downsampled median Spearman", round(np.median(offS), 3), 0.552, 0.001)

CMP = pd.read_csv(f"{O}/panelB_scale_compare.csv").set_index("protocol")
for p, exp in [("pooled_ap", 3), ("pooled_ef1", 3), ("pooled_ef10", 2),
               ("macro_ef1", 19), ("macro_ef5", 23), ("macro_ef10", 24),
               ("macro_ef20", 20)]:
    chk(f"B full-set ties {p}", int(CMP.loc[p, "tie_B_full"]), exp)

# ============================================================ C
MC = pd.read_csv(f"{O}/panelC_matrix.csv")
COV = pd.read_csv(f"{O}/panelC_protocol_coverage.csv")
for ts, exp_w, exp_c in [("10_targets", 5, 12), ("cao5_targets", 4, 12)]:
    cm = COV[(COV.targetset == ts) & COV.comparable].protocol.tolist()
    sub = MC[MC.targetset == ts].set_index("generator")[cm]
    chk(f"C {ts} winners", len(set(sub.idxmax().dropna())), exp_w)
    chk(f"C {ts} comparable protocols", len(cm), exp_c)
DT = pd.read_csv(f"{O}/panelC_determinism.csv")
acc = [c for c in DT.columns if "acc" in c.lower()][0]
chk("C deterministic pairs (file,definition)", len(DT), 43)
chk("C min determinism training accuracy", round(float(DT[acc].min()), 6), 1.0, 1e-6)

# ============================================================ D
MD = pd.read_csv(f"{O}/panelD_matrix.csv")
T1 = pd.read_csv(f"{O}/panelD_top1.csv")
MP = list(T1.protocol)
chk("D metrics", len(MD), 35)
chk("D protocols", len(MP), 48)
chk("D distinct winners", len(set(MD.set_index("method")[MP].idxmax())), 11)
DIN = pd.read_csv(f"{O}/panelD_instability.csv")
r2, p2 = spearmanr(DIN.rank_sd, DIN.pooled_auc_L_bind)
chk("D hypothesis-H rho", round(r2, 3), 0.078, 0.002)
chk("D hypothesis-H p", round(p2, 3), 0.658, 0.002)

# ============================================================ E / cross-panel
CARD = pd.read_csv(f"{O}/protocol_audit/cross_panel_card.csv")
chk("E cross-panel cells", len(CARD), 5)
chk("E non-STABLE cells", int((~CARD.verdict.str.startswith("STABLE")).sum()), 5)

# ============================================================ summary
npass = sum(1 for r in rows if r[0])
print(f"\n{npass} PASS / {len(rows)-npass} FAIL")
if npass != len(rows):
    print("\nFAILURES:")
    for okk, n, g, c in rows:
        if not okk:
            print(f"   {n}: computed {g} vs manuscript {c}")
sys.exit(0 if npass == len(rows) else 1)
