"""
Demo driver: run `protocol_audit` on the two design-level tables of the paper.

  (1) ProtDBench generative benchmark (Panel C)  -> zero-label-noise control
      items = 30,000 designs (<=3,000 per target, seeded) from 10_targets
      scores = the per-design in-silico metrics shipped with the benchmark
      labels = the published `*_success` definitions
      groups = Target

  (2) Adaptyv EGFR round 2 (Panel D) -> prospective, real SPR
      items = the 378 designs with a known label and all descriptors finite
      scores = 16 sequence / structure descriptors
      labels = 4 label definitions
      groups = design provenance (design_models)

Both audit cards land in ./reports/<name>/.
"""
import pathlib
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from protocol_audit import ProtocolAudit                       # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
TASK = str(ROOT)
OUT = f"{TASK}/out"
REP = f"{HERE}/reports"
os.makedirs(REP, exist_ok=True)


# --------------------------------------------------------------------- (1) ProtDBench
def run_protdbench():
    table = f"{TASK}/work/panelC_designs.parquet"
    if not os.path.isfile(table):
        raise SystemExit(
            "[protdbench] work/panelC_designs.parquet not found -- build it first:\n"
            "            PROTDBENCH_ROOT=/path/to/protdbench_meta \
python work/panelC_matrix.py"
            "            See DATA_SOURCES.md.")
    df = pd.read_parquet(table)
    df = df[df.targetset == "10_targets"].reset_index(drop=True)
    scores = ["score_iptm", "ptx_iptm_binder", "pLDDT", "pLDDT_MONOMER", "pAE",
              "i_pAE", "i_pTM", "Rg", "alpha", "beta", "loop", "ref_ratio"]
    labels = ["af2_opt_success", "af2_easy_success", "ptx_mini_success",
              "bindcraft_success", "ptx_large_success"]
    scores = [c for c in scores if c in df.columns]
    labels = [c for c in labels if c in df.columns]
    print(f"[protdbench] rows={len(df):,}  scores={len(scores)}  labels={len(labels)}")
    au = ProtocolAudit(df, scores, labels, group_col="Target",
                       n_boot=200, seed=20261014, subsample_per_group=3000)
    au.run()
    au.write(f"{REP}/protdbench", title="Adaptyv/ProtDBench protocol audit card")
    (open(f"{REP}/protdbench/audit_card.md", "w")
     .write(au.markdown("ProtDBench 10_targets — protocol audit card")))
    return au


# ------------------------------------------------------------------------ (2) Adaptyv
def run_adaptyv():
    B = os.environ.get("ADAPTYV_ROOT", "")
    if not B or not os.path.isdir(f"{B}/round2"):
        raise SystemExit(
            "[adaptyv] ADAPTYV_ROOT is not set to the Adaptyv EGFR dataset directory\n"
            "          (expected $ADAPTYV_ROOT/round2/result_summary.csv).\n"
            "          The source dataset is not redistributed here -- see DATA_SOURCES.md.")
    r2 = pd.read_csv(f"{B}/round2/result_summary.csv").rename(columns={"name": "design_name"})
    ds = pd.read_csv(f"{B}/round2/destress_binder_with_egfr.csv")
    D = r2.merge(ds, on="design_name", how="left", suffixes=("", "_ds"))

    def tb(s):
        return s.astype(str).str.strip().str.lower().map({"true": True, "false": False})

    D["L_bind"] = tb(D["binding"])
    D["L_medium_plus"] = D["binding_strength"].isin(["medium", "strong"]).where(D["binding"] != "unknown")
    D["L_strong"] = (D["binding_strength"] == "strong").where(D["binding"] != "unknown")
    D["L_bind_expressed"] = (D["L_bind"] & D["expression"].eq("high")).where(D["binding"] != "unknown")
    scores = ["iptm", "plddt", "pae_interaction", "esm_pll",
              "evoef2_interS_total_per_aa", "evoef2_total_per_aa", "budeff_total_per_aa",
              "budeff_desolvation_per_aa", "rosetta_total_per_aa", "aggrescan3d_avg_value",
              "hydrophobic_fitness_per_aa", "packing_density",
              "ss_prop_loop", "ss_prop_alpha_helix", "isoelectric_point", "num_residues"]
    labels = ["L_bind", "L_medium_plus", "L_strong", "L_bind_expressed"]
    need = [c for c in scores if np.isfinite(pd.to_numeric(D[c], errors="coerce")).any()]
    keep = D[D["L_bind"].notna() & np.isfinite(D[need].astype(float)).all(axis=1)].reset_index(drop=True)
    keep["provenance"] = keep["design_models"].fillna("(unreported)")
    print(f"[adaptyv] rows={len(keep)}  scores={len(need)}  labels={len(labels)}  "
          f"groups={keep.provenance.nunique()}  binders={int(keep.L_bind.sum())}")
    au = ProtocolAudit(keep, need, labels, group_col="provenance", n_boot=400, seed=20261014)
    au.run()
    au.write(f"{REP}/adaptyv", title="Adaptyv EGFR round 2 — protocol audit card")
    return au


if __name__ == "__main__":
    t0 = time.time()
    a = run_adaptyv()
    b = run_protdbench()
    print(f"\ntotal {time.time()-t0:.0f}s")
    for nm, au in (("adaptyv", a), ("protdbench", b)):
        m = au.meta
        print(f"{nm:12s} verdict={m['verdict']:28s} winners={m['n_distinct_winners']:2d}/"
              f"{m['n_comparable_protocols']:2d}  rho_min={m['pairwise_spearman_min']}  "
              f"rho_med={m['pairwise_spearman_median']}  "
              f"tied_med={m.get('median_tied_with_winner')}/{m['n_methods']}")
