# `protocol_audit` -- an evaluation-protocol audit toolkit

**In one line**: given a table of designs x candidate scores, it emits an **audit card**
telling you how much the leaderboard depends on the evaluation protocol, and which apparent
"winners" are actually statistical noise.

> Its role is the counterpart of PoseBusters. PoseBusters checks whether a **prediction**
> satisfies basic chemistry; `protocol_audit` checks whether an **evaluation protocol**
> satisfies basic statistics. They are complementary, and neither of them makes predictions.

- Version: `0.1.0`
- Requirements: `numpy / pandas / scipy / scikit-learn` (see `requirements.txt`)
- Entry points: `protocol_audit.py` (library + CLI), `demo_runs.py` (two real-dataset demos)

---

## 1. The problem it addresses

A leaderboard ("which in-silico metric / generator / model is best?") is **not a property of
the data, it is a property of the evaluation protocol**. A protocol is built from these axes:

| Axis | Values |
|---|---|
| aggregation | `pooled` (everything together) / `macro` (average within group) / `median` (group median) |
| label definition | any binary column you pass in ("binds" / "above medium" / "zero-noise success" ...) |
| metric | `auc` `ap` `ef1` `ef5` `ef10` `ef20` `ndcg10` |
| grouping | target / design source / batch |

Change one axis and the winner can change -- **even when the labels carry no noise at all**.
The toolkit turns that observation into a check anyone can re-run.

---

## 2. Three checks that are almost always skipped

1. **Comparability (coverage).** If only 3 of 40 methods have a value under some protocol, that
   protocol cannot rank anything. The toolkit measures per-protocol method coverage and
   **excludes the incomparable protocols explicitly**, instead of silently averaging them. The
   same rule applies to columns: score/label columns with coverage below 50% are dropped and
   recorded in `dropped.csv` (**this is the trap where ProtDBench `bindcraft_success` covers
   only 1/7 of the designs**).
2. **Orientation.** A score with an unknown sign is not a score. Each column is oriented
   against the primary label (so that AUC >= 0.5), and **both the raw and the flipped state are
   recorded** in `orientation.csv`, so a reader can see which columns were reversed.
3. **Separability.** A `#1` without a paired test means nothing. For each protocol the toolkit
   runs a **two-level paired bootstrap** (resampling groups, then items within groups) and
   counts the methods statistically indistinguishable from the winner as ties. **Marginal
   confidence intervals are wrong here**: the methods are evaluated on the same items, so the
   correlation has to be preserved.

---

## 3. Quick start

```python
import pandas as pd
from protocol_audit import ProtocolAudit

df = pd.read_parquet("designs.parquet")          # one row per design
au = ProtocolAudit(
        df,
        score_cols=["plddt", "iptm", "pae_interaction", "num_residues"],
        label_cols=["binds", "binds_strongly"],  # each binary column = one label definition
        group_col="target",
        n_boot=200, seed=0,               # n_boot=0 skips the bootstrap
        orient="primary",                 # or "none"
        subsample_per_group=None)         # set 3000 for large datasets
au.run()
au.write("reports/mydata", title="My dataset audit")
print(au.markdown())
```

CLI:

```bash
python3 protocol_audit.py --table designs.parquet \
    --scores plddt iptm pae_interaction --labels binds binds_strong \
    --group target --n-boot 200 --out reports/mydata
```

Already have a leaderboard matrix (method x protocol) and only want the audit:

```python
from protocol_audit import audit_from_matrix
card = audit_from_matrix(matrix_df, boot=boot_dict)   # boot optional
print(card["meta"])
```

---

## 4. Output

| File | Contents |
|---|---|
| `matrix.csv` | the full method x protocol value table |
| `coverage.csv` | how many methods have a value under each protocol + comparability |
| `agreement.csv` | pairwise rank Spearman between protocols |
| `top1.csv` | the winner under each protocol |
| `instability.csv` | per-method mean rank / rank SD / rank range |
| `separability.csv` | per-protocol count of methods tied with the winner (paired bootstrap) |
| `orientation.csv` | per score column: coverage, raw AUC, whether it was flipped |
| `dropped.csv` | columns dropped for coverage or degeneracy, with the reason |
| `boot.npz` | the raw bootstrap tensors (for reproducible secondary analysis) |
| `audit.json` | machine-readable audit summary + `verdict` |
| `audit_card.md` | human-readable audit card |

`verdict` values:

| verdict | Meaning |
|---|---|
| `STABLE` | every comparable protocol points to one winner |
| `MODERATE` | number of winners <= max(2, 20% of the methods) |
| `FRAGILE` | more winners than that: the leaderboard is mostly decided by the protocol |
| `+SIGN-DISAGREEMENT` | appended when some protocol pair has rank Spearman < 0 (a reversal) |

---

## 5. Two real demonstrations (`demo_runs.py`)

```
$ python3 demo_runs.py            # about 1 minute
adaptyv      verdict=FRAGILE+SIGN-DISAGREEMENT  winners= 8/84  rho_min=-0.590  rho_med=0.420  tied_med=16/16
protdbench   verdict=FRAGILE                    winners= 5/63  rho_min=-0.024  rho_med=0.816  tied_med=2/11
```

| Demo | Data | Items | Groups | Scores | Labels | Protocols | Winners | rho_min | rho_med | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Adaptyv EGFR r2 (prospective, real SPR) | 378 designs | design source (20) | 16 descriptors | 4 definitions | 84 | **8** | **-0.590** | 0.420 | FRAGILE + sign disagreement |
| ProtDBench 10_targets (zero label noise) | 30,000 designs (<=3,000 per target) | targets (10) | 12 structural metrics | 3 usable definitions | 63 | **5** | -0.024 | 0.816 | FRAGILE |

A few audit findings from those reports that are worth stating:

- **ProtDBench**: `bindcraft_success` / `ptx_large_success` / `ptx_iptm_binder` cover only
  **18.6%** of the designs, so they are dropped automatically and recorded. Without that step
  you would compare a definition that exists for 1/7 of the generators against all methods.
- **Adaptyv**: **8 of the 16** descriptors have a raw AUC < 0.5 (`pae_interaction` 0.388,
  `rosetta_total_per_aa` 0.384, `ss_prop_alpha_helix` 0.383, `evoef2_total_per_aa` 0.392,
  `isoelectric_point` 0.432, `evoef2_interS_total_per_aa` 0.433, `budeff_total_per_aa` 0.438,
  `packing_density` 0.466), so without orientation you get the absurd "larger is worse"
  leaderboard, and **half** of the descriptors point the wrong way. After orientation,
  `pae_interaction` still ranks near the top under several protocols.
- **Adaptyv**: over 378 entries with 53 positives, **the winner under almost every protocol is
  statistically tied with all 16 methods** (`median_tied_with_winner = 16/16`). That number is
  itself the proof that this dataset cannot separate any of the metrics, which is firmer
  ground than any sentence about sample size.

---

## 6. Honest limitations

| Item | Note |
|---|---|
| **Orientation uses the primary label** | `orient="primary"` orients against the first label column, which is **mild information use**. A production pipeline should use prior directions or an independent calibration set; `orientation.csv` keeps the raw AUC so the choice can be reviewed. |
| **`ef1` (top 1%) is very unstable at small n** | 378 entries put only 4 designs in the top 1%. Prefer `ef5`/`ef10` at small n, or use `ndcg10` directly. |
| **The two-level bootstrap assumes exchangeable groups** | With few groups (e.g. 7) the intervals come out too narrow. The toolkit does not enforce a groups >= 10 rule, but `n_groups` is written into `audit.json`. |
| **No multiple-comparison correction** | The per-protocol tie counts have no family-wise error control; the `+SIGN-DISAGREEMENT` flag only says that a reversal exists, it is not a test. |
| **The `macro` group filter** | Default: >=10 items and >=2 positives within a group; both thresholds are themselves protocol choices, and changing them changes the result, so they must be reported. |
| **Not done** | No feature selection, no model training, no significance correction, no missing-value imputation. |

---

## 7. Re-running

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/mpl python3 out/protocol_audit/demo_runs.py         # ~1 min -> out/protocol_audit/reports/
MPLCONFIGDIR=/tmp/mpl python3 out/protocol_audit/cross_panel_card.py  # ~5 s   -> out/protocol_audit/cross_panel_card.{csv,md}
```

Both demonstrations read third-party source datasets that are not redistributed with this
repository; see `DATA_SOURCES.md` for the environment variables they expect.
