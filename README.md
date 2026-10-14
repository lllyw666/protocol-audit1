# protocol-audit

Code and derived results for the manuscript *How evaluation protocol shapes de novo binder-design leaderboards* (Lin, 2026; submitted to
*Computational Biology and Chemistry*).

The paper asks how much of a de novo binder-design leaderboard is set by the **evaluation
protocol** rather than by the designs, and answers it by re-ranking the same predictions
under a grid of alternative protocols on four independent source datasets. This repository
contains the audit toolkit, the derived per-panel results, the bootstrap tensors, and the
scripts that generate Figures 1-15.

## What is here

| Path | Contents |
|---|---|
| `out/protocol_audit/` | the `protocol_audit` toolkit: protocol primitives, an audit-card generator, the cross-panel card, and two real-data demonstrations |
| `out/protocol_audit/reports/` | generated audit cards, one per analysis row (`adaptyv/`, `protdbench/`) |
| `out/panel*.csv`, `out/panel*.npz` | derived per-panel result tables and the bootstrap tensors |
| `out/figs/` | Figures 1-15, vector PDF and 300-dpi PNG |
| `out/README.md` | index of the derived results |
| `work/` | analysis and figure-generation scripts |
| `work/number_audit.py` | the number-recognition audit: every printed number traced to its source |
| `DATA_SOURCES.md` | how to obtain the four third-party source datasets |

`out/` holds results and `work/` holds the code that produces them.

## Quick start

```bash
pip install -r requirements.txt

# regenerate the cross-panel audit card (out/protocol_audit/cross_panel_card.csv + .md)
python out/protocol_audit/cross_panel_card.py

# the toolkit's own CLI
python out/protocol_audit/protocol_audit.py --help
```

`out/protocol_audit/demo_runs.py` demonstrates the toolkit on two real datasets end to end.
Both are third-party and are **not** redistributed here; it needs two environment variables
and says so if they are missing. See `DATA_SOURCES.md`.

```bash
export ADAPTYV_ROOT=/path/to/adaptyv_egfr     # expects $ADAPTYV_ROOT/round2/result_summary.csv
python out/protocol_audit/demo_runs.py
```

The same applies to `work/panelA_figs_vec.py` and `work/number_audit.py`, which read the
per-design Overath feature table (`OVERATH_FEATURES`); without it the audit skips the one
check that needs it and reports the rest.

## Headline results

All of the following are reproducible from the files in `out/`.

* **14 protocols on 3,669 wet-lab designs produce 7 different first-place methods**, and the
  top two are indistinguishable under all 14.
* **Scaling 74-fold does not settle it.** On 236,246 designs the pooled-label family
  collapses toward consensus (tied winners 32 -> 3) while the per-target enrichment family
  stays contested (19-24 near-ties).
* **Zero label noise is not enough.** On a benchmark whose success labels are deterministic
  functions of the designs' own predicted structures, disagreement does not vanish
  (minimum Spearman -0.964 over 7 generators).
* **Protocol stability does not predict prospective performance.** On 48 protocols over real
  prospective SPR data, 11 distinct winners emerge, and the association between
  protocol-stability and prospective accuracy is absent (rho = 0.078, p = 0.658).
* The toolkit's audit card grades **no** analysis row STABLE.

## Licence

Code: MIT (`LICENSE`). Derived data, figures and documentation: CC BY 4.0 (`LICENSE-DATA`).

## Citing

See `CITATION.cff`. If you use the toolkit, please cite this repository and the paper.

## Third-party data

The four source datasets (Overath et al. 2025; Cao et al. 2022; Liu et al. 2026; Cotet et al.
2025) are not redistributed here. `DATA_SOURCES.md` gives their identifiers and the
environment variables this code expects.
