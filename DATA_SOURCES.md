# Source data

This repository ships only **derived** results and code. The four third-party datasets the
audit re-analyses are not redistributed here; obtain each from its own source and point the
code at it with the environment variable listed below. Please observe each dataset's own
licence and terms of use.

## 1. Overath et al. (2025) - wet-lab binder meta-analysis

*Reference.* Overath, M. D., Rygaard, A. S. H., Jacobsen, C. P., et al. (2025). Predicting
experimental success in de novo binder design: a meta-analysis of 3,766 experimentally
characterised binders. bioRxiv preprint. doi:10.1101/2025.08.14.670059

*Used by.* Panel A (14 protocols x 243 methods on 3,669 designs), and the small-scale arm of
Panel B. The derived matrix and bootstrap tensor are in `out/panelA_*` and
`out/panelB_scale_small_*`, and are sufficient to reproduce every panel-A number without the
source data.

*Per-design table.* Two scripts need row-level per-design values, which this repository does
not ship: `work/panelA_figs_vec.py` (Figures 1-6) and `work/number_audit.py` (one of its 35
checks). Set `OVERATH_FEATURES` to a table with one row per design and the columns
`label`, `target` plus the per-design metric columns:

```bash
export OVERATH_FEATURES=/path/to/overath_rankfeatures.parquet
python work/panelA_figs_vec.py     # regenerates Figures 1-6 (out/figs/F1..F6)
python work/number_audit.py        # all 35 checks
```

## 2. Cao et al. (2022) - large-scale computational design set

*Reference.* Cao, L., Coventry, B., Goreshnik, I., et al. (2022). Design of protein-binding
proteins from the target structure alone. *Nature*, 605(7910), 551-560.
doi:10.1038/s41586-022-04654-9

*Used by.* Panel B, the 74-fold scale comparison (236,246 designs). Derived matrices:
`out/panelB_*`.

## 3. ProtDBench (Liu et al., 2026) - generative benchmark

*Reference.* Liu, C., Ren, M., Guan, J., et al. (2026). ProtDBench: a unified benchmark of
protein binder design and evaluation. arXiv preprint 2605.04118.
doi:10.48550/arXiv.2605.04118

*Used by.* Panel C (four competing success definitions on one dataset = the zero-external-noise
control), and the ProtDBench demonstration in `demo_runs.py`.

*How to point the code at it.* Set `PROTDBENCH_ROOT` to the cloned `protdbench_meta` directory,
then rebuild the panel-C table, which is the input the demonstration reads:

```bash
export PROTDBENCH_ROOT=/path/to/protdbench_meta
python work/panelC_matrix.py            # writes work/panelC_designs.parquet
```

The intermediate `work/panelC_designs.parquet` is not shipped, because it is a row-level
extract of the source data. Derived aggregates (`out/panelC_*`) are shipped.

## 4. Adaptyv Bio EGFR binder competition (Cotet et al., 2025) - prospective SPR

*Reference.* Cotet, T.-S., Krawczuk, I., Stocco, F., et al. (2025). Crowdsourced protein
design: lessons from the Adaptyv EGFR binder competition. bioRxiv preprint.
doi:10.1101/2025.04.17.648362

*Used by.* Panel D, the prospective arbitration arm (real SPR measurements, so the closest
available analogue of the deployment setting), and the Adaptyv demonstration in `demo_runs.py`.
Figure 13b additionally reads two raw round-2 files from the same directory.

*How to point the code at it.* Set `ADAPTYV_ROOT` to the dataset directory that contains
`round2/result_summary.csv`:

```bash
export ADAPTYV_ROOT=/path/to/adaptyv_egfr
python out/protocol_audit/demo_runs.py
python work/panelD_figs_vec.py          # regenerates Figures 12 and 13 sources
```

The derived tables for Panel D are in `out/panelD_*`, and the audit cards under
`out/protocol_audit/reports/adaptyv/` were generated from them.
