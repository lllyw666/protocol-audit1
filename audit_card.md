# Adaptyv EGFR round 2 — protocol audit card

`protocol_audit` v0.1.0 — 378 items · 20 groups · 16 methods · 4 label definitions · 84 protocols.

**Verdict: `FRAGILE+SIGN-DISAGREEMENT`** — 8 distinct winners over 84 comparable protocols; pairwise leaderboard Spearman min -0.5901 / median 0.4201.

## 1. Input audit

| | |
|---|---|
| items | 378 |
| groups | 20 |
| methods (score columns) | 16 |
| label definitions | 4 |
| protocols enumerated | 84 |
| protocols actually comparable | 84 |

**Orientation** (each score flipped so that higher = better vs the primary label; the raw un-oriented AUC is preserved):

| score | coverage | raw AUC | flipped |
|---|---|---|---|
| `iptm` | 100.0% | 0.636 | no |
| `plddt` | 100.0% | 0.656 | no |
| `pae_interaction` | 100.0% | 0.388 | yes |
| `esm_pll` | 100.0% | 0.547 | no |
| `evoef2_interS_total_per_aa` | 100.0% | 0.433 | yes |
| `evoef2_total_per_aa` | 100.0% | 0.392 | yes |
| `budeff_total_per_aa` | 100.0% | 0.438 | yes |
| `budeff_desolvation_per_aa` | 100.0% | 0.559 | no |
| `rosetta_total_per_aa` | 100.0% | 0.384 | yes |
| `aggrescan3d_avg_value` | 100.0% | 0.602 | no |
| `hydrophobic_fitness_per_aa` | 100.0% | 0.660 | no |
| `packing_density` | 100.0% | 0.466 | yes |
| `ss_prop_loop` | 100.0% | 0.553 | no |
| `ss_prop_alpha_helix` | 100.0% | 0.383 | yes |
| `isoelectric_point` | 100.0% | 0.432 | yes |
| `num_residues` | 100.0% | 0.575 | no |

## 2. Winner diversity

| winner | protocols won |
|---|---|
| `isoelectric_point` | 20 |
| `ss_prop_alpha_helix` | 18 |
| `aggrescan3d_avg_value` | 11 |
| `hydrophobic_fitness_per_aa` | 10 |
| `iptm` | 9 |
| `num_residues` | 9 |
| `plddt` | 5 |
| `esm_pll` | 2 |

> 8 distinct winners over 84 comparable protocols. Any claim of the form “method X is best” is therefore a claim about the protocol, not about the method.

## 3. Separability (two-level paired bootstrap)

| protocol | winner | # tied with winner | sample |
|---|---|---|---|
| `pooled_auc[L_bind]` | `hydrophobic_fitness_per_aa` | **16 / 16** | 100% |
| `pooled_ap[L_bind]` | `ss_prop_alpha_helix` | **16 / 16** | 100% |
| `pooled_ef1[L_bind]` | `num_residues` | **16 / 16** | 100% |
| `pooled_ef5[L_bind]` | `ss_prop_alpha_helix` | **16 / 16** | 100% |
| `pooled_ef10[L_bind]` | `iptm` | **16 / 16** | 100% |
| `pooled_ef20[L_bind]` | `iptm` | **16 / 16** | 100% |
| `pooled_ndcg10[L_bind]` | `ss_prop_alpha_helix` | **16 / 16** | 100% |
| `macro_ef1[L_bind]` | `num_residues` | **16 / 16** | 100% |
| `macro_ef10[L_bind]` | `ss_prop_alpha_helix` | **16 / 16** | 100% |
| `macro_ef5[L_bind]` | `ss_prop_alpha_helix` | **16 / 16** | 100% |
| `macro_ndcg10[L_bind]` | `ss_prop_alpha_helix` | **16 / 16** | 100% |
| `macro_ef20[L_bind]` | `ss_prop_alpha_helix` | **16 / 16** | 100% |
| `median_ef10[L_bind]` | `ss_prop_alpha_helix` | **16 / 16** | 100% |
| `median_ef5[L_bind]` | `ss_prop_alpha_helix` | **16 / 16** | 100% |
| `median_ap[L_bind]` | `num_residues` | **16 / 16** | 100% |

> Median protocol has **16 of 16** methods statistically tied with its winner. If that number is close to the method count, the leaderboard carries no ranking information at all at this sample size.

## 4. Stability of individual methods

| method | mean rank | rank SD | worst-case rank range |
|---|---|---|---|
| `aggrescan3d_avg_value` | 7.0 | 4.0 | 15 |
| `esm_pll` | 10.0 | 3.9 | 14 |
| `isoelectric_point` | 5.3 | 3.9 | 14 |
| `hydrophobic_fitness_per_aa` | 5.3 | 3.7 | 14 |
| `iptm` | 6.7 | 3.4 | 14 |
| `budeff_desolvation_per_aa` | 12.2 | 2.6 | 12 |
| `evoef2_interS_total_per_aa` | 10.7 | 2.4 | 11 |
| `evoef2_total_per_aa` | 9.7 | 2.4 | 12 |

## 5. What to report

- The **number of comparable protocols** and the **winner set**, not a single ranking.

- The **paired-bootstrap tied set** for the top method, not a bare `#1`.

- Any protocol in which the leaderboard Spearman is < 0 against the headline protocol — those pairs are the ones a reviewer will find.
