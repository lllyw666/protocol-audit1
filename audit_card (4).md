# ProtDBench 10_targets — protocol audit card

`protocol_audit` v0.1.0 — 30,000 items · 10 groups · 11 methods · 3 label definitions · 63 protocols.

**Verdict: `FRAGILE`** — 5 distinct winners over 63 comparable protocols; pairwise leaderboard Spearman min -0.0239 / median 0.8155.

> subsampled to <= 3000 items per group (seeded), n = 30000

## 1. Input audit

| | |
|---|---|
| items | 30,000 |
| groups | 10 |
| methods (score columns) | 11 |
| label definitions | 3 |
| protocols enumerated | 63 |
| protocols actually comparable | 63 |

**Dropped columns** (silently averaging over these is the most common audit failure):

| kind | name | reason |
|---|---|---|
| label | `bindcraft_success` | coverage 18.6% |
| label | `ptx_large_success` | coverage 18.6% |
| score | `ptx_iptm_binder` | coverage 18.6% |

**Orientation** (each score flipped so that higher = better vs the primary label; the raw un-oriented AUC is preserved):

| score | coverage | raw AUC | flipped |
|---|---|---|---|
| `score_iptm` | 100.0% | 0.745 | no |
| `pLDDT` | 100.0% | 0.884 | no |
| `pLDDT_MONOMER` | 100.0% | 0.809 | no |
| `pAE` | 100.0% | 0.017 | yes |
| `i_pAE` | 100.0% | 0.023 | yes |
| `i_pTM` | 100.0% | 0.946 | no |
| `Rg` | 100.0% | 0.522 | no |
| `alpha` | 100.0% | 0.715 | no |
| `beta` | 100.0% | 0.325 | yes |
| `loop` | 100.0% | 0.289 | yes |
| `ref_ratio` | 100.0% | 0.553 | no |

## 2. Winner diversity

| winner | protocols won |
|---|---|
| `pAE` | 32 |
| `score_iptm` | 22 |
| `i_pAE` | 4 |
| `i_pTM` | 4 |
| `alpha` | 1 |

> 5 distinct winners over 63 comparable protocols. Any claim of the form “method X is best” is therefore a claim about the protocol, not about the method.

## 3. Separability (two-level paired bootstrap)

| protocol | winner | # tied with winner | sample |
|---|---|---|---|
| `pooled_ndcg10[af2_easy_success]` | `alpha` | **11 / 11** | 100% |
| `median_ef1[af2_easy_success]` | `score_iptm` | **6 / 11** | 55% |
| `macro_ndcg10[af2_easy_success]` | `i_pTM` | **6 / 11** | 55% |
| `median_ndcg10[af2_easy_success]` | `i_pTM` | **6 / 11** | 55% |
| `median_ef5[af2_easy_success]` | `i_pTM` | **4 / 11** | 36% |
| `median_ef10[ptx_mini_success]` | `score_iptm` | **4 / 11** | 36% |
| `median_ef20[ptx_mini_success]` | `score_iptm` | **4 / 11** | 36% |
| `macro_ef1[af2_easy_success]` | `i_pTM` | **3 / 11** | 27% |
| `median_ef20[af2_opt_success]` | `pAE` | **3 / 11** | 27% |
| `pooled_ef20[af2_opt_success]` | `pAE` | **3 / 11** | 27% |
| `pooled_ndcg10[af2_opt_success]` | `i_pAE` | **3 / 11** | 27% |
| `median_ef5[af2_opt_success]` | `pAE` | **3 / 11** | 27% |
| `median_ndcg10[af2_opt_success]` | `pAE` | **3 / 11** | 27% |
| `median_ef10[af2_opt_success]` | `pAE` | **3 / 11** | 27% |
| `pooled_ef20[af2_easy_success]` | `pAE` | **3 / 11** | 27% |

> Median protocol has **2 of 11** methods statistically tied with its winner. If that number is close to the method count, the leaderboard carries no ranking information at all at this sample size.

## 4. Stability of individual methods

| method | mean rank | rank SD | worst-case rank range |
|---|---|---|---|
| `pLDDT` | 5.3 | 2.2 | 8 |
| `beta` | 8.5 | 2.0 | 6 |
| `score_iptm` | 3.3 | 1.9 | 7 |
| `alpha` | 7.1 | 1.4 | 8 |
| `loop` | 7.4 | 1.4 | 8 |
| `i_pAE` | 2.7 | 1.1 | 6 |
| `ref_ratio` | 10.1 | 0.8 | 4 |
| `Rg` | 10.1 | 0.8 | 3 |

## 5. What to report

- The **number of comparable protocols** and the **winner set**, not a single ranking.

- The **paired-bootstrap tied set** for the top method, not a bare `#1`.

- Any protocol in which the leaderboard Spearman is < 0 against the headline protocol — those pairs are the ones a reviewer will find.
