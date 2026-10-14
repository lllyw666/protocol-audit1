# Cross-panel protocol audit card (four panels, one code path)

Recomputed by `protocol_audit.audit_from_matrix` from the leaderboard matrices already computed for the four panels.

| Panel | Data | Object ranked | Protocols (comparable) | Winners | Min rank Spearman | Median | Sign-reversed pairs | verdict |
|---|---|---|---|---|---|---|---|---|
| **A** | Overath 3,669 · wet-lab labels | 243 | 14/14 | **7** | **+0.639** | 0.859 | 0.0% | `MODERATE` |
| **B** | cao 236,246 · wet-lab labels | 37 | 14/14 | **6** | **+0.226** | 0.682 | 0.0% | `MODERATE` |
| **C1** | ProtDBench 10_targets · DETERMINISTIC labels | 7 | 12/12 | **5** | **-0.964** | 0.486 | 39.4% | `FRAGILE+SIGN-DISAGREEMENT` |
| **C2** | ProtDBench cao5_targets · DETERMINISTIC labels | 7 | 12/12 | **4** | **-0.771** | 0.571 | 24.2% | `FRAGILE+SIGN-DISAGREEMENT` |
| **D** | Adaptyv EGFR r2 · real SPR (prospective) | 35 | 48/48 | **11** | **-0.403** | 0.473 | 6.1% | `FRAGILE+SIGN-DISAGREEMENT` |

## Winner sets

- **A** (14 comparable protocols, 7 winners): af3_ipSAE_d0chn, af3_ipSAE_d0dom, af3_ipSAE_min, af3_iptm_avg, af3_max_pae_contact, af3_min_pae_contact, af3_ptm_avg
- **B** (14 comparable protocols, 6 winners): colabfold_ipae, consensus_all, consensus_iptm, protenix_iptm, protenix_mini_ipae, protenix_plddt
- **C1** (12 comparable protocols, 5 winners): BindCraft, Boltzgen, Odesign, PXDesign, Protpardelle-1c
- **C2** (12 comparable protocols, 4 winners): BindCraft, Odesign, PXDesign, RFDiffusion-3
- **D** (48 comparable protocols, 11 winners): aggrescan3d_avg_value, consensus_all, hydrophobic_fitness_per_aa, iptm, isoelectric_point, num_residues, pct_aggrescan3d_avg_value, pct_esm_pll, pct_num_residues, plddt, ss_prop_alpha_helix

## How to read

- `STABLE` = every comparable protocol points to the same winner; `FRAGILE` = more winners than max(2, 20% of the objects ranked).
- `+SIGN-DISAGREEMENT` = at least one protocol pair has a rank Spearman < 0 (the leaderboard reverses).
- **None** of the four panels is `STABLE`. All four reach the conclusion that the leaderboard is a property of the protocol, before any null-hypothesis test.