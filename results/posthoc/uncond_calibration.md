## Unconditional entropy-calibration benchmark (EVAL.md Addendum B)

Signed offset: mean(H_model − H_engine) over all voxels, 95% t-interval across the 8 conditions/env; negative = under-dispersed. Verdict compares |signed| (resp. MAE) to the engine split-half band.

| environment | signed offset (bits) ± CI | MAE (bits) ± CI | band (MAE) | verdict |
|---|---|---|---|---|
| lobe | -0.0205 ± 0.0261 | 0.0580 ± 0.0441 | 0.0433 | near |
| channel:PV_SHOESTRING | -0.0550 ± 0.0335 | 0.1091 ± 0.0135 | 0.0734 | near |
| channel:CB_LABYRINTH | -0.1156 ± 0.0405 | 0.1421 ± 0.0331 | 0.0621 | outside |
| channel:CB_JIGSAW | -0.1479 ± 0.0370 | 0.1602 ± 0.0342 | 0.0589 | outside |
| channel:SH_DISTAL | -0.1989 ± 0.0425 | 0.2196 ± 0.0397 | 0.0619 | outside |
| channel:SH_PROXIMAL | -0.1932 ± 0.0440 | 0.2209 ± 0.0446 | 0.0642 | outside |
| channel:MEANDER_OXBOW | -0.1017 ± 0.0499 | 0.1342 ± 0.0361 | 0.0516 | outside |
| delta | -0.1015 ± 0.0230 | 0.1253 ± 0.0118 | 0.0511 | outside |
| **pooled (64 conds)** | -0.1168 | 0.1462 | 0.0583 | outside |

Estimator note: plug-in entropy bias differs by ≈ 0.002 bits between K = 128 (model) and N = 256 (engine) ensembles — negligible against the observed offsets.
