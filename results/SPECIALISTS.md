# Foundation vs specialist comparison — results (EVAL.md Addendum E)

Status 2026-07-28 03:00 CDT. Protocol: EVAL.md Addendum E (frozen before
training) + Amendments E-1/E-2. Matched-exposure (40-epoch) specialist
results are final. The pre-registered anti-undertraining rule (E.3)
**fired for both environments** (best validation loss at the final
checkpoint in both cases). The PV_SHOESTRING 80-epoch extension is
**complete and scored** (below); the delegated truncation rule never
triggered (validation still improving at every checkpoint through 80),
so the run used its full registered budget. The lobe extension is
queued (Amendment E-2 update) and pending launch. All specialist artifacts in
`results/specialists/`; assets: `comparison.parquet`, `comparison.md`,
figures (PDF+PNG), `compartmentalization.{md,json}`, `parity.json`.

## Main comparison table

Split-half band and foundation rows are copied verbatim from the frozen
master table (`RESULTS.md` / `results/metrics.parquet`); specialist
verdicts are computed against the frozen band (E.5). Specialists are the
epoch-40 EMA checkpoints selected by validation argmin (E.3; both were
also the final checkpoints, which is what fired the extension rule).

| environment | source | \|dNTG\| | variogram MAE/sill | connectivity MAE | geobody W1 (log10) | \|d largest frac\| | well mismatch % |
|---|---|---|---|---|---|---|---|
| channel:PV_SHOESTRING | split-half band | 0.0050 | 0.0115 | 0.0014 | 0.0125 | 0.0117 | n/a |
| channel:PV_SHOESTRING | foundation | 0.0069 (near) | 0.0202 (near) | 0.0034 (outside†) | 0.0210 (near) | 0.0443 (outside) | 0.000 (inside) |
| channel:PV_SHOESTRING | specialist (40 ep) | 0.0398 (outside) | 0.1067 (outside) | 0.1326 (outside) | 0.0670 (outside) | 0.0806 (outside) | n/a\* |
| channel:PV_SHOESTRING | specialist (ext, union argmin: 80-run ep 80) | 0.0438 (outside) | 0.1023 (outside) | 0.0100 (outside) | 0.1581 (outside) | 0.0638 (outside) | n/a\* |
| lobe | split-half band | 0.0253 | 0.0065 | 0.0274 | 0.0305 | 0.0386 | n/a |
| lobe | foundation | 0.0026 (inside) | 0.0121 (near) | 0.0356 (near) | 0.0588 (near) | 0.0227 (inside) | 0.000 (inside) |
| lobe | specialist (40 ep) | 0.0271 (near) | 0.0895 (outside) | 0.0223 (inside) | 0.2473 (outside) | 0.0284 (inside) | n/a\* |

\* Well-conditioned generation is out of scope for specialists (E.1):
exactitude is guaranteed by the hard-replacement pipeline independent of
model weights, so the column cannot discriminate the models.
† Band saturation in the master table (percolating sand phase); see the
master-table footnote.

## Parity outcome (pre-registered criterion, E.5)

The foundation claim is substantiated on an environment iff the
foundation row's verdict tier (inside ≻ near ≻ outside, same frozen
band) is at least as good as the specialist row's in every metric
column.

- **channel:PV_SHOESTRING: SUBSTANTIATED** — against both specialist
  variants. Foundation tier ≥ specialist tier in all five columns for
  the matched-exposure specialist AND for the extension specialist
  (union-argmin selection, E.3: 80-run epoch 80, official val 0.2085 vs
  the 40-run's best 0.4307). Foundation absolute values are also
  smaller in every column vs the extension specialist (|dNTG| 6.3×,
  variogram 5.1×, connectivity 2.9×, geobody W1 7.5×, largest frac
  1.4×).
- **lobe: NOT substantiated.** The specialist's connectivity-MAE tier
  (inside, 0.0223) exceeds the foundation's (near, 0.0356). Foundation
  tiers are strictly better in |dNTG|, variogram MAE/sill, and geobody
  W1; the |d largest frac| tier is equal (both inside).

## Per-column summary (absolute values, compared descriptively)

- **|dNTG|.** PV: reference mean NTG 0.2249; foundation 0.2318
  (|d| = 0.0069); specialist 0.1850 (|d| = 0.0398, under-produces sand).
  Lobe: reference 0.5028; foundation 0.5054 (0.0026); specialist 0.4757
  (0.0271). Foundation closer in both environments; per-volume NTG
  histograms in `ntg_hist_{pv_shoestring,lobe}.{pdf,png}`.
- **Variogram MAE/sill.** Foundation 0.0202 (PV) and 0.0121 (lobe);
  specialist 0.1067 and 0.0895 — 5.3× and 7.4× larger respectively;
  overlays with the split-half band in
  `variogram_overlay_*.{pdf,png}` (x/y/z panels).
- **Connectivity MAE.** PV: foundation 0.0034 (outside on a saturated
  band of 0.0014), specialist 0.1326 — 39× larger. Lobe: foundation
  0.0356 (near), specialist 0.0223 (inside) — the one column where the
  specialist's tier is better; overlays in `conn_overlay_*.{pdf,png}`.
- **Geobody W1 (log10).** Foundation 0.0210 (PV) / 0.0588 (lobe);
  specialist 0.0670 / 0.2473. Foundation closer in both; body-size CDFs
  in `geobody_cdf_*.{pdf,png}`.
- **|d largest frac|.** PV: foundation 0.0443 (outside), specialist
  0.0806 (outside; median largest-body fraction 0.838 vs reference
  0.995). Lobe: foundation 0.0227 (inside), specialist 0.0284 (inside).

## Secondary diagnostic (pre-registered, E.6)

Question: does the specialist reproduce the foundation model's
over-connectivity signature (elevated largest-body fraction, γ_z
plateau undershoot, under-reproduced compartmentalization tail)?
**Answer: no — both specialists deviate in the opposite direction
(over-fragmentation).** From `compartmentalization.md`:

| environment | source | frac τ_z<0.99 [95% CI] | floor-to-surface span frac [95% CI] | largest-body frac median [IQR] |
|---|---|---|---|---|
| channel:PV_SHOESTRING | reference | 0.287 [0.250, 0.328] | 0.740 [0.701, 0.776] | 0.995 [0.903, 0.998] |
| channel:PV_SHOESTRING | foundation | 0.270 [0.233, 0.310] | 0.852 [0.818, 0.880] | 0.994 [0.980, 0.998] |
| channel:PV_SHOESTRING | specialist (40 ep) | 1.000 [0.993, 1.000] | 0.783 [0.746, 0.817] | 0.838 [0.764, 0.912] |
| lobe | reference | 0.885 [0.854, 0.910] | 0.334 [0.294, 0.376] | 0.292 [0.172, 0.730] |
| lobe | foundation | 0.822 [0.787, 0.853] | 0.373 [0.332, 0.416] | 0.295 [0.170, 0.830] |
| lobe | specialist (40 ep) | 0.947 [0.924, 0.964] | 0.336 [0.296, 0.378] | 0.337 [0.208, 0.752] |

The extension run reverses this reading for PV. The converged extension
specialist reproduces — and exceeds — the over-connectivity signature:
floor-to-surface span fraction 0.943 [0.920, 0.960] vs the foundation's
0.852 and the reference's 0.740; largest-body-fraction median 0.990
(reference 0.995); compartmentalized fraction 0.398 vs reference 0.287
(the 40-ep specialist's 1.000 was an undertraining artifact). Under the
pre-registered E.6 reading, the over-connectivity bias is therefore
**attributable to architecture or CFG rather than to weight sharing**:
a single-environment model with identical architecture and inference
settings, trained to a lower validation loss than any other candidate,
shows the same bias more strongly. The 40-ep specialists' opposite
(over-fragmenting) deviation is explained by their EMA undertraining.

## Anti-undertraining rule outcomes (E.3) and EMA context

- Validation curves (paired draws, seed 20260901, K=4):
  PV 1.397 → 0.431 (epochs 5→40, no plateau); lobe 1.271 → 0.211
  (epochs 5→40, decelerating, no plateau). Foundation validation loss
  0.1385 for scale. **Best = final for both ⇒ the rule fired for both.**
- EMA context (E.7): decay 0.9999 ⇒ time constant 10,000 steps. PV ran
  9,360 steps (fraction of the EMA average predating training ≈ 39%);
  lobe 18,720 steps (≈ 15%); foundation 93,720 (≈ 0.01%).
- Branch taken: PV 80-epoch from-scratch retrain (4× GH200, Amendment
  E-2) COMPLETE. Official paired validation on its 16 checkpoints:
  1.348 (ep 5) → 0.236 (70) → 0.222 (75) → 0.2085 (80), monotone;
  the delegated truncation rule never triggered. Union argmin selected
  the 80-run epoch-80 checkpoint (md5
  `7a2c6b07896a8c0c65593ce3273ed345`; run manifest in
  `/scratch/08405/ilgar/specialist_runs/pv_shoestring_80ep/`, world
  size 4, global batch 384 = 4×96, ResFlow commit `e7d0552`).
  Validation was still improving at the terminal registered budget
  (epoch 80); no further extension is registered — disclosed. At 80
  epochs the run spans 18,720 steps ≈ 1.9 EMA time constants (≈ 15%
  stale fraction), the same EMA position as the lobe 40-ep run.
- Lobe 80-epoch retrain queued (Amendment E-2 update), pending launch.

## Run manifests and provenance

- Training wrappers and pipeline: ResFlow repo commit `fda489d`
  (+ `dc49722`-referenced protocol amendments in ResBench).
  No file under `resflow/` or `resbench/` modified.
- PV specialist (40 ep): run dir
  `/scratch/08405/ilgar/specialist_runs/pv_shoestring`, seed 8101,
  1× GH200 (epochs 1–5 on c608-122, 6–40 on c608-061, pure resume),
  scored EMA checkpoint `inference_epoch040.pt`
  md5 `0e02ee0a48122a299866009b4a557961`.
- Lobe specialist (40 ep): run dir
  `/scratch/08405/ilgar/specialist_runs/lobe`, seed 8102, epochs 1–5 on
  1× GH200, 6–40 as 2× GH200 DDP (Amendment E-1, loss-continuous
  resume), scored EMA checkpoint `inference_epoch040.pt`
  md5 `dfad1933e663b4e19ed098544bfa7435`.
- Both runs: global batch 384, peak LR 3.4641e-3, 2-epoch warmup +
  cosine (T_max 38), EMA 0.9999/step, CFG drop 0.1, foundation
  `cond_stats.npz` reused (md5 `76f805f8ab1a2665096e8bf3ff8b37ac`),
  loader seed 42; torch 2.10.0+cu126; full resolved configs in each run
  dir's `run_manifest.json`.
- Generation: `generate_ensembles.py` unchanged, `--self-test` passed
  (bit-identity vs library sampler) for both; Table 6 settings read from
  `figure2.py` (Euler, NFE 50, CFG 3.0, binarize x > 0); noise seeds =
  (manifest `fresh_noise_seed` + 500000)·1000; outputs
  `resbench_eval/specialist_{pv_shoestring,lobe}/ensemble_a/` with
  generation manifests alongside.
- Scoring: `resbench.run` unchanged, reference and split-half band
  identical to the master run; specialist verdicts recomputed against
  the frozen band (E.5).

## Anomalies

- The anti-undertraining rule fired in both environments — the
  matched-exposure budget leaves the EMA average far from its
  asymptote at 1/10 and 1/5 data shares (see EMA context above). This
  is the E.7 disclosed caveat materializing, not an unexpected failure.
- The PV specialist-only `resbench.run` invocation seeds its split-half
  permutation by the filtered environment index, so its self-computed
  band differs from the frozen band; verdicts here use the frozen
  master band per E.5. Sanity check: the lobe run's self-computed band
  reproduces the frozen band exactly (index coincides).
- PV specialist under-produces sand (mean NTG 0.185 vs 0.225) and
  over-fragments (100% of volumes with per-volume τ_z < 0.99 vs 28.7%
  in the reference) — consistent with an under-converged EMA
  checkpoint.
- Mid-run platform changes (disclosed, gradient-identical): lobe
  switched to 2-node DDP at its epoch-5 checkpoint (E-1); PV relocated
  between nodes at its epoch-5 checkpoint (pure resume). Loss curves
  continuous across both transitions.
- PV 80-run: transient training-loss excursion at epoch 10 (epoch mean
  0.42 vs ~0.14 baseline) during the stretched schedule's peak-LR
  plateau; recovered fully by epoch 13; EMA/validation curve unaffected
  (val at epochs 10/15 tracked the 40-run's within 0.002). Logged, no
  action taken.
- PV extension specialist still under-produces sand (mean NTG 0.1811 vs
  reference 0.2249; the 40-ep specialist's 0.1850) — the |dNTG| cell
  barely moves across a 2× training-budget change, i.e. the NTG bias of
  the PV specialist appears training-budget-independent.
- PV extension: validation still improving at the terminal 80-epoch
  budget (0.222 → 0.2085 over the last 5 epochs); the registered
  protocol ends at 80, disclosed above.
- Wave 2 (delta, channel:SH_DISTAL) not started; listed as planned in
  E.1.
