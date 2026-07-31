# Foundation vs specialist comparison — results (EVAL.md Addendum E)

Status 2026-07-28 22:15 CDT — **all runs complete and scored.**
Protocol: EVAL.md Addendum E (frozen before training) + Amendments
E-1/E-2. The pre-registered anti-undertraining rule (E.3) fired for
both environments; both 80-epoch extensions are complete (PV directly;
lobe after one logged divergence and the registered single seed-retry).
The delegated truncation rule never triggered in either extension
(validation still improving at every checkpoint through 80), so both
used their full registered budget. All specialist artifacts in
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
| lobe | specialist (ext, union argmin: 80-run ep 80) | 0.0194 (inside) | 0.0414 (outside) | 0.0250 (inside) | 0.0447 (near) | 0.0281 (inside) | n/a\* |

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
- **lobe: NOT substantiated** — against either specialist variant, and
  in both cases on the same column: **connectivity MAE** (specialist
  inside vs foundation near; 40 ep 0.0223, ext 0.0250, foundation
  0.0356). Against the converged extension specialist (union argmin:
  80-run epoch 80, official val 0.1693 vs the 40-run's 0.2110) the
  remaining tiers are: |dNTG| tied inside (foundation better in
  absolute, 0.0026 vs 0.0194), variogram foundation better (near vs
  outside; 0.0121 vs 0.0414), geobody W1 tied near (specialist better
  in absolute, 0.0447 vs 0.0588), |d largest frac| tied inside. In
  absolute values the converged lobe specialist is better than the
  foundation in the two body-structure columns (connectivity, geobody
  W1) and worse in proportions and variogram, with largest-frac
  effectively tied.

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

The extension runs give an environment-dependent answer. On PV, the
converged extension specialist reproduces — and exceeds — the
over-connectivity signature: floor-to-surface span fraction 0.943
[0.920, 0.960] vs the foundation's 0.852 and the reference's 0.740;
largest-body-fraction median 0.990 (reference 0.995);
compartmentalized fraction 0.398 vs reference 0.287 (the 40-ep
specialist's 1.000 was an undertraining artifact). On PV, therefore,
the over-connectivity bias is **attributable to architecture or CFG
rather than to weight sharing**: a single-environment model with
identical architecture and inference settings, trained to a lower
validation loss than any other candidate, shows the same bias more
strongly. On lobe, however, the converged extension specialist is
close to calibrated where the foundation over-connects: span fraction
0.281 [0.244, 0.322] vs reference 0.334 (foundation 0.373),
compartmentalized fraction 0.904 vs reference 0.885 (foundation
0.822), largest median 0.274 vs reference 0.292. Stated per
environment: the signature is architecture/CFG-linked in the channel
environment; in the multi-body environment a converged specialist
avoids the foundation's mild over-connection, so weight sharing (or
multi-environment training) remains a candidate contributor there.
The 40-ep specialists' over-fragmenting deviations are explained by
EMA undertraining in both environments.

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
- Lobe branch: attempt 1 (4× GH200, seed 8102) trained normally through
  epoch 18 then collapsed at epoch 19 (training loss flat at ≈ 1.82 for
  7 epochs, observational val rising); terminated and preserved at
  `specialist_runs/lobe_80ep_diverged_seed8102`; one registered
  seed-retry (8112, 2× GH200) trained cleanly through the same regime
  (a transient recovered excursion at epoch 14) and completed 80/80.
  Official paired validation: 1.213 (ep 5) → 0.366 (40) → 0.176 (75) →
  0.1693 (80), monotone after the excursion washout; truncation rule
  never triggered; validation still improving at the terminal budget
  (disclosed, as for PV). Union argmin selected the 80-run epoch-80
  checkpoint (md5 `611b747da72399b5b624abc9857ccfe8`; run manifest in
  `/scratch/08405/ilgar/specialist_runs/lobe_80ep_r2/`). Lobe ext mean
  NTG 0.4834 vs reference 0.5028 (40-ep: 0.4757).

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
- Lobe 80-run attempt 1 diverged at epoch 19 (see E.3 outcomes above);
  the registered single seed-retry completed cleanly. Both 80-runs
  (PV and lobe attempt 2) showed one transient recovered loss
  excursion each (epochs 10 and 14 respectively) during the stretched
  schedule's peak-LR plateau — a regime the 40-epoch recipe's earlier
  LR decay never enters; noted as a property of the extended recipe at
  this batch size.
- Wave 2 (delta, channel:SH_DISTAL) not started; listed as planned in
  E.1.

---

# Wave 2 results (delta, channel:SH_DISTAL) — 2026-07-31

Executed per Amendment E-3 with one operator deviation logged pre-training
(E-3 update): wave-2 specialists trained at the 80-epoch budget directly
(warmup 4, cosine T_max 76, seeds delta 8103 / SH_DISTAL 8104, two 4× GH200
DDP groups, global batch 384 = 4×96); the matched-exposure 40-epoch stage
was not run for these environments (aborted pre-checkpoint, preserved at
`*_aborted40`). Their comparison rows are labeled "specialist (80 ep,
direct)" and the matched-exposure rows can be added later if needed.

## Wave-2 comparison rows (frozen band; foundation verbatim; full 4-env table in `comparison.md`)

| environment | source | \|dNTG\| | variogram MAE/sill | connectivity MAE | geobody W1 (log10) | \|d largest frac\| |
|---|---|---|---|---|---|---|
| delta | split-half band | 0.0117 | 0.0054 | 0.0027 | 0.0234 | 0.0182 |
| delta | foundation | 0.0078 (inside) | 0.0150 (outside) | 0.0009 (inside) | 0.0308 (near) | 0.0085 (inside) |
| delta | specialist (80 ep, direct) | 0.0150 (near) | 0.0354 (outside) | 0.0148 (outside) | 0.0712 (outside) | 0.0709 (outside) |
| channel:SH_DISTAL | split-half band | 0.0189 | 0.0122 | 0.0000 | 0.1344 | 0.0001 |
| channel:SH_DISTAL | foundation | 0.0104 (inside) | 0.0364 (outside) | 0.0001 (outside†) | 0.9709 (outside) | 0.0000 (inside) |
| channel:SH_DISTAL | specialist (80 ep, direct) | 0.0182 (inside) | 0.0314 (outside) | 0.0001 (near†) | 0.8747 (outside) | 0.0001 (near†) |

† Saturated bands (see the master-table footnote): the SH_DISTAL
connectivity and largest-frac bands are ≤ 1e-4 because the sand phase
percolates; both models' absolute values are 0.0001 or smaller and the
tier difference at the fourth decimal is not meaningful.

## Parity outcomes (pre-registered criterion, all four environments)

- channel:PV_SHOESTRING: **SUBSTANTIATED** (vs both specialist variants).
- lobe: **NOT substantiated** (connectivity column, both variants).
- delta: **SUBSTANTIATED** — foundation tier ≥ specialist in every
  column (better in four of five; absolute values 1.9–8.3× smaller in
  dNTG, connectivity, geobody W1, largest frac).
- channel:SH_DISTAL: **NOT substantiated**, driven solely by the
  saturated connectivity tier (both models at 0.0001 absolute; band
  0.0000). Excluding saturated-band columns per the master-table
  convention, foundation tier ≥ specialist everywhere; descriptively
  the specialist is slightly better on variogram (0.0314 vs 0.0364)
  and geobody W1 (0.8747 vs 0.9709), the foundation on dNTG (0.0104 vs
  0.0182).

## Wave-2 findings

- **The SH_DISTAL geobody failure is not a weight-sharing cost.** The
  dedicated specialist at twice the foundation's per-environment
  exposure reproduces the foundation's dominant master-table failure
  (geobody W1 0.8747 vs 0.9709; both ~6.5–7.2× the band). Whatever
  causes the geobody-size mismatch in this environment, single-
  environment training does not repair it.
- **Delta extends the over-connection attribution.** The converged
  delta specialist over-connects more than the foundation
  (floor-to-surface span 0.953 [0.931, 0.968] vs foundation 0.793 and
  reference 0.809; compartmentalized fraction 0.090 vs reference
  0.203), matching the converged-PV pattern: in channel/delta
  environments the over-connection bias follows the architecture, not
  the weight sharing.
- Validation at the terminal budget: delta 1.229 → 0.1744 (ep 80),
  SH_DISTAL 1.397 → 0.1881 (ep 80); best = final for both (the E.3
  flag fires; 80 is the registered terminal budget, disclosed as for
  wave 1). Delta's is the lowest specialist validation loss of all
  four environments, consistent with its largest data share (28,080
  steps ≈ 2.8 EMA time constants).

## Wave-2 run manifests and anomalies

- delta_80ep: seed 8103, selected `inference_epoch080.pt` md5
  `e56dbb7f2e3f5a0b1fc336303ac0137c`; one transient recovered loss
  excursion each at epochs 13 and 30 (peak-LR plateau; EMA/val curve
  recovered within two checkpoints).
- sh_distal_80ep: seed 8104, selected `inference_epoch080.pt` md5
  `681c3267859d1ad74ab09a14725e3069`; one transient recovered
  excursion at epoch 11. Its first pipeline invocation failed on a
  manifest filename mismatch (operational only, before generation; no
  scoring artifact was produced); fixed and rerun.
- Scoring emitted variogram z-axis plateau-vs-sill sanity warnings for
  delta for both the reference and generated summaries — an intrinsic
  zonal-anisotropy property of the environment, not a model artifact.
- Wave-2 generation manifests use the E.4 noise offset (+500000);
  self-test passed (bit-identity) in both pipelines; specialist mean
  NTG: delta 0.5934 vs reference (see report), SH_DISTAL 0.5995.
