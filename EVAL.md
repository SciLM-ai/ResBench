# ResBench — Frozen Evaluation Protocol

Quantitative validation of a generative model of binary sand/shale reservoir
facies against the held-out SiliciclasticReservoirs test distribution, following
the geostatistical minimum acceptance criteria of Leuangthong, McLennan &
Deutsch (2004), with categorical/MPS extensions (Boisvert, Pyrcz & Deutsch,
2010) and GenAI-specific checks (Merzoug, Pyrcz et al., 2025): distribution
recovery, variogram/connectivity reproduction, and data exactitude.

**Every choice below was fixed before any metric was computed.** This document
is the public benchmark protocol. ResBench takes directories of saved volumes
as input and imports nothing from ResFlow or ResMill.

## 1. Data

- Dataset: `AnonymouScientist/SiliciclasticReservoirs` (HuggingFace), binary
  facies channel only, volumes of shape (X, Y, Z) = (64, 64, 32) int8 {0, 1},
  z (depth) last axis.
- Splits: the dataset's shipped `splits/test.parquet` (90/5/5
  train/validation/test, stratified by `layer_type`, split seed 42, built by
  ResMill `build_splits.py`). Instance identity is the tuple
  `(layer_type, shard_dir, sample_idx)`.
- The 8 environments (canonical order and slugs):
  `lobe`, `channel:PV_SHOESTRING`, `channel:CB_LABYRINTH`,
  `channel:CB_JIGSAW`, `channel:SH_DISTAL`, `channel:SH_PROXIMAL`,
  `channel:MEANDER_OXBOW`, `delta`
  (directory slugs replace `:` with `_`). Test-split sizes are unequal by
  design (lobe 10,000; CB_JIGSAW and delta 7,500; the rest 5,000).

## 2. Ensembles

- **Reference ensemble**: 512 test-split instances per environment, sampled
  uniformly at random without replacement with the manifest seed (§6),
  loaded from the dataset by instance id. 512 × 8 = 4,096 volumes.
- **Ensemble (a) — distribution recovery**: for each reference instance, one
  generated volume conditioned on the *same parameter vector* (identical
  conditioning input as the reference instance), empty well mask, fresh noise
  seed from the manifest. 512 × 8 = 4,096 volumes.
- **Ensemble (b) — data exactitude**: the first 256 manifest rows per
  environment. Wells are extracted from the reference volume with the exact
  Figure 3 well builder of the ResFlow paper and one conditioned sample is
  generated per configuration (`samples_per_condition = 1`; the flag exists
  for a later calibration study). 256 × 8 = 2,048 volumes.
- No down-scoping was needed: generation ran locally on one GH200
  (no queue), so ensemble (a) is the full 512 per environment.

### Well configurations for ensemble (b)

Vertical, full-depth (all 32 z-voxels) wells at y = 32, x = round(64·f) for
well-position fractions f — exactly the paper's Figure 3 XZ-row builder and
its three published configurations:

| config | fractions f | well columns x |
|---|---|---|
| 1 well  | [0.50]             | {32} |
| 2 wells | [0.33, 0.66]       | {21, 42} |
| 3 wells | [0.25, 0.50, 0.75] | {16, 32, 48} |

Configs are assigned deterministically by reference index: `row_index mod 3`
→ 0 = 1 well, 1 = 2 wells, 2 = 3 wells; the assignment is recorded per row in
the manifest. Well *values* are carved from the reference volume at the mask
voxels. Exactitude is reported pooled and broken out by well count.

## 3. Inference settings (verbatim from the paper, Table 6 / `figure2.py`)

Read from the ResFlow repo's config (`examples/reservoirs/paper_figures/figure2.py`
constants, identical to Table `tab:inference-hparams` of the paper). Not tuned
per metric; identical for every generation in this study.

| setting | value |
|---|---|
| Checkpoint | `flow_matching.pt` (raw, non-EMA — the paper-figure weights) |
| ODE solver | explicit Euler |
| NFE | 50 (Δt = 0.02) |
| CFG | scale g = 3.0, `v = v_uncond + g (v_cond − v_uncond)` |
| Conditioning normalization | `cond_stats.npz` stored beside the checkpoint |
| Hard data replacement | well voxels overwritten with conditioning facies after the final ODE step |
| Binarization | facies = (x > 0) on the continuous output in [−1, 1] |

Checkpoint path + md5, package versions, and every seed are logged in the run
manifest (§6).

## 4. Metrics (implemented in `resbench/metrics.py`, unit-tested vs brute force)

All metrics operate on int8 {0,1} volumes, sand = 1. Axes: 0 = x, 1 = y,
2 = z. **Max lag = half the axis extent: L = (32, 32, 16)**; lags h = 1..L.

1. **Facies proportion (NTG)** — mean of the binary volume. Per volume;
   environment statistic = mean over the ensemble.
2. **Indicator variogram** γ_a(h) = 0.5 · mean over all voxel pairs at lag h
   along axis a of the squared indicator difference, a ∈ {x, y, z},
   h = 1..L_a. All volumes share one shape, so the environment curve is the
   pooled pair mean (= mean of per-volume curves).
3. **Connectivity function** τ_a(h) (Renard & Allard): sand geobodies labeled
   with 6-connectivity (faces only); over voxel pairs at lag h along axis a
   with *both* voxels sand, τ = P(same geobody). Environment curve pools
   numerator and denominator counts over the ensemble (denominators vary per
   volume, so pooling — not averaging of ratios — is the frozen choice).
4. **Geobody statistics** from the same 6-connected labeling: component size
   distribution (voxel counts, pooled over the ensemble per environment) and
   largest-geobody fraction = largest component / total sand voxels per
   volume (0 if no sand), environment statistic = ensemble mean.
5. **Well exactitude** (ensemble (b) only): fraction of conditioning well
   voxels where the generated facies ≠ the reference facies, pooled over the
   ensemble; also broken out by well-count config.

Sanity assertions (checked at run time): variograms non-negative and
plateauing near p(1−p); τ non-increasing in expectation (monotone after
ensemble pooling, tolerance 0.02 for Monte-Carlo noise); exactitude near
zero; proportions in [0, 1].

## 5. Master-table scalar summaries and acceptance criterion

One row per environment plus a pooled row (pooled = all 8 ensembles
concatenated). Columns:

| column | definition |
|---|---|
| \|ΔNTG\| | \|mean NTG(gen) − mean NTG(ref)\| |
| variogram MAE | mean over all (axis, lag) of \|γ_gen − γ_ref\|, ÷ reference sill p̄(1−p̄), p̄ = ref mean NTG |
| connectivity MAE | mean over all (axis, lag) of \|τ_gen − τ_ref\| (τ is already in [0,1]; unnormalized) |
| geobody W1 | Wasserstein-1 distance between pooled distributions of log10(component size) |
| Δ largest frac | \|mean largest-geobody fraction (gen) − (ref)\| |
| well mismatch % | exactitude metric × 100 (ensemble (b)) |

**Null band (split-half)**: each 512-volume reference ensemble is split into
two disjoint random halves of 256 (fixed seed, §6); every column above is
computed half-A-vs-half-B with the same estimator. That discrepancy is the
Monte-Carlo noise of the data engine itself and defines "matching".
**Acceptance**: a cell is **inside** if ≤ its split-half band value, **near**
if ≤ 2× band, **outside** otherwise. Scalar metrics additionally get 95%
bootstrap CIs (1,000 resamples over volumes, fixed seed).

For the well-mismatch column the split-half band is not defined (the
reference reproduces wells exactly by construction); the frozen criterion is
mismatch ≤ 1% = inside, ≤ 2% = near (hard replacement guarantees mask voxels
match; this column instead audits the pipeline end-to-end and should be ~0).

## 6. Seeds and determinism

- `MANIFEST_SEED = 20260726` — samples the 512 rows/env and draws one
  `fresh_noise_seed` per manifest row (`numpy.random.default_rng`).
- `SPLIT_HALF_SEED = 20260727`, `BOOTSTRAP_SEED = 20260728`.
- Generation noise: per row, `x0 = torch.randn` from a CPU
  `torch.Generator().manual_seed(fresh_noise_seed)`, moved to device — the
  full sampler input is reproducible from the manifest alone.
- The run manifest (`run_manifest.json`) records: checkpoint path + md5,
  cond_stats path + md5, git SHAs of ResFlow and ResBench, package versions,
  GPU model, and all seeds above. Bitwise GPU reproducibility across hardware
  is not asserted (cuDNN kernels); statistical reproducibility is.

## 7. Out of scope

- **Porosity/permeability recovery.** The dataset's property channels are
  produced inside the ResMill engine from its internal 6-code (Alluvsim)
  facies plus stochastic per-event fields drawn during geometry simulation —
  they are stochastic ResMill *annotations* of a realization, not
  deterministic functions of the released binary facies, and ResFlow does not
  generate them. Property-generation evaluation is therefore out of scope for
  this harness.
- Multi-block (MultiDiffusion) generation; only native 64×64×32 volumes are
  scored.

## 8. Input contract (for scoring any model with this benchmark)

A *volume directory* holds `volumes_*.npz` shards, each with:
`ids` — array of row-id strings `"{layer_type}|{shard_dir}|{sample_idx}"`;
`volumes` — int8 array (N, 64, 64, 32) with values in {0, 1}.
Well-mask directories use the same layout with key `masks` (uint8, 1 = known).
Prediction, reference, and mask directories are joined on `ids`.

CLI: `python -m resbench.run --pred-dir A --ref-dir REF --out metrics.parquet
[--mask-dir MASKS] [--band] [--figures]`.

---

# Addendum A — Conditional-entropy calibration (frozen 2026-07-26, before computation)

Scored follow-up study; the §1–§8 master table is unchanged by it. Goal:
calibrate the model's conditional voxelwise uncertainty against the data
engine's conditional uncertainty under identical (parameters, wells).

**A.1 Conditions.** Per environment, the ensemble-(b) manifest rows with
row_index 0–3 (well configs 1/2/3/1 wells): 32 conditions total.

**A.2 Model entropy field.** K = 128 conditioned samples per condition
(Table 6 settings; per-sample noise seed = fresh_noise_seed·1000 + k).
Voxelwise p̂ = mean sand indicator over K; H(x) = −p̂ log₂ p̂ −
(1−p̂) log₂(1−p̂), with H = 0 at p̂ ∈ {0, 1}. H = 0 at well voxels holds by
construction (hard replacement); it is stated, not evidence of calibration.

**A.3 Reference far-field entropy.** For each condition's parameter vector:
N = 512 unconditional engine realizations (same `create_geology` kwargs,
seeds drawn from a logged rng), validated by first reproducing stored
dataset instances bit-exactly from (params, seed). Voxelwise entropy of
that ensemble is the *unconditional-under-parameters* level; beyond the
wells' correlation range the model's conditional entropy must converge to
it. If measured engine cost makes 512 infeasible, N may be lowered
(never below 128) with the actual N logged per condition.

**A.4 Reference near-field entropy (rejection, feasibility-gated).**
Acceptance = an unconditional realization matching ALL conditioning well
voxels exactly. Acceptance rates are measured on the A.3 draws (no extra
cost), and CPU cost to reach 200 accepted realizations per condition is
projected from the measured per-realization time. Rejection proceeds only
where projected cost is reasonable (1-well configs prioritized); elsewhere
the study reports far-field-only calibration and says so.

**A.5 Metrics.** Voxels binned by Chebyshev distance to the nearest well
voxel (bin width 2, up to 24). Deliverables per environment: mean-entropy
vs distance curves (model; rejection reference where available;
unconditional level as asymptote), one side-by-side voxelwise entropy-map
figure for a representative condition, scalar calibration error
mean |H_model − H_ref| per bin, the distance beyond which the model curve
is within the A.3 split-half band of the reference level, and an explicit
flag when H_model < H_ref (under-dispersion — the direction that matters
for uncertainty quantification).

**A.6 Out of protocol.** Items run as *post-hoc analyses* (geobody-W1
minimum-size sensitivity, per-volume compartmentalization table, CFG
sweep) are presentation/diagnostics: clearly labeled, never a re-scoring
of the master table.

---

# Addendum B — Unconditional entropy-calibration benchmark component (frozen 2026-07-27, before computation)

Promotes far-field calibration into a reusable, well-free benchmark
component. The engine reference fields are published with ResBench so
future models are scored without running the data engine.

**B.1 Conditions.** Manifest rows with row_index 4–11 per environment
(disjoint from Addendum A's rows 0–3): 64 conditions, parameter vectors
from the frozen manifest.

**B.2 Reference.** N = 256 unconditional engine realizations per condition
(same bit-exact-validated reproduction path as A.3), seeds from
`default_rng([20260801, condition_index])`. Published asset: the voxelwise
sand-probability field p̂ (float16) per condition.

**B.3 Model ensembles.** K = 128 samples per condition, empty well mask,
Table 6 settings, noise seed = fresh_noise_seed·1000 + k (k = 0..K−1; k = 0
coincides with the ensemble-(a) sample of the same row by construction).

**B.4 Metric.** Per condition, over all voxels: signed mean entropy offset
mean(H_model − H_ref) and mean absolute error mean|H_model − H_ref|
(bits). Per environment: mean over its 8 conditions with a 95% t-interval
across conditions. Yardstick: the same statistics computed between the two
halves of the engine ensemble (split rng `default_rng([20260802,
condition_index])`) — inside/near/outside at 1×/2× band as in §5.
Negative signed offset flags under-dispersion.

**B.5 Reporting.** One benchmark table (8 environment rows + pooled).
The §5 master table remains unchanged; this is an additional scored
component of the benchmark.

---

# Addendum C — Well-conditional entropy reference via modal-pattern rejection (frozen 2026-07-27, before computation)

Extends the near-well calibration reference to **all 8 environments** by
conditioning on the *modal* well pattern instead of a pre-specified one:
carve the well column from every unconditional engine draw, histogram the
binary patterns, and condition on the most frequent. All draws showing that
pattern are exact samples of P(volume | parameters, well data); acceptance
is maximal by construction. The accepted ensembles are published so the
benchmark never needs rejection again.

**C.1 Well geometry.** One vertical full-depth well at (x, y) = (32, 32)
(the paper's 1-well config), pattern length 32.

**C.2 Condition selection (deterministic).** Per environment, the
(parameter-vector) condition whose modal pattern count over all
pre-existing unconditional draws (A.3's N = 512 × 4 and B.2's N = 256 × 8)
is largest. Selected conditions and patterns are recorded in
`modal_conditions.json`. Known trade-off, stated: the modal pattern is a
*typical* (high-likelihood, low-information) well — all-shale "dry well"
in low-NTG environments, all-sand column in sheet-like ones; rare
informative patterns are out of scope.

**C.3 Reference ensemble.** Existing accepted draws + top-up rejection
(seed base 20260803, chunked rng as in A.4) until 200 accepted or the
per-environment draw cap (12,000) is reached; minimum publishable size 50.
Published asset: accepted volumes (int8), the pattern, the well voxels,
and the parameter pointer, per environment.

**C.4 Model ensembles & metric.** K = 128 model samples conditioned on
(parameters, the modal pattern as well data) with Table 6 settings, noise
seed = fresh_noise_seed·1000 + 500 + k (offset avoids reuse of A/B
streams). Metrics as A.5 (entropy vs Chebyshev distance to the well, mean
|H_model − H_ref| per bin, near-field under-dispersion flag), with the
engine ensemble split-half band (rng [20260804, env_index]) as yardstick.

**C.5 Informative-pattern tier (mixed wells).** The modal pattern is
typically all-shale or all-sand (C.2 trade-off). A second tier conditions
on the most frequent *mixed* pattern — the modal pattern among draws with
**≥ 4 voxels of each facies** in the well column — selected by the same
deterministic rule over the same pre-existing draws
(`mixed_conditions.json`; e.g. a real shoestring-channel intersection in
PV, thin shale breaks in delta). Model ensembles as C.4 with noise-seed
offset 700. Both tiers are published and reported side by side.

**C.6 Top-up method (amended 2026-07-27, before mixed-tier computation).**
Mixed-tier top-ups do NOT use discard-mode rejection: new unconditional
draws are generated and **stored in full** (seed base 20260806, N per the
per-environment draw budgets), then the conditional ensembles are
assembled by dictionary-mining the pooled draws (pre-existing A.3/B.2
ensembles + stored top-up pools) for the frozen pattern. Same engine cost,
but every draw remains reusable (several environments share one parameter
vector between the modal and mixed tiers, and the pools extend the
unconditional references). The modal tier's top-ups (completed earlier)
used discard-mode rejection with seed base 20260803; its accepted-only
output is statistically equivalent, just not reusable. CB jigsaw's mixed
tier is reported infeasible: measured acceptance ~0.03% (probe estimate
0.6% was 3 lucky hits in 512); reaching the 50-realization floor would
cost ~11 engine-hours and is out of budget.

**C.7 Most-matches re-selection (amended 2026-07-27, before mixed-tier
scoring).** Small-probe pattern frequencies proved systematically
optimistic. Final rule: mine ALL stored pools for each environment's mixed
condition; if the frozen pattern's total matches fall below the
50-realization floor, the pattern is re-selected as the pool-wide
most-matched mixed pattern (ties broken by count, then lexicographic byte
order). Frozen patterns are kept as provenance in
`mixed_conditions.json`; re-selection is flagged per environment.
Environments still below the floor after re-selection (labyrinth 25,
SH proximal 36, meander-oxbow 11, jigsaw 3) are published with an
**indicative-only** flag; their split-half bands are correspondingly wide
and verdicts must not be quoted without the n. Sampling budget ended
2026-07-27 ~12:20 CDT on one node (the second node was returned to the
user mid-run).

# Addendum D — Baseline protocol (frozen 2026-07-27, before baseline training)

Pre-registers one published-baseline comparison for the rebuttal. Everything
in this section is fixed before any baseline training step; the frozen
protocol above (§1–§8) is unchanged and the baseline is scored by the same
`resbench.run` CLI with no harness modification.

## D.1 Scope and environment

- Single environment: `channel:PV_SHOESTRING`. Justification: it is the
  baseline's demonstrated domain (channelized reservoirs, per GANSim's
  publications) and simultaneously ResFlow's weakest master-table
  environment (2 of 5 banded cells `outside`), i.e. a conservative choice
  for us. Distribution-level comparison only (ensemble-(a)-style columns);
  well-conditioned baseline generation is out of scope and the
  `well mismatch %` column is n/a for all baseline-comparison rows.
- Baseline model: GANSim-3D (Song, Mukerji & Hou 2022, WRR,
  doi:10.1029/2021WR031865), authors' official updated code
  **GANSim3D_v2** (github.com/SuihongSong/GANSim3D_v2, commit `02677daf`,
  vendored in `ResBaselines/vendor/`). The original WRR-2022 repository
  hard-wires well/probability conditioning into the training loop with no
  unconditional path; v2 is the same authors' same-method release whose
  shipped default is exactly the unconditional configuration used here
  (`cond_label = cond_well = cond_prob = False`). Four logged port shims
  (Python 3.12 `imp` stub, Pillow install, one `int()` cast for TF 2.17
  buffer-size type checks, one reload-path type-check widening) are
  recorded in `ResBaselines/manifests/`; none alter training semantics.

## D.2 Training data

- The full PV_SHOESTRING training split: 90,000 volumes addressed by
  `splits/train.parquet`, facies channel only, converted to GANSim
  multi-resolution TFRecords by `ResBaselines/scripts/make_tfrecords.py`
  (insertion shuffle seed 123, the authors' notebook constant; round-trip
  verified bit-exact on 10 volumes). No test or validation contact at any
  point before final scoring.
- Grid adaptation: none required. The native grid (64, 64, 32) is
  power-of-two per axis and matches the architecture's anisotropic growth
  (4×4×4 → 8×8×4 → 16×16×8 → 32×32×16 → 64×64×32), verified by a probe
  run. No padding, no cropping; ResBench only ever sees native-grid
  volumes. (Contingency, unused: symmetric shale padding + pre-scoring
  crop would have been used had the grid been incompatible.)

## D.3 Training configuration (one shot, pre-registered)

- Platform: 1× NVIDIA GH200 (TACC Vista) under apptainer image
  `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` (arm64; SIF sha256 in
  `ResBaselines/manifests/probe_20260727.json`).
- Schedule: the WRR-2022 paper defaults from the original repository's
  `config.py`/training log, adapted only to the native grid's resolution
  ladder: `lod_training_kimg = lod_transition_kimg =
  {4:160, 8:320, 16:320, 32:480, 64:640}`; minibatch
  `{4:32, 8:32, 16:32, 32:32, 64:16}`; G/D learning rates
  `{4:0.0025, 8:0.005, 16:0.005, 32:0.0035, 64:0.0025}` (Adam β1=0,
  β2=0.99); `total_kimg = 3600` (the paper run's stopping point; full
  resolution is reached at kimg 2560). Facies codes `[0, 1]`,
  soft-argmax β = 8e3 (v2 default), D_repeats = 1, G EMA β = 0.999.
  Global seed 8001 (v2 default). Measured projection ≈ 9 GPU-h;
  hard ceiling 48 GPU-h.
- Snapshots: every tick (`tick_kimg = {4:160, 8:160, 16:240, 32:240,
  64:80}`, original-repo defaults), i.e. a ~13-snapshot ladder across the
  full-resolution phase.
- No hyperparameter iteration after any score is seen. If training
  diverges (NaN loss, or generator collapse visible as constant output in
  snapshot previews), exactly one retry with seed 8002, logged. A second
  failure triggers the fallback (D.7), not tuning.

## D.4 Checkpoint selection (training-split proxy only)

- Reference statistics: 512 training-split volumes drawn uniformly
  without replacement with `numpy.random.default_rng(20260805)` from the
  90,000-row training index. The test reference ensemble is never
  consulted for selection.
- Proxy score per snapshot, computed on 64 volumes generated from that
  snapshot (latent seed 9000 + snapshot kimg): `|ΔNTG| +
  variogram MAE / (p̄(1−p̄))`, both computed exactly as in §4/§5 but
  against the training-split reference sample above. Lowest score wins;
  ties break to the later snapshot. Only full-resolution snapshots
  (kimg ≥ 2560) are eligible.
- Binarization everywhere (proxy and scoring, identically): the
  generator's soft-argmax output lies in [0, 1]; facies = output > 0.5 —
  the midpoint of the output range, equivalent to arg-max over the two
  facies channels and to the original repository's published rule
  (`np.where(out < 0, −1, 1)` on its [−1, 1] range).

## D.5 Scored ensembles (512 volumes each, PV_SHOESTRING)

1. **GANSim-3D unconditional**: 512 volumes from the selected checkpoint,
   latent seed 20260806, binarized per D.4, saved int8 (64, 64, 32).
2. **ResFlow environment-only (marginalized)**: ResFlow has no per-field
   null token (CFG dropout nulls the entire 18-D conditioning embedding),
   so "environment-only" is implemented as marginalization: for each of
   512 samples, a parameter vector is drawn uniformly with replacement
   from the 90,000 PV_SHOESTRING *training-split* rows
   (`default_rng(20260807)`), environment one-hot fixed, empty well mask,
   Table 6 inference settings (Euler, NFE 50, CFG 3.0, `x > 0`), fresh
   noise seeds `20260808*1000 + k`. This is the like-for-like partner for
   row 1: both models produce "a random plausible PV_SHOESTRING volume"
   with no per-instance information.
3. **ResFlow parameter-conditioned**: the frozen master-table row copied
   verbatim, labeled as conditioning on per-instance parameters (an
   advantage rows 1–2 do not have). Not re-generated, not re-scored.
- Scoring join: `resbench.run` aligns predictions to reference by
  instance id. Rows 1–2 have no per-instance correspondence, so generated
  volumes are assigned the 512 reference ids in generation order,
  nominally, purely to satisfy the join; every compared metric is
  ensemble-level. Comparison verdicts use the same split-half band as the
  master table.

## D.6 Budget

- GANSim-3D: paper-default schedule above; measured probe throughput
  16.3 s/kimg at full resolution projects ≈ 9 GPU-h; hard ceiling 48
  GPU-h. If the ceiling interrupts training, the snapshot ladder up to
  that point feeds D.4 unchanged.
- Fallback DDPM (if triggered): ≈ 20 GPU-h, capacity-matched to one
  per-environment share of ResFlow's training compute (166 GPU-h / 8
  environments); reuses the ResFlow UNet3D backbone (5.37 M parameters,
  `in_channels = out_channels = 1`) with `resflow/methods/diffusion.py`,
  unconditional, trained from scratch on the same TFRecords-equivalent
  split with the same D.4 proxy rule and the same binarization
  discipline.

## D.7 Fallback trigger (pre-registered)

Switch to the DDPM fallback and record the reason if either: (a) the
GANSim-3D code is not training successfully within 1 calendar day of
hands-on effort from first launch attempt (environment/container issues
included; the completed feasibility probe makes this unlikely), or (b)
training fails twice under the divergence rule in D.3. Both baselines are
pre-registered here so a switch is not post-hoc.
