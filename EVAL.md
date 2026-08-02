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

# Addendum E — Foundation vs specialist protocol (frozen 2026-07-27, before specialist training)

Pre-registers the direct test of the foundation-model claim requested by
the Area Chair and reviewer jfkD: does one shared-weights model trained on
all 8 environments match dedicated per-environment specialists? The frozen
protocol above (§1–§8) is unchanged; specialists are scored by the same
`resbench.run` CLI with no harness modification. Design principle: a
specialist differs from the foundation model in exactly one way — its
training set is restricted to one environment's training split.

## E.1 Scope and environments

- Wave 1 (this addendum): `channel:PV_SHOESTRING` — the foundation
  model's weakest master-table environment (2 of 5 banded cells
  `outside`), the adversarial pick; and `lobe` — the only multi-body
  environment, the discriminating one for connectivity metrics. Wave 2
  (`delta`, `channel:SH_DISTAL`) is planned for camera-ready if resources
  allow and is not part of this freeze.
- Distribution-level comparison only (ensemble-(a)-style columns).
  Well-conditioned generation is out of scope for specialists: exactitude
  is guaranteed by the hard-replacement pipeline (§ masking,
  `apply_inpaint_output`) independent of model weights, so it cannot
  discriminate foundation from specialist. The `well mismatch %` column
  is n/a for specialist rows, footnoted to this section.

## E.2 Specialist training configuration (single-difference, pre-registered)

The foundation reference run is `examples/reservoirs/inpainting/train.py`
via `run_A100.sh` (LS6 Slurm job 3135586, completed 2026-04-30/05-01;
checkpoints in `genflows_runs_backup_ls6/reservoirs_inpainting/`). Its
resolved configuration, recovered from code + job log, and the specialist
configuration next to it:

| quantity | foundation (resolved) | specialist (pre-registered) |
|---|---|---|
| training data | full train split, 900,000 volumes | one environment's train split only: PV 90,000 / lobe 180,000 (`Subset` over the loader's `layer_idx`; no other loader change) |
| architecture | UNet3D `in=3, out=1`, 18-D cond, GroupNorm, 5.37 M params | identical; environment one-hot stays in the model, fed as a constant |
| cond normalization | `cont_min/cont_max` over the full train split (`cond_stats.npz`) | foundation `cond_stats.npz` reused verbatim (bit-identical conditioning surface; subset-derived stats would differ and can be all-NaN in family columns) |
| method / CFG | Flow Matching, `drop_prob = 0.1` (whole 18-D embedding nulled) | identical |
| mask distribution | 30% empty / 70% 1–5 straight wells | identical |
| epochs | 40 | 40 over the environment subset (matched per-sample exposure; 1/10 resp. 1/5 of foundation compute) |
| global batch | 384 (12 ranks × 32, DDP mean; `drop_last`) | 384 on 1 GPU — single batch of 384 if it fits (smoke-test-resolved), else gradient accumulation of 384 / feasible per-device batch with EMA updated once per optimizer step. GroupNorm-only architecture + mean-reduced loss make either path gradient-identical to the foundation's 12-rank DDP mean. Resolved choice logged in the run manifest. |
| optimizer | AdamW, base lr 1e-3 × √12 (world-size rule) = 3.4641e-3 peak, default wd 0.01, grad clip 1.0 | identical, with peak lr 3.4641e-3 passed explicitly (world size 1 would otherwise skip the √-scaling; decision approved 2026-07-27) |
| LR schedule | epoch-parametrized: 2-epoch linear warmup (start factor 0.5) → cosine `T_max = 38`, stepped once per epoch | identical in epoch space. Nothing in the foundation recipe is step-parametrized, so the "rescale absolute-step quantities" clause is vacuous — recorded here as a resolved fact. |
| optimizer steps | 2,343/epoch; 93,720 total | PV 234/epoch (9,360 total); lobe 468/epoch (18,720 total) |
| EMA | decay 0.9999 per optimizer step; checkpoints are EMA-applied | identical decay; shortened horizon disclosed in E.7 |
| loader shuffle seed | 42 | 42 |
| global torch seed | not set (model init unseeded; disclosed) | PV 8101, lobe 8102, logged |
| checkpoint cadence | every 5 epochs + auto-resume `training_state.pt` | identical (every 5 epochs; approved 2026-07-27) |
| platform | 4 nodes × 3 A100 (LS6), collaborator env | 1 GH200 per specialist (Vista idev, nodes c608-122 / c611-041), env `genflows` (torch 2.10.0+cu126, accelerate 1.13.0); numerics differences disclosed in E.7 |

Implementation is a thin wrapper (`ResFlow_ls6/scripts/specialists/`,
new directory): environment filter, foundation-stats reuse, explicit
lr/seed, then the unmodified `resflow.utils.training.train_model_inpaint`
loop (or a verbatim copy extended only with gradient accumulation if the
384 batch does not fit — the fallback carries its own smoke-test
equivalence check). No file under `resflow/` or `resbench/` is modified.

## E.3 Validation protocol, checkpoint selection, anti-undertraining rule

- Validation loss per saved checkpoint: the environment's validation
  split only (PV 5,000 / lobe 10,000), FM velocity MSE exactly as in
  `eval_losses.py` (K = 4 random `(t, noise)` draws per cube averaged,
  `drop_prob = 0.1`, on-the-fly masks), with one fixed evaluation seed
  20260901 applied identically before each checkpoint's pass so all
  checkpoints see paired draws.
- Selection: the scored specialist checkpoint is the argmin of validation
  loss over the saved EMA checkpoints (`inference_epoch{005..040}.pt`).
  The test split is never touched before final scoring.
- Anti-undertraining rule (operational): if the argmin is epoch 40 — the
  only saved checkpoint inside the final 10% of the 40-epoch run at the
  5-epoch cadence — that specialist is retrained from scratch at the
  80-epoch horizon under the identical code path (its own epoch
  semantics: warmup `max(1, 80//20) = 4` epochs, cosine `T_max = 76`,
  checkpoints every 5). In-place continuation is not used because the
  cosine floor at epoch 40 (LR ≈ 0) makes it semantically broken, and the
  restored scheduler state would cycle the LR back upward — not the
  recipe's shape. The final checkpoint is then the argmin of validation
  loss over the union of both runs' checkpoints. Which branch fired is
  logged per environment in `SPECIALISTS.md`.

## E.4 Generation (512 volumes per specialist)

- Entry point: `scripts/rebuttal_eval/generate_ensembles.py` unchanged —
  the same script, solver replica (`--self-test` on), and Table 6
  settings read at run time from `paper_figures/figure2.py` (Euler,
  NFE 50, CFG 3.0, binarization `x > 0`, empty well masks, no hard
  replacement) that produced ensemble (a).
- Conditioning: the same manifest rows for that environment (identical
  parameter vectors via the same `conds.npz`; the reuse of
  `cond_stats.npz` in E.2 keeps the vectors bit-identical to
  ensemble (a)).
- Fresh noise (pre-registered offset 500000): the specialist run passes a
  manifest copy whose `fresh_noise_seed` column is the original value
  + 500000; the unchanged script then derives torch CPU seed
  `(fresh_noise_seed + 500000)·1000 + k`, `k = 0`.
- Output: script-native layout and naming, int8 `.npz` with the same ids —
  `resbench_eval/specialist_pv_shoestring/ensemble_a/channel_PV_SHOESTRING/volumes_r0000-r0511.npz`
  and
  `resbench_eval/specialist_lobe/ensemble_a/lobe/volumes_r0000-r0511.npz`
  — with the script's generation manifest JSON alongside; selected
  checkpoint md5 recorded in the run manifest.

## E.5 Scoring, comparison table, parity criterion

- `resbench.run` unchanged: each specialist directory against the same
  reference directory and the same split-half band as the master table.
- Comparison table (parquet + rendered markdown), three rows per
  environment: split-half band; foundation — copied verbatim from the
  master table in `RESULTS.md`; specialist. Columns identical to the
  master table; `well mismatch %` n/a per E.1. Verdict tiers and absolute
  values in every cell.
- Parity criterion (pre-registered): the foundation claim is
  substantiated on an environment iff the foundation row's verdict tier
  (inside ≻ near ≻ outside, against the same band) is at least as good as
  the specialist row's in every metric column. Absolute values are
  reported alongside and compared descriptively regardless of tiers.

## E.6 Secondary diagnostic (pre-registered)

Does the specialist reproduce the foundation model's over-connectivity
signature — elevated largest-body fraction, γ_z plateau undershoot, and
under-reproduced compartmentalization tail (per-volume τ_z < 0.99
fraction)? If yes, the bias is attributable to architecture or CFG rather
than to weight sharing; if no, weight sharing remains a candidate cause.
Stated in advance of any specialist score.

## E.7 Disclosed caveats (inherent to the design, not corrected)

- EMA horizon: decay 0.9999 implies a ~10,000-step time constant, which
  exceeds the PV specialist's entire 9,360-step run and half the lobe
  specialist's. Matched per-sample exposure necessarily shortens the EMA
  horizon by the subset ratio; disclosed, not corrected.
- The foundation run's model-init seed was not recorded (collaborator
  run, unseeded); specialist init seeds are fixed and logged. One seed
  per specialist — matching the single foundation run — so no seed-
  variance band accompanies either row.
- Hardware/software numerics differ (12× A100 DDP on LS6 vs 1× GH200 on
  Vista, different torch builds); gradient math is equivalent per E.2.
- Subset ratios are 1/10 (PV) and 1/5 (lobe), not the planning-language
  "1/8"; matched exposure is defined by 40 epochs, not by a uniform
  ratio.

## Amendment E-1 — lobe specialist platform change (2026-07-27, during training, before any scoring)

Recorded during training, before any specialist validation loss or score
was computed. To meet the rebuttal deadline, the lobe specialist run is
switched from 1× GH200 to 2× GH200 DDP (nodes c608-122 + c611-041) at
its epoch-5 checkpoint, resuming from `training_state.pt`. Training
semantics are unchanged: same global batch 384 (now 2 ranks × 192, each
accumulated as 96 × 2), same peak LR 3.4641e-3 (the wrapper deliberately
skips the library's world-size re-scaling), same epoch-parametrized
schedule and EMA decay with one update per optimizer step, and the same
468 optimizer steps/epoch (DistributedSampler with drop_last shards
90,000 samples per rank). DDP's cross-rank gradient mean of per-rank
accumulated means equals the single-GPU accumulated mean (GroupNorm-only
model), so the gradient computation is identical to E.2. Disclosed
differences: within-epoch data order (per-rank DistributedSampler
shuffle, seed 42, vs one global shuffle stream) and per-rank FM
noise/CFG draws (torch seeds 8102 + rank). The PV_SHOESTRING run stays
1× GH200 throughout; it is relocated to node c608-061 at its epoch-10
checkpoint (pure resume, no semantic change) to free c608-122 for the
lobe DDP pair. Both variants of the launcher and the
DDP code path are in `scripts/specialists/` (ResFlow repo), smoke-tested
before the switch.

## Amendment E-2 — PV extension-run platform (2026-07-27, before any extension score)

The fired-branch 80-epoch PV_SHOESTRING retrain (E.3) runs as 4× GH200
DDP (c608-061, c611-041, c608-122, c622-042), global batch unchanged at
384 (4 ranks × 96, no accumulation), peak LR/schedule/EMA per E.2 with
the code path of Amendment E-1 (per-rank torch seeds 8101 + rank). A
per-checkpoint validation watcher runs alongside training (batch 32, the
E.3 paired-seed protocol); it is observational only — checkpoint
selection remains argmin validation loss over the union of both PV runs'
saved checkpoints. If the validation curve demonstrably plateaus or
degrades before epoch 80, the run may be truncated with the operator's
explicit approval; any such truncation and its evidence will be logged
in SPECIALISTS.md. The lobe extension (also fired) is deferred to
camera-ready under the same 80-epoch recipe.

(Update 2026-07-27 23:15, pre-scoring: the lobe extension is queued to
start on the same 4-node platform immediately after the PV extension
completes, same 80-epoch recipe and observational val watcher with the
E-2 truncation clause; its results remain camera-ready-bound.)

(Update 2026-07-27 23:20, pre-scoring — delegated truncation rule: the
operator granted standing approval for automatic truncation of the
extension runs. Concrete rule, fixed before the first extension
checkpoint was evaluated: watching the observational val series v_k at
checkpoints k = 5, 10, …, 80, stop the run if (a) k ≥ 50 and the
5-epoch improvement (v_{k−5} − v_k) < 0.02·v_k on two consecutive
checkpoints, or (b) k ≥ 45 and v increases on two consecutive
checkpoints; never truncate after k = 75 (finish instead). Definitive
checkpoint selection is unchanged: argmin of the official paired
validation protocol (batch 128, seed 20260901, K = 4) over the union of
that environment's runs' checkpoints.)

# Addendum E — MultiDiffusion assembly-consistency benchmark (frozen 2026-07-28, before computation)

## E.1 Question

Whether volumes assembled beyond the training extent (MultiDiffusion tiling,
Table 6 settings) preserve the statistics of native-tile generation and of
the engine, beyond visual seamlessness. Requested by Reviewer jfkD (Q4) and
the Area Chair.

## E.2 Environment and shared condition

`channel:PV_SHOESTRING`. One parameter vector shared by all three
ensembles: the training-split instance whose slim parameters (ntg,
width_cells, depth_cells, mCHsinu, mFFCHprop, probAvulInside) minimize the
Euclidean distance to the per-type training medians after min-max
normalization with the checkpoint's `cond_stats.npz` ranges; azimuth is
overridden to 95 degrees (midpoint of the paper's Figure-4 gradient)
everywhere. The selected instance id and full parameter row are recorded in
the run manifest. Training-split contact only; the §2 test reference is not
used anywhere in this addendum.

## E.3 Ensembles

- (A) **native**: 250 model volumes (64, 64, 32), Table 6 inference
  (Euler, NFE 50, CFG 3.0, empty well mask), noise seeds `20260811000 + k`.
- (B) **assembly tiles**: 10 assemblies, 10 x 10 block grid, overlap 24
  (the deployed Table 6 tiling), uniform conditioning per E.2, torch seed
  `20260809000 + i` per assembly; from each 424 x 424 x 32 assembly a
  5 x 5 grid of non-overlapping (64, 64, 32) tiles cut from the centered
  320 x 320 region (origin cell (52, 52), stride 64) = 250 tiles.
- (C) **engine reference**: 256 ResMill realizations at the selected
  instance's engine parameters (azimuth 95), seeds `2026081000 + k` (amended from `20260810000 + k` before any realization was generated: the engine RNG requires seeds below 2^32),
  engine source unmodified; published under
  `results/assembly_reference/` as a reusable asset.

## E.4 Scored metrics

Identical estimators to §4 plus one addition: ensemble |dNTG|; W1 between
per-volume NTG distributions; sill-normalized variogram MAE; connectivity
tau(h) MAE; geobody-size W1 (log10 sizes); and body lateral-extent W1
(log10 of each body's maximum x-y bounding-box extent, pooled) as the
channel-length statistic. Scored comparisons: **B vs C** (headline),
**A vs C** (native control), **B vs A** (isolated MultiDiffusion effect).
Verdict band: split-half of (C), `default_rng(20260812)`, same estimator
per metric, tiers as in §5.

## E.5 Seam diagnostic (figure, unscored)

Column-mean sand-fraction profiles along x and y for each assembly; the
discrete-spectrum amplitude at the block-stride period (40 cells) compared
against the same statistic computed on profiles of tiled-together engine
volumes (which have true seams). A stride-locked periodicity in (B) that
exceeds the engine-concatenation level indicates blending artifacts.

## E.6 Standing benchmark component

This addendum defines the "assembly consistency" task: a model claiming
beyond-training-extent generation submits ensemble (B)-equivalent tiles;
they are scored against the published (C) with the E.4 metrics and band.

## E.7 Lobe tier (frozen 2026-07-28, before computation)

Same design as E.2-E.5 for environment `lobe` (the multi-body
environment), with: shared condition = the training-split lobe instance
nearest (same normalized-distance rule) the **medians of the 180,000 indexed lobe training rows** over lobe's defined slim columns (ntg,
width_cells, depth_cells, asp), azimuth overridden to 95 degrees; seeds:
engine `2026082000 + k`, assemblies `20260813000 + i`, native
`20260814000 + k`, split-half `default_rng(20260815)`; identical grid
(10 x 10, overlap 24), tile cutting, metrics, and comparisons. Engine
reference published under `results/assembly_reference/lobe/`.

(Update 2026-07-28 06:58, pre-scoring — lobe extension divergence and
retry: attempt 1 (4× GH200, seed 8102) trained normally through epoch 18
(loss ≈ 0.163) then collapsed at epoch 19 (training loss flat at ≈ 1.82
for 7 consecutive epochs, observational val rising 0.535 → 1.489); the
run cannot improve on the 40-epoch run's validation argmin and was
terminated; its logs, checkpoints and manifest are preserved at
`specialist_runs/lobe_80ep_diverged_seed8102`. Following the one-retry
convention of Addendum D.3, a single restart with torch seed 8112 is
launched on 2× GH200 (the topology that trained the lobe 40-epoch run
without incident), all other settings per E-2/E.3 unchanged, same
observational watcher and truncation clause. Recorded before any retry
checkpoint was evaluated.)

## Amendment E-3 — Wave 2 execution (2026-07-30, before wave-2 training)

Wave 2 (E.1: `delta`, `channel:SH_DISTAL`) executes under the identical
E.2/E.3 recipe and rules: matched-exposure 40 epochs first, validation
argmin selection, the fired-branch 80-epoch from-scratch extension with
the E-2 delegated truncation clause, and the single seed-retry
convention on divergence. Torch init seeds: delta 8103, SH_DISTAL 8104
(loader seed 42 unchanged). Platform: two concurrent 4× GH200 DDP
groups (global batch unchanged at 384 = 4×96), nodes
c636-[092,101,102,111] (delta) and c636-112 + c637-[052,061,062]
(SH_DISTAL), job 876236. Generation manifests use the E.4 noise offset
(+500000) verbatim. Recorded before any wave-2 training step.

(Update 2026-07-30 20:35, pre-scoring: by operator decision the wave-2
environments train at the 80-epoch budget directly (same recipe as the
wave-1 extensions: warmup 4, cosine T_max 76, seeds per E-3); the
40-epoch matched-exposure attempts were aborted at ~10 minutes, before
any checkpoint, and preserved at `*_aborted40`. Selection per E.3 is
the validation argmin over each environment's completed runs.
Matched-exposure 40-epoch runs for these environments may be added
later; their absence from any interim wave-2 table will be marked.)

## Amendment E-4 — Wave 3 execution (2026-07-31, before wave-3 training)

Wave 3 extends the specialist comparison to `channel:CB_LABYRINTH`
(torch seed 8105) and `channel:CB_JIGSAW` (seed 8106), at the 80-epoch
budget directly per the standing operator decision recorded in the E-3
update (no 40-epoch stage), all other settings per E.2/E-2/E-3
unchanged: global batch 384 as two concurrent 4× GH200 DDP groups (job
876236), warmup 4 / cosine T_max 76, observational watchers with the
delegated truncation clause, single seed-retry on divergence, E.4 noise
offset (+500000), validation-argmin selection, frozen-band verdicts.
Recorded before any wave-3 training step.

## Amendment E-5 — Wave 4 execution (2026-07-31, before wave-4 training)

Wave 4 completes the specialist sweep over all eight environments:
`channel:SH_PROXIMAL` (torch seed 8107) and `channel:MEANDER_OXBOW`
(seed 8108), 80-epoch budget directly per the standing operator
decision, all other settings per E.2/E-2/E-3/E-4 unchanged (global
batch 384, two 4× GH200 groups launched per group as its wave-3
pipeline completes, observational watchers, delegated truncation
clause, single seed-retry on divergence, E.4 noise offset,
validation-argmin selection, frozen-band verdicts). Recorded before any
wave-4 training step.

---

# Addendum F — MPS and GenAI acceptance-check extensions (frozen 2026-07-31, before computation)

Post-hoc batch 2 (Boisvert et al. 2010; Merzoug et al. 2025). CPU-only on
existing volumes except F.4's 512 generations. Master table unchanged.
Split-half machinery applied wherever a null band is meaningful; verdicts
as §5 (inside ≤ band, near ≤ 2×, outside).

**F.1 Multiple-point histograms.** All overlapping 2×2×2 binary patterns
per volume (63×63×31 per volume); pattern code = Σ v(x+i, y+j, z+k)·2^(4i+2j+k),
(i,j,k) ∈ {0,1}³ — 256 bins. Pooled histogram per environment for the
aligned reference and ensemble-(a) stacks; metric = Jensen-Shannon
divergence (base 2) between normalized pooled histograms; band =
reference split-half JSD (rng [20260813, env_index], 256/256 halves).
Figure: top-20 reference patterns for the two environments with largest JSD.

**F.2 Runs statistics.** Maximal sand run lengths along z per (x, y)
column, and along x (control), pooled per environment. Report median, p90,
and W1(ref, gen) on the run-length distribution; band = split-half W1 and
|Δ median| (same rng family [20260813, ·]). An explicit statement connects
the sign of the vertical-runs shift to the γ_z plateau undershoot.

**F.3 Reliability diagram (accuracy-plot analog).** For each Addendum C
condition (both tiers, all with engine-conditional references): off-well
voxels binned by model p̂ (K = 128) into deciles; per bin the mean
engine-conditional sand frequency; ECE = Σ_b w_b |mean p̂_b − freq_b|
(w_b = bin voxel fraction). Pooled per environment over its two
conditions. An *unconditional* version from the Addendum B ensembles
(64 conditions, model K = 128 vs engine N = 256) is reported alongside and
labeled as such. Band: engine self-ECE from split halves of the engine
ensemble (rng [20260814, condition_index]). Well voxels are excluded
(hard replacement makes them trivially calibrated).

**F.4 Memorization / novelty check.** 64 training-split rows per
environment (rng [20260812, env_index] over the env's rows sorted by
(shard_dir, sample_idx)); one generation each, Table 6 settings, empty
mask, noise seed drawn per row from the same rng. Metrics: (i)
distribution of voxel agreement between each generation and its
corresponding training volume; reference level = mean pairwise agreement
among same-parameter engine realizations (200 pairs subsampled from the
Addendum C modal pools, rng [20260815, env_index]); the engine
regeneration at the exact (params, seed) equals the stored volume
bit-for-bit (validated), i.e. the memorization ceiling is 1.0 by
construction and any generation approaching it is flagged. (ii) Classical
MDS (eigendecomposition of the double-centered squared-distance matrix)
of per-volume statistics vectors [NTG, largest-body fraction,
log10(1 + body count), γ_z plateau (mean of lags 13–16), γ_x(16), median
z-run length], z-scored per environment, for training-64 / generated-64 /
reference-64 (first 64 aligned reference rows). Purpose: verify novelty.

**F.5 Artifact rate.** Artifacts = 6-connected sand bodies < 8 voxels.
Reference vs ensemble (a): mean artifacts/volume per environment, band =
split-half discrepancy (rng [20260816, env_index]). Ensemble (b): mean
artifacts/volume and fraction of volumes with ≥ 1 artifact vs well count
(1/2/3) with Wilson 95% CIs; an explicit statement on whether the rate
rises with conditioning density (Merzoug failure mode) or stays flat.

**F.6 Coverage map.** RESULTS.md ends with a factual table mapping every
check in Leuangthong et al. (2004), Boisvert et al. (2010), and Merzoug et
al. (2025) to its implementation (section/figure) or "deferred (dynamic)".

---

# Addendum G — Assembly-aware training and segmentation-free body-scale metrics (frozen 2026-08-01, before any G-metric computation)

## G.1 Question

Addendum E established that MultiDiffusion assemblies preserve NTG and
positions in `lobe` but distort body-scale structure (geobody W1 0.2121
vs a 0.0190 band; extent W1 0.0785 vs 0.0100). Two causes were
hypothesized: (i) cross-environment contamination from the intrinsically
elongated `channel:*` families, and (ii) velocity averaging as the fusion
rule. This addendum records the tests of both, and a candidate fix.

## G.2 Pre-scoring result: contamination refuted (computed 2026-08-01)

The lobe specialist (`lobe_80ep_r2/checkpoints/inference_epoch080.pt`,
union argmin, val 0.169334) was run through the unmodified E.3/E.4
pipeline at the identical E.7 condition, seeds, grid and overlap, with
`--split-seed 20260815`. Result (outputs
`$WORK/resbench_eval/assembly_lobe_specialist/`):

| comparison | \|dNTG\| | vario MAE | conn. MAE | geobody W1 | extent W1 |
|---|---|---|---|---|---|
| band | 0.0001 | 0.0041 | 0.0040 | 0.0190 | 0.0100 |
| A vs C (native) | 0.0189 | 0.0052 | 0.0128 | 0.0828 | 0.0371 |
| B vs C (assembly) | 0.0175 | 0.0127 | 0.0175 | 0.2592 | 0.0981 |
| B vs A (MD effect) | 0.0014 | 0.0117 | 0.0210 | 0.2240 | 0.0976 |

The lobe-only specialist degrades under tiling by the same margin as the
foundation (B vs A geobody 0.2240 vs 0.2423), so hypothesis (i) is
**refuted**: the effect is a property of the fusion rule, not of the
training mix. The specialist additionally acquires stride-locking the
foundation lacked (stride-40 amplitude 0.0231 mean / 0.0440 max vs an
engine-concatenation floor of 0.0112 / 0.0283).

Disclosed asymmetry: the foundation assembly used raw `flow_matching.pt`
while specialists have EMA checkpoints only. Uncorrected, as in E.

## G.3 Pre-scoring result: the crop marginal is not the driver

A second hypothesis — that training on whole 64^3 ResMill domains rather
than windows of a larger field distorts the tile marginal — was tested
directly. A new 200,000-volume `lobe` sweep at 192x192x32 (build grid
208x208x50, crop 8/8/9) reuses `build_jobs` with the production seed and
count, so it realizes the SAME parameter draws and volumes pair exactly
by ResMill seed. Over 150 matched pairs, native 64^3 vs a centred 64-crop
of 192^3: NTG +2.5%, body count +4.0%, median body +3.5%, mean chord
(major) +0.7%, chord anisotropy +0.6%, largest fraction -10.3%; binned by
requested `width_cells` there is no systematic gap for bodies wider than
the 64-cell cube. A few percent cannot account for a geobody W1 of 0.21,
so this mechanism is **minor**. Script:
`ResFlow_ls6/scripts/tier2/paired_marginal_check.py`.

## G.4 Intervention under test

Replace velocity averaging with conditioning. Blocks are generated in
raster order and each block receives its already-generated neighbours'
overlap region as known data through the existing 3-channel inpainting
interface ("outpainting"). Blocks on an anti-diagonal i + j = k are
mutually independent and are sampled as one batch.

This requires slab-shaped training masks, which no existing checkpoint
has seen. Two models are trained, both 80 epochs / 180,000 volumes /
468 steps per epoch at global batch 384 — **identical exposure and
schedule to the lobe specialist** — and identical to it in every other
respect (UNet3D, FlowMatching drop_prob 0.1, peak LR 1e-3*sqrt(12),
warmup 4 -> cosine T_max 76, EMA 0.9999, foundation `cond_stats.npz`):

- **native64** (torch seed 8201): original 64^3 lobe training split, mask
  distribution changed to 30% empty / 35% wells / 35% context slab.
  Isolates the MASK change.
- **crops192** (torch seed 8202): same masks, each sample a random
  64x64x32 crop of a 192x192x32 volume (deterministic 180,000-volume
  subset, `default_rng(20260801)` permutation; tail 20,000 held out).
  Isolates the DATA change on top of native64.

Context slabs are L-shaped (left and/or top arm), the configuration drawn
with raster-order frequencies for a 10x10 grid (1 : 9 : 9 : 81 for
none : left : top : both), and the slab width drawn uniformly from
[8, 32] cells so generation-time overlap can be swept without retraining.
Conditioning for crops is read from the PARENT volume, not recomputed on
the crop.

Checkpoint selection: validation argmin under the **unmodified E.3 rule**
(`eval_val_specialist.py`, lobe validation split, wells-only masks, K=4,
seed 20260901), so selection is identical across specialist, native64 and
crops192. Disclosed: native64/crops192 saw wells in 35% of training
samples rather than 70%, so this yardstick mildly disfavours them; the
bias is conservative with respect to any claim that they improve.

## G.5 Scored comparisons

At the E.7 condition, seeds, grid (10x10) and overlap (24), each model
produces: (A) 250 native volumes, (B1) 10 MultiDiffusion assemblies,
(B2) 10 outpainting assemblies. B1 and B2 differ **only** in the fusion
rule. Scored with the E.4 metrics and the frozen E.7 band
(`--split-seed 20260815`) against the published engine reference:
B1 vs C, B2 vs C, A vs C, and B2 vs B1. Headline claim is that
B2 vs C improves on B1 vs C in geobody W1 and extent W1.

Comparison chain: specialist -> native64 isolates the mask change;
native64 -> crops192 isolates the data change.

## G.6 New metrics (segmentation-free; `resbench/anisotropy.py`)

The E.4 body metrics rest on 6-connected labelling, which merges bodies
wherever sand amalgamates — and in `lobe` amalgamation is physically
real, so the engine reference has merged bodies too. Per-object width,
length or aspect ratio therefore inherits an arbitrary decision about
where one lobe ends. Primary estimators avoid segmentation entirely:

1. **Directional chord lengths** — linear intercepts along parallel scan
   lines at a given azimuth, pooled over z. Sampling is linear (order=1)
   thresholded at 0.5; nearest-neighbour sampling shatters chords that
   graze a rasterized boundary (measured aspect ratio 1.58 instead of
   2.00 at azimuth 45 on an analytic ellipse, and *worse* with a finer
   step). Chords touching a scan line's end are censored and dropped.
   `chord_anisotropy` = mean chord along the azimuth / mean chord
   perpendicular; W1 is taken on log10 chord lengths, matching the
   `geobody_w1` convention.
2. **Directional variogram ranges** — gamma(h) along the azimuth and its
   perpendicular; practical range = first lag reaching 0.95 of the
   **theoretical** indicator sill p(1-p) (not the empirical max of gamma,
   which a hole effect can push out of reach); anisotropy = range ratio.
   Structure longer than max_lag returns NaN rather than max_lag.

Validation: for an isolated ellipse the mean chord along a semi-axis is
analytically a*pi/2, so the anisotropy ratio must recover the aspect
ratio exactly, independent of size; recovered to within 3% at azimuths
0/45/90 (`tests/test_anisotropy.py`, 19 tests).

Disclosed limitation: a residual rasterization bias up to ~13% remains at
off-axis azimuths (chord splitting). It cancels whenever both sides are
measured at the same azimuth, which every scored comparison here is, and
absolute anisotropy is therefore **never** compared against a requested
`asp`. A `gap_close` parameter reduces it to ~3% (saturating at 2 cells)
but **defaults to 0 and stays there for body-scale scoring**, because
closing 1-2 cell gaps would erase exactly the thin mud drapes whose
welding is the effect under study; a regression test guards this.

3. **Conditioning adherence** — realized size estimators regressed
   against requested `width_cells` / `depth_cells` / `asp` over a Sobol
   sweep of the conditioning range. The reference is the **engine's own**
   calibration curve at the same requested values (the engine does not
   hit its targets exactly either); the score is the distance between the
   model's curve and the engine's. Size metrics are always
   model-vs-engine, never model-vs-requested-value.

## G.7 Standing component

A model claiming beyond-training-extent generation may submit either
fusion rule; both are scored against the same published reference with
the E.4 metrics plus G.6, and the fusion rule is reported alongside.
