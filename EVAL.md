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
