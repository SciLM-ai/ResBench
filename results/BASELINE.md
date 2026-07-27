# Published-baseline comparison — channel:PV_SHOESTRING (2026-07-27)

Executed per the pre-registered protocol in `EVAL.md` Addendum D (frozen
before baseline training). Scoring used the unmodified `resbench.run` CLI
and the frozen PV_SHOESTRING test reference (512 volumes), touched once per
ensemble at final scoring. The frozen master table is unchanged.

## Which path ran

**The primary path ran: GANSim-3D via the authors' official GANSim3D_v2
code** (github.com/SuihongSong/GANSim3D_v2 @ `02677daf`), unconditional
configuration (its shipped default), trained from scratch on the full
90,000-volume PV_SHOESTRING training split at the native (64, 64, 32) grid
(no padding), WRR-2022 paper-default schedule (3,600 kimg), one attempt,
seed 8001. Training completed with no divergence and no retry;
**the DDPM fallback was never triggered.** Cost: 7.9 GPU-h on one GH200
(ceiling: 48). Checkpoint selected by the pre-registered training-split
proxy (D.4): `network-snapshot-002720.pkl` (proxy 0.01488; full trace in
`baseline/manifests/proxy_scores.json`).

## Combined table

Rows scored against the frozen PV_SHOESTRING reference; verdicts vs the
master-table split-half band (per Addendum D.5). `well mismatch %` is n/a
for all three rows (distribution-level comparison; Addendum D.1).

| row | \|dNTG\| | variogram MAE/sill | connectivity MAE | geobody W1 (log10) | \|d largest frac\| | well mismatch % |
|---|---|---|---|---|---|---|
| split-half band (master) | 0.0050 | 0.0115 | 0.0014 | 0.0125 | 0.0117 | — |
| GANSim-3D (unconditional) | 0.0013 (inside) | 0.0114 (inside) | 0.0097 (outside) | 0.3881 (outside) | 0.0195 (near) | n/a |
| ResFlow env-only (marginalized) | 0.0117 (outside) | 0.0266 (outside†) | 0.0041 (outside) | 0.0239 (near) | 0.0501 (outside) | n/a |
| ResFlow param-conditioned (copied from master table; conditions on per-instance parameters, an input rows 1–2 do not receive) | 0.0069 (near) | 0.0202 (near) | 0.0034 (outside) | 0.0210 (near) | 0.0443 (outside) | 0.0 (inside) |

† Single-environment scoring runs draw a different split-half permutation
than the 8-environment master run (the permutation seed includes the
environment's index within the run), giving a run-native band of 0.0138
for this cell; under that band this verdict is `near` (0.0266 ≤ 2×0.0138).
This is the only cell whose verdict differs between the two bands. Raw
run outputs are preserved unmodified in `baseline/manifests/*.parquet`.

## Per-metric summaries (factual)

**Facies proportion.** GANSim-3D ensemble mean NTG 0.2235 vs reference
0.2249 (|Δ| = 0.0013, inside the band). ResFlow env-only 0.2365
(|Δ| = 0.0117); the parameter-conditioned row, which receives each test
instance's NTG target as input, sits at 0.2318 (|Δ| = 0.0069).

**Indicator variograms.** GANSim-3D's sill-normalized MAE (0.0114) is
inside the band and is the lowest of the three rows; the overlays
(`baseline/variogram_overlay_baseline.pdf`) show all three models tracking
the reference curves in x, y, z, with the largest deviations at mid-range
x-lags (GANSim) and long y-lags (ResFlow env-only).

**Connectivity.** All three rows are outside the band. GANSim-3D's τ(h)
MAE (0.0097) is 2.9× the ResFlow env-only value (0.0041) and 2.4× the
range of the two ResFlow rows; the τ overlays show GANSim's connectivity
decaying faster with lag on all three axes.

**Geobody sizes.** GANSim-3D's W1 on log10 sizes is 0.3881 — 16× the
ResFlow env-only value (0.0239) and ~31× the band. The size-filter
sensitivity (post-hoc machinery, `min_size ∈ {1,2,8}`) gives 0.388 /
0.322 / 0.674, i.e. the mismatch is not explained by single-voxel
speckle alone. The CDF panel shows GANSim producing a substantially
heavier population of small-to-mid-size bodies than the reference.

**Largest-geobody fraction.** GANSim-3D |Δ| = 0.0195 (near; median
largest-body fraction 0.982 vs reference 0.995). Both ResFlow rows are
outside on this cell (0.0443 / 0.0501), in the opposite direction
(over-connected single body, medians ≈ 0.994–0.995 with narrowed IQR).

**Per-volume NTG spread (mode-collapse diagnostic).** Reference sd 0.0811
(range 0.067–0.412); GANSim-3D sd 0.0711 (range 0.057–0.440); ResFlow
env-only sd 0.0852 (range 0.081–0.404). GANSim shows a mild narrowing
concentrated near the mode (histogram overlay,
`baseline/ntg_hist_baseline.pdf`), not a collapsed spread.

## Per-volume compartmentalization (extension of the post-hoc table)

Reference column recomputed identically in each run (values agree bit-for-bit).

| quantity | reference | GANSim-3D | ResFlow env-only | ResFlow param-cond |
|---|---|---|---|---|
| % volumes with τ_z < 0.99 | 28.7 [25.0, 32.8] | 45.5 [41.2, 49.8] | 27.0 [23.3, 31.0] | 27.0 [23.3, 31.0] |
| % volumes floor-to-surface spanning | 74.0 [70.1, 77.6] | 86.5 [83.3, 89.2] | 85.4 [82.0, 88.2] | 85.2 [81.8, 88.0] |
| largest-body fraction, median | 0.9951 | 0.9821 | 0.9947 | 0.9943 |
| largest-body fraction, IQR | [0.903, 0.998] | [0.938, 0.994] | [0.979, 0.998] | [0.980, 0.998] |

## Figures (PDF + PNG in `results/baseline/`)

1. `variogram_overlay_baseline` — γ(h) x/y/z panels, three model curves +
   reference and split-half band.
2. `connectivity_geobody_baseline` — τ(h) x/y/z panels and pooled
   geobody-size CDF (four panels; the τ overlays and the size CDF are the
   two protocol figures, combined in one file).
3. `ntg_hist_baseline` — per-volume NTG histograms (reference, GANSim-3D,
   ResFlow env-only), 512 volumes each.

## Run manifests

`results/baseline/manifests/`: feasibility probe (container SHA-256, port
shims, measured throughput), training launch (node, config, commit,
90,000-volume TFRecords conversion with bit-exact round-trip check),
proxy trace (14 snapshots, selection), generation manifests for both
ensembles (seeds 20260806 / 20260807+20260808·1000+k, checkpoint MD5s
`0f79097a2026cccc7de17d4a2b3202fc` (GANSim snapshot 2720) /
`759a047f0ea609b9e0c1926aaf55fbf1` (ResFlow `flow_matching.pt`)), and the
raw scoring parquets. Code: `ResBaselines` repo (sibling to this one).

## Anomalies

- First probe training attempt OOM'd on node c608-061: the GH200 exposes
  its HBM as a NUMA node and ~90 GB was occupied by Linux page cache from
  prior file I/O; the run was moved to c622-042. (Cache was later
  reclaimed with a node-bound allocation; no code implication.)
- Four port shims were required to run the 2021-era codebase under the
  2025 container (Python 3.12 `imp` stub, Pillow install, one `int()`
  cast in `dataset.py`, one reload-path type-check widening); listed in
  Addendum D.1 and the probe manifest; none affect training semantics.
- The first proxy-evaluation invocation crashed before producing any
  scores (a graph-reset bug in the new `proxy_eval.py` tooling, not in
  vendor code); fixed by loading all snapshots into one session. No
  selection numbers were produced by the crashed run.
- Training throughput blipped once (tick 16: 19.9 s/kimg vs 15.8–16.6
  steady) with no recurrence.
- The proxy trace is non-monotone across snapshots (range 0.0149–0.0819),
  and the selected snapshot (kimg 2,720) precedes the final one; the
  pre-registered minimum rule was applied as frozen.
- Split-half-band permutation differs between single-environment runs and
  the master run (see † above); verdicts in the combined table use the
  master band as pre-registered.
- Unconditional ensembles have no per-instance correspondence to the
  reference; reference ids were assigned to generated volumes nominally,
  in generation order, solely to satisfy the scoring join (Addendum D.5).
- `EVAL.md` Addendum D shares its file with the uncommitted Addendum C
  workstream; committing `EVAL.md` should be coordinated with that
  workstream's owner.
