# ResFlow vs SiliciclasticReservoirs test distribution — ResBench results

Run 2026-07-26 under the frozen protocol in [EVAL.md](EVAL.md). Reference:
512 held-out test instances per environment (manifest seed 20260726).
Ensemble (a): 4,096 parameter-conditioned generations (Table 6 settings:
Euler, NFE 50, CFG 3.0, raw `flow_matching.pt`, md5 in
`results/run_manifest.json`). Ensemble (b): 2,048 well-conditioned
generations (Figure-3 well configs cycled 1/2/3 wells). Generation: single
GH200, 23.2 min total. Verdicts are against the ResMill split-half
Monte-Carlo band: **inside** (≤ band), **near** (≤ 2× band), **outside**.

## Master table

| environment | \|dNTG\| | variogram MAE/sill | connectivity MAE | geobody W1 (log10) | \|d largest frac\| | well mismatch % |
|---|---|---|---|---|---|---|
| lobe | 0.0026 (inside; band 0.0253) | 0.0121 (near; band 0.0065) | 0.0356 (near; band 0.0274) | 0.0588 (near; band 0.0305) | 0.0227 (inside; band 0.0386) | 0.000 (inside) |
| channel:PV_SHOESTRING | 0.0069 (near; band 0.0050) | 0.0202 (near; band 0.0115) | 0.0034† (outside; band 0.0014) | 0.0210 (near; band 0.0125) | 0.0443 (outside; band 0.0117) | 0.000 (inside) |
| channel:CB_LABYRINTH | 0.0025 (inside; band 0.0058) | 0.0176 (inside; band 0.0226) | 0.0024† (outside; band 0.0009) | 0.0139 (inside; band 0.0176) | 0.0189 (outside; band 0.0011) | 0.000 (inside) |
| channel:CB_JIGSAW | 0.0048 (inside; band 0.0091) | 0.0186 (near; band 0.0113) | 0.0010 (inside; band 0.0010) | 0.0129 (near; band 0.0067) | 0.0087 (outside; band 0.0038) | 0.000 (inside) |
| channel:SH_DISTAL | 0.0104 (inside; band 0.0189) | 0.0364 (outside; band 0.0122) | 0.0001† (outside; band 0.0000) | 0.9709 (outside; band 0.1344) | 0.0000 (inside; band 0.0001) | 0.000 (inside) |
| channel:SH_PROXIMAL | 0.0087 (inside; band 0.0171) | 0.0276 (outside; band 0.0067) | 0.0003 (inside; band 0.0003) | 0.0177 (inside; band 0.0189) | 0.0011 (inside; band 0.0017) | 0.000 (inside) |
| channel:MEANDER_OXBOW | 0.0023 (inside; band 0.0044) | 0.0136 (near; band 0.0069) | 0.0005† (outside; band 0.0001) | 0.0077 (outside; band 0.0028) | 0.0026 (near; band 0.0016) | 0.000 (inside) |
| delta | 0.0078 (inside; band 0.0117) | 0.0150 (outside; band 0.0054) | 0.0009 (inside; band 0.0027) | 0.0308 (near; band 0.0234) | 0.0085 (inside; band 0.0182) | 0.000 (inside) |
| **pooled** | 0.0034 (inside; band 0.0060) | 0.0133 (outside; band 0.0039) | 0.0054 (outside; band 0.0021) | 0.0143 (inside; band 0.0219) | 0.0109 (near; band 0.0075) | 0.000 (inside) |

† **Band saturation.** In these environments the sand phase percolates, so
τ(h) ≈ 1 with almost no ensemble variance and the split-half band collapses
to 10⁻⁵–10⁻³. The absolute MAE printed in the cell is the primary read
(0.0001–0.0034 — τ reproduced to within ~0.3% of its full [0, 1] scale);
the inside/near/outside verdict is secondary wherever the band saturates.

Well mismatch by config, pooled: 1 well 0.000% · 2 wells 0.000% · 3 wells 0.000%.

*Pooled geobody-W1 arithmetic:* the pooled row compares the *pooled body
populations* of all environments, so environments with many bodies per
volume (CB jigsaw ~120, meander-oxbow ~78) dominate the pooled distribution
while SH distal's ~1–3 bodies/volume barely register — which is why pooled
W1 (0.0143, inside) can coexist with the large SH-distal cell. The
per-environment rows are the informative ones for this column.

Figures: [NTG parity](results/ntg_parity.pdf) ·
[variogram overlays](results/variogram_overlays.pdf) ·
[connectivity + geobody CDFs](results/connectivity_geobody.pdf)
(PNG versions alongside; all curves/CIs in `results/report.npy`).

## Per-metric interpretation

**Facies proportion (NTG).** Inside the band in 7 of 8 environments and
pooled (|ΔNTG| ≤ 0.0104 everywhere); PV shoestring is *near* (0.0069 vs band
0.0050). The parity plot sits on the diagonal across the full NTG range
0.22–0.62 with overlapping bootstrap CIs. Six of eight deltas are negative
(slight under-generation of sand), a consistent sign worth noting though
each is small.

**Indicator variograms.** Normalized MAE is 1.2–3.6% of the sill. The x/y
(horizontal) curves overlay the reference within or at the split-half band
in every environment; essentially the entire MAE comes from the **z axis**,
where ResFlow's short-range γ_z plateau undershoots the reference by
~0.02–0.03 in the sheet/channel environments (SH distal, SH proximal, delta
are *outside* at 2–5× band). Generated volumes are slightly more continuous
vertically than ResMill's. The delta z-curve plateaus far below the sill in
*both* reference and generated volumes (zonal anisotropy of the data
itself; this triggers the expected sanity warning, not a model artifact).

**Connectivity function τ(h).** In the six high-NTG channel/sheet
environments the sand phase percolates, so τ ≈ 1 at all lags and the
split-half bands are extremely tight (≤ 0.003); four *outside* verdicts
there correspond to absolute MAEs of 0.0001–0.0034 — statistically
detectable, practically negligible. The informative panel is lobe (τ decays
to ~0.6): ResFlow's τ sits above the reference on all three axes
(MAE 0.036, *near* a 0.027 band), i.e. mild over-connection of lobes.

**Geobody size distribution (W1 on log₁₀ sizes).** Inside or near in 6 of 8
environments and inside pooled (0.0143 vs band 0.0219). Meander-oxbow is
*outside* but small in absolute terms (0.0077 ≈ 1.8% size shift). SH distal
is the one large deviation (0.97 decades) — see anomalies.

**Largest-geobody fraction.** Inside for lobe, SH distal/proximal, delta.
The three channel-belt families are *outside* with generated > reference:
PV shoestring 0.904 → 0.949, CB labyrinth 0.963 → 0.982 (bootstrap CIs
disjoint in both), CB jigsaw 0.978 → 0.986. Together with the z-variogram
undershoot and the lobe τ excess this is a coherent mild
**over-connectivity** signature.

**Well exactitude.** 0.000% mismatch over 8 environments × {1, 2, 3}-well
configs (130,816 conditioning voxels). This is exactitude *by construction*:
Table 6 specifies hard replacement of well voxels after the final ODE step,
so the number verifies **pipeline integrity** — that masks, carved well
values, and generated volumes stay aligned end to end through the whole
harness — rather than a learned ability. How well the model *learns* to
honor well data away from the replaced voxels is assessed by the
conditional-entropy calibration (EVAL.md Addendum A; results below).

## Supplementary connectivity diagnostics (post-hoc, reference ensemble)

Checks run after the frozen-protocol scoring to interpret the τ(h) and
geobody columns; they do not alter the master table.

**How τ(h) aggregates.** The plotted/scored curve is the *pooled* ratio
(Σ same-body pairs ÷ Σ both-sand pairs over all 512 volumes), so it is
pair-count-weighted: low-NTG fragmented volumes contribute fewer sand pairs
and are down-weighted. τ ≈ 1 therefore means "one body holds nearly all the
sand *mass*", not "there is one body": pair counts scale ~quadratically
with body size, so speck populations are invisible to τ.

**Body-mass structure per environment** (medians over 128 reference
volumes; "big" = ≥1,000 voxels):

| environment | bodies/vol | big/vol | largest body % of sand | top-3 % |
|---|---|---|---|---|
| lobe | 56 | **8** | **31%** | 60% |
| channel:PV_SHOESTRING | 39 | 1 | 99% | 100% |
| channel:CB_LABYRINTH | 58 | 1 | 100% | 100% |
| channel:CB_JIGSAW | 120 | 1 | 99% | 100% |
| channel:SH_DISTAL | 1 | 1 | 100% | 100% |
| channel:SH_PROXIMAL | 27 | 1 | 100% | 100% |
| channel:MEANDER_OXBOW | 78 | 1 | 100% | 100% |
| delta | 33 | 1 | 100% | 100% |

Seven of eight environments carry a single dominant sand network plus a
cloud of small fragments; **lobe is the only multi-body environment**
(~8 comparable bodies), which is why it is the only panel where τ(h)
decays and the most discriminating environment for connectivity.

**Connectivity is structural, not a sand-fraction artifact.** A real
PV-shoestring volume at NTG 0.309 has 26 bodies with the largest holding
99.1% of sand; the same number of sand voxels placed at random shatters
into 7,577 bodies with the largest holding 5.2% (random site percolation
at p = 0.31 < p_c ≈ 0.312 does not percolate under 6-connectivity).
Long channel bodies crossing anywhere merge into one spanning network.

**Vertical (z) connection is usually indirect and not universal**
(per-volume, all 512 reference volumes):

- PV shoestring: 147/512 volumes (29%) have per-volume τ_z < 0.99
  (minimum 0.41 with largest-body fraction 0.27 — genuinely
  compartmentalized, isolated shoestrings in mud); 379/512 (74%) have a
  single body spanning floor to surface. Ensemble mean largest-fraction
  0.90 vs median 0.99 — a compartmentalization tail the pooled τ curve
  down-weights. Delta shows a smaller tail (mean 0.91 vs median ~1.0).
- Multi-storey meander-oxbow: 506/512 span floor-to-surface, 14/512 with
  τ_z < 0.99 — vertical amalgamation by construction.
- "Connected along z" in τ_z routes through lateral amalgamation points,
  not necessarily continuous vertical sand columns.

**Rebuttal-relevant reading.** The discriminating connectivity targets in
this benchmark are (i) lobe body-scale separation and (ii) the
PV-shoestring/delta compartmentalization tails — and ResFlow's one
systematic bias (largest-fraction 0.90 → 0.95 PV, 0.96 → 0.98 labyrinth,
lobe τ above reference) sits exactly there: slight over-merging in the
environments where merging is the live geological variable. A per-volume
compartmentalization-frequency comparison (reference vs ResFlow) would
quantify this and is a natural supplementary table if reviewers press on
connectivity.

## Post-hoc analyses (EVAL.md Addendum A.6 — diagnostics, not re-scoring)

### Geobody W1 vs minimum body size (item 2)

Full table: `results/posthoc/posthoc_geobody.md`. Key cells (W1, verdict
against the same-filter split-half band):

| environment | min ≥ 1 (frozen) | min ≥ 2 | min ≥ 8 |
|---|---|---|---|
| channel:SH_DISTAL | 0.9709 (outside; 0.1344) | 0.7078 (outside; 0.1122) | 0.5410 (outside; 0.1083) |
| lobe | 0.0588 (near) | 0.1565 (outside) | 0.0538 (inside) |
| pooled | 0.0143 | 0.0491 | 0.1079 |

**The speck-filter expectation is REFUTED for SH distal**: removing 1-voxel
(min ≥ 2) and sub-8-voxel bodies lowers W1 (0.97 → 0.71 → 0.54) but it stays
far outside its band — the generated excess of small fragments extends into
the 2–100-voxel range, not just single-voxel debris. Elsewhere the filter is
not monotone (removing speck mass re-weights the comparison toward mid-size
bodies where discrepancies differ), so the min ≥ 1 frozen column remains the
primary read and the anomalies list below is updated accordingly.

### Per-volume compartmentalization: reference vs ResFlow (item 3)

Full table with Wilson 95% CIs: `results/posthoc/posthoc_geobody.md`.
Headline rows (% volumes with per-volume τ_z < 0.99; % volumes with a
floor-to-surface spanning body; largest-frac median [IQR]):

- **channel:PV_SHOESTRING** — compartmentalization *frequency* is
  reproduced: 28.7% [25.0, 32.8] ref vs 27.0% [23.3, 31.0] gen
  (overlapping CIs). But vertical *spanning* is over-produced: 74.0%
  [70.1, 77.6] ref vs 85.2% [81.8, 88.0] gen (disjoint), and the
  largest-frac lower quartile collapses (ref IQR [0.903, 0.998] vs gen
  [0.980, 0.998]): ResFlow generates compartmentalized volumes at the right
  *rate* but under-produces the *severely* compartmentalized ones.
- **channel:CB_LABYRINTH** — fragmentation under-produced outright: 18.4%
  [15.2, 21.9] ref vs 11.5% [9.0, 14.6] gen; spanning 90.8% vs 95.1%
  (both disjoint).
- **lobe** — 88.5% [85.4, 91.0] ref vs 82.2% [78.7, 85.3] gen (disjoint):
  modest over-connection, consistent with the τ overlay.
- SH distal/proximal, meander-oxbow, CB jigsaw, delta: ref and gen agree
  within CIs on all three statistics.

### CFG sensitivity sweep (item 4)

lobe + channel:PV_SHOESTRING, 128 samples per point at CFG {1.0, 1.5, 2.0,
3.0}, fresh logged seeds, deltas vs the **matched 128 reference rows**
(`results/posthoc/cfg_sweep.md`, figure `cfg_sweep.pdf/png`):

- **The over-smoothing/over-connection signature persists at CFG 1.0** —
  γ_z plateau deviation is flat in CFG for lobe (−0.006 ± 0.001 at every
  scale) and small at all scales for PV; lobe compartmentalization
  frequency sits at ~76% vs ref 84% at *every* scale; PV largest-fraction
  excess (~0.92–0.94 vs ref 0.90) persists at every scale. The
  mode-seeking-CFG hypothesis is **refuted as the primary cause**: the
  signature is intrinsic to the learned distribution (training data
  processing, architecture, or the 50-step Euler discretization).
- What CFG *does* modulate: NTG rises monotonically with scale (lobe
  −0.011 at CFG 1.0 → +0.002 at 3.0, crossing zero near the deployed
  setting; PV +0.003 → +0.007), and PV compartmentalization frequency
  *improves* with scale (39.1% at 1.0 → 30.5% at 3.0, ref 27.3%) — at the
  deployed CFG 3.0 both sweep environments are at or near their best
  operating point among the scales tested.
- Caveat: 128 samples/point → Wilson ±8% on frequency estimates;
  directional conclusions above exceed that, per-cell values are noisier
  than the master table's.

## Conditional-entropy calibration (EVAL.md Addendum A — scored)

Design per Addendum A: 32 conditions (8 envs × 4 ensemble-(b) rows covering
1/2/3-well configs); model entropy from K = 128 conditioned samples per
condition; reference from N = 512 unconditional-under-parameters ResMill
realizations per condition (engine reproduction validated **bit-exact**
against stored dataset instances in all 8 environments before generation).
H = 0 at well voxels holds by construction (hard replacement) and is
asserted, not counted as calibration evidence. Full per-condition table:
`results/posthoc/entropy_calibration.md`; figures
`entropy_vs_distance.{pdf,png}`, `entropy_maps.{pdf,png}`.

**Rejection feasibility (A.4).** Acceptance measured on the 512 draws per
condition: only two conditions are tractable — SH distal r0 1-well (4.1%)
and SH proximal r3 1-well (2.0%). Targeted rejection delivered **203
accepted / 6,656 draws** and **175 / 12,000 (draw-capped)** respectively.
All other conditions: 0/512 accepted (rate < 0.6% at 95% confidence);
with measured engine costs of 1.6–94 s/realization, reaching 200 accepted
projects to hundreds of core-hours or worse (e.g. CB labyrinth 1-well:
0.2% → ~450 core-h) — those conditions get far-field-only calibration, as
the protocol provides.

**Far field (the headline).** Mean model-minus-unconditional entropy
offset in bins at Chebyshev distance ≥ 15 (split-half band of the
reference level in parentheses):

| lobe | PV shoe. | CB lab. | CB jig. | SH dist. | SH prox. | m-oxbow | delta |
|---|---|---|---|---|---|---|---|
| **−0.02 (0.04)** | −0.12 (0.04) | −0.18 (0.03) | −0.15 (0.04) | −0.20 (0.03) | −0.21 (0.05) | −0.17 (0.03) | −0.15 (0.04) |

**Lobe is calibrated** (inside its band; per-condition convergence
distances 3–19 voxels). **All seven channel/delta environments are
systematically under-dispersed**: conditional entropy sits 0.12–0.21 bits
below the engine's own uncertainty level even far from wells (3–6× the
band), so the model curve never enters the reference band (conv. distance
> 24). This is the miscalibration direction that matters for UQ — the
model is over-confident about voxels the wells say nothing about — and it
is the entropy-space image of the over-smoothing/over-connection signature
(both survive at CFG 1.0).

**Near field (where a rejection reference exists).** Mean |H_model −
H_ref| binned by distance: 0.097 bits (SH distal, n = 203) and 0.120 bits
(SH proximal, n = 175); both flagged **under-dispersed** in the near-field
bins as well. The entropy-map figure (SH distal, 1-well) shows the
mechanism: both fields correctly close the uncertainty funnel at the well,
but ResFlow prints large confident (near-zero-entropy) regions between its
uncertainty bands where the engine keeps broad ~0.8-bit uncertainty.

Estimator caveats (small vs the effect): plug-in entropy bias is ≈ −0.006
bits at K = 128, ≈ −0.004 at n ≈ 200, and the near-field reference exists
only for 1-well configs in two sheet environments.

## Anomalies to look at before writing the rebuttal

- **Systematic conditional under-dispersion (entropy calibration)** — the
  strongest new finding: in all seven channel/delta environments the
  model's conditional voxelwise entropy runs 0.12–0.21 bits below the
  engine's true uncertainty at all distances from wells (3–6× the
  reference noise band); lobe is calibrated (−0.02). Confirmed near-field
  by rejection references in both tractable conditions (0.10–0.12 bits,
  under-dispersed). Any downstream use of ResFlow ensembles for
  uncertainty quantification in the channel/delta families will
  understate uncertainty by roughly that margin.
- **SH distal geobody W1 = 0.97 (7× band)**: an excess population of small
  fragments (5.2 vs 2.6 bodies/volume; 50% vs 30% singletons) around a
  correctly reproduced giant sheet (max size, p90, |Δ largest frac| =
  0.0000 all match). *Post-hoc update (item 2)*: a minimum-body-size filter
  does **not** move the cell inside the band (0.97 → 0.71 at min ≥ 2 →
  0.54 at min ≥ 8, band ≈ 0.11) — the fragment excess extends into the
  2–100-voxel range, not just 1-voxel debris. Still ~0.004% of voxels; it
  dominates only because the environment has ~3 bodies/volume.
- **Systematic vertical over-smoothing / over-connection**: γ_z plateau
  undershoot in five environments, largest-fraction up in the channel-belt
  families, lobe τ up. *Post-hoc update (item 4)*: the CFG sweep **refutes
  guidance mode-seeking as the primary cause** — the signature persists
  essentially unchanged at CFG 1.0. It is intrinsic to the learned
  distribution (candidate causes: training-data processing, purely
  convolutional architecture's receptive field in z, or the 50-step Euler
  discretization). CFG mainly shifts NTG (monotone, crossing zero near the
  deployed 3.0 for lobe) and *improves* PV compartmentalization frequency.
- **PV shoestring largest-fraction +0.044 with disjoint CIs**: *post-hoc
  update (item 3)*: the model reproduces the *rate* of compartmentalized
  volumes (27.0% vs 28.7% ref, overlapping CIs) but under-produces the
  *severely* compartmentalized tail (largest-frac lower quartile 0.980 vs
  0.903 ref) and over-produces floor-to-surface spanning (85.2% vs 74.0%,
  disjoint). CB labyrinth under-produces fragmentation outright (11.5% vs
  18.4%). Also the only environment whose NTG delta is *near* rather than
  inside.
- **Tight-band caveat for τ**: split-half bands of 10⁻⁴–10⁻³ make
  connectivity verdicts hypersensitive; absolute MAEs are ≤ 0.0034 in all
  four *outside* cells. Report absolute values alongside verdicts.
- **NTG sign consistency**: 6/8 environments slightly under-generate sand;
  all within or near band individually, pooled |ΔNTG| = 0.0034 (inside).
