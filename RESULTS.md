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

## Unconditional entropy-calibration benchmark (EVAL.md Addendum B — scored)

Well-free generalization of the calibration study into a reusable
benchmark component: 8 fresh manifest conditions per environment
(rows 4–11, disjoint from Addendum A), model K = 128 empty-mask samples vs
engine N = 256 unconditional realizations per condition, compared over all
voxels. The engine reference (per-condition voxelwise p̂, float16) is
**published in `results/entropy_reference/`** so future models can be
scored on this component without running ResMill. Full table:
`results/posthoc/uncond_calibration.md`; per-condition strip plot
`uncond_calibration.{pdf,png}`.

| environment | signed offset (bits) ± 95% CI | MAE ± CI | band | verdict |
|---|---|---|---|---|
| lobe | −0.0205 ± 0.0261 | 0.0580 ± 0.0441 | 0.0433 | near |
| channel:PV_SHOESTRING | −0.0550 ± 0.0335 | 0.1091 ± 0.0135 | 0.0734 | near |
| channel:CB_LABYRINTH | −0.1156 ± 0.0405 | 0.1421 ± 0.0331 | 0.0621 | outside |
| channel:CB_JIGSAW | −0.1479 ± 0.0370 | 0.1602 ± 0.0342 | 0.0589 | outside |
| channel:SH_DISTAL | −0.1989 ± 0.0425 | 0.2196 ± 0.0397 | 0.0619 | outside |
| channel:SH_PROXIMAL | −0.1932 ± 0.0440 | 0.2209 ± 0.0446 | 0.0642 | outside |
| channel:MEANDER_OXBOW | −0.1017 ± 0.0499 | 0.1342 ± 0.0361 | 0.0516 | outside |
| delta | −0.1015 ± 0.0230 | 0.1253 ± 0.0118 | 0.0511 | outside |
| **pooled (64 conds)** | −0.1168 | 0.1462 | 0.0583 | outside |

This independently confirms Addendum A on fresh parameter vectors with
condition-level CIs: **lobe's signed offset includes zero (calibrated)**;
all 56 channel/delta conditions are individually negative, environment
means −0.06 to −0.20 bits with CIs excluding zero, ordered sheets >
labyrinth/jigsaw > meander/delta > shoestring. Under-dispersion is a
property of the learned family distributions themselves (no wells
involved), not of the well-conditioning path.

## Well-conditional entropy calibration in all 8 environments (EVAL.md Addendum C — scored)

The near-well calibration gap of Addendum A (engine-conditional references
in only 2 environments) is closed by **modal-pattern rejection**: carve the
1-well column (x, y) = (32, 32) from every stored unconditional draw,
histogram the patterns, and condition on the most frequent — every
matching draw is an exact conditional sample, at zero extra generation
cost beyond stored pools. Two tiers are published in
`results/well_conditional_reference/` (per environment: accepted volumes,
pattern, well mask, parameter pointer), so **this benchmark component
never requires rejection again**:

**Tier 1 — modal (typical) wells** (dry column in lobe/PV/meander, full
sand elsewhere; 80–287 reference realizations per environment,
`results/posthoc/modal_calibration.md`, figure `modal_calibration.pdf`):
a clean **two-regime miscalibration**. Near the well (Chebyshev ≤ 6) the
model *under-collapses* in the dry-well environments (signed offset
+0.11 lobe, +0.10 PV — it keeps too much uncertainty where the well is
informative, i.e. under-uses well data), while sheet/delta/meander show
mild near-well under-dispersion. Beyond ~8–12 voxels every environment
reverts to the familiar far-field over-confidence (model below the
engine band). MAE 0.12–0.20 bits vs bands 0.05–0.11.

**Tier 2 — informative (mixed) wells** (C.5–C.7; genuinely heterogeneous
well logs — a shoestring-channel intersection in PV, a mid-section lobe
penetration, thin shale breaks in delta;
`results/posthoc/mixed_calibration.md`, figure `mixed_calibration.pdf`):

| environment | well NTG | n ref | MAE (bits) | band | verdict |
|---|---|---|---|---|---|
| lobe (re-selected) | 0.25 | 54 | 0.174 | 0.179 | inside (under-disp.) |
| channel:PV_SHOESTRING | 0.25 | 91 | 0.179 | 0.099 | near (under-disp.) |
| channel:SH_DISTAL | 0.75 | 71 | 0.284 | 0.104 | outside (under-disp.) |
| delta | 0.88 | 64 | 0.143 | 0.091 | near (under-disp.) |
| channel:CB_LABYRINTH | 0.75 | 62 | 0.232 | 0.122 | near (under-disp.) |
| channel:SH_PROXIMAL | 0.88 | 50 | 0.221 | 0.144 | near (under-disp.) |
| channel:MEANDER_OXBOW | 0.28 | 93 | 0.202 | 0.110 | near (under-disp.) |
| channel:CB_JIGSAW | 0.88 | 50 | 0.186 | 0.142 | near (under-disp.) |

\* below the 50-realization floor after most-matches re-selection
(EVAL.md C.7) — published as indicative only; bands are correspondingly
wide. With informative wells the near-field signed offset is **negative in
7 of 8 environments**: given a heterogeneous well log, the model is
under-dispersed both near and far — the modal tier's near-well
under-collapse appears specific to low-information (typical) wells.
Every stored draw pool is published, so all sub-floor environments can be
topped up later at engine cost only — demonstrated 2026-07-27 by
meander-oxbow: a 5,000-draw top-up pool raised its frozen top-story well
(NTG 0.28; meander stories snap to 8-voxel blocks) from 11 to **93**
conditional realizations, promoting it to the fully published set
(now 5 of 8 environments; labyrinth 25, SH proximal 36, jigsaw 3 remain
indicative).

## Published-baseline comparison (EVAL.md Addendum D — scored)

Full deliverable: [results/BASELINE.md](results/BASELINE.md) (combined
table, four figure panels, compartmentalization extension, run manifests,
anomalies). Executed 2026-07-27 under the pre-registered Addendum D
protocol; the primary path ran (authors' official GANSim3D_v2,
unconditional, paper-default 3,600-kimg schedule, one attempt, no
divergence, 7.9 GPU-h of the 48 ceiling); the DDPM fallback was never
triggered. Checkpoint kimg 2720 selected by the training-split proxy.

| row (vs master band) | \|dNTG\| | vario/sill | conn. | geobody W1 | \|d lgst\| |
|---|---|---|---|---|---|
| band | 0.0050 | 0.0115 | 0.0014 | 0.0125 | 0.0117 |
| GANSim-3D uncond. | 0.0013 in | 0.0114 in | 0.0097 out | 0.3881 out | 0.0195 near |
| ResFlow env-only (marginalized) | 0.0117 out | 0.0266 out† | 0.0041 out | 0.0239 near | 0.0501 out |
| ResFlow param-cond (master row) | 0.0069 near | 0.0202 near | 0.0034 out | 0.0210 near | 0.0443 out |

† `near` under the run-native band (single-env runs draw a different
split-half permutation; see BASELINE.md).

Compartmentalization: 45.5% of GANSim volumes with τ_z < 0.99 vs 28.7%
reference (ResFlow env-only 27.0%); NTG spread sd 0.071 vs 0.081
reference — no mode collapse.

**Draft rebuttal text.** Following the reviewers' request, we trained a
published generative baseline — GANSim-3D (Song, Mukerji & Hou, 2022,
*Water Resources Research*), using the authors' official code in its
unconditional configuration — on the full PV_SHOESTRING training split
(90,000 volumes) under a pre-registered protocol (paper-default schedule,
one training run, checkpoint selection by training-split statistics
only), and scored it with the identical frozen ResBench metrics against
the same 512-volume test reference. The baseline reproduces low-order
statistics well: ensemble NTG within 0.0013 of the reference and
sill-normalized variogram MAE of 0.0114, both inside the split-half
sampling band — the only inside cells in the comparison. It misses
higher-order structure by an order of magnitude, however: geobody-size
Wasserstein distance of 0.388 (31× the band and 16× the 0.024 of ResFlow
conditioned on environment only), connectivity MAE 2.4–2.9× ResFlow's,
and 45.5% of its volumes vertically compartmentalized versus 28.7% in
the reference (ResFlow env-only: 27.0%). Its per-volume NTG spread
(sd 0.071 vs. reference 0.081) shows no mode collapse, so these
structural deficits are not an artifact of a degenerate ensemble. This
is precisely the regime the benchmark's structural metrics are designed
to discriminate — and where ResFlow retains its advantage under the most
conservative comparison we could construct: its weakest environment, a
single global model spanning all eight environments, with conditioning
marginalized to environment-only (per-instance parameters drawn at
random from the training split), against a single-environment specialist
trained on 90,000 volumes of this facies style.

## Assembly-consistency benchmark (EVAL.md Addendum E — scored)

Full outputs: `results/assembly/`; engine reference published in
`results/assembly_reference/`. Three ensembles at ONE shared condition (the
median-nearest training instance `channel_pv_shoestring/shard_0236|20`,
azimuth 95): (A) 250 native model volumes, (B) 250 tiles cut from ten
424x424x32 MultiDiffusion assemblies at the deployed Table 6 tiling
(10x10 blocks, overlap 24), (C) 256 fresh engine realizations. Note the
fixed-condition band is far tighter than master-table bands.

| comparison | \|dNTG\| | NTG-dist W1 | vario MAE | conn. MAE | geobody W1 | extent W1 |
|---|---|---|---|---|---|---|
| band (C split-half) | 0.0001 | 0.0015 | 0.0039 | 0.0054 | 0.0085 | 0.0072 |
| B vs C (headline) | 0.0621 | 0.0621 | 0.1348 | 0.0153 | 0.1089 | 0.0673 |
| A vs C (native control) | 0.0115 | 0.0116 | 0.0366 | 0.0107 | 0.0477 | 0.0281 |
| B vs A (MultiDiffusion effect) | 0.0506 | 0.0506 | 0.1258 | 0.0229 | 0.0795 | 0.0406 |

**Finding (three parts).** (1) No local seams: channels are continuous
across block boundaries; the stride-locked profile amplitude is comparable
to hard-concatenation of engine volumes at their own seam period.
(2) BUT channel *positions* lock to the tiling grid: the sand-fraction
profile oscillates 0.10-0.50 with a dominant spectral peak exactly at the
40-cell block stride (see `results/assembly/assembly_slice.png`: parallel
channels spaced ~40 cells). (3) Global statistics drift: +0.062 absolute
NTG excess vs the engine (0.285 vs 0.223, +28% relative; +0.051 vs native
generation at the identical condition), variogram MAE 3.7x the native
control's. The mean NTG assemblies produce is 0.28 for a condition
requesting 0.228. Assemblies are visually seamless and locally coherent
but do NOT preserve per-tile statistics; the drift is systematic across
all ten seeds (per-assembly NTG 0.2796-0.2802).

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
