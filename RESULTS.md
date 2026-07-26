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
| channel:PV_SHOESTRING | 0.0069 (near; band 0.0050) | 0.0202 (near; band 0.0115) | 0.0034 (outside; band 0.0014) | 0.0210 (near; band 0.0125) | 0.0443 (outside; band 0.0117) | 0.000 (inside) |
| channel:CB_LABYRINTH | 0.0025 (inside; band 0.0058) | 0.0176 (inside; band 0.0226) | 0.0024 (outside; band 0.0009) | 0.0139 (inside; band 0.0176) | 0.0189 (outside; band 0.0011) | 0.000 (inside) |
| channel:CB_JIGSAW | 0.0048 (inside; band 0.0091) | 0.0186 (near; band 0.0113) | 0.0010 (inside; band 0.0010) | 0.0129 (near; band 0.0067) | 0.0087 (outside; band 0.0038) | 0.000 (inside) |
| channel:SH_DISTAL | 0.0104 (inside; band 0.0189) | 0.0364 (outside; band 0.0122) | 0.0001 (outside; band 0.0000) | 0.9709 (outside; band 0.1344) | 0.0000 (inside; band 0.0001) | 0.000 (inside) |
| channel:SH_PROXIMAL | 0.0087 (inside; band 0.0171) | 0.0276 (outside; band 0.0067) | 0.0003 (inside; band 0.0003) | 0.0177 (inside; band 0.0189) | 0.0011 (inside; band 0.0017) | 0.000 (inside) |
| channel:MEANDER_OXBOW | 0.0023 (inside; band 0.0044) | 0.0136 (near; band 0.0069) | 0.0005 (outside; band 0.0001) | 0.0077 (outside; band 0.0028) | 0.0026 (near; band 0.0016) | 0.000 (inside) |
| delta | 0.0078 (inside; band 0.0117) | 0.0150 (outside; band 0.0054) | 0.0009 (inside; band 0.0027) | 0.0308 (near; band 0.0234) | 0.0085 (inside; band 0.0182) | 0.000 (inside) |
| **pooled** | 0.0034 (inside; band 0.0060) | 0.0133 (outside; band 0.0039) | 0.0054 (outside; band 0.0021) | 0.0143 (inside; band 0.0219) | 0.0109 (near; band 0.0075) | 0.000 (inside) |

Well mismatch by config, pooled: 1 well 0.000% · 2 wells 0.000% · 3 wells 0.000%.

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
configs (130,816 conditioning voxels): the hard-replacement guarantee of the
sampling pipeline is verified end to end, and the audit confirms masks,
carved well values, and generated volumes stay aligned through the whole
harness.

## Anomalies to look at before writing the rebuttal

- **SH distal geobody W1 = 0.97 (7× band)** is *speck debris*, not missing
  structure: the reference has 2.6 bodies/volume (one ~10⁵-voxel sheet plus
  a few fragments; 30% singletons), ResFlow has 5.2 bodies/volume (50%
  singletons, median size 2 voxels) while reproducing the sheet itself
  (max size, p90, and largest-fraction all match; |Δ largest frac| = 0.0000).
  ≈ 5 extra isolated sand voxels per 131k-voxel volume, i.e. ~0.004% of
  voxels, but they dominate the pooled size *distribution* because the
  environment has so few bodies. A 1–2-voxel minimum-body-size filter would
  likely move this cell inside the band — worth a sensitivity note.
- **Systematic vertical over-smoothing**: γ_z plateau undershoot in five
  environments plus over-connection (largest-fraction up, lobe τ up).
  Consistent with CFG = 3.0 mode-seeking; a CFG-scale sensitivity sweep
  (e.g. 1.0/2.0/3.0 on one environment) would localize the cause.
- **PV shoestring largest-fraction +0.044 with disjoint CIs**: isolated
  shoestring bodies merge more often than in ResMill; also the only
  environment whose NTG delta is *near* rather than inside.
- **Tight-band caveat for τ**: split-half bands of 10⁻⁴–10⁻³ make
  connectivity verdicts hypersensitive; absolute MAEs are ≤ 0.0034 in all
  four *outside* cells. Report absolute values alongside verdicts.
- **NTG sign consistency**: 6/8 environments slightly under-generate sand;
  all within or near band individually, pooled |ΔNTG| = 0.0034 (inside).
