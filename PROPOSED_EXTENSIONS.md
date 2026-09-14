# Proposed extensions and one bug fix (2026-09-13)

Nothing frozen in [EVAL.md](EVAL.md) is changed by this document. It records
(a) one implementation bug in the Addendum E scorer, fixed and regression
tested, and (b) additive diagnostics that apply metrics **the benchmark
already defines** to the assembly task, plus two scalars from the
connectivity review EVAL.md already cites.

Everything below came out of four days of comparing ~20 generative models on
the lobe assembly task (2026-09-10..13).

## 1. Bug: the E.2 tile grid is not centred at extents other than 424

`analysis/assembly_stats.py` cut its 5x5 grid of 64-cubes at a hard-coded
origin `(52, 52)`. For the 424-cell overlap-24 assembly that E.2 was frozen
on, 52 = (424 - 5*64) // 2 is exactly centred. At any other extent it is
not:

| assembly extent | tiles cover | margins | area scored |
|---|---|---|---|
| 424 (overlap 24) | [52, 372) | 52 / 52 | 57% (centred) |
| 532 (overlap 12) | [52, 372) | 52 / **160** | 36% (a corner) |
| 496 (overlap 16) | [52, 372) | 52 / 124 | 42% (off-centre) |
| 1572 (30x30 field) | [52, 372) | 52 / **1200** | **4%** (a corner) |

Fixed by deriving the origin from the array (`tile_origin`). **The 424
layout is bit-identical**, so no published number changes; a regression test
asserts `tile_origin(424) == 52` and a second test asserts the centre of a
532-cell field is now scored. `STRIDE_PERIOD` was likewise hard-coded to 40
(the overlap-24 block stride), so the E.5 seam diagnostic reported the
amplitude at a period the assembly did not have whenever the overlap
differed, and measured nothing at all for tiling-free generation; it is now
`--stride-period`.

## 2. The assembly table uses none of Addendum F or G

Addendum E scores six columns. Addendum F (F.1 multiple-point histograms,
F.2 runs, F.5 artifact rate) and Addendum G.6 (segmentation-free directional
chords, `resbench/anisotropy.py`) are defined, tested and banded — and are
applied only to the master table. Before this commit, nothing in the
repository imported `anisotropy.py`.

That matters because the two failure modes we measured are both invisible in
those six columns and both are covered by F/G:

* **Sharpening.** Raising CFG from 3 to 7.5 improved geobody W1 fourfold
  (0.231 -> 0.047) while lobes visibly merged and NTG drifted +1.2 pp.
  Geobody W1 rewards deleting small bodies.
* **Speckling.** A convolutional decoder head improved connectivity MAE
  (0.033 -> 0.026) while doubling geobody W1 (0.089 -> 0.191), by adding
  isolated cells: 90% more shale bodies, 93% more single shale voxels.
  F.5's artifact rate measures exactly this.

Across seven independently trained models geobody W1 and connectivity MAE
are **anti-correlated** (0.053/0.032, 0.058/0.030, 0.070/0.029, 0.072/0.027,
0.093/0.027): they are two ends of one merge-versus-fragment axis, not two
independent quality axes. A useful rule that costs nothing: a real
improvement moves geobody, extent AND connectivity in the same direction.

## 3. Two scalars from Renard & Allard (2013) that are cited but not implemented

EVAL.md §4 cites Renard & Allard for the connectivity function tau(h). The
same review defines two companions that ResBench does not compute and that
measure the amalgamation Addendum G was written to chase:

* **Global connectivity** Gamma = sum(n_i^2) / (sum n_i)^2, the probability
  that two randomly chosen cells of the phase are in the same cluster.
* **Traversing / percolation probability** per direction.

Added, with the **Euler characteristic** (V - E + F - C of the foreground
cubical complex), the topological member of the Minkowski functionals and
the standard descriptor that penalises excess fragments *and* excess holes.
All three are unit-tested against shapes of known topology
(`tests/test_assembly_ext.py`).

## 4. Geobody W1 is largely a speckle statistic

It weights every body equally and 62% of bodies in a scored ensemble are
smaller than 27 voxels. `mass_geobody_w1` (weighted by body volume) and
`geobody_w1_min27` are reported alongside. This matters concretely: on the
frozen count-weighted metric a whole-field model's assembled field scored
0.061 against 0.112 for its own independently generated blocks, suggesting
assembly *improves* tiles; volume-weighted the two are 0.114 and 0.122, i.e.
the same, and the apparent effect was a difference in speckle populations.

Also: `ens_summary()` computes per-volume largest-body fractions via
`ensemble_geobodies()` and discards them, although EVAL.md §5 has that
column. Restored.

## 5. The deepest limitation: the reference is 64 cubes

Ensemble C is 256 volumes of 64x64x32, so the protocol must cut assemblies
back into 64-cubes and **nothing above 64 cells is measurable against ground
truth**. A model whose lobes amalgamate across 128 cells can score well.

`ResFlow_ls6/scripts/tier2/../specialist_runs/gen_big_engine_reference.py`
generates ResMill realisations at the scored extent (532x532x32) at the
identical E.7 conditioning and grid convention (dx = dy = 100 m, dz = 1 m,
8-cell lateral / 9-cell vertical build margin cropped off), so field-scale
statistics finally have a reference. `--engine-big-dir` enables that
comparison, with its own split-half band.

### 5.1 CONFIRMED (16 independent fields, 400 tiles): the engine does not
### match itself under the assembly protocol

64-cubes cut from large engine fields do **not** reproduce the frozen
64-cube reference. Both sides are ResMill at identical E.7 conditioning; the
only difference is that one set was generated as 64-cube volumes and the
other cut out of 532-cube fields:

| metric | engine tiles vs frozen ref | split-half band | best model B_vs_C | offset as share of model score |
|---|---|---|---|---|
| geobody W1 | 0.0510 | 0.0190 | 0.0627 | 81% |
| connectivity MAE | 0.0410 | 0.0040 | 0.0318 | **129%** |
| variogram MAE | 0.0096 | 0.0041 | 0.0133 | 72% |
| \|dNTG\| | 0.0073 | 0.0001 | 0.0081 | 90% |

Mechanism: a natively generated 64-cube gets a full complement of lobes
fitted into a small box; a 64-cube window cut from a large field sees
whatever crosses it, including bodies the window truncates that were whole
in the larger domain. Addendum G.3 measured a related offset (native 64^3 vs
a centred 64-crop of 192^3: NTG +2.5%, largest fraction -10.3%) and judged
it minor for its purpose; at assembly-scoring scale it is not.

**Consequences.** (i) The effective floor for the assembly task is ~0.051
geobody and ~0.041 connectivity, not the 0.019 / 0.004 split-half band.
(ii) Model rankings are unaffected in direction — every model went through
the same biased comparison — but absolute distances have been overstated for
all of them. (iii) The correct comparison is assemblies against large engine
fields, which `--engine-big-dir` now performs.

### 5.2 What changes when the reference is right

Same model (77M whole-field RoPE DiT), same assemblies, two references:

| metric | vs 64-cube tiles | vs 532-cell engine fields |
|---|---|---|
| global connectivity Gamma | 0.0536 | **0.0005** |
| largest-body fraction | 0.0994 | **0.0058** |
| chord anisotropy | 0.0421 | **0.0024 (inside band)** |
| chord W1 along azimuth | 0.0276 | 0.0162 |
| mass-weighted geobody W1 | 0.1169 | 0.0525 |
| Euler characteristic / 1e6 | 49.1 | 8.96 |

Amalgamation and lobe shape — the properties Addendum G was written to
chase — are essentially exact against the right reference. What remains
genuinely outside band is FRAGMENTATION (Euler 3.4x band, artifact rate
near band, mass geobody 8x band): the model makes too many small pieces,
which independently confirms a vertical-stacking deficit found in the
model work (plan-view statistics match the engine while 3D body counts are
2.8x too high).

## Usage

```bash
python analysis/assembly_stats.py     --native-dir A --assembly-dir B \
    --engine-dir results/assembly_reference --env-slug lobe \
    --split-seed 20260815 --stride-period 52 --out-dir OUT        # frozen
python analysis/assembly_stats_ext.py --native-dir A --assembly-dir B \
    --engine-dir results/assembly_reference --env-slug lobe \
    --engine-big-dir BIG --azimuth 95 --out-dir OUT/ext           # additive
```

### 5.3 The ranking of methods changes when the reference has the right extent

Every model generated at 424x424x32 (the overlap-24 layout) and scored
against 16 engine fields of the SAME extent, identical band:

| model | mass geobody W1 | Gamma | largest frac | Euler /1e6 | chord aniso |
|---|---|---|---|---|---|
| engine split-half band | 0.0126 | 0.0002 | 0.0010 | 20.64 | 0.0250 |
| **whole-field RoPE 77M** | **0.0245** | **0.0004** | **0.0025** | **4.18 (inside)** | **0.0002 (inside)** |
| whole-field RoPE 33M | 0.0289 | 0.0005 | 0.0042 | 27.36 | 0.0560 |
| tiled DiT 77M, MultiDiffusion | 0.0857 | 0.0019 | 0.0114 | 14.10 (inside) | 0.0118 (inside) |
| tiled DiT p442, 4-stage | 0.1464 | 0.0042 | 0.0266 | 38.70 | 0.0613 |
| UNet EMA, outpaint | 0.2023 | 0.0045 | 0.0206 | 46.29 | 0.0716 |
| UNet EMA, MultiDiffusion | 0.3657 | 0.0139 | 0.0643 | 49.83 | 0.0831 |

**The frozen protocol ranks UNet-outpaint (geobody W1 0.087) ahead of tiled
p442 4-stage (0.118). At field scale against a matched reference that
reverses: 0.2023 vs 0.1464.** Method ranking, not only magnitude, depends on
whether the reference has the extent of the thing being scored.

Caveat: this table mixes model sizes across the tiled rows, so it supports
"whole-field beats tiled" but is not a clean fusion-rule comparison.

Caveat on the bands: they are split-halves of 16 engine fields (8 vs 8) and
are themselves noisy — the Euler band is 20.64 at 424 cells but 2.66 at 532,
a gap larger than the change in field size explains. Absolute deviations are
the robust part. More engine fields are being generated to tighten them.
