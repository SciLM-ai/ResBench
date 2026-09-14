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

**Caution, measured on a 2-realisation pilot and to be confirmed at n=16:**
64-cubes cut from large engine fields do **not** reproduce the frozen
64-cube reference — geobody W1 0.044 against a 0.019 band, connectivity MAE
0.044 against 0.004. Addendum G.3 documented a related offset (native 64^3
vs a centred 64-crop of 192^3: NTG +2.5%, largest fraction -10.3%) and
judged it minor. If the larger measurement holds, then part of every model's
B_vs_C score is an engine-level protocol offset rather than model error, and
the effective floor for the assembly task is ~0.044 rather than the 0.019
split-half band.

## Usage

```bash
python analysis/assembly_stats.py     --native-dir A --assembly-dir B \
    --engine-dir results/assembly_reference --env-slug lobe \
    --split-seed 20260815 --stride-period 52 --out-dir OUT        # frozen
python analysis/assembly_stats_ext.py --native-dir A --assembly-dir B \
    --engine-dir results/assembly_reference --env-slug lobe \
    --engine-big-dir BIG --azimuth 95 --out-dir OUT/ext           # additive
```
