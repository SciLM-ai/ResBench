# ResBench v1 — frozen constants

README.md is the protocol. This file is the exact numbers an implementer needs,
so nothing here is a judgment call at scoring time.

## Data

| | |
|---|---|
| dataset | `SiliciclasticReservoirs` (HuggingFace) |
| split | 90 / 5 / 5 train / validation / test, stratified by environment, seed 42 |
| volume | `64 x 64 x 32` int8, `1 = sand`, axes `(x, y, z)`, depth last |
| origin of a volume | one window of a `128 x 128 x 64` ResMill volume, origin from `default_rng([42, seed])`: x, y uniform on 0..64, z uniform on 1..31, redrawn while the window has no sand (`resmill.dataset.windows`); the dataset row records it as `crop_x0, crop_y0, crop_z0` |
| cell size | `lobe` 100 x 100 x 1 m; the other seven 10 x 10 x 1 m |
| reference | 512 test-split volumes per environment, ids in `reference/manifest.csv` |

Environments, canonical order:

```
lobe  channel:PV_SHOESTRING  channel:CB_LABYRINTH  channel:CB_JIGSAW
channel:SH_DISTAL  channel:SH_PROXIMAL  channel:MEANDER_OXBOW  delta
```

## Field extent

| environment | extent | notes |
|---|---|---|
| `lobe`, `delta` | `512 x 512 x 32` | square |
| `channel:*` | `512 x 128 x 32` | elongated along flow; `azimuth = 0` |

Cell size is each environment's own and is never changed: only the number of
cells grows. The corridor is 128 wide, not 64: a sinuous channel that reaches
a side wall ends there, and at 64 wide the far end of the corridor starves
(PV_SHOESTRING sand in the last fifth over the first fifth: 0.34 at 64 wide,
1.0 to 1.3 at 128).

A field is a complete 32 m column, as tall as a training window, generated on
the environment's own cell size at the field extent. The dataset simulates a
64 m column and windows 32 m of it, so a field keeps the row's **aggradation
ratio** (level spacing over channel depth) rather than its level count:
`nlevel_field = round((32 - depth) / (ratio x depth)) + 1`, the same rule the
dataset sampler used for 64 m.

ResMill's event budget does not scale with the domain, so the generator scales
it from the dataset box (`128 x 128`) to the field, with laws calibrated per
family against each row's native `128 x 128 x 64` volume (2026-09-23,
`tools/gen_field_reference.py`):

- **channels, avulsion-dominated** (`PV_SHOESTRING`, `CB_*`, `SH_*`): `ntime`
  (per level) scales with the **area** ratio, 4x at `512 x 128`. Their sand
  targets are reached, so the field net-to-gross matches the native volume
  whatever the budget: field / native 0.96 to 1.17 measured.
- **`MEANDER_OXBOW`** (one migrating belt per level, budget-limited): `ntime`
  scales with the **edge** ratio, 2x at `512 x 128`: field / native 0.98 and
  1.08, against 0.73 to 0.83 unscaled and 1.18 to 1.59 with the area law.
- **delta** (distributary trees, no net-to-gross target): the number of
  networks per generation `n_trees` and the bifurcations per network
  `n_bifurcations` both scale with the **edge** ratio, 4x at `512 x 512`:
  field / native 1.13, 0.65, 1.11; bifurcations alone leave the field at a
  third of the native value because terminal branches are capped by `q_min`.

These are safety nets rather than stopping rules for the channel families: the
per-level loop stops on its sand target. Only the top-level `azimuth` rotates
the model; `mCHazi` is engine-internal and is left alone.

Every reference volume is generated with the ResMill named in `REF/ENGINE.txt`,
the engine that generated the dataset itself (`3ffd173` or later; the fluvial
walker's entries lie on the grid boundary, walks are clipped by the real grid,
and the walk cap is separate from the node density). `--regenerate` proves it:
every selected test-split row, run again with its seed on the dataset grid and
windowed at its recorded origin, reproduces the dataset volume bit for bit.

## Lags

`h = 1 .. floor(extent / 2)` per axis. A `64 x 64 x 32` volume gives
`32 + 32 + 16 = 80` lags; extents differ per task, so bands are always computed
at the same extent as the comparison.

## Seeds

| constant | value | what it fixes |
|---|---|---|
| dataset split seed | 42 | train / validation / test |
| `MANIFEST_SEED` | 20260726 | which 512 test volumes are the reference, and their noise seeds (frozen protocol, EVAL.md 6) |
| `WELL_XY_SEED` | 20260919 | `(well_x, well_y)` per reference row |
| repeats `SEED_BASE` | 20260921 | ResMill seeds for `cond<i>`, via `default_rng([SEED_BASE, env_index, i])` |
| `SPLIT_SEED` | 20260918 | the half/half partition used for every band |
| noise seed | one per manifest row | your model's starting noise |
| field `SEED_BASE` | 2026091800 | ResMill seeds for the field reference |
| `WELL_POOL_SEED` | 20260922 | ResMill seeds of the well pools, via `default_rng([WELL_POOL_SEED, env_index, row_index])` |
| window `crop_seed` | 42 | the dataset's window origin per sample, via `default_rng([42, seed])` (`resmill.dataset.windows`) |

## Reference layout and manifest

```
REF/manifest.csv
REF/volumes/<slug>/volumes.npz          ids, volumes            512 per environment
REF/repeats/<slug>/cond<i>.npz          ids, volumes, seeds      i = 0..4, 256 ResMill runs of reference row i
REF/repeats/<slug>/well<i>.npz          ids, volumes, pattern, well_mask, well_xy, source_id   i = 1..5
REF/fields/<slug>/fields.npz            ids, volumes            32 per environment
REF/ENGINE.txt                          the ResMill commit every volume above was generated with
```

The reference is built in this order, all with the ResMill named in `ENGINE.txt`:

```
tools/build_unconditional_reference.py --out REF --regenerate
tools/gen_repeats_reference.py --ref REF
tools/mine_wells.py --ref REF --env <env> --out WELLS                    (one per environment; five rows each)
tools/build_wells_reference.py --ref REF --wells-dir WELLS
tools/gen_field_reference.py --out FIELDS --n 32
tools/build_reference.py --out REF --fields all=FIELDS --expect 32
tools/reference_self_check.py REF OUT
```

`--regenerate` runs the selected test-split rows again, same parameters and
seeds, on the dataset's `128 x 128 x 64` grid and cuts the window at the row's
recorded origin; every row must reproduce the dataset volume exactly. Repeats
run fresh seeds on the same grid and window each volume with the dataset's rule
for that seed, so an ensemble member is distributed like a dataset sample of
the same parameters.

**Wells are the centre column of the cube, five geologies per environment.**
The five wells of an environment come from five different reference rows: the
512 unconditional rows are ranked by realised net-to-gross and split into
quintiles, and well i takes from quintile i the row whose characteristic size
(`depth_cells`: channel depth, or lobe thickness) sits at the i-th quintile of
that bin, so the wells step from thin, low-NTG geology to thick, high-NTG
geology. A dataset sample is a window of a `128 x 128 x 64` volume at a random
origin (x0, y0 in 0..64, z0 in 1..31), so its centre column is the volume
column at (x0 + 32, y0 + 32), z0..z0 + 31, and the `65 x 65 x 31` such columns
of a volume are every centre column a window of it can have. Each row's pool
runs 500 fresh volumes and records all of them; a volume matches a candidate
column when the column occurs anywhere in that block, and the member is the
window at its first occurrence, one window per volume. This marginalises the
window origin exactly instead of by one random draw per volume, so 500 volumes
give the matches that hundreds of thousands of single-window runs would (a
40,000-window CB_JIGSAW pool gave the best informative column 6 exact matches
at a fixed location). Members are windows of distinct volumes with the well at
(32, 32). What this benchmark's well task covers, stated plainly: conditioning
on the centre column of a 64-cube, for five geologies per environment; the
`well_conditioned` samples task keeps its per-row interior locations.

A candidate column is **informative** when it has at least two sand bodies,
each at least 2 cells thick, separated by at least 2 cells of mud (one-cell
specks and one-cell breaks do not count), all of them inside the column (mud
in its top and base cell, so no body is a stub cut by the cube face),
**representative** when its sand fraction is within 0.15 of the row's realised
`ntg`, and **estimable** when at least 50 distinct volumes of the row's pool
contain it; the most frequent such column of the row that no earlier well of
the environment already uses is the well, so the five wells are five distinct
columns of five different geologies. Wells follow the published rule (C.2). A
well is a location `x, y in [8, 56]` and a 32-cell column that ResMill produces
there under the source row's parameters; candidates are every (location,
column) seen in a pool of fresh runs of that row, and a candidate must be
informative (two sand bodies of at least 2 cells separated by at least 2 cells
of mud, all inside the column), representative (column sand fraction within 0.15 of the row's `ntg`)
and estimable (at least 50 distinct volumes contain it). The most frequent such
column of each of the five rows is a well; the matching volumes' windows are
the ensemble.

`manifest.csv` has one row per reference item and a `task` column:

| task | id | what the row records |
|---|---|---|
| `unconditional` | `<env>\|<shard_dir>\|<sample_idx>` | `ntg` (realized), `requested_ntg`, `azimuth`, slim parameters, `noise_seed`, `well_x`, `well_y` |
| `repeats_unconditional` | `<env>\|cond<i>` | the same, copied from reference row i; `source_id` names that row |
| `repeats_well` | `<env>\|well<i>` | `source_id` = the parameter row the ensemble was drawn from, `well_x`, `well_y` (32, 32), `pattern` |
| `field_scale` | `field\|<slug>\|<seed>` | `ntg` = the FIELD's realized sand fraction, `ntg_source_cube`, `requested_ntg`, `azimuth` |

`ntg` is always the number the model is conditioned on for that item. The well
location for the `well_conditioned` samples task is one interior column per
reference row, `(well_x, well_y)` drawn from `default_rng(20260919)` uniformly
in `[8, 56]`; the submitter reads the borehole from the reference volume there.

## Per-check constants

| check | constant |
|---|---|
| `net_to_gross` | tolerance `0.01` sand fraction (a choice, not a band) |
| `patterns` | 256 bins, every offset, Jensen-Shannon base 2 |
| `bed_thickness` | runs along z, truncated runs kept, histogram to 512 cells |
| `connectivity`, `compartments`, `body_size`, `speckle` | 6-connectivity, faces only |
| `body_size` | log10 histogram, 0 to 8 dex, 800 bins |
| `chord_lengths` | linear sampling, threshold 0.5, censored chords dropped, log10 histogram 0 to 4 dex, 400 bins |
| `speckle` | bodies `< 8` cells |
| `variety`, `well_blending`, `calibration` | `K = 128` model runs per condition, 5 conditions per environment per task; reference 256 runs (`cond`) or up to 256 exact matches (`well`); band = split-half of the reference at the same condition |
| `well_blending` | profile to Chebyshev distance 20; signed offset over distance `<= 6` |
| `calibration` | 10 equal-width probability bins, well cells excluded |

## Score

```
band  = check( half of the reference , the other half )     # SPLIT_SEED
s     = check( model , reference ) / band
```

A check with several sub-parts bands each part separately and averages them, so
it contributes exactly one number. The task score is the mean of s over checks
and environments; the overall score is the mean over tasks. Verdicts:
`s <= 1` matched, `s <= 2` close, otherwise distinguishable.

The aggregation is an average, never a median. Measured on 47 scored
checkpoints the two rank models at Spearman `-0.63`: the order reverses, and
the median's top model misses one check by 5.9x its band and another by 8.0x.

## Check-to-task matrix

| check | uncond | well | field |
|---|:--:|:--:|:--:|
| `net_to_gross` `variogram` `patterns` `bed_thickness` `connectivity` `compartments` `body_size` `chord_lengths` `speckle` | yes | yes | yes |
| `variety` | yes | no | no |
| `well_blending` | no | yes | no |
| `calibration` | yes | yes | no |

11 / 11 / 9.
