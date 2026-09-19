# ResBench v1 — frozen constants

README.md is the protocol. This file is the exact numbers an implementer needs,
so nothing here is a judgment call at scoring time.

## Data

| | |
|---|---|
| dataset | `SiliciclasticReservoirs` (HuggingFace) |
| split | 90 / 5 / 5 train / validation / test, stratified by environment, seed 42 |
| volume | `64 x 64 x 32` int8, `1 = sand`, axes `(x, y, z)`, depth last |
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
| `channel:*` | `512 x 64 x 32` | elongated along flow; `azimuth = 0` |

Cell size is each environment's own and is never changed: only the number of
cells grows. ResMill's `ntime` is a **global** event cap across all levels, not
a per-level one, so it is scaled by the area ratio when generating fields —
without that, level 0 exhausts the budget on its now-larger sand target and the
upper levels never run at all.

## Lags

`h = 1 .. floor(extent / 2)` per axis. A `64 x 64 x 32` volume gives
`32 + 32 + 16 = 80` lags; extents differ per task, so bands are always computed
at the same extent as the comparison.

## Seeds

| constant | value | what it fixes |
|---|---|---|
| dataset split seed | 42 | train / validation / test |
| index seed | published in `manifest.csv` | which 512 test volumes are the reference |
| `SPLIT_SEED` | 20260918 | the half/half partition used for every band |
| noise seed | one per manifest row | your model's starting noise |
| field `SEED_BASE` | 2026091800 | ResMill seeds for the field reference |

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
| `variety`, `well_blending`, `calibration` | `K = 128` model runs per condition |
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
