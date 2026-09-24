# Status

What is finished, what is running, and what is deliberately out of scope.
Dates are 2026-09-23 unless stated.

## Engine and dataset

The dataset (`SciLM/SiliciclasticReservoirs`, 1,000,000 samples) is now a set
of `64 x 64 x 32` windows cut at seeded random origins out of `128 x 128 x 64`
ResMill volumes, generated with ResMill `3ffd173` (fluvial entries on the grid
boundary, walks clipped by the real grid, walk cap separate from the node
density, several entry points, sampled entry scatter, levels from a sampled
aggradation ratio, tree delta, correlated porosity texture). Every reference
volume is generated with that engine (`REF/ENGINE.txt`), so a model trained on
the dataset is scored against references from its own engine: the offset the
earlier reference carried against the earlier dataset is gone.

All builders generate on the dataset grid and apply the dataset's window rule
(`resmill.dataset.windows`): `--regenerate` cuts the row's recorded window,
repeats and well pools cut the window their own seed would get. Fields are
the dataset's 64 m column at the field extent, windowed to 32 cells by the same rule.

Cost per 128 x 128 x 64 volume on one core: lobe 6 s, PV 5, CB_LABYRINTH 13,
CB_JIGSAW 16, SH_DISTAL 21, SH_PROXIMAL 18, MEANDER_OXBOW 56, delta 11.

## Reference build

Root: `/scratch/08405/ilgar/resbench_v1_ref`, hosted as `SciLM/ResBench-reference` (swapped in 2026-09-23, engine
`3ffd173`); the previous reference (`9c0bd15`, earlier dataset) is kept at
`resbench_v1_ref.stale_9c0bd15`.

| part | state |
|---|---|
| `unconditional` | done: 512 regenerated test-split windows x 8 environments, every row bit-identical to the dataset (`--regenerate`), `ntg`, `noise_seed`, `well_x`, `well_y` per row |
| `repeats/cond0-4` | done: 256 ResMill runs (fresh seeds, dataset grid, each windowed by its own seed) of reference rows 0-4, x 8 environments; every condition's first member regenerated bit for bit; 38 min on 96 workers |
| `fields` | done (rebuilt 2026-09-24): 32 per environment, lobe and delta 512 x 512 x 32, channels 512 x 128 x 32, each the dataset-rule 32-cell window of a 64-cell engine column with the row's own level count; mean NTG unchanged against the earlier direct-32 fields (within 0.01), floor/roof artefacts gone; 19 min on this node (lobe 21 GB a worker, 6 at a time); every channel corridor inside the along-flow gate (last/first fifth 0.57 to 1.64) |
| `repeats/well1-5` | done: 5 wells x 8 environments, one well per reference row from five rows stepping through net-to-gross and size (depth 3 to 20 cells); the well is the cube's centre column, mined over every window origin of 500-volume pools; every ensemble 72 to 256 members (cap 256), each a window of a distinct volume; 70 minutes on 140 workers for all eight. Rules: informative = at least two sand bodies of at least 2 cells separated by at least 2 cells of mud, all inside the column; representative = within 0.15 of the row's own ntg; estimable = 50 volumes; the five columns of an environment are distinct. Dropped on the way: one window per volume (about 400,000 volumes for CB_JIGSAW), one source row per environment (five near-identical wells, two-speck delta columns), bodies touching the cube face (two-cell stubs) |
| self-check | passed 2026-09-24 after the field rebuild (`tools/reference_self_check.py REF OUT`, 1.5 h on one gg node): 40 parts readable, 31/31 cells matched, overall 0.06; every samples-based check exactly 0 in all three tasks; repeats-based checks variety 0.47, calibration 0.46 / 0.47, well_blending 0.55. Results in `/scratch/08405/ilgar/resbench_selfcheck/results.json`; the direct-32 fields reference is kept at `resbench_v1_ref.stale_fields32` |

Field budgets were recalibrated for the 128 x 128 dataset box (SPEC.md "Field
extent"): area law for the avulsion-dominated channels, edge law for
MEANDER_OXBOW's `ntime`, edge law for the delta's `n_trees` and
`n_bifurcations`.

## Hosting

The reference is on HuggingFace at `SciLM/ResBench-reference` (public, uploaded
2026-09-24, revision `9087d11`, pinned in `resbench/cli.py`). `resbench download`
fetches it; `resbench score` without `--reference` uses the cached copy or
downloads it. Nothing blocks release.

## Why fields are scored at their own extent

Field-scale statistics are compared against ResMill fields of the same size.
Cutting 64-cubes out of a large field and comparing them to natively built
64-cubes is a different distribution: both sides ResMill, identical
conditioning, 16 fields and 400 tiles, body size 0.0510 against a 0.0190 band,
connectivity 0.0410 against 0.0040. A 64-cube can never contain a body wider
than 64 cells. Nothing is cut into tiles, on either side.
