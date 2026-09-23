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
complete 32 m columns at the field extent with the row's aggradation ratio.

Cost per 128 x 128 x 64 volume on one core: lobe 6 s, PV 5, CB_LABYRINTH 13,
CB_JIGSAW 16, SH_DISTAL 21, SH_PROXIMAL 18, MEANDER_OXBOW 56, delta 11.

## Reference build

Root: `/scratch/08405/ilgar/resbench_v1_ref` (swapped in 2026-09-23, engine
`3ffd173`); the previous reference (`9c0bd15`, earlier dataset) is kept at
`resbench_v1_ref.stale_9c0bd15`.

| part | state |
|---|---|
| `unconditional` | done: 512 regenerated test-split windows x 8 environments, every row bit-identical to the dataset (`--regenerate`), `ntg`, `noise_seed`, `well_x`, `well_y` per row |
| `repeats/cond0-4` | done: 256 ResMill runs (fresh seeds, dataset grid, each windowed by its own seed) of reference rows 0-4, x 8 environments; every condition's first member regenerated bit for bit; 38 min on 96 workers |
| `fields` | done: 32 per environment, lobe and delta 512 x 512 x 32, channels 512 x 128 x 32, 23 min on 24 workers; every channel corridor inside the along-flow gate (last/first fifth PV 0.68, CB_LABYRINTH 1.16, CB_JIGSAW 1.58, SH_DISTAL 1.07, SH_PROXIMAL 1.63, MEANDER_OXBOW 0.57); mean field NTG lobe 0.503, PV 0.176, CB_LABYRINTH 0.320, CB_JIGSAW 0.390, SH_DISTAL 0.623, SH_PROXIMAL 0.477, MEANDER_OXBOW 0.384, delta 0.142 |
| `repeats/well1-5` | done: 5 wells x 8 environments, one well per reference row from five rows stepping through net-to-gross and size (depth 3 to 20 cells); the well is the cube's centre column, mined over every window origin of 500-volume pools; every ensemble 72 to 256 members (cap 256), each a window of a distinct volume; 70 minutes on 140 workers for all eight. Rules: informative = at least two sand bodies of at least 2 cells separated by at least 2 cells of mud, all inside the column; representative = within 0.15 of the row's own ntg; estimable = 50 volumes; the five columns of an environment are distinct. Dropped on the way: one window per volume (about 400,000 volumes for CB_JIGSAW), one source row per environment (five near-identical wells, two-speck delta columns), bodies touching the cube face (two-cell stubs) |
| self-check | pending: `tools/reference_self_check.py REF OUT` started 2026-09-23 13:47 on the same files, result not yet recorded; the previous reference's self-check came out matched in all 31 cells |

Field budgets were recalibrated for the 128 x 128 dataset box (SPEC.md "Field
extent"): area law for the avulsion-dominated channels, edge law for
MEANDER_OXBOW's `ntime`, edge law for the delta's `n_trees` and
`n_bifurcations`.

## Blocks release

- Hosting: `resbench download` says the reference is not hosted; document the
  path or host the directory.

## Why fields are scored at their own extent

Field-scale statistics are compared against ResMill fields of the same size.
Cutting 64-cubes out of a large field and comparing them to natively built
64-cubes is a different distribution: both sides ResMill, identical
conditioning, 16 fields and 400 tiles, body size 0.0510 against a 0.0190 band,
connectivity 0.0410 against 0.0040. A 64-cube can never contain a body wider
than 64 cells. Nothing is cut into tiles, on either side.
