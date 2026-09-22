# Status

What is finished, what is running, and what is deliberately out of scope.
Dates are 2026-09-21 unless stated.

## Engine

Every reference volume is generated with ResMill `9c0bd15` (recorded in
`REF/ENGINE.txt`). Three defects were fixed on the way, all in the fluvial
walker (`resmill/layers/_fluvial.py`), see SPEC.md "Field extent":

1. `fa75137` walks clipped in the unrotated frame (corners empty, net-to-gross
   azimuth-dependent);
2. `9c0bd15` streamline entries inside the grid at off-axis azimuth (channels
   and delta fans starting in the open);
3. `9c0bd15` the walk cap doubled as the node count, so raising it for long
   grids changed the per-node geometry (MEANDER_OXBOW NTG 0.45 to 0.26).

The published 64-cube dataset predates all three. The reference is therefore
**regenerated** from the same rows and seeds (`--regenerate`), not copied, so
it is the fixed engine's output. Lobe is untouched by the fixes and regenerates
bit for bit (checked on every build). Until the dataset is regenerated with
`9c0bd15` and models retrained, a model trained on the published data is scored
against references from a corrected engine; at 64 cells the differences sit at
off-axis azimuths (entries, corners), see the verification numbers in SPEC.md.

Cost per 64-cube on one core, either engine: PV 2 s, CB_LABYRINTH 15 s,
CB_JIGSAW 28 s, SH_DISTAL 33 s, SH_PROXIMAL 36 s, MEANDER_OXBOW 91 s, delta
112 s, lobe well under a second.

## Reference build: complete (2026-09-22)

Local root: `/scratch/08405/ilgar/resbench_v1_ref`, every volume from ResMill
`9c0bd15` (`ENGINE.txt`). The fa75137 build is kept as
`resbench_v1_ref.stale_fa75137` for the old-versus-new comparison below.

| part | contents |
|---|---|
| `unconditional` | 512 regenerated test-split volumes x 8 environments, `ntg`, `noise_seed`, `well_x`, `well_y` per row |
| `repeats/cond0-4` | 256 ResMill runs of reference rows 0-4, x 8 |
| `repeats/well1-5` | 5 wells x 8 environments; members: lobe 52-71, PV 72-86, CB_JIGSAW 67-90, CB_LABYRINTH 152-206, SH_DISTAL 162-224, SH_PROXIMAL 245-256, MEANDER 216-256, delta 158-256 |
| `fields` | 32 per environment: lobe and delta 512 x 512 x 32, channels 512 x 128 x 32; every corridor inside the along-flow gate (last/first fifth 0.56 to 1.83) |
| `manifest.csv` | 4096 + 40 + 40 + 256 rows |

`tools/reference_self_check.py` scores the reference against itself: every
samples-based check is 0 in all 31 (task, check) cells and the repeats-based
checks land at 0.02 to 0.83 (first 128 members against the whole ensemble),
overall 0.05, 31/31 matched.

What it cost (gg nodes, 144 cores): unconditional 10 min; repeats ~2.5 h on 40
cores; fields ~5 h on 40 to 56 cores; wells lobe (514k runs) and PV (60k) on one
node, the other six on a 4-node 13-hour job (55k to 120k runs each). Per-run
costs on one core: lobe 1.8 s, PV 3.4, CB_LABYRINTH 12.7, CB_JIGSAW 14.6,
SH_DISTAL 18.8, SH_PROXIMAL 19.7, MEANDER 68.5, delta 89.6.

## Models trained on the published dataset

The published dataset predates the three engine fixes. Its own 512 test
volumes per environment, scored as a submission against this reference on the
nine samples-based checks, come out matched on most cells, close (1 to 1.7) on
several, and distinguishable on CB_LABYRINTH bed thickness (4.0) and chord
lengths (3.1); in absolute terms the shifts are a few percent (CB_LABYRINTH mean
bed thickness 5.26 to 5.11 cells). That is the offset any model trained on the
published data carries here. Regenerating the dataset with `9c0bd15` removes
it: `ResMill/examples/dataset_generation/vista/` holds the launch scripts and
the measured cost (7,861 core-hours, 55 gg node-hours, about 18 SU).

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

## Not in v1

**More than one well at a time.** Two boreholes produce so many distinct
combinations that exact-match rejection stops being affordable; a single well
already takes 60k to 500k ResMill runs.

**Porosity, permeability and flow response.** The dataset is facies only, and a
flow simulator would break the "runs on a laptop" property.
