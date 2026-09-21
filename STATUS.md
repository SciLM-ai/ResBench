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

## Reference build

Local root: `/scratch/08405/ilgar/resbench_v1_ref` (the fa75137 build is kept
as `resbench_v1_ref.stale_fa75137`).

| part | state |
|---|---|
| `unconditional`, 512 regenerated volumes x 8, manifest rows with `ntg`, `noise_seed`, `well_x`, `well_y` | built (`build_unconditional_reference.py --regenerate --verify`) |
| `repeats/cond0-4`, 256 runs x 5 x 8 | running (`gen_repeats_reference.py`, ~2.5 h on 40 cores) |
| `repeats/well1-5` lobe | built from a 514k-run pool, 71 / 67 / 52 / 54 / 52 members |
| `repeats/well1-5` PV_SHOESTRING | mining, 60k runs (`mine_wells.py`) |
| `repeats/well1-5` the other six | mining on a 4-node job (`/scratch/08405/ilgar/mine_wells_4nodes.sbatch`, job 1012769): MEANDER 70k runs, delta 60k, CB/SH 100k to 120k, then `build_wells_reference.py --wells-dir` for all eight |
| `fields` six channels, 32 x 512 x 128 x 32 | generating (`gen_field_reference.py`, `/scratch/08405/ilgar/field_reference_v2`) |
| `fields` delta, 32 x 512 x 512 x 32, row azimuth | chained after the channels |
| `fields` lobe, 32 x 512 x 512 x 32 | done earlier (`/scratch/08405/ilgar/resbench_v1_reference/ref/fields/lobe`); lobe is unaffected by the fixes |
| assembly (`build_reference.py --fields`) and the reference-as-submission gate (every samples-based check 0) | after the fields |

Well pools are the expensive part: an exact-match ensemble needs about 120k
runs of the source row (the published pools were 110k to 130k), which is 3,000
core-hours for MEANDER or delta. `mine_wells.py` saves its pass-1 pool and
resumes, so a pool can be grown across allocations.

## Blocks release

- Well ensembles for six environments (job above, 14 h wall).
- Hosting: `resbench download` says the reference is not hosted; document the
  path or host the directory once complete.

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
