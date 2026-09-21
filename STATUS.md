# Status

What is finished, what is not, and what is deliberately out of scope.

## Blocks release: ResMill fields at the scored extent

Field-scale statistics have to be compared against ResMill fields of the **same
size**. Cutting 64-cubes out of a large field and comparing them to natively
built 64-cubes is a different distribution, and measurably so. Both sides
ResMill, identical conditioning, 16 fields and 400 tiles:

| check | ResMill tiles vs ResMill 64-cubes | band | best model | offset as share of model score |
|---|---|---|---|---|
| body size | 0.0510 | 0.0190 | 0.0627 | 81% |
| connectivity | 0.0410 | 0.0040 | 0.0318 | **129%** |
| variogram | 0.0096 | 0.0041 | 0.0133 | 72% |
| net-to-gross | 0.0073 | 0.0001 | 0.0081 | 90% |

Not an edge effect: ResMill clips bodies at the box in both cases. **A 64-cube
can never contain a body wider than 64 cells.** In a large field a lobe complex
runs hundreds of cells and a 64-cell window cut across it sees a body filling
the whole window, which a natively built 64-cube cannot produce.

ResMill is not broken; the comparison was, and it charged every model for a gap
ResMill opens against itself. On connectivity that gap was larger than the
distance being blamed on the model. With the right reference, same model and
same fields: global connectivity 0.0536 to 0.0005, largest-body fraction 0.0994
to 0.0058, chord anisotropy 0.0421 to 0.0024 and inside the band. The ranking of
methods changes too.

**Where this stands.** `tools/gen_field_reference.py` generates the reference for
all eight environments, CPU only. Remaining: run it at `--n 32`.

One correction is baked in. ResMill's event budget does not scale with the
domain: `ntime` for channels, `ntime_per_gen` for delta. Every channel row has
`ntime_per_level = True`, so `ntime` is a *per-level* budget; each level's sand
target grows with the field while its budget does not, and every level
under-fills equally. Scaling by the area ratio fixes it, measured on
`channel:PV_SHOESTRING` at `512 x 64`:

| | ntime | NTG | sand by depth quartile |
|---|---|---|---|
| native `64 x 64` | 30 | 0.1722 | 0.163 0.155 0.163 0.208 |
| `512 x 64` unscaled | 30 | 0.1163 | 0.089 0.078 0.112 0.187 |
| `512 x 64` scaled x8 | 240 | **0.1672** | 0.185 0.158 0.172 0.154 |

Delta is the awkward one: it does not honor its NTG target even at native size
(target 0.231, dataset realized 0.532), so its field reference is checked
against the realized value rather than the target.

## Well ensembles: seven environments of eight

| environment | wells | ResMill matches per well | status |
|---|---|---|---|
| `channel:CB_JIGSAW` | 5 | 158 – 169 | ready |
| `channel:CB_LABYRINTH` | 5 | 148 – 167 | ready |
| `channel:PV_SHOESTRING` | 5 | 217 – 235 | ready |
| `channel:MEANDER_OXBOW` | 5 | 256 (capped) | ready |
| `channel:SH_DISTAL` | 5 | 256 (capped) | ready |
| `channel:SH_PROXIMAL` | 5 | 256 (capped) | ready |
| `delta` | 5 | 256 (capped) | ready |
| `lobe` | 5 | **34 – 40** | needs ~1.6x more ResMill runs |

## The reference directory does not exist yet

There is no `ref/` a submitter can point `--reference` at. What exists, and in
what form:

| reference | state |
|---|---|
| `field_scale` lobe, 32 fields + manifest rows | done |
| `field_scale` six channels and delta, 32 each | generating (`tools/gen_field_reference.py`) |
| `unconditional`: 512 test-split volumes per environment | **never assembled** — the old protocol only recorded their ids; nothing copies them into `ref/volumes/` |
| `well_conditioned`: 40 well ensembles | exist in the old `references/well_conditional/` layout, 7 of 8 environments adequate, lobe short (see below); not in `ref/wells/` |
| repeat ensembles for `variety` / `calibration` | old layout holds per-condition probability maps only, not volumes; nothing in `ref/repeats/` |
| `manifest.csv` | written by `tools/build_reference.py` (field rows so far) |

`tools/build_reference.py` assembles `ref/fields/` and the manifest from
generator output. The `unconditional` and `well_conditioned` halves of the
reference still need building, and `cmd_score` does not yet run the
repeats-based checks at all (`variety`, `well_blending`, `calibration`), so at
most 9 of 12 checks can currently be scored. Once the directory is complete,
host it or document the local path; `resbench download` currently says it is
not hosted rather than failing later.

## Not in v1

**More than one well at a time.** Two boreholes produce so many distinct
combinations that exact-match rejection stops being affordable; a *single* well
in `lobe` already took about a quarter of a million ResMill runs.

**Porosity, permeability and flow response.** The dataset is facies only, and a
flow simulator would break the "runs on a laptop" property.
