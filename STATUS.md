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
all eight environments, CPU only. Remaining: run it at `--n 32`. One correction
is baked in — ResMill's `ntime` is a global event cap, so it must be scaled by
the area ratio or the upper channel levels never run (NTG collapses from 0.169
to 0.041 for shoestring).

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

## Not hosted yet

The ~12 GB reference has no download location. `resbench download` says so
rather than failing later; point ResBench at a local copy with `--reference` or
`RESBENCH_REFERENCE`.

## Not in v1

**More than one well at a time.** Two boreholes produce so many distinct
combinations that exact-match rejection stops being affordable; a *single* well
in `lobe` already took about a quarter of a million ResMill runs.

**Porosity, permeability and flow response.** The dataset is facies only, and a
flow simulator would break the "runs on a laptop" property.
