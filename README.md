# ResBench

**ResMill** is a geological process simulator that builds 3D sand-and-shale reservoirs by
simulating deposition: channels migrating and avulsing, lobes spreading and stacking.

**ResBench** asks whether a generative AI model can build reservoirs that no check in this
benchmark can tell apart from ResMill's.

```bash
pip install "resbench[download]"
resbench download                   # ResMill reference from HuggingFace, 250 MB
resbench validate ./my_submission   # shapes, ids, counts; fails fast
resbench score    ./my_submission   # the table below, plus results.json
```

The reference is [`SciLM/ResBench-reference`](https://huggingface.co/datasets/SciLM/ResBench-reference),
pinned in the code to one revision so scores never drift. `score` downloads it on first
use if `download` was skipped; `--reference PATH` scores against a local copy instead.

Python 3.10+, five packages (`numpy`, `scipy`, `pandas`, `pyarrow`, `matplotlib`), plus
`huggingface_hub` for the download.
**No GPU, no PyTorch, no geomodeling software** — it reads saved volumes off disk and
does arithmetic on them. ResBench never loads your checkpoint and never imports your
code, so any architecture, framework, language or sampler can be scored.

---

## 1. Data

`SiliciclasticReservoirs` on HuggingFace. Volumes are `64 x 64 x 32` cells, one byte each,
**1 = sand, 0 = shale**. Split 90 / 5 / 5; ResBench only ever uses the **test** split.

| environment | cell size | volume covers |
|---|---|---|
| `lobe` | 100 m | 6.4 x 6.4 km, 32 m thick |
| the other seven | 10 m | 0.64 x 0.64 km, 32 m thick |

```
lobe · channel:PV_SHOESTRING · channel:CB_LABYRINTH · channel:CB_JIGSAW
channel:SH_DISTAL · channel:SH_PROXIMAL · channel:MEANDER_OXBOW · delta
```

A submission covers all eight or it does not get a rank.

Your model is compared against three references:

- **512 reference volumes per environment.** Test-split volumes, selected with the
  published index seed; `manifest.csv` lists each one's id, its parameters — **its
  realized sand fraction**, body width and depth, sinuosity, azimuth — a noise seed, and
  the map location `(well_x, well_y)` of the borehole used in the well-conditioned task.
- **5 repeat conditions per environment**, `cond0`–`cond4`: the parameters of reference
  rows 0–4, each with 256 ResMill runs, for the checks that need many runs of one input.
- **5 wells per environment**, `well1`–`well5`. A well is one vertical borehole: the 32
  cells under a map location. ResMill was run repeatedly from one parameter row and every
  volume whose column matched the pattern *exactly* was kept, up to 256. Each well has
  >= 2 separate sand intervals, a sand fraction within 0.15 of the environment's, and
  >= 50 exact matches. `manifest.csv` names the parameter row (`source_id`), the
  location and the pattern.
- **32 large fields per environment**, each a single ResMill run. See §3.

---

## 2. Tasks

| task | input | output |
|---|---|---|
| `unconditional` | geological parameters only | `64 x 64 x 32` |
| `well_conditioned` | parameters + one vertical well to honor exactly | `64 x 64 x 32` |
| `field_scale` | parameters, at a much larger extent | see §3 |

They count equally toward the score. `unconditional` and `well_conditioned` are scored
against the **same** 512 reference volumes: each conditioned volume takes its well from
its own reference volume, so across the ensemble the wells are as varied as the geology.
The gap between the two scores is published as **well drift** — if it is large, the model
is bending the geology to fit the borehole.

---

## 3. Field scale

| environment | field extent | covers |
|---|---|---|
| `lobe`, `delta` | `512 x 512 x 32` | square |
| `channel:*` | `512 x 128 x 32` | elongated **along flow**, azimuth forced to 0 |

Channels are elongated in the flow direction because that is how a channel belt extends;
a square box clips every channel at the same length and makes long-range statistics
meaningless.

**Every check runs on the whole field. Nothing is cut into tiles, on either side.**
A 64-cell box can never hold a body wider than 64 cells, so scoring a large field through
small windows throws away the only thing this task tests. Measured on 16 ResMill fields:
cutting ResMill's *own* fields into 64-cubes and scoring them against natively built
64-cubes, ResMill fails its own test at body size 0.0510 against a 0.0190 band.

---

## 4. The twelve checks

Each is one specific way a reservoir model can be wrong. Nine run in every task;
`variety`, `well_blending` and `calibration` need 128 runs of one fixed input, which at
field scale is unaffordable.

**W1** below is the earth-mover distance: sort both sets of values and average the gap
between them. Used wherever the quantity is a pile of numbers with no natural pairing.

| check | what it asks | how it is computed | number |
|---|---|---|---|
| `net_to_gross` | Does the model honor the sand fraction it was given? | The conditioned sand fraction is the reference volume's *realized* value, so the target is the condition itself. Needs no reference data. | mean abs. error, over a **0.01 tolerance** (a chosen constant, not a band — ResMill is never asked to reproduce a realized sand fraction) |
| `variogram` | How long and wide are the bodies, which way do they point? | For each axis and separation `h`, count cell pairs that *disagree*, divide by pair count, halve. `h` to half the axis: 80 numbers for a 64-cube. Counts summed over the ensemble before dividing. | mean abs. difference over the 80 points, ÷ sill `p(1-p)` |
| `patterns` | Is the local texture right even where the variogram is? | Every 2x2x2 block is one of 256 fillings — read its 8 cells in a fixed order and it *is* an 8-bit number. Every offset, so 123,039 blocks per volume. | Jensen-Shannon divergence, bits (KL is asymmetric and goes infinite when one side lacks a pattern) |
| `bed_thickness` | Are the beds the right thickness? | Read down each of the 4,096 map columns, exactly what a vertical well logs; each unbroken sand run is one bed. Intercept thicknesses, not body thicknesses. | W1 |
| `connectivity` | Can you walk from one sand cell to another through sand? | Cells touch only if they share a **face**; flood-fill gives body numbers. Among pairs `h` apart that are both sand, what fraction share a body? Tallies summed over the ensemble, divided once. | mean abs. difference over the 80 lags |
| `compartments` | One connected reservoir or many pockets? | `Γ = Σnᵢ²/(Σnᵢ)²`, the chance two random sand cells share a body — the fraction one well reaches. Plus the Euler characteristic, `pieces − tunnels + cavities` per million cells, which separates rocks Γ cannot. | abs. difference on each |
| `body_size` | Are the blobs the right size? | Same flood-fill; pool every blob's cell count, take log₁₀ so a body twice too big counts the same at 10 cells as at 10,000. | W1 on log₁₀ sizes |
| `chord_lengths` | How wide are bodies where blobs are ambiguous, and how elongated? | Amalgamated lobes flood-fill into one blob, so `body_size` measures welded clumps. A chord needs no such decision: scan lines within each depth slice, along the azimuth and across it. | W1 on log₁₀ lengths + abs. difference in the along/across ratio |
| `speckle` | Stray grains, and worse under conditioning? | Sand blobs under 8 cells per volume, also reported against well count. | abs. difference |
| `variety` | Given identical inputs, does it vary as much as ResMill? | 128 runs on one input; per-cell P(sand) → entropy in bits. Unconditionally ResMill's map is nearly flat (mean 0.98, sd 0.02), so the signal is the *level*: a collapsing model falls away from it. | mean abs. difference, bits |
| `well_blending` | Does the borehole blend into the rock, or leave a seam? | Same entropy map with a well present, binned by distance. ResMill runs 0.00 at the well then 0.13, 0.22, 0.39, 0.55, 0.81, 0.96 at distance 1, 2, 4, 6, 10, 20. **The only check that measures conditioning** — well reproduction error is zero for everybody by construction. | mean abs. difference of the decay curve, bits; signed near-field offset reported beside it |
| `calibration` | Of cells called sand 70% of the time, are 70% sand? | Same 128 runs, different question. Bin off-well cells by stated probability, compare against ResMill's actual frequency. **Not `variety` again**: entropy is symmetric, `H(0.3) = H(0.7)`, so a model that inverts every probability scores 0.000 on `variety` and 0.375 here. | expected calibration error |

Deliberately absent: **well reproduction error** (zero for everybody by construction),
**power spectra** (the Fourier dual of the variogram), **sand-fraction spread** (implied by
`net_to_gross`), **porosity / permeability / flow** (the dataset is facies only).

---

## 5. Score

```
band  = check( half of ResMill , the other half )
s     = check( your model , ResMill )  /  band
score = average of s over every check, task and environment
```

**`s = 1` means your model is as close to ResMill as ResMill is to itself.** Lower is
better and **1.0 is the floor, not zero**.

| s | verdict |
|---|---|
| ≤ 1 | **matched** — inside ResMill's own scatter |
| 1 – 2 | **close** — detectable, within twice that scatter |
| > 2 | **distinguishable** |

An **average**, not a median. On 47 scored checkpoints the two rank models at a
correlation of **−0.63** — the order reverses, and the median's top model misses one check
by 5.9x its band and another by 8.0x.

```
$ resbench score ./my_submission

ResBench v1 · 8 environments · sampler heun-100 · NFE 100

                       score   matched   weakest
  unconditional         0.94     10/11   connectivity   2.4
  well_conditioned      1.31      8/11   well_blending  3.1  (near-field -0.029)
  field_scale           1.44      6/9    body_size      2.9  [outpainting]

  overall               1.23     24/31
  well drift            0.12   conditioning does not bend the geology
```

---

## 6. Submitting

```
my_submission/
├── submission.yaml                    # model, params, sampler, NFE, training set
├── unconditional/<slug>/
│   ├── samples/*.npz                  # 512 volumes, one per reference volume, same ids
│   └── repeats/cond0.npz … cond4.npz  # 128 runs each of the condition's parameters
├── well_conditioned/<slug>/
│   ├── samples/*.npz                  # 512 volumes, same ids, borehole read from the reference
│   └── repeats/well1.npz … well5.npz  # 128 runs each, honoring the published well
└── field_scale/<slug>/
    └── fields/*.npz                   # 32 fields at the §3 extent, ids from the manifest
```

`<slug>` is the environment with `:` replaced by `_`. Every `.npz` holds `ids` and
`volumes` (int8, 1 = sand). Ids are what line your output up with the reference:

| part | id | where the condition comes from |
|---|---|---|
| `samples` | the reference volume's id, e.g. `lobe\|lobe/shard_0002\|192` | that manifest row: parameters, and for the well task the borehole at `(well_x, well_y)` of the reference volume |
| `repeats/cond<i>.npz` | `cond<i>\|<k>`, k = 0…127 | manifest row `<env>\|cond<i>` (task `repeats_unconditional`) |
| `repeats/well<i>.npz` | `well<i>\|<k>`, k = 0…127 | manifest row `<env>\|well<i>` (task `repeats_well`): its `source_id` parameters plus the pattern at `(well_x, well_y)` |
| `fields` | `field\|<slug>\|<seed>` from the manifest | manifest row of that field: its realized sand fraction is the condition |

Well cells in a well-conditioned volume must equal the pattern exactly; the harness treats
them as given and excludes them from `calibration`.

```bash
resbench validate ./my_submission --envs lobe              # one environment while you develop
resbench score ./my_submission --reference REF --envs lobe
resbench score ./my_submission --reference REF --target-column ntg_source_cube
#   ^ only for a submission generated BEFORE the manifest existed; see SPEC.md
```

**Fixed for everyone:** the dataset and splits, the 512 reference volumes and their seed,
the parameters handed to your model, the 40 wells, the twelve checks, the bands, the field
extents.

**Yours, and printed on the leaderboard:** architecture, training recipe, sampler and step
count, guidance, precision, how you reached field scale. The sampler is deliberately not
fixed — a step count from one architecture's paper penalizes every other architecture; our
own model scores 0.086 at 50 Euler steps and 0.058 at 100 Heun steps. **NFE is a
leaderboard column** instead.

**Disqualifying:** training or tuning on the test split · misdeclaring your training set ·
hand-picking outputs · leaving out environments or tasks · returning volumes retrieved or
copied from the dataset instead of generating them.

Enforced by declaration and reproducibility, the same way Matbench Discovery does it.
There is no detector behind these rules: a model that quietly trained on the test split
without memorizing any single volume leaves no fingerprint in a folder of saved volumes.

---

## 7. Repository

```
resbench/
├── resbench/
│   ├── checks/            one module per check, one test each
│   ├── metrics.py         the low-level estimators
│   ├── bands.py           split-half bands
│   ├── score.py           normalizing, averaging, verdicts
│   └── cli.py             download · validate · score · figures
├── reference/
│   ├── manifest.csv       one row per reference item; `task` says which kind
│   ├── volumes/<slug>/    512 test volumes per environment
│   ├── repeats/<slug>/    cond0–4 (256 ResMill runs each), well1–5 (up to 256 exact matches each)
│   └── fields/<slug>/     32 ResMill fields at the §3 extent
├── tools/                 build_*_reference.py, gen_*_reference.py, mine_wells.py, relabel_submission.py
├── example/               make_example.py writes a valid one-environment submission
└── tests/
```

Every module in `checks/` exports `summarize(volumes, ctx)`, `merge(parts)` and
`compare(model, ref)`. Summaries merge across files, so nothing needs a whole ensemble in
memory. Adding a check means one module and one test.

See **SPEC.md** for the full protocol and **STATUS.md** for what is not yet finished.
