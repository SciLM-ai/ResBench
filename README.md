# ResBench

A minimal, self-contained benchmark for generative models of binary
sand/shale reservoir facies, built on the SiliciclasticReservoirs dataset
(64×64×32 int8 volumes, 8 depositional environments). Model-agnostic: it
scores **directories of saved volumes** and imports nothing from any
generative model or data engine. The core is ~800 lines of Python with four
dependencies; all reference data needed for scoring ships in this repo
(~26 MB). Metrics follow the minimum acceptance criteria of Leuangthong et
al. (2004) with categorical/MPS extensions (Boisvert et al., 2010) and
GenAI-specific checks (Merzoug et al., 2025).

The frozen protocol — every metric definition, seed, ensemble size, and
acceptance criterion, fixed before computation — is [EVAL.md](EVAL.md).
[RESULTS.md](RESULTS.md) is the worked example: the full evaluation of the
ResFlow foundation model.

## Benchmark components

| component | protocol | reference data |
|---|---|---|
| Geostatistical master table — NTG, indicator variograms, connectivity τ(h), geobody statistics, well exactitude; verdicts vs the engine's split-half noise band | EVAL.md §1–8 | test-split ensemble (from the dataset) |
| Unconditional entropy calibration — is ensemble variability the right amount, per environment? | Addendum B | `references/entropy_unconditional/` (64 conditions) |
| Well-conditional entropy calibration — the same, given well data; exact engine-conditional ensembles, no rejection sampling ever needed | Addendum C | `references/well_conditional/` (16 wells × 50–287 realizations) |
| MultiDiffusion assembly consistency — do statistics survive large-domain tiled generation? | Addendum E | `references/assembly/` |
| Baseline protocol | Addendum D | `results/baseline/` |

## Install

```bash
pip install -e .          # numpy, scipy, pandas, pyarrow, matplotlib
pytest tests/             # metrics verified against brute force
```

## Score a model

Save your model's volumes as npz shards (`ids` + `volumes`, int8 {0,1},
one subdirectory per environment — full contract in EVAL.md §8), then:

```bash
python -m resbench.run --pred-dir YOUR_VOLUMES --ref-dir REFERENCE \
    --out results/metrics.parquet --figures
```

Output: the master table (parquet + markdown) with inside/near/outside
verdicts against the engine's own Monte-Carlo noise band, bootstrap CIs,
and publication figures. For the calibration components, generate K = 128
samples per condition listed in the `references/` metadata (each npz
carries the well pattern, mask, and parameter pointer) and compare
voxelwise entropy — see `paper/analysis/` for worked implementations.

## Adapting to your own dataset

The metrics, split-half band logic, and CLI are shape- and domain-agnostic.
To benchmark on different data:

1. Edit **one file**, `resbench/__init__.py`: set `VOLUME_SHAPE`,
   `MAX_LAGS` (half of each axis extent), and your category list
   (`LAYER_TYPES`).
2. Build your reference ensemble: a held-out sample of real/simulated
   volumes per category, saved in the same npz format.
3. Run the same CLI. The split-half band machinery automatically defines
   "matching" at your reference ensemble's own noise level.

The entropy-calibration components additionally require an ensemble
generator for your data (any simulator that can redraw realizations under
fixed parameters); the mining approach is described in EVAL.md Addendum C.

## Repository layout

```
EVAL.md            frozen protocol (read this first)
RESULTS.md         worked example: ResFlow scored on all components
resbench/          the package (~800 lines): metrics, stats, io, figures, CLI
analysis/          runnable scorers for the Addendum E/F/G components
tests/             brute-force verification of every metric
references/        benchmark reference data (self-contained, ~26 MB)
results/           the worked example's outputs (tables, figures, manifests)
paper/analysis/    one-off study scripts behind RESULTS.md (archaeology)
PROPOSED_EXTENSIONS.md   known limitations of the frozen protocol
```

Scoring an assembly (Addendum E) against the shipped reference:

```bash
python analysis/assembly_stats.py --env-slug lobe \
  --native-dir <dir of natively generated 64-cubes> \
  --assembly-dir <dir of assembled fields> \
  --engine-dir references/assembly --out-dir scored
```

Two caveats are documented in `PROPOSED_EXTENSIONS.md` and are worth reading
before quoting a number. The Addendum E tile grid scores a 5x5 subgrid of
64-cubes whose placement is a free parameter at extents other than 424, and it
moves geobody W1 by as much as models typically differ from one another --
`--full-coverage` removes the free parameter, and `--tile-origin` exposes it for
sensitivity studies. Separately, the reference volumes are generated natively at
64 cubed, so bodies in them are truncated by construction in a way that bodies
cut out of a larger field are not; the engine does not match itself under this
protocol, and the effective floor is well above the quoted split-half band.

Raw engine draw pools (~17 GB; for mining new well patterns or extending
the references) are published separately with the dataset release.

## License

MIT — see [LICENSE](LICENSE).
