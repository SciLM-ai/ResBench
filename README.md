# ResBench

Geostatistical minimum-acceptance benchmark for generative models of binary
sand/shale reservoir facies, built for the SiliciclasticReservoirs dataset
(64×64×32 int8 volumes, 8 depositional environments). Model-agnostic: it
scores **directories of saved volumes** and imports nothing from any
generative model or data engine.

The frozen protocol — ensembles, metric definitions, inference settings,
acceptance criterion — is [EVAL.md](EVAL.md). Metrics follow the minimum
acceptance criteria of Leuangthong et al. (2004) with categorical/MPS
extensions (Boisvert et al., 2010) and GenAI-specific checks (Merzoug et
al., 2025): distribution recovery (facies proportions, indicator variograms,
connectivity, geobody statistics) and data exactitude at conditioning wells,
all judged against the data engine's own split-half Monte-Carlo noise band.

## Install

```bash
conda create -n resbench python=3.12 -y && conda activate resbench
pip install -e .
```

## Usage

```bash
python -m resbench.run \
    --pred-dir  /path/ensemble_a       # generated volumes (dirs per env slug)
    --ref-dir   /path/reference        # reference volumes, same layout
    --out       results/metrics.parquet \
    --well-pred-dir /path/ensemble_b   # optional: well-conditioned ensemble
    --mask-dir  /path/ensemble_b_masks # optional: its well masks
    --manifest  /path/manifest.csv     # optional: well-config per row id
    --figures                          # render the three EVAL.md figures
```

Outputs beside `--out`: `metrics.parquet` + `metrics.md` (master table with
split-half-band verdicts), `report.npy` (all curves/CIs), optional
`well_exactitude.json` and figures (PDF + PNG).

### Input contract

Each ensemble root has one subdirectory per environment slug
(`lobe`, `channel_PV_SHOESTRING`, …) holding `*.npz` shards with keys
`ids` (strings `"{layer_type}|{shard_dir}|{sample_idx}"`) and `volumes`
(int8, `(N, 64, 64, 32)`, values {0, 1}). Mask directories use key `masks`
(uint8, 1 = known). Rows are joined on `ids`; a `#k` suffix on prediction
ids (multiple samples per condition) is stripped for the join.

## Tests

```bash
pytest tests/   # every metric is verified against brute force on 8x8x8 arrays
```

## Repository layout

```
EVAL.md               frozen benchmark protocol (read this first)
RESULTS.md            ResFlow evaluation results (master table + figures)
resbench/metrics.py   NTG, indicator variograms, connectivity, geobodies, exactitude
resbench/stats.py     ensemble summaries, split-half null band, bootstrap, verdicts
resbench/io.py        volume-directory loading / id alignment
resbench/figures.py   the three publication figures
resbench/run.py       CLI entry point (python -m resbench.run)
```

Volume generation for ResFlow lives in the ResFlow repo
(`scripts/rebuttal_eval/`) and communicates with ResBench only through saved
volume directories.
