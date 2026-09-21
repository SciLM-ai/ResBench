"""Build the `unconditional` reference and designate its repeat conditions.

Selects 512 test-split volumes per environment EXACTLY as the frozen protocol
did (EVAL.md section 6: sort the test split by (layer_type, shard_dir,
sample_idx), one `default_rng(20260726)` stream, environments in canonical
order, `choice(n, 512, replace=False)` per environment, then one noise seed
per row from the same stream). v1 scores and the frozen master table therefore
describe the same reference ensemble. Pass --verify with the frozen lobe
manifest and the script refuses to write anything unless every lobe id
matches.

Writes ref/volumes/<slug>/volumes.npz (ids + int8 volumes, straight copies of
the dataset, or with --regenerate the same rows and seeds run again through the
ResMill checked out at RESMILL_REPO, recorded in ref/ENGINE.txt) and manifest
rows:
    task=unconditional          one per volume; id = `<env>|<shard_dir>|<sample_idx>`
    task=repeats_unconditional  cond0..cond4 = rows 0..4 of each environment

The well location for the `well_conditioned` samples task is also fixed here,
one interior column per row from its own seeded stream, so a submitter reads
the borehole from the reference volume at (well_x, well_y).

    python tools/build_unconditional_reference.py --out REF \\
        --verify /scratch/08405/ilgar/specialist_runs/manifest_lobe.csv
"""
import argparse, csv, json, os, subprocess, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resbench.io import ENVIRONMENTS, SLUG, CONDITIONS  # noqa: E402
from tools._manifest import upsert                      # noqa: E402
from tools.gen_field_reference import RESMILL_REPO      # noqa: E402

MANIFEST_SEED = 20260726       # frozen protocol, EVAL.md section 6
N_PER_ENV = 512
WELL_XY_SEED = 20260919        # well column per row for the well_conditioned samples task
WELL_INTERIOR = (8, 56)        # inclusive; same interior the published wells were mined over
SLIM = ['requested_ntg', 'azimuth', 'width_cells', 'depth_cells', 'asp',
        'mCHsinu', 'mFFCHprop', 'probAvulInside', 'trunk_length_fraction']
DATA_DIR = Path(os.environ.get('RESERVOIR_DATA_DIR',
                               os.path.join(os.environ.get('SCRATCH', '.'),
                                            'SiliciclasticReservoirs')))


def frozen_selection(data_dir):
    split = pd.read_parquet(data_dir / 'splits' / 'test.parquet')
    split = split.sort_values(['layer_type', 'shard_dir', 'sample_idx'],
                              kind='mergesort').reset_index(drop=True)
    rng = np.random.default_rng(MANIFEST_SEED)
    sel = {}
    for lt in ENVIRONMENTS:
        env = split[split['layer_type'] == lt].reset_index(drop=True)
        pick = np.sort(rng.choice(len(env), size=N_PER_ENV, replace=False))
        sel[lt] = env.iloc[pick].reset_index(drop=True)
    noise = rng.integers(0, 2**31 - 1, size=N_PER_ENV * len(ENVIRONMENTS))
    return sel, noise


def _regen_one(task):
    env, row, seed = task
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from tools.gen_repeats_reference import native_kwargs, CONFIG_DIR, CONFIG_FOR_ENV
    from resmill.dataset.generate import generate_sample
    grid = json.loads((CONFIG_DIR / CONFIG_FOR_ENV[env]).read_text())['grid']
    f, _, _, _, _ = generate_sample({'layer_type': env.split(':')[0],
                                     'params': native_kwargs(row, env), 'seed': int(seed)}, grid)
    return np.asarray(f, np.int8)


def engine_stamp():
    """`ResMill <commit>` of the checkout the volumes came from; refuses a dirty tree."""
    head = subprocess.run(['git', '-C', str(RESMILL_REPO), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(RESMILL_REPO), 'status', '--porcelain', '--', 'resmill'],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f'{RESMILL_REPO} has uncommitted changes under resmill/; commit them '
                         f'first so ENGINE.txt names the code that made the reference')
    return f'ResMill {head}  ({RESMILL_REPO})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--verify', help='frozen manifest_lobe.csv; refuse unless ids match')
    ap.add_argument('--data-dir', default=str(DATA_DIR))
    ap.add_argument('--regenerate', action='store_true',
                    help='run the selected rows again with the checked-out ResMill instead of copying '
                         'the dataset volumes; ntg becomes the regenerated volume\'s sand fraction')
    ap.add_argument('--jobs', type=int, default=40)
    a = ap.parse_args()
    data = Path(a.data_dir); out = Path(a.out)
    engine = engine_stamp() if a.regenerate else None

    sel, noise = frozen_selection(data)

    if a.verify:
        frozen = list(csv.DictReader(open(a.verify, newline='')))
        theirs = [r['row_id'] for r in frozen]
        mine = [f"{r['layer_type']}|{r['shard_dir']}|{int(r['sample_idx'])}"
                for _, r in sel['lobe'].iterrows()]
        if mine != theirs:
            bad = next(i for i, (x, y) in enumerate(zip(mine, theirs)) if x != y)
            raise SystemExit(f'selection does NOT reproduce the frozen protocol: '
                             f'first mismatch at lobe row {bad}: {mine[bad]} vs {theirs[bad]}')
        seeds_ok = [int(r['fresh_noise_seed']) for r in frozen] == list(noise[:N_PER_ENV])
        print(f'verified: all {len(mine)} lobe ids reproduce the frozen manifest; '
              f'noise seeds {"match" if seeds_ok else "DIFFER"}')
        if not seeds_ok:
            raise SystemExit('noise seeds differ from the frozen manifest')

    xy_rng = np.random.default_rng(WELL_XY_SEED)
    rows_vol, rows_rep = [], []
    for ei, lt in enumerate(ENVIRONMENTS):
        env = sel[lt]
        vols = np.empty((N_PER_ENV, 64, 64, 32), np.int8)
        ids = []
        params_ntg = np.empty(N_PER_ENV)
        slim = {c: [None] * N_PER_ENV for c in SLIM}
        full_rows = [None] * N_PER_ENV          # every parameter of the row, for --regenerate
        for shard, grp in env.groupby('shard_dir', sort=False):
            fac = np.load(data / shard / 'facies.npy', mmap_mode='r')
            arrow = pq.read_table(data / shard / 'params.parquet')
            tab = arrow.to_pandas()
            for i, r in grp.iterrows():
                si = int(r['sample_idx'])
                vols[i] = fac[si]
                params_ntg[i] = float(tab['ntg'].iloc[si])
                full_rows[i] = {c: arrow[c][si].as_py() for c in arrow.column_names}
                for c in SLIM:
                    if c in tab.columns:
                        v = tab[c].iloc[si]
                        slim[c][i] = None if pd.isna(v) else float(v)
                    elif c == 'requested_ntg' and 'NTGtarget' in tab.columns:
                        slim[c][i] = float(tab['NTGtarget'].iloc[si])
        for i, r in env.iterrows():
            ids.append(f"{lt}|{r['shard_dir']}|{int(r['sample_idx'])}")
        realized = vols.reshape(N_PER_ENV, -1).mean(1)
        # The dataset's `ntg` column IS the realized mean; anything else means
        # the wrong volume was read.
        gap = np.abs(realized - params_ntg).max()
        if gap > 1e-5:
            raise SystemExit(f'{lt}: volume mean disagrees with params ntg by {gap:.2e}')
        if a.regenerate:
            with Pool(a.jobs) as pool:
                regen = pool.map(_regen_one, [(lt, full_rows[i], int(full_rows[i]['seed']))
                                              for i in range(N_PER_ENV)], chunksize=4)
            regen = np.stack(regen)
            same = float((regen == vols).all(axis=(1, 2, 3)).mean())
            if lt == 'lobe' and same < 1.0:
                # Only the fluvial walker changed; a lobe that no longer reproduces
                # means the checkout differs in more than that.
                raise SystemExit(f'lobe: only {same:.0%} of the rows reproduce the dataset '
                                 f'under {engine}')
            print(f'{lt:<24} regenerated: {same:>4.0%} identical to the dataset, '
                  f'mean ntg {regen.mean():.4f} (dataset {realized.mean():.4f})')
            vols = regen
            realized = vols.reshape(N_PER_ENV, -1).mean(1)
        d = out / 'volumes' / SLUG[lt]; d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / 'volumes.npz', ids=np.array(ids, dtype=object), volumes=vols)
        for i in range(N_PER_ENV):
            wx, wy = (int(v) for v in xy_rng.integers(WELL_INTERIOR[0], WELL_INTERIOR[1] + 1, 2))
            row = {'environment': lt, 'id': ids[i], 'row_index': i,
                   'shard_dir': env['shard_dir'].iloc[i], 'sample_idx': int(env['sample_idx'].iloc[i]),
                   'ntg': float(realized[i]), 'noise_seed': int(noise[ei * N_PER_ENV + i]),
                   'well_x': wx, 'well_y': wy}
            row.update({c: slim[c][i] for c in SLIM})
            rows_vol.append(row)
            if i < len(CONDITIONS['unconditional']):
                # Manifest ids are global, so the condition id carries the
                # environment; the FILE is still <slug>/repeats/cond<i>.npz.
                rep = dict(row); rep.update({'id': f"{lt}|{CONDITIONS['unconditional'][i]}",
                                             'source_id': ids[i], 'well_x': None, 'well_y': None})
                rows_rep.append(rep)
        print(f'{lt:<24} 512 volumes  mean ntg {realized.mean():.4f}  -> {d / "volumes.npz"}')
    if engine:
        (out / 'ENGINE.txt').write_text(engine + '\n')
    n1, k1 = upsert(out, 'unconditional', rows_vol)
    n2, k2 = upsert(out, 'repeats_unconditional', rows_rep)
    print(f'manifest.csv: {n1} unconditional rows, {n2} repeat conditions; {k2} other rows kept')


if __name__ == '__main__':
    main()
