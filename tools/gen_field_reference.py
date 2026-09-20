"""Generate the ResBench field-scale reference: whole ResMill fields at the
scored extent, for every environment.

Why this exists: field-scale statistics have to be compared against ResMill
fields of the SAME size. Cutting 64-cubes out of a large field and comparing
them to natively built 64-cubes is a different distribution -- measured on 16
fields, ResMill fails its own test that way (body-size distance 0.0510 against
a 0.0190 band). See SPEC.md section 7.

Extent per environment:
  lobe, delta            square,    512 x 512 x 32   (64x the training area)
  channel:*              elongated, 512 x  64 x 32   (8x, all of it along flow)

Channels are extended along flow and not sideways because that is how a channel
belt actually grows; a square box clips every channel at the same length and
makes the long-range statistics meaningless. Cell size is taken from each
environment's own ResMill config and never changed (lobe 100 m, the rest 10 m),
so only the number of cells grows.

ResMill's event budget does NOT scale with the domain, so it has to be scaled
here or the fields come out starved. For channels `ntime` is per level
(`ntime_per_level=True` in every channel row): each level's sand target grows
with the domain while its event budget stays fixed, so every level under-fills
equally. Measured on PV_SHOESTRING at 512 x 64: NTG 0.1163 unscaled against
0.1722 native, and 0.1672 once `ntime` is multiplied by the area ratio. Delta
uses `ntime_per_gen` instead and is scaled the same way.

  python tools/gen_field_reference.py --out DIR [--n 32] [--envs lobe,delta]
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RESMILL_REPO = Path(os.environ.get(
    'RESMILL_REPO', '/work/08405/ilgar/vista/codes/ResMill_ls6'))
sys.path.insert(0, str(RESMILL_REPO))
CONFIG_DIR = RESMILL_REPO / 'examples' / 'dataset_generation'

DATA_DIR = Path(os.environ.get(
    'RESERVOIR_DATA_DIR',
    os.path.join(os.environ.get('SCRATCH', '.'), 'SiliciclasticReservoirs')))

CONFIG_FOR_ENV = {
    'lobe': 'config_full_lobes.json',
    'channel:PV_SHOESTRING': 'config_full_pv_shoestring.json',
    'channel:CB_LABYRINTH': 'config_full_cb_labyrinth.json',
    'channel:CB_JIGSAW': 'config_full_cb_jigsaw.json',
    'channel:SH_DISTAL': 'config_full_sh_distal.json',
    'channel:SH_PROXIMAL': 'config_full_sh_proximal.json',
    'channel:MEANDER_OXBOW': 'config_full_meander_oxbow.json',
    'delta': 'config_full_delta.json',
}
ENVS = list(CONFIG_FOR_ENV)

# Meta/derived columns that are not create_geology kwargs.
ENGINE_IGNORE = {
    'layer_type', 'seed', 'caption', 'ntg', 'requested_ntg',
    'poro_ave', 'perm_ave',
    'r_ave_m', 'r_ave_cells', 'r_major_m', 'r_major_cells',
    'dh_ave_m', 'dh_ave_cells',
    'mCHdepth_m', 'mCHdepth_cells', 'mCHwidth_m', 'mCHwidth_cells',
    'width_cells', 'depth_cells',
}

SQUARE = (512, 512)
ELONGATED = (512, 64)            # extended along flow only
NATIVE_XY = (64, 64)             # the training extent the budgets were tuned for
MARGIN_XY, SEED_BASE = 8, 2026091800


def extent_for(env):
    return SQUARE if env in ('lobe', 'delta') else ELONGATED


def grid_for(env):
    """The environment's own grid, widened to the field extent. Cell size,
    depth, crop margins and every other convention are left untouched."""
    g = dict(json.loads((CONFIG_DIR / CONFIG_FOR_ENV[env]).read_text())['grid'])
    dx, dy = g['x_len'] / g['nx'], g['y_len'] / g['ny']
    ex, ey = extent_for(env)
    g['nx'], g['ny'] = ex + 2 * MARGIN_XY, ey + 2 * MARGIN_XY
    g['x_len'], g['y_len'] = dx * g['nx'], dy * g['ny']
    return g, (ex, ey), (dx, dy)


def engine_kwargs(row, env):
    kw = {k: v for k, v in row.items()
          if k not in ENGINE_IGNORE and v is not None
          and not (isinstance(v, float) and np.isnan(v))}
    if 'NTGtarget' not in row or row.get('NTGtarget') is None:
        kw['ntg'] = row['requested_ntg']
    if env == 'lobe':
        kw['poro_ave'], kw['perm_ave'] = row['poro_ave'], row['perm_ave']
    if env.startswith('channel:'):
        # Point the channels down the long axis. Only the top-level `azimuth`
        # rotates the model; `mCHazi` is engine-internal and is left alone.
        if 'azimuth' in kw:
            kw['azimuth'] = 0.0
    # Scale the event budget with the domain, or the field starves: the sand
    # target grows with the field while the budget does not. The two layer
    # families need different laws, both measured:
    #
    #   channels  budget x AREA. Each event fills a swath, so covering 8x the
    #             area needs 8x the swaths. PV_SHOESTRING at 512x64: NTG
    #             0.1163 unscaled vs 0.1722 native, 0.1672 scaled.
    #   delta     budget x LINEAR extent (sqrt of area). A delta grows outward
    #             from a point apex and branches, so it needs generations in
    #             proportion to how far it must build, not to the area it
    #             covers. At 16x area, 4x events gives NTG 0.5608 against
    #             0.5416 native; 4x events at 4x area overshoots to 0.6673 and
    #             1x undershoots to 0.4012.
    ex, ey = extent_for(env)
    area = (ex * ey) / (NATIVE_XY[0] * NATIVE_XY[1])
    for k, ratio in (('ntime', area), ('ntime_per_gen', area ** 0.5)):
        if k in kw and kw[k]:
            kw[k] = int(round(kw[k] * ratio))
    return kw


def _one(task):
    env, row, seed = task
    from resmill.dataset.generate import generate_sample
    grid, (ex, ey), _ = grid_for(env)
    t0 = time.time()
    facies, _, _, _, _ = generate_sample(
        {'layer_type': env.split(':')[0], 'params': engine_kwargs(row, env),
         'seed': int(seed)}, grid)
    f = np.asarray(facies, np.int8)
    assert f.shape == (ex, ey, 32), f'{env}: got {f.shape}, want {(ex, ey, 32)}'
    return env, f, int(seed), float(f.mean()), time.time() - t0


def pick_rows(env, n):
    """n test-split conditions spread evenly across the realized sand
    fraction, so the fields span the same range as the 64-cube reference."""
    test = pd.read_parquet(DATA_DIR / 'splits' / 'test.parquet')
    test = test[test['layer_type'] == env].reset_index(drop=True)
    rows = []
    for _, r in test.iterrows():
        rows.append(r.to_dict())
    idx = np.linspace(0, len(rows) - 1, n).round().astype(int)
    out = []
    for i in idx:
        r = rows[int(i)]
        t = pq.read_table(DATA_DIR / r['shard_dir'] / 'params.parquet')
        out.append({c: t[c][int(r['sample_idx'])].as_py() for c in t.column_names})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--envs', default=','.join(ENVS))
    ap.add_argument('--jobs', type=int, default=min(140, os.cpu_count() or 8))
    a = ap.parse_args()
    os.environ.setdefault('MPLBACKEND', 'Agg')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # Build per environment, then round-robin, so the memory-heavy 512x512
    # fields (~4 GB a process, against ~1.7 GB for a 512x64 one) are spread
    # through the queue instead of all landing in the first wave.
    per_env = []
    for env in (e for e in a.envs.split(',') if e):
        per_env.append([(env, row, SEED_BASE + 1000 * ENVS.index(env) + k)
                        for k, row in enumerate(pick_rows(env, a.n))])
    tasks = [t for group in zip(*per_env) for t in group] if per_env else []
    tasks += [t for g in per_env for t in g[min(map(len, per_env)):]]
    print(f'{len(tasks)} fields over {a.jobs} workers', flush=True)

    want = {}
    for env, _, _ in tasks:
        want[env] = want.get(env, 0) + 1

    got, t0 = {}, time.time()
    with Pool(a.jobs) as pool:
        for i, (env, f, seed, ntg, dt) in enumerate(
                pool.imap_unordered(_one, tasks, chunksize=1), 1):
            got.setdefault(env, []).append((seed, f, ntg))
            print(f'  {i}/{len(tasks)} {env:<24} seed {seed} '
                  f'shape {f.shape} ntg {ntg:.4f} {dt:.0f}s', flush=True)
            # Checkpoint after every field, not only when an environment is
            # complete. A 12-hour limit once killed a run at 123/128 and threw
            # away 15 finished delta fields because the 16th never arrived.
            _write_env(out, env, got[env], partial=len(got[env]) < want[env])
            if len(got[env]) == want[env]:
                got.pop(env)

    for env, items in list(got.items()):
        _write_env(out, env, items, partial=len(items) < want.get(env, 0))
    print(f'done in {time.time()-t0:.0f}s', flush=True)


def _write_env(out, env, items, partial=False):
    items.sort(key=lambda t: t[0])
    slug = env.replace(':', '_')
    d = out / slug; d.mkdir(parents=True, exist_ok=True)
    vols = np.stack([f for _, f, _ in items])
    ids = np.array([f'field|{slug}|{s}' for s, _, _ in items], dtype=object)
    np.savez_compressed(d / 'fields.npz', ids=ids, volumes=vols)
    grid, ext, (dx, dy) = grid_for(env)
    (d / 'manifest.json').write_text(json.dumps({
        'environment': env, 'n': len(items),
        'extent': [ext[0], ext[1], 32], 'cell_m': [dx, dy, 1.0],
        'elongated_along_flow': env.startswith('channel:'),
        'grid': grid, 'seed_base': SEED_BASE,
        'mean_ntg': float(np.mean([n for _, _, n in items])),
    }, indent=2))
    print(f'WROTE {env:<24} {len(items)} fields {vols.shape} '
          f'mean ntg {np.mean([n for _,_,n in items]):.4f}', flush=True)


if __name__ == '__main__':
    main()
