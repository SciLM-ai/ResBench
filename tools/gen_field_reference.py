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
    # window provenance of a dataset sample (the dataset stores one 64x64x32
    # window of every 128x128x64 engine volume; resmill.dataset.windows)
    'crop_x0', 'crop_y0', 'crop_z0', 'crop_nx', 'crop_ny', 'crop_nz', 'crop_seed',
    'source_shard', 'source_row',
}

SQUARE = (512, 512)
ELONGATED = (512, 128)           # extended along flow only; 128 wide so sinuous
                                 # channels do not end on a side wall
FIELD_NZ = 32                    # the scored field is 32 cells tall, like a training window
MARGIN_XY, SEED_BASE = 0, 2026091800
# Event-budget scaling from the dataset box (128 x 128) to the field, as
# powers of the area ratio: 1.0 = with the area, 0.5 = with the edge, 0 = none.
# Calibrated per family against native volumes (see SPEC.md "Field extent").
# Calibrated 2026-09-23 on 512 x 128 (channels, 4x area) and 512 x 512 (delta,
# 16x area) against each row's native 128 x 128 x 64 volume:
#   PV, CB_LABYRINTH at x4:            field / native 1.06, 1.00, 0.96, 1.15
#   SH_DISTAL, SH_PROXIMAL, CB_JIGSAW: 1.00 to 1.17 whatever the budget (targets are reached)
#   MEANDER_OXBOW (budget-limited):    x1 0.83 / 0.73, x2 0.98 / 1.08, x4 1.59 / 1.18 -> edge law
#   delta n_trees x edge (x4):         1.13, 0.65, 1.11; x2: 0.68, 0.43, 0.78 -> edge law
CHANNEL_NTIME_AREA_EXP = {'default': 1.0, 'channel:MEANDER_OXBOW': 0.5}
DELTA_BIFURCATION_EDGE_EXP = 1.0        # bifurcations per network, power of the edge ratio
DELTA_TREES_EDGE_EXP = 1.0              # networks per generation, power of the edge ratio


def extent_for(env):
    return SQUARE if env in ('lobe', 'delta') else ELONGATED


def engine_grid(env):
    """The environment's own dataset grid: the volume the dataset windows are
    cut from (128 x 128 x 64, no crop)."""
    g = {k: v for k, v in json.loads((CONFIG_DIR / CONFIG_FOR_ENV[env]).read_text())['grid'].items()
         if not k.startswith('_')}
    g.pop('crop', None)
    return g


def grid_for(env):
    """The environment's own grid at the field extent: cell size unchanged,
    x and y widened, 32 cells tall (a field is a complete 32 m column, as tall
    as a training window; the dataset simulates 64 m and windows 32 of them)."""
    g = engine_grid(env)
    dx, dy = g['x_len'] / g['nx'], g['y_len'] / g['ny']
    dz = g['z_len'] / g['nz']
    ex, ey = extent_for(env)
    g['nx'], g['ny'] = ex + 2 * MARGIN_XY, ey + 2 * MARGIN_XY
    g['x_len'], g['y_len'] = dx * g['nx'], dy * g['ny']
    g['nz'], g['z_len'] = FIELD_NZ, dz * FIELD_NZ
    return g, (ex, ey), (dx, dy)


def levels_for_column(row, z_len):
    """The row's number of levels for a column of height ``z_len``, keeping its
    aggradation ratio (level spacing over channel depth): the dataset sampled
    the ratio and derived ``nlevel`` for its 64 m column as
    round((64 - depth) / (ratio * depth)) + 1, both layers anchoring the bottom
    level's base on the floor."""
    key = 'n_generations' if 'n_generations' in row and row['n_generations'] is not None else 'nlevel'
    n = int(row[key]); d = float(row['mCHdepth'])
    g = engine_grid_z(row)
    ratio = (g - d) / (max(n - 1, 1) * d) if n > 1 else 1.0
    return key, max(1, int(round((z_len - d) / (ratio * d))) + 1)


def engine_grid_z(row):
    """z_len the row's level count was derived for (the dataset grid)."""
    return 64.0


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
    g0 = engine_grid(env)
    area = (ex * ey) / (g0['nx'] * g0['ny'])          # the box the budgets were sampled for
    exp = CHANNEL_NTIME_AREA_EXP.get(env, CHANNEL_NTIME_AREA_EXP['default'])
    for k, ratio in (('ntime', area ** exp), ('ntime_per_gen', area ** 0.5)):
        if k in kw and kw[k]:
            kw[k] = int(round(kw[k] * ratio))
    if env == 'delta' and kw.get('bifurcate'):
        edge = ex / g0['nx']
        if kw.get('n_bifurcations'):
            kw['n_bifurcations'] = int(round(kw['n_bifurcations'] * edge ** DELTA_BIFURCATION_EDGE_EXP))
        if kw.get('n_trees'):
            kw['n_trees'] = int(round(kw['n_trees'] * edge ** DELTA_TREES_EDGE_EXP))
    # a field is 32 cells tall: keep the row's aggradation ratio, not its level count
    if 'mCHdepth' in kw:
        key, n = levels_for_column(row, FIELD_NZ * (g0['z_len'] / g0['nz']))
        kw[key] = n
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
    assert f.shape == (ex, ey, FIELD_NZ), f'{env}: got {f.shape}, want {(ex, ey, FIELD_NZ)}'
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

    # Resume: fields already checkpointed under OUT are kept and their seeds
    # skipped, so a killed run only loses the fields that were in flight.
    got = {}
    for env in want:
        f = out / env.replace(':', '_') / 'fields.npz'
        if f.exists():
            z = np.load(f, allow_pickle=True)
            for i, v in zip(z['ids'], z['volumes']):
                v = np.asarray(v, np.int8)
                got.setdefault(env, []).append((int(str(i).rsplit('|', 1)[1]), v, float(v.mean())))
            print(f'resuming {env}: {len(got[env])} of {want[env]} fields already on disk', flush=True)
    have = {(env, s) for env, items in got.items() for s, _, _ in items}
    tasks = [t for t in tasks if (t[0], t[2]) not in have]
    for env in [e for e, items in got.items() if len(items) >= want[e]]:
        got.pop(env)
    t0 = time.time()
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


def flow_profile(vols):
    """Sand fraction in each fifth along x (the flow axis for channels), and
    the last/first ratio.

    Net-to-gross matching is NOT evidence that a field is right. The engine
    keeps adding channels until each level's sand target is met, so channels
    that die early against a side wall pile their sand near the entry and the
    total still comes out correct. PV_SHOESTRING at 512x64: fifths 0.306
    0.270 0.145 0.079 0.044 with the right overall NTG; the same row and seed
    at 512x256: 0.132 0.152 0.173 0.167 0.164, and zero sand on the walls.
    """
    v = np.asarray(vols, np.float32); nx = v.shape[1]
    fifths = [float(v[:, j * nx // 5:(j + 1) * nx // 5].mean()) for j in range(5)]
    ratio = fifths[4] / fifths[0] if fifths[0] > 0 else float('nan')
    return fifths, ratio


def _write_env(out, env, items, partial=False):
    items.sort(key=lambda t: t[0])
    slug = env.replace(':', '_')
    d = out / slug; d.mkdir(parents=True, exist_ok=True)
    vols = np.stack([f for _, f, _ in items])
    fifths, ratio = flow_profile(vols)
    if env.startswith('channel:') and not partial and not (0.5 <= ratio <= 2.0):
        print(f'WARNING {env}: sand along flow is not uniform, fifths '
              + ' '.join(f'{x:.3f}' for x in fifths) + f', last/first {ratio:.2f}: '
              f'channels terminate before the far end (corridor too narrow for '
              f'this sinuosity?)', flush=True)
    ids = np.array([f'field|{slug}|{s}' for s, _, _ in items], dtype=object)
    np.savez_compressed(d / 'fields.npz', ids=ids, volumes=vols)
    grid, ext, (dx, dy) = grid_for(env)
    (d / 'manifest.json').write_text(json.dumps({
        'environment': env, 'n': len(items),
        'extent': [ext[0], ext[1], FIELD_NZ], 'cell_m': [dx, dy, grid['z_len'] / grid['nz']],
        'elongated_along_flow': env.startswith('channel:'),
        'grid': grid, 'seed_base': SEED_BASE,
        'mean_ntg': float(np.mean([n for _, _, n in items])),
        'flow_fifths': fifths, 'flow_last_over_first': ratio,
        'complete': not partial,
    }, indent=2))
    print(f'{"saved" if partial else "WROTE"} {env:<24} {len(items)} fields {vols.shape} '
          f'mean ntg {np.mean([n for _,_,n in items]):.4f}', flush=True)


if __name__ == '__main__':
    main()
