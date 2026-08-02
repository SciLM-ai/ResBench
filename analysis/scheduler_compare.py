"""Score several block schedulers against the engine reference (Addendum G).

Takes any number of `label=assembly_dir` pairs plus optional
`label=native_dir` entries, cuts tiles with the same generalised rule as
analysis/overlap_sweep.py (a centred 5x5 grid of 64-cell tiles, origin
(extent - 320)//2, which reproduces the frozen 52 at the deployed 424),
and reports the E.4 metric set against the published engine ensemble with
its split-half band.

Everything but the scheduler is held fixed upstream, so differences
between rows are attributable to the scheduler alone.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import wasserstein_distance

from resbench import metrics

TILE = 64
TILE_N = 5
COLS = ['abs_dntg', 'ntg_dist_w1', 'variogram_mae', 'connectivity_mae',
        'geobody_w1', 'extent_w1']


def load_npz_dir(d, slug='lobe'):
    p = Path(d, slug)
    f = sorted(p.glob('volumes_*.npz')) or sorted(Path(d).glob('volumes_*.npz'))
    return np.load(f[0], allow_pickle=True)['volumes']


def cut_tiles(assembly_dir):
    tiles = []
    files = sorted(Path(assembly_dir).glob('assembly_*.npz'))
    if not files:
        return None, None
    extent = None
    for f in files:
        b = np.load(f)['binary']
        extent = b.shape[0]
        origin = (extent - TILE_N * TILE) // 2
        if origin < 0:
            raise SystemExit(f'{assembly_dir}: extent {extent} too small')
        for i in range(TILE_N):
            for j in range(TILE_N):
                x0, y0 = origin + i * TILE, origin + j * TILE
                tiles.append(b[x0:x0 + TILE, y0:y0 + TILE, :])
    return np.stack(tiles), extent


def body_extents(vols):
    out = []
    for v in vols:
        labels, _ = metrics.label_geobodies(v)
        for sl in ndimage.find_objects(labels):
            if sl is not None:
                out.append(max(sl[0].stop - sl[0].start,
                               sl[1].stop - sl[1].start))
    return np.asarray(out, dtype=np.float64)


def compare(a, b):
    ntg_a = np.array([v.mean() for v in a])
    ntg_b = np.array([v.mean() for v in b])
    ga, gb = metrics.ensemble_variogram(a), metrics.ensemble_variogram(b)
    ta, tb = metrics.ensemble_connectivity(a), metrics.ensemble_connectivity(b)
    sill = float(ntg_b.mean() * (1 - ntg_b.mean()))
    sza, _ = metrics.ensemble_geobodies(a)
    szb, _ = metrics.ensemble_geobodies(b)
    ea, eb = body_extents(a), body_extents(b)
    return {
        'abs_dntg': float(abs(ntg_a.mean() - ntg_b.mean())),
        'ntg_dist_w1': float(wasserstein_distance(ntg_a, ntg_b)),
        'variogram_mae': float(metrics.variogram_mae(ga, gb, sill)),
        'connectivity_mae': float(metrics.connectivity_mae(ta, tb)),
        'geobody_w1': float(metrics.geobody_w1(sza, szb)),
        'extent_w1': float(wasserstein_distance(np.log10(ea), np.log10(eb))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('entries', nargs='+',
                    help='label=dir (assembly dirs, or native dirs holding '
                         'lobe/volumes_*.npz)')
    ap.add_argument('--engine-dir', default='results/assembly_reference')
    ap.add_argument('--env-slug', default='lobe')
    ap.add_argument('--split-seed', type=int, default=20260815)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    C = load_npz_dir(args.engine_dir, args.env_slug)
    perm = np.random.default_rng(args.split_seed).permutation(len(C))
    band = compare(C[perm[:len(C) // 2]], C[perm[len(C) // 2:]])

    print('| ensemble | n | extent | ' + ' | '.join(COLS) + ' |')
    print('|' + '---|' * (len(COLS) + 3))
    print('| band | — | — | ' + ' | '.join(f'{band[c]:.4f}' for c in COLS) + ' |')

    rows = {'band': band}
    for e in args.entries:
        label, d = e.split('=', 1)
        B, extent = cut_tiles(d)
        if B is None:                      # native ensemble
            try:
                B = load_npz_dir(d, args.env_slug)
                extent = 'native'
            except Exception:
                print(f'| {label} | MISSING | | ' +
                      ' | '.join('—' for _ in COLS) + ' |')
                continue
        r = compare(B, C)
        rows[label] = {'n': len(B), 'extent': str(extent), **r}
        print(f'| {label} | {len(B)} | {extent} | '
              + ' | '.join(f'{r[c]:.4f}' for c in COLS) + ' |')

    if args.out:
        Path(args.out).write_text(json.dumps(
            {'split_seed': args.split_seed, 'rows': rows}, indent=2))
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
