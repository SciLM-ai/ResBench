"""Score outpainting assemblies across generation-time overlaps (Addendum G).

The frozen E.4 scorer hardcodes the deployed tiling: a 424x424x32 assembly
and a 5x5 grid of 64-cell tiles from origin (52, 52). Sweeping the overlap
changes the assembly extent (stride = 64 - overlap, so overlap 8 gives 568
and overlap 32 gives 352), so the origin has to follow.

The rule is kept identical in substance -- a centred 5x5 grid of
non-overlapping 64-cell tiles -- and generalised as
origin = (extent - 5*64) // 2, which reproduces the frozen 52 exactly at
the deployed 424. Metrics and the split-half band are the E.4 estimators
verbatim, so the swept numbers sit on the same scale as the frozen table.

The E.5 seam diagnostic is NOT reported here: its period is the block
stride, which changes with overlap, so a single hardcoded period would be
meaningless across the sweep.
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


def load_engine(engine_dir, slug='lobe'):
    f = sorted(Path(engine_dir, slug).glob('volumes_*.npz'))[0]
    return np.load(f, allow_pickle=True)['volumes']


def cut_tiles(assembly_dir):
    tiles = []
    files = sorted(Path(assembly_dir).glob('assembly_*.npz'))
    if not files:
        raise SystemExit(f'no assemblies in {assembly_dir}')
    extent = None
    for f in files:
        b = np.load(f)['binary']
        extent = b.shape[0]
        origin = (extent - TILE_N * TILE) // 2
        if origin < 0:
            raise SystemExit(f'assembly {extent} too small for a 5x5 tiling')
        for i in range(TILE_N):
            for j in range(TILE_N):
                x0, y0 = origin + i * TILE, origin + j * TILE
                tiles.append(b[x0:x0 + TILE, y0:y0 + TILE, :])
    return np.stack(tiles), extent, origin


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
    """E.4 metric set between two ensembles of (64,64,32) volumes."""
    ntg_a = np.array([v.mean() for v in a])
    ntg_b = np.array([v.mean() for v in b])
    ga = metrics.ensemble_variogram(a)
    gb = metrics.ensemble_variogram(b)
    ta = metrics.ensemble_connectivity(a)
    tb = metrics.ensemble_connectivity(b)
    sill = float(ntg_b.mean() * (1 - ntg_b.mean()))
    sza, _ = metrics.ensemble_geobodies(a)   # (pooled sizes, largest fracs)
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
    ap.add_argument('--sweep-root', required=True,
                    help='dir holding ovsweep_<N>/outpaint/assembly_*.npz')
    ap.add_argument('--engine-dir',
                    default='results/assembly_reference')
    ap.add_argument('--env-slug', default='lobe')
    ap.add_argument('--split-seed', type=int, default=20260815)
    ap.add_argument('--overlaps', type=int, nargs='+',
                    default=[8, 12, 16, 24, 32])
    ap.add_argument('--deployed-dir', default=None,
                    help='assembly dir for the deployed overlap 24')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    C = load_engine(args.engine_dir, args.env_slug)
    perm = np.random.default_rng(args.split_seed).permutation(len(C))
    h1, h2 = C[perm[:len(C) // 2]], C[perm[len(C) // 2:]]
    band = compare(h1, h2)

    cols = ['abs_dntg', 'variogram_mae', 'connectivity_mae', 'geobody_w1',
            'extent_w1']
    rows = {}
    print('| overlap | extent | origin | ' + ' | '.join(cols) + ' |')
    print('|' + '---|' * (len(cols) + 3))
    print('| band | — | — | ' + ' | '.join(f'{band[c]:.4f}' for c in cols) + ' |')
    for ov in args.overlaps:
        d = (Path(args.deployed_dir) if (ov == 24 and args.deployed_dir)
             else Path(args.sweep_root) / f'ovsweep_{ov}' / 'outpaint')
        if not d.exists():
            print(f'| {ov} | missing | | ' + ' | '.join('—' for _ in cols) + ' |')
            continue
        B, extent, origin = cut_tiles(d)
        r = compare(B, C)
        rows[ov] = {'extent': extent, 'origin': origin, 'n_tiles': len(B), **r}
        print(f'| {ov} | {extent} | {origin} | '
              + ' | '.join(f'{r[c]:.4f}' for c in cols) + ' |')

    if args.out:
        Path(args.out).write_text(json.dumps(
            {'band': band, 'split_seed': args.split_seed, 'by_overlap': rows},
            indent=2))
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
