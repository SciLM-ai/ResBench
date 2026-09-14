"""Assembly-consistency scoring (EVAL.md Addendum E.4/E.5).

Cuts 5x5 non-overlapping (64,64,32) tiles from each assembly (origin (52,52),
stride 64), then scores B vs C, A vs C, B vs A with the E.4 metrics against
the split-half band of the engine ensemble C. Also produces the seam
diagnostic figure (E.5).

Usage:
  python analysis/assembly_stats.py --native-dir X --assembly-dir Y \
      --engine-dir Z --out-dir W
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.stats import wasserstein_distance

from resbench import metrics

TILE_N = 5
TILE = 64
# E.2 fixed the tile grid at origin (52, 52) for the 424-cell overlap-24
# assembly, where 52 = (424 - 5*64) // 2 is exactly centred. Deriving the
# origin instead keeps that layout bit-identical (a regression test below
# asserts it) and keeps the grid centred for any other extent. Before this,
# a 532-cell overlap-12 assembly was sampled at [52, 372) — a 52 / 160 cell
# margin split, i.e. a corner, not the centre — and a 1572-cell field had
# 4% of its area scored, all of it in one corner.
TILE_ORIGIN_424 = 52
DEFAULT_STRIDE_PERIOD = 40  # block stride of the overlap-24 tiling


def tile_origin(extent, tile_n=TILE_N, tile=TILE):
    """Centred origin for a `tile_n` x `tile_n` grid of `tile`-cubes."""
    return max((extent - tile_n * tile) // 2, 0)


assert tile_origin(424) == TILE_ORIGIN_424, 'E.2 layout must be unchanged'


def load_npz_dir(d, slug):
    f = sorted(Path(d, slug).glob('volumes_*.npz'))[0]
    return np.load(f, allow_pickle=True)['volumes']


def n_disjoint(extent, tile=TILE):
    """How many disjoint `tile`-cubes fit along one axis of `extent` cells."""
    return max(extent // tile, 1)


def cut_tiles(assembly_dir, origin=None, full_coverage=False):
    """Cut the scoring tiles out of each assembled field.

    Defaults reproduce the frozen E.2 layout exactly: a centred TILE_N x TILE_N
    subgrid. `origin` pins the subgrid corner instead of centring it, which is
    what the placement-sensitivity study varies. `full_coverage` instead takes
    every disjoint tile that fits, which removes the origin as a free parameter
    altogether (see PROPOSED_EXTENSIONS.md section 6).
    """
    tiles, profiles_x, profiles_y = [], [], []
    for f in sorted(Path(assembly_dir).glob('assembly_*.npz')):
        b = np.load(f)['binary']
        profiles_x.append(b.mean(axis=(1, 2)))
        profiles_y.append(b.mean(axis=(0, 2)))
        if full_coverage:
            nx, ny = n_disjoint(b.shape[0]), n_disjoint(b.shape[1])
            ox = (b.shape[0] - nx * TILE) // 2
            oy = (b.shape[1] - ny * TILE) // 2
        else:
            nx = ny = TILE_N
            ox = tile_origin(b.shape[0]) if origin is None else origin
            oy = tile_origin(b.shape[1]) if origin is None else origin
        for i in range(nx):
            for j in range(ny):
                x0 = ox + i * TILE
                y0 = oy + j * TILE
                tiles.append(b[x0:x0 + TILE, y0:y0 + TILE, :])
    return np.stack(tiles), profiles_x, profiles_y


def body_extents(vols):
    """Pooled per-body max lateral (x-y) bounding-box extent."""
    out = []
    for v in vols:
        labels, n = metrics.label_geobodies(v)
        for sl in ndimage.find_objects(labels):
            if sl is not None:
                out.append(max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start))
    return np.asarray(out, dtype=np.float64)


def ens_summary(vols):
    pooled_sizes, _ = metrics.ensemble_geobodies(vols)
    return {
        'ntg': metrics.ensemble_ntg(vols),
        'gamma': metrics.ensemble_variogram(vols),
        'tau': metrics.ensemble_connectivity(vols),
        'sizes': pooled_sizes,
        'extents': body_extents(vols),
    }


def compare(a, b, sill):
    tau_diffs = np.concatenate([np.abs(a['tau'][ax] - b['tau'][ax]) for ax in (0, 1, 2)])
    return {
        'abs_dntg': abs(float(a['ntg'].mean()) - float(b['ntg'].mean())),
        'ntg_dist_w1': float(wasserstein_distance(a['ntg'], b['ntg'])),
        'variogram_mae': metrics.variogram_mae(a['gamma'], b['gamma'], sill),
        'connectivity_mae': float(tau_diffs.mean()),
        'geobody_w1': float(wasserstein_distance(np.log10(a['sizes']), np.log10(b['sizes']))),
        'extent_w1': float(wasserstein_distance(np.log10(a['extents']), np.log10(b['extents']))),
    }


def stride_amplitude(profile, period):
    x = np.arange(len(profile))
    p = profile - profile.mean()
    c = np.abs(np.sum(p * np.exp(-2j * np.pi * x / period))) * 2 / len(p)
    return float(c)


def verdict(v, band):
    return 'inside' if v <= band else 'near' if v <= 2 * band else 'outside'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--native-dir', required=True)
    ap.add_argument('--assembly-dir', required=True)
    ap.add_argument('--engine-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--env-slug', default='channel_PV_SHOESTRING')
    ap.add_argument('--split-seed', type=int, default=20260812)
    ap.add_argument('--stride-period', type=int, default=DEFAULT_STRIDE_PERIOD,
                    help='block stride of the tiling under test, for the E.5 '
                         'seam diagnostic. The frozen default 40 is the '
                         'overlap-24 stride; an overlap-12 run has stride 52 '
                         'and a tiling-free run has none, in which case the '
                         'amplitude at 40 measures nothing.')
    ap.add_argument('--tile-origin', type=int, default=None,
                    help='pin the E.2 subgrid corner instead of centring it. '
                         'The score depends on this at the 0.003-0.012 level '
                         'on geobody W1, comparable to between-model gaps, so '
                         'it is exposed for sensitivity studies only.')
    ap.add_argument('--full-coverage', action='store_true',
                    help='score every disjoint 64-cube that fits instead of a '
                         'centred 5x5 subgrid, removing the origin as a free '
                         'parameter. Not the frozen E.2 layout: report it '
                         'alongside, not instead of, the default.')
    args = ap.parse_args()
    stride_period = args.stride_period
    SPLIT_SEED = args.split_seed
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    A = load_npz_dir(args.native_dir, args.env_slug)
    B, profs_x, profs_y = cut_tiles(args.assembly_dir, origin=args.tile_origin,
                                    full_coverage=args.full_coverage)
    C = load_npz_dir(args.engine_dir, args.env_slug)
    print(f'A(native)={A.shape} B(tiles)={B.shape} C(engine)={C.shape}')

    sA, sB, sC = ens_summary(A), ens_summary(B), ens_summary(C)
    p = float(sC['ntg'].mean())
    sill = p * (1 - p)

    # Split-half band of C, same estimators.
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(C))
    h1, h2 = C[perm[:len(C) // 2]], C[perm[len(C) // 2:]]
    band = compare(ens_summary(h1), ens_summary(h2), sill)

    rows = {}
    for name, x, y in (('B_vs_C', sB, sC), ('A_vs_C', sA, sC), ('B_vs_A', sB, sA)):
        c = compare(x, y, sill)
        rows[name] = {k: {'value': v, 'band': band[k], 'verdict': verdict(v, band[k])}
                      for k, v in c.items()}

    # Seam diagnostic: stride-period amplitude on assembly profiles vs
    # profiles of engine volumes concatenated at the tile size (true seams).
    asm_amp = [stride_amplitude(pr, stride_period) for pr in profs_x + profs_y]
    rng = np.random.default_rng(SPLIT_SEED + 1)
    eng_amp = []
    for _ in range(20):
        pick = rng.choice(len(C), size=7, replace=False)
        concat = np.concatenate([C[k] for k in pick], axis=0)
        eng_amp.append(stride_amplitude(concat.mean(axis=(1, 2)), TILE))
    seam = {'stride_period': stride_period,
            f'assembly_amp_at_stride{stride_period}_mean': float(np.mean(asm_amp)),
            f'assembly_amp_at_stride{stride_period}_max': float(np.max(asm_amp)),
            'engine_concat_amp_at_period64_mean': float(np.mean(eng_amp)),
            'engine_concat_amp_at_period64_max': float(np.max(eng_amp))}

    result = {'ensembles': {'A_native': len(A), 'B_tiles': len(B), 'C_engine': len(C)},
              'mean_ntg': {'A': float(sA['ntg'].mean()), 'B': float(sB['ntg'].mean()),
                           'C': float(sC['ntg'].mean())},
              'band_split_seed': SPLIT_SEED, 'comparisons': rows,
              'seam_diagnostic': seam}
    (out / 'assembly_stats.json').write_text(json.dumps(result, indent=2))

    cols = ['abs_dntg', 'ntg_dist_w1', 'variogram_mae', 'connectivity_mae',
            'geobody_w1', 'extent_w1']
    lines = ['# Assembly-consistency results (EVAL.md Addendum E)', '',
             '| comparison | ' + ' | '.join(cols) + ' |',
             '|---' * (len(cols) + 1) + '|',
             '| band (C split-half) | ' + ' | '.join(f"{band[c]:.4f}" for c in cols) + ' |']
    for name in ('B_vs_C', 'A_vs_C', 'B_vs_A'):
        cells = [f"{rows[name][c]['value']:.4f} ({rows[name][c]['verdict']})" for c in cols]
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lines += ['', f"Seam diagnostic: {json.dumps(seam)}"]
    (out / 'assembly_stats.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
    for pr in profs_x[:3]:
        axes[0].plot(pr, lw=0.9)
    axes[0].set_title('assembly x sand-fraction profiles', fontsize=10)
    axes[0].set_xlabel('x (cells)')
    axes[0].set_ylabel('sand fraction')
    periods = np.arange(8, 130)
    spec = np.mean([[stride_amplitude(pr, per) for per in periods]
                    for pr in profs_x + profs_y], axis=0)
    axes[1].plot(periods, spec, c='#2a78d6', lw=1.4)
    axes[1].axvline(stride_period, c='#eb6834', lw=0.9, ls=':')
    axes[1].axvline(TILE, c='#9a9a9a', lw=0.9, ls=':')
    axes[1].set_title(f'mean profile spectrum (dotted: {stride_period}, {TILE})', fontsize=10)
    axes[1].set_xlabel('period (cells)')
    bins = np.linspace(0, max(sA['ntg'].max(), sB['ntg'].max(), sC['ntg'].max()) * 1.05, 30)
    for arr, lab, c, ls in ((sC['ntg'], 'engine', '#222222', '-'),
                            (sA['ntg'], 'native', '#1baf7a', (0, (1, 1.5))),
                            (sB['ntg'], 'assembly tiles', '#eb6834', (0, (4, 2)))):
        axes[2].hist(arr, bins=bins, density=True, histtype='step', lw=1.6,
                     color=c, ls=ls, label=lab)
    axes[2].set_title('per-volume NTG', fontsize=10)
    axes[2].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'assembly_stats.{ext}', dpi=200)
    print('wrote figure')


if __name__ == '__main__':
    main()
