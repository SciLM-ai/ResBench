"""Bird's-eye facies maps of the field-scale reference.

Three depth slices per field -- near the base, the middle, and near the top --
so the vertical stacking of channel levels is visible, not just the plan view
of whichever level happens to dominate.

  python tools/plot_fields.py --ref DIR --out DIR [--index 0]
"""
import argparse, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

CMAP = ListedColormap(['#e8e2d5', '#b8641e'])      # shale, sand


def panels(nz):
    lo, mid, hi = max(1, int(0.08 * nz)), nz // 2, min(nz - 2, int(0.92 * nz))
    return [(lo, 'near base'), (mid, 'middle'), (hi, 'near top')]


def plot_one(vol, title, path_stem, cell_m):
    nx, ny, nz = vol.shape
    aspect = ny / nx
    fig, axes = plt.subplots(3, 1, figsize=(11, max(3.2, 11 * aspect * 3 + 1.4)),
                             constrained_layout=True)
    for ax, (k, label) in zip(np.atleast_1d(axes), panels(nz)):
        ax.imshow(vol[:, :, k].T, origin='lower', cmap=CMAP, vmin=0, vmax=1,
                  interpolation='nearest', aspect='equal')
        ax.set_title(f'{label}   z = {k} of {nz}    '
                     f'sand {vol[:, :, k].mean() * 100:.1f}%',
                     fontsize=9, loc='left')
        ax.set_xticks([0, nx // 2, nx - 1])
        ax.set_xticklabels([f'{v * cell_m[0] / 1000:.1f}' for v in (0, nx // 2, nx - 1)])
        ax.set_yticks([0, ny - 1])
        ax.set_yticklabels([f'{v * cell_m[1] / 1000:.1f}' for v in (0, ny - 1)])
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[-1].set_xlabel('km', fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.savefig(f'{path_stem}.pdf')          # vector, for papers
    fig.savefig(f'{path_stem}.png', dpi=130)  # raster, only so it can be viewed inline
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--index', type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for d in sorted(Path(a.ref).iterdir()):
        f = d / 'fields.npz'
        if not f.exists():
            continue
        man = json.loads((d / 'manifest.json').read_text())
        vols = np.load(f, allow_pickle=True)['volumes']
        v = vols[min(a.index, len(vols) - 1)]
        cell = man.get('cell_m', [100.0, 100.0, 1.0])
        plot_one(v, f"{man['environment']}   {v.shape[0]}x{v.shape[1]}x{v.shape[2]} cells"
                    f"   {cell[0]:.0f} m cells   sand {v.mean() * 100:.1f}%",
                 str(out / d.name), cell)
        print(f'{d.name:<26} {v.shape}  sand {v.mean()*100:.1f}%', flush=True)


if __name__ == '__main__':
    main()
