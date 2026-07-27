"""Side-by-side voxelwise entropy maps: ResFlow vs the ResMill conditional
reference, for the informative-well (mixed-tier) conditions with n >= 50.

Rows = environments; columns = [ResFlow XY, ResMill XY, ResFlow XZ,
ResMill XZ]; XY slice at z = 18, XZ slice through the well (y = 32); the
well column is outlined in white on the XZ panels.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resbench import env_slug   # noqa: E402
from resbench.figures import SHORT   # noqa: E402

FLOOR = 50
Z_SLICE = 18


def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return np.where((p <= eps) | (p >= 1 - eps), 0.0, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modal-json', required=True)
    ap.add_argument('--asset-dir', required=True)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tag', default='mixed')
    args = ap.parse_args()

    conds = json.loads(Path(args.modal_json).read_text())
    keep = []
    for rec in conds:
        slug = env_slug(rec['environment'])
        ref = np.load(Path(args.asset_dir) / f'{slug}_{args.tag}.npz',
                      allow_pickle=True)
        if len(ref['volumes']) >= FLOOR:
            mod = np.load(Path(args.model_dir) / f'{slug}_{args.tag}.npz',
                          allow_pickle=True)
            keep.append((rec, ref, mod))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5})

    n = len(keep)
    fig, axes = plt.subplots(n, 4, figsize=(11.5, 2.6 * n))
    for r, (rec, ref, mod) in enumerate(keep):
        H_m = entropy(mod['volumes'].mean(0))
        H_r = entropy(ref['volumes'].astype(np.float32).mean(0))
        K, nr = len(mod['volumes']), len(ref['volumes'])
        panels = [(H_m[:, :, Z_SLICE].T, f'ResFlow (K={K}) — XY @ z={Z_SLICE}'),
                  (H_r[:, :, Z_SLICE].T, f'ResMill (n={nr}) — XY @ z={Z_SLICE}'),
                  (H_m[:, 32, :].T, 'ResFlow — XZ @ y=32'),
                  (H_r[:, 32, :].T, 'ResMill — XZ @ y=32')]
        for c, (img, title) in enumerate(panels):
            ax = axes[r, c] if n > 1 else axes[c]
            im = ax.imshow(img, origin='lower', vmin=0, vmax=1, cmap='magma',
                           aspect='auto')
            if c >= 2:
                ax.axvline(32, c='white', lw=0.7, alpha=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=7.5)
        lbl = axes[r, 0] if n > 1 else axes[0]
        wntg = float(np.asarray(ref['pattern']).mean())
        lbl.set_ylabel(f"{SHORT[rec['environment']]}\nwell NTG {wntg:.2f}",
                       fontsize=8.5)
    fig.colorbar(im, ax=axes, shrink=0.7, label='entropy (bits)')
    fig.suptitle('Conditional voxelwise entropy: ResFlow vs ResMill reference '
                 f'({args.tag}-tier wells, n ≥ {FLOOR})', y=0.995)
    out = Path(args.out_dir)
    fig.savefig(out / f'entropy_map_compare_{args.tag}.pdf', bbox_inches='tight')
    fig.savefig(out / f'entropy_map_compare_{args.tag}.png', bbox_inches='tight',
                dpi=250)
    print('saved', out / f'entropy_map_compare_{args.tag}.pdf')


if __name__ == '__main__':
    main()
