"""Entropy maps of the well ensembles: one page per environment, one row per
well: the column, the ensemble's per-cell entropy in the XZ and YZ sections
through the well and in an XY slice inside a sand body, and P(sand) in XZ.

    python tools/plot_well_entropy.py REF OUT.pdf [--wells-dir WELLS]

Reads REF/repeats/<slug>/well<i>.npz (or WELLS/<slug>_well<i>.npz)."""
import argparse, csv, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resbench.io import ENVIRONMENTS, SLUG  # noqa: E402


def entropy(p):
    q = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref'); ap.add_argument('out')
    ap.add_argument('--wells-dir', default=None)
    a = ap.parse_args()
    ref = Path(a.ref)
    rows = {}
    mp = ref / 'manifest.csv'
    if mp.exists():
        for r in csv.DictReader(open(mp)):
            if r['task'] == 'unconditional':
                rows[(r['environment'], int(r['row_index']))] = r
    with PdfPages(a.out) as pdf:
        for env in ENVIRONMENTS:
            slug = SLUG[env]
            files = [(Path(a.wells_dir) / f'{slug}_well{i}.npz') if a.wells_dir else ref / 'repeats' / slug / f'well{i}.npz' for i in range(1, 6)]
            files = [f for f in files if f.exists()]
            if not files:
                continue
            fig, axes = plt.subplots(len(files), 5, figsize=(26, 4.8 * len(files) + 1), squeeze=False, gridspec_kw={'width_ratios': [0.35, 1, 1, 1, 1]})
            for k, f in enumerate(files):
                z = np.load(f, allow_pickle=True); v = z['volumes'].astype(np.float32); pat = z['pattern'].astype(int)
                p = v.mean(0); H = entropy(p); x, y = int(z['well_xy'][0]), int(z['well_xy'][1])
                ri = int(z['cond_row_index']) if 'cond_row_index' in z.files else None
                src = rows.get((env, ri)); info = ''
                if src:
                    info = f"row {ri}: ntg {float(src['ntg']):.2f}, depth {float(src['depth_cells'] or 0):.1f} cells, width {float(src['width_cells'] or 0):.1f} cells"
                zs = [j for j in range(32) if pat[j] == 1]; zmid = zs[len(zs) // 2] if zs else 16
                ax = axes[k, 0]; ax.imshow(pat[None, :].T, cmap='Greys', vmin=0, vmax=1, aspect=0.25, origin='lower'); ax.set_xticks([]); ax.set_yticks([0, 8, 16, 24, 31]); ax.set_ylabel('z'); ax.set_title(f'well{k + 1}  column\nsand {pat.mean():.2f}', fontsize=11)
                ax = axes[k, 1]; im = ax.imshow(H[:, y, :].T, origin='lower', cmap='magma', vmin=0, vmax=1, aspect='auto'); ax.axvline(x, color='cyan', lw=0.8); ax.set_title(f'entropy  XZ section through the well (y={y})\n{info}', fontsize=10); ax.set_xlabel('x'); ax.set_ylabel('z')
                ax = axes[k, 2]; ax.imshow(H[x, :, :].T, origin='lower', cmap='magma', vmin=0, vmax=1, aspect='auto'); ax.axvline(y, color='cyan', lw=0.8); ax.set_title(f'entropy  YZ section (x={x})', fontsize=11); ax.set_xlabel('y')
                ax = axes[k, 3]; ax.imshow(H[:, :, zmid].T, origin='lower', cmap='magma', vmin=0, vmax=1, aspect='equal'); ax.plot(x, y, 'c+', ms=12, mew=1.5); ax.set_title(f'entropy  XY slice inside a sand body (z={zmid})', fontsize=11); ax.set_xlabel('x'); ax.set_ylabel('y')
                ax = axes[k, 4]; im2 = ax.imshow(p[:, y, :].T, origin='lower', cmap='viridis', vmin=0, vmax=1, aspect='auto'); ax.axvline(x, color='w', lw=0.8); ax.set_title(f'P(sand)  XZ (y={y})   {len(v)} members, mean entropy {H.mean():.2f}, ensemble ntg {p.mean():.2f}', fontsize=10); ax.set_xlabel('x')
                for ax in axes[k, 1:]: ax.tick_params(labelsize=8)
            fig.colorbar(im, ax=axes[:, 1:4].ravel().tolist(), fraction=0.01, pad=0.01).set_label('entropy (bits)  0 = every member agrees, 1 = coin flip')
            fig.colorbar(im2, ax=axes[:, 4].ravel().tolist(), fraction=0.02, pad=0.01).set_label('P(sand)')
            fig.suptitle(f'{env}   well ensembles: windows of distinct volumes, well at the cube centre (32, 32)', fontsize=15)
            pdf.savefig(fig); plt.close(fig)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
