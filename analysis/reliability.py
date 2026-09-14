"""Addendum F.3: reliability diagrams (accuracy-plot analog) + ECE.

Well-conditional version: model p-hat (K=128, Addendum C ensembles, both
tiers) vs engine-conditional sand frequency (published reference assets),
off-well voxels, pooled per environment. Unconditional version: Addendum B
ensembles (model K=128 vs engine N=256), labeled as such. Bands: engine
self-ECE from split halves (rng [20260814, condition_index]).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resbench import LAYER_TYPES, env_slug     # noqa: E402
from resbench.stats import verdict             # noqa: E402

SPLIT_SEED = 20260814
BINS = np.linspace(0, 1, 11)


def ece_curve(p_model, freq_ref, w_mask=None):
    """Returns (bin mean p, bin mean freq, bin weight, ECE)."""
    pm = p_model.ravel() if w_mask is None else p_model[w_mask]
    fr = freq_ref.ravel() if w_mask is None else freq_ref[w_mask]
    idx = np.clip(np.digitize(pm, BINS) - 1, 0, 9)
    mp, mf, w = np.full(10, np.nan), np.full(10, np.nan), np.zeros(10)
    for b in range(10):
        m = idx == b
        if m.any():
            mp[b], mf[b], w[b] = pm[m].mean(), fr[m].mean(), m.sum()
    w = w / w.sum()
    ece = float(np.nansum(w * np.abs(mp - mf)))
    return mp, mf, w, ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-dir', required=True)
    ap.add_argument('--wc-asset-dir', required=True,
                    help='well-conditional reference assets')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    E = Path(args.eval_dir)

    rows, curves = [], {}
    ci = 0
    for lt in LAYER_TYPES:
        slug = env_slug(lt)
        env = {'environment': lt}
        # -- well-conditional (pool both tiers) --------------------------
        pm_all, fr_all, bands = [], [], []
        for tag in ('modal', 'mixed'):
            md = np.load(E / 'entropy_model_modal' / f'{slug}_{tag}.npz',
                         allow_pickle=True)
            ref = np.load(Path(args.wc_asset_dir) / f'{slug}_{tag}.npz',
                          allow_pickle=True)
            off = md['mask'].astype(bool) == 0
            vr = ref['volumes']
            pm_all.append(md['volumes'].mean(0)[off])
            fr_all.append(vr.mean(0)[off])
            perm = np.random.default_rng([SPLIT_SEED, ci]).permutation(len(vr))
            *_, b = ece_curve(vr[perm[:len(vr) // 2]].mean(0)[off],
                              vr[perm[len(vr) // 2:]].mean(0)[off])
            bands.append(b)
            ci += 1
        mp, mf, w, ece = ece_curve(np.concatenate(pm_all),
                                   np.concatenate(fr_all))
        env['wc'] = {'ece': ece, 'band': float(np.mean(bands))}
        curves.setdefault(lt, {})['wc'] = (mp.tolist(), mf.tolist())

        # -- unconditional (Addendum B, 8 conditions) --------------------
        pm_all, fr_all, bands = [], [], []
        for f in sorted((E / 'entropy_model_uncond' / slug).glob('cond_*.npz')):
            ridx = int(re.match(r'cond_r(\d+)_', f.name).group(1))
            rf = list((E / 'entropy_ref_uncond' / slug).glob(
                f'cond_r{ridx:04d}_*.npz'))[0]
            vm = np.load(f, allow_pickle=True)['volumes']
            vr = np.load(rf, allow_pickle=True)['volumes']
            pm_all.append(vm.mean(0).ravel())
            fr_all.append(vr.mean(0).ravel())
            perm = np.random.default_rng([SPLIT_SEED, ci]).permutation(len(vr))
            *_, b = ece_curve(vr[perm[:len(vr) // 2]].mean(0),
                              vr[perm[len(vr) // 2:]].mean(0))
            bands.append(b)
            ci += 1
        mp, mf, w, ece = ece_curve(np.concatenate(pm_all),
                                   np.concatenate(fr_all))
        env['uc'] = {'ece': ece, 'band': float(np.mean(bands))}
        curves[lt]['uc'] = (mp.tolist(), mf.tolist())
        rows.append(env)
        print(f"{lt}: ECE wc={env['wc']['ece']:.4f} (band {env['wc']['band']:.4f}) "
              f"uc={env['uc']['ece']:.4f} (band {env['uc']['band']:.4f})",
              flush=True)

    out = Path(args.out_dir)
    json.dump({'rows': rows, 'curves': curves},
              open(out / 'reliability.json', 'w'), indent=2)
    md = ['## Reliability / expected calibration error (F.3)', '',
          '| environment | ECE well-cond. (band) | verdict '
          '| ECE uncond. (band) | verdict |', '|' + '---|' * 5]
    for r in rows:
        md.append(f"| {r['environment']} "
                  f"| {r['wc']['ece']:.4f} ({r['wc']['band']:.4f}) "
                  f"| {verdict(r['wc']['ece'], r['wc']['band'])} "
                  f"| {r['uc']['ece']:.4f} ({r['uc']['band']:.4f}) "
                  f"| {verdict(r['uc']['ece'], r['uc']['band'])} |")
    (out / 'reliability.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from resbench.figures import AXIS_COLORS, DIAG_GRAY, SHORT
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), sharex=True, sharey=True)
    for i, lt in enumerate(LAYER_TYPES):
        ax = axes.flat[i]
        ax.plot([0, 1], [0, 1], ls='--', lw=0.9, c=DIAG_GRAY)
        for key, color, ls, lbl in (('uc', AXIS_COLORS[0], '-', 'unconditional'),
                                    ('wc', AXIS_COLORS[1], (0, (4, 2)),
                                     'well-conditional')):
            mp, mf = (np.array(v, float) for v in curves[lt][key])
            ax.plot(mp, mf, 'o-', ms=3, lw=1.3, c=color, ls=ls, label=lbl)
        ax.set_title(SHORT[lt], fontsize=9)
        if i >= 4:
            ax.set_xlabel('model p(sand)')
        if i % 4 == 0:
            ax.set_ylabel('engine frequency')
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.suptitle('Reliability diagrams (voxelwise, decile bins)', y=1.0)
    fig.tight_layout()
    fig.savefig(out / 'reliability.pdf', bbox_inches='tight')
    fig.savefig(out / 'reliability.png', bbox_inches='tight', dpi=250)
    print('done ->', out / 'reliability.md')


if __name__ == '__main__':
    main()
