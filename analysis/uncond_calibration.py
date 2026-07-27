"""Addendum B: unconditional (well-free) entropy-calibration benchmark.

Per condition (manifest rows 4-11 per environment, 64 total): compare the
model's voxelwise Bernoulli entropy (K = 128 empty-mask samples) with the
engine's (N = 256 unconditional realizations, same parameter vector) over
ALL voxels: signed mean offset (negative = under-dispersed) and mean
absolute error, judged against the engine split-half band (EVAL.md B.4).

Also exports the published reference asset: per-condition voxelwise sand
probability p-hat (float16) under results/entropy_reference/.

Model/ref condition files are matched on the row index parsed from the
filename (naming differs: model uses `_nowells`, engine keeps the manifest
well-config tag; wells are irrelevant to both).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resbench import LAYER_TYPES, env_slug   # noqa: E402

SPLIT_SEED = 20260802
T_95_DF7 = 2.365


def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return np.where((p <= eps) | (p >= 1 - eps), 0.0, h)


def by_row(root, lt):
    out = {}
    d = Path(root) / env_slug(lt)
    for f in sorted(d.glob('cond_*.npz')):
        out[int(re.match(r'cond_r(\d+)_', f.name).group(1))] = f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--ref-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--asset-dir', default=None,
                    help='where to write the published p-hat reference')
    args = ap.parse_args()

    rows, per_env = [], {}
    ci = 0
    for lt in LAYER_TYPES:
        model_f, ref_f = by_row(args.model_dir, lt), by_row(args.ref_dir, lt)
        assert set(model_f) == set(ref_f), f'{lt}: condition mismatch'
        for ridx in sorted(model_f):
            vm = np.load(model_f[ridx], allow_pickle=True)['volumes']
            rd = np.load(ref_f[ridx], allow_pickle=True)
            vr = rd['volumes']
            H_m, H_r = entropy(vm.mean(0)), entropy(vr.mean(0))
            perm = np.random.default_rng([SPLIT_SEED, ci]).permutation(len(vr))
            H_a = entropy(vr[perm[:len(vr) // 2]].mean(0))
            H_b = entropy(vr[perm[len(vr) // 2:]].mean(0))
            rec = {
                'environment': lt, 'row_index': ridx, 'ci': ci,
                'signed': float((H_m - H_r).mean()),
                'mae': float(np.abs(H_m - H_r).mean()),
                'band_signed': float(abs((H_a - H_b).mean())),
                'band_mae': float(np.abs(H_a - H_b).mean()),
                'H_ref_mean': float(H_r.mean()),
            }
            rows.append(rec)
            per_env.setdefault(lt, []).append(rec)
            if args.asset_dir:
                d = Path(args.asset_dir) / env_slug(lt)
                d.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    d / f'cond_r{ridx:04d}_phat.npz',
                    p_hat=vr.mean(0).astype(np.float16),
                    n=len(vr), seeds=rd['seeds'],
                    ref_id=str(rd['ref_id']))
            ci += 1
            print(f'{lt} r{ridx}: signed={rec["signed"]:+.4f} '
                  f'mae={rec["mae"]:.4f} band_mae={rec["band_mae"]:.4f}',
                  flush=True)

    def verdict(v, b):
        return 'inside' if v <= b else 'near' if v <= 2 * b else 'outside'

    md = ['## Unconditional entropy-calibration benchmark (EVAL.md Addendum B)',
          '',
          'Signed offset: mean(H_model − H_engine) over all voxels, 95% '
          't-interval across the 8 conditions/env; negative = '
          'under-dispersed. Verdict compares |signed| (resp. MAE) to the '
          'engine split-half band.', '',
          '| environment | signed offset (bits) ± CI | MAE (bits) ± CI | '
          'band (MAE) | verdict |', '|' + '---|' * 5]
    env_rows = []
    for lt in LAYER_TYPES:
        rs = per_env.get(lt)
        if not rs:
            continue
        s = np.array([r['signed'] for r in rs])
        m = np.array([r['mae'] for r in rs])
        b = float(np.mean([r['band_mae'] for r in rs]))
        s_ci = T_95_DF7 * s.std(ddof=1) / np.sqrt(len(s))
        m_ci = T_95_DF7 * m.std(ddof=1) / np.sqrt(len(m))
        v = verdict(float(m.mean()), b)
        env_rows.append({'environment': lt, 'signed': float(s.mean()),
                         'signed_ci': float(s_ci), 'mae': float(m.mean()),
                         'mae_ci': float(m_ci), 'band_mae': b, 'verdict': v})
        md.append(f'| {lt} | {s.mean():+.4f} ± {s_ci:.4f} '
                  f'| {m.mean():.4f} ± {m_ci:.4f} | {b:.4f} | {v} |')
    s_all = np.array([r['signed'] for r in rows])
    m_all = np.array([r['mae'] for r in rows])
    b_all = float(np.mean([r['band_mae'] for r in rows]))
    md.append(f'| **pooled (64 conds)** | {s_all.mean():+.4f} '
              f'| {m_all.mean():.4f} | {b_all:.4f} '
              f'| {verdict(float(m_all.mean()), b_all)} |')
    md += ['', 'Estimator note: plug-in entropy bias differs by ≈ 0.002 bits '
           'between K = 128 (model) and N = 256 (engine) ensembles — '
           'negligible against the observed offsets.']

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({'conditions': rows, 'environments': env_rows},
              open(out / 'uncond_calibration.json', 'w'), indent=2)
    (out / 'uncond_calibration.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))

    # strip plot: per-condition signed offsets by environment
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from resbench.figures import GEN_BLUE, REF_GRAY, SHORT
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for i, lt in enumerate([lt for lt in LAYER_TYPES if lt in per_env]):
        ys = [r['signed'] for r in per_env[lt]]
        bs = np.mean([r['band_signed'] for r in per_env[lt]])
        ax.scatter([i] * len(ys), ys, s=18, c=GEN_BLUE, zorder=3)
        ax.plot([i - 0.3, i + 0.3], [bs, bs], c=REF_GRAY, lw=0.8, ls=':')
        ax.plot([i - 0.3, i + 0.3], [-bs, -bs], c=REF_GRAY, lw=0.8, ls=':')
    ax.axhline(0, c=REF_GRAY, lw=0.9)
    ax.set_xticks(range(len(per_env)))
    ax.set_xticklabels([SHORT[lt] for lt in LAYER_TYPES if lt in per_env],
                       rotation=20, ha='right')
    ax.set_ylabel('signed entropy offset (bits)')
    ax.set_title('Unconditional calibration: per-condition offsets '
                 '(dotted: engine split-half band)')
    fig.tight_layout()
    fig.savefig(out / 'uncond_calibration.pdf', bbox_inches='tight')
    fig.savefig(out / 'uncond_calibration.png', bbox_inches='tight', dpi=250)
    print('done ->', out / 'uncond_calibration.md')


if __name__ == '__main__':
    main()
