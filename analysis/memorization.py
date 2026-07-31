"""Addendum F.4: memorization / novelty check.

(i) Voxel agreement between each training-conditioned generation and its
corresponding training volume, vs the engine same-parameter pair level
(200 pairs from the Addendum A unconditional pools, rng [20260815, ei]).
The exact-(params, seed) engine regeneration equals the stored volume
bit-for-bit, so the memorization ceiling is 1.0; flag any generation
exceeding the engine-pair p99.
(ii) Classical MDS of per-volume statistics vectors for training /
generated / reference volumes, per environment.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resbench import LAYER_TYPES, env_slug     # noqa: E402
from resbench import io, metrics               # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from acceptance_ext import run_lengths         # noqa: E402

PAIR_SEED = 20260815


def gamma_axis(v, lags, axis):
    out = []
    for h in lags:
        a = np.take(v, np.arange(h, v.shape[axis]), axis=axis)
        b = np.take(v, np.arange(0, v.shape[axis] - h), axis=axis)
        out.append(0.5 * (a != b).mean())
    return np.mean(out)


def features(v):
    sizes = metrics.geobody_sizes(v)
    tot = max(int(sizes.sum()), 1)
    zr = run_lengths(v[None], 2)
    return [float(v.mean()),
            float(sizes[0]) / tot if len(sizes) else 0.0,
            float(np.log10(1 + len(sizes))),
            float(gamma_axis(v, range(13, 17), 2)),
            float(gamma_axis(v, [16], 0)),
            float(np.median(zr)) if len(zr) else 0.0]


def classical_mds(X, dim=2):
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    n = len(D2)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:dim]
    return V[:, idx] * np.sqrt(np.maximum(w[idx], 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-check-dir', required=True)
    ap.add_argument('--ref-dir', required=True)
    ap.add_argument('--pools-dir', required=True,
                    help='Addendum A unconditional pools (entropy_ref)')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    rows, embeds = [], {}
    for ei, lt in enumerate(LAYER_TYPES):
        slug = env_slug(lt)
        d = np.load(Path(args.train_check_dir) / f'{slug}.npz',
                    allow_pickle=True)
        tv, gv = d['train_volumes'], d['gen_volumes']
        agree = (gv == tv).mean(axis=(1, 2, 3))

        pool = np.load(sorted((Path(args.pools_dir) / slug)
                              .glob('cond_r0000_*.npz'))[0],
                       allow_pickle=True)['volumes']
        rng = np.random.default_rng([PAIR_SEED, ei])
        ii = rng.integers(0, len(pool), 200)
        jj = rng.integers(0, len(pool), 200)
        ok = ii != jj
        pair = (pool[ii[ok]] == pool[jj[ok]]).mean(axis=(1, 2, 3))

        p99 = float(np.percentile(pair, 99))
        rows.append({
            'environment': lt,
            'agree_median': float(np.median(agree)),
            'agree_max': float(agree.max()),
            'engine_pair_median': float(np.median(pair)),
            'engine_pair_p99': p99,
            'n_flagged': int((agree > p99).sum()),
        })
        print(f"{lt}: gen-train med={np.median(agree):.3f} "
              f"max={agree.max():.3f} | engine-pair med={np.median(pair):.3f} "
              f"p99={p99:.3f} | flagged={int((agree > p99).sum())}", flush=True)

        rid, rv = io.load_volume_dir(Path(args.ref_dir) / slug)
        F = np.array([features(v) for v in
                      list(tv) + list(gv) + list(rv[:64])])
        F = (F - F.mean(0)) / np.maximum(F.std(0), 1e-9)
        embeds[lt] = classical_mds(F).tolist()

    out = Path(args.out_dir)
    json.dump({'rows': rows, 'embeds': embeds},
              open(out / 'memorization.json', 'w'), indent=2)
    md = ['## Memorization / novelty check (F.4)', '',
          '| environment | gen-vs-train agreement med / max '
          '| engine same-params pair med / p99 | flagged (> p99) |',
          '|' + '---|' * 4]
    for r in rows:
        md.append(f"| {r['environment']} "
                  f"| {r['agree_median']:.3f} / {r['agree_max']:.3f} "
                  f"| {r['engine_pair_median']:.3f} / {r['engine_pair_p99']:.3f} "
                  f"| {r['n_flagged']} |")
    (out / 'memorization.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from resbench.figures import AXIS_COLORS, REF_GRAY, SHORT
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.4))
    for i, lt in enumerate(LAYER_TYPES):
        ax = axes.flat[i]
        Y = np.array(embeds[lt])
        for sl, c, m, lbl in ((slice(0, 64), REF_GRAY, 'o', 'training'),
                              (slice(64, 128), AXIS_COLORS[0], '^', 'generated'),
                              (slice(128, None), AXIS_COLORS[2], 's', 'reference')):
            ax.scatter(Y[sl, 0], Y[sl, 1], s=9, c=c, marker=m, alpha=0.6,
                       label=lbl)
        ax.set_title(SHORT[lt], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.suptitle('MDS of per-volume harness statistics '
                 '(training / generated / reference)', y=1.0)
    fig.tight_layout()
    fig.savefig(out / 'memorization_mds.pdf', bbox_inches='tight')
    fig.savefig(out / 'memorization_mds.png', bbox_inches='tight', dpi=250)
    print('done ->', out / 'memorization.md')


if __name__ == '__main__':
    main()
