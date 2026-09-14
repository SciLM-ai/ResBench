"""Addendum F items F.1/F.2/F.5: MPS histograms, runs statistics, artifact
rate. CPU-only, post-hoc on existing volumes (EVAL.md Addendum F).

Usage:
  python analysis/acceptance_ext.py --ref-dir REF --pred-dir ENS_A \
      --well-pred-dir ENS_B --manifest manifest.csv --out-dir results/posthoc
"""
import argparse
import json
import multiprocessing as mp
from itertools import product
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resbench import LAYER_TYPES                       # noqa: E402
from resbench import io, metrics                       # noqa: E402
from resbench.stats import verdict                     # noqa: E402

MPS_SEED, ART_SEED = 20260813, 20260816
_G = {}


# ---------------- F.1 MPS ------------------------------------------------

def mps_hist(vol):
    """Pooled 2x2x2 pattern histogram (256 bins) of one volume."""
    v = np.asarray(vol, np.uint8)
    acc = np.zeros((v.shape[0] - 1, v.shape[1] - 1, v.shape[2] - 1), np.uint8)
    for i, j, k in product((0, 1), repeat=3):
        acc |= (v[i:v.shape[0] - 1 + i, j:v.shape[1] - 1 + j,
                  k:v.shape[2] - 1 + k] << (4 * i + 2 * j + k))
    return np.bincount(acc.ravel(), minlength=256).astype(np.int64)


def jsd_bits(p, q, eps=1e-300):
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        s = a > 0
        return float((a[s] * np.log2(a[s] / np.maximum(b[s], eps))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ---------------- F.2 runs -----------------------------------------------

def run_lengths(vols, axis):
    """Pooled maximal sand run lengths along `axis` over a volume stack."""
    out = []
    for v in vols:
        a = np.moveaxis(np.asarray(v, np.int8), axis, -1)
        pad = np.zeros(a.shape[:-1] + (a.shape[-1] + 2,), np.int8)
        pad[..., 1:-1] = a
        d = np.diff(pad, axis=-1)
        starts = np.argwhere(d == 1)
        ends = np.argwhere(d == -1)
        out.append(ends[:, -1] - starts[:, -1])
    return np.concatenate(out) if out else np.empty(0, np.int64)


def w1(a, b):
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(a, b)) if len(a) and len(b) else np.nan


# ---------------- F.5 artifacts ------------------------------------------

def artifact_counts(vols):
    # the F.5 threshold lives in resbench.metrics so it cannot drift
    return metrics.artifact_counts(vols)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ---------------- per-environment worker ---------------------------------

def _env_worker(args_):
    lt, ei = args_
    rv, gv = _G[lt]
    res = {'environment': lt}

    # F.1 MPS
    h_ref = sum(mps_hist(v) for v in rv)
    h_gen = sum(mps_hist(v) for v in gv)
    perm = np.random.default_rng([MPS_SEED, ei]).permutation(len(rv))
    h_a = sum(mps_hist(rv[i]) for i in perm[:len(rv) // 2])
    h_b = sum(mps_hist(rv[i]) for i in perm[len(rv) // 2:])
    res['mps'] = {'jsd': jsd_bits(h_ref, h_gen),
                  'band': jsd_bits(h_a, h_b),
                  'hist_ref': (h_ref / h_ref.sum()).tolist(),
                  'hist_gen': (h_gen / h_gen.sum()).tolist()}

    # F.2 runs (z + x control)
    res['runs'] = {}
    for name, axis in (('z', 2), ('x', 0)):
        r_ref = run_lengths(rv, axis)
        r_gen = run_lengths(gv, axis)
        r_a = run_lengths(rv[perm[:len(rv) // 2]], axis)
        r_b = run_lengths(rv[perm[len(rv) // 2:]], axis)
        res['runs'][name] = {
            'median_ref': float(np.median(r_ref)),
            'median_gen': float(np.median(r_gen)),
            'p90_ref': float(np.percentile(r_ref, 90)),
            'p90_gen': float(np.percentile(r_gen, 90)),
            'w1': w1(r_ref, r_gen), 'w1_band': w1(r_a, r_b),
            'median_band': float(abs(np.median(r_a) - np.median(r_b))),
        }

    # F.5 artifacts, ref vs (a)
    perm2 = np.random.default_rng([ART_SEED, ei]).permutation(len(rv))
    c_ref = artifact_counts(rv)
    c_gen = artifact_counts(gv)
    res['artifacts'] = {
        'mean_ref': float(c_ref.mean()), 'mean_gen': float(c_gen.mean()),
        'band': float(abs(c_ref[perm2[:len(rv) // 2]].mean()
                          - c_ref[perm2[len(rv) // 2:]].mean())),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-dir', required=True)
    ap.add_argument('--pred-dir', required=True)
    ap.add_argument('--well-pred-dir', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--label', default='model',
                    help='name for the generated ensemble in figure legends')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    refs = io.load_ensemble(args.ref_dir)
    preds = io.load_ensemble(args.pred_dir)
    envs = [lt for lt in LAYER_TYPES if lt in refs and lt in preds]
    for lt in envs:
        pid, pv = preds[lt]
        rid, rv = refs[lt]
        pv, rv, _ = io.align(pid, pv, rid, rv)
        _G[lt] = (rv, pv)

    with mp.get_context('fork').Pool(min(8, len(envs))) as pool:
        rows = pool.map(_env_worker, [(lt, i) for i, lt in enumerate(envs)])

    # F.5 ensemble (b): artifacts vs well count
    import pandas as pd
    mf = pd.read_csv(args.manifest, keep_default_na=False)
    cfg_by_id = dict(zip(mf['row_id'], mf['well_config']))
    wellb = {}
    b = io.load_ensemble(args.well_pred_dir)
    for lt in envs:
        bid, bv = b[lt]
        counts = artifact_counts(bv)
        for i, c in zip(bid, counts):
            cfg = cfg_by_id.get(i.split('#')[0], '?')
            wellb.setdefault(cfg, []).append(int(c))
    by_cfg = {}
    for cfg in sorted(wellb):
        cs = np.array(wellb[cfg])
        k = int((cs > 0).sum())
        lo, hi = wilson(k, len(cs))
        by_cfg[cfg] = {'n': len(cs), 'mean_count': float(cs.mean()),
                       'frac_any': k / len(cs),
                       'wilson': (round(lo, 4), round(hi, 4))}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({'rows': rows, 'artifacts_by_wellcount': by_cfg},
              open(out / 'acceptance_ext.json', 'w'), indent=2)

    # markdown
    md = ['## MPS 2x2x2 pattern histograms (F.1)', '',
          '| environment | JSD (bits) | band | verdict |', '|' + '---|' * 4]
    for r in rows:
        m = r['mps']
        md.append(f"| {r['environment']} | {m['jsd']:.5f} | {m['band']:.5f} "
                  f"| {verdict(m['jsd'], m['band'])} |")
    md += ['', '## Sand run-length statistics (F.2)', '',
           '| environment | axis | median ref→gen | p90 ref→gen | W1 (band) '
           '| verdict |', '|' + '---|' * 6]
    for r in rows:
        for ax in ('z', 'x'):
            s = r['runs'][ax]
            md.append(
                f"| {r['environment']} | {ax} "
                f"| {s['median_ref']:.0f}→{s['median_gen']:.0f} "
                f"| {s['p90_ref']:.0f}→{s['p90_gen']:.0f} "
                f"| {s['w1']:.3f} ({s['w1_band']:.3f}) "
                f"| {verdict(s['w1'], s['w1_band'])} |")
    md += ['', '## Artifact rate: bodies < 8 voxels (F.5)', '',
           '| environment | mean/vol ref | gen | band | verdict |',
           '|' + '---|' * 5]
    for r in rows:
        a = r['artifacts']
        md.append(f"| {r['environment']} | {a['mean_ref']:.2f} "
                  f"| {a['mean_gen']:.2f} | {a['band']:.2f} "
                  f"| {verdict(abs(a['mean_gen'] - a['mean_ref']), a['band'])} |")
    md += ['', '### Ensemble (b): artifacts vs well count', '',
           '| config | n vols | mean artifacts/vol | frac ≥1 artifact '
           '[Wilson 95%] |', '|' + '---|' * 4]
    for cfg, s in by_cfg.items():
        md.append(f"| {cfg} | {s['n']} | {s['mean_count']:.2f} "
                  f"| {s['frac_any']:.3f} [{s['wilson'][0]:.3f}, "
                  f"{s['wilson'][1]:.3f}] |")
    (out / 'acceptance_ext.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))

    # F.1 figure: top-20 patterns for the two worst-JSD environments
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from resbench.figures import AXIS_COLORS, REF_GRAY, SHORT
    worst = sorted(rows, key=lambda r: -r['mps']['jsd'])[:2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, r in zip(axes, worst):
        p = np.array(r['mps']['hist_ref'])
        q = np.array(r['mps']['hist_gen'])
        top = np.argsort(-p)[:20]
        x = np.arange(20)
        ax.bar(x - 0.2, p[top], 0.4, color=REF_GRAY, label='reference')
        ax.bar(x + 0.2, q[top], 0.4, color=AXIS_COLORS[0], label=args.label)
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{t:08b}' for t in top], rotation=90, fontsize=5.5)
        ax.set_title(f"{SHORT[r['environment']]} "
                     f"(JSD {r['mps']['jsd']:.4f}, band {r['mps']['band']:.4f})",
                     fontsize=9)
        ax.set_ylabel('pattern probability')
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Top-20 2×2×2 patterns — two largest-JSD environments', y=1.02)
    fig.tight_layout()
    fig.savefig(out / 'mps_top20.pdf', bbox_inches='tight')
    fig.savefig(out / 'mps_top20.png', bbox_inches='tight', dpi=250)
    print('done ->', out / 'acceptance_ext.md')


if __name__ == '__main__':
    main()
