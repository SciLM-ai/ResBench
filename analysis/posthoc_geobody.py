"""Post-hoc analyses (RESULTS.md items 2-3; EVAL.md Addendum A.6 — not a
re-scoring of the master table).

1. Geobody-W1 sensitivity to a minimum-body-size filter {1, 2, 8} voxels,
   with the split-half band recomputed under the same filter and the SAME
   per-environment split permutation as the frozen scoring run.
2. Per-volume compartmentalization: fraction of volumes with per-volume
   tau_z < 0.99 (pair counts pooled over lags 1..16), fraction with a single
   body spanning floor (z=0) to surface (z=Z-1), median/IQR of per-volume
   largest-body fraction; Wilson 95% CIs on fractions. Reference vs
   ensemble (a), side by side.

Usage:
  python analysis/posthoc_geobody.py --ref-dir ... --pred-dir ... --out-dir results/posthoc
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resbench import LAYER_TYPES, SPLIT_HALF_SEED, env_slug   # noqa: E402
from resbench import io, metrics                              # noqa: E402
from resbench.stats import verdict                            # noqa: E402

SIZE_FILTERS = [1, 2, 8]
TAU_Z_THRESHOLD = 0.99

_G = {}


def per_volume(v):
    labels, _ = metrics.label_geobodies(v)
    same, pairs = metrics.connectivity_counts(v, (1, 1, 16), labels=labels)[2]
    sizes = metrics.geobody_sizes(v, labels=labels)
    top = set(np.unique(labels[:, :, -1])) - {0}
    bot = set(np.unique(labels[:, :, 0])) - {0}
    tot = int(sizes.sum())
    return {
        'sizes': sizes,
        'tau_z': float(same.sum() / pairs.sum()) if pairs.sum() else np.nan,
        'span': bool(top & bot),
        'largest': float(sizes[0]) / tot if tot else 0.0,
    }


def _stack_worker(key):
    return key, [per_volume(v) for v in _G[key]]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def filtered_w1(recs_a, recs_b, k):
    sa = np.concatenate([r['sizes'] for r in recs_a])
    sb = np.concatenate([r['sizes'] for r in recs_b])
    sa, sb = sa[sa >= k], sb[sb >= k]
    return metrics.geobody_w1(sa, sb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-dir', required=True)
    ap.add_argument('--pred-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    refs = io.load_ensemble(args.ref_dir)
    preds = io.load_ensemble(args.pred_dir)
    envs = [lt for lt in LAYER_TYPES if lt in refs and lt in preds]
    for lt in envs:
        pid, pv = preds[lt]
        rid, rv = refs[lt]
        pv, rv, _ = io.align(pid, pv, rid, rv)
        _G[f'{lt}|ref'] = rv
        _G[f'{lt}|gen'] = pv

    with mp.get_context('fork').Pool(16) as pool:
        recs = dict(pool.map(_stack_worker, list(_G)))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- item 2: W1 vs minimum body size -------------------------------
    w1_rows = []
    for i, lt in enumerate(envs):
        ra, ga = recs[f'{lt}|ref'], recs[f'{lt}|gen']
        rng = np.random.default_rng([SPLIT_HALF_SEED, i])   # same perm as scoring
        perm = rng.permutation(len(ra))
        ha = [ra[j] for j in perm[:len(ra) // 2]]
        hb = [ra[j] for j in perm[len(ra) // 2:]]
        row = {'environment': lt}
        for k in SIZE_FILTERS:
            w1 = filtered_w1(ra, ga, k)
            band = filtered_w1(ha, hb, k)
            row[f'w1_min{k}'] = w1
            row[f'band_min{k}'] = band
            row[f'verdict_min{k}'] = verdict(w1, band)
        w1_rows.append(row)
    # pooled row
    row = {'environment': 'pooled'}
    all_ref = sum((recs[f'{lt}|ref'] for lt in envs), [])
    all_gen = sum((recs[f'{lt}|gen'] for lt in envs), [])
    for k in SIZE_FILTERS:
        row[f'w1_min{k}'] = filtered_w1(all_ref, all_gen, k)
        row[f'band_min{k}'] = np.nan
        row[f'verdict_min{k}'] = 'n/a'
    w1_rows.append(row)

    # ---- item 3: compartmentalization table ----------------------------
    comp_rows = []
    for lt in envs:
        row = {'environment': lt}
        for side in ('ref', 'gen'):
            r = recs[f'{lt}|{side}']
            n = len(r)
            k_frag = sum(x['tau_z'] < TAU_Z_THRESHOLD for x in r)
            k_span = sum(x['span'] for x in r)
            lf = np.array([x['largest'] for x in r])
            lo1, hi1 = wilson_ci(k_frag, n)
            lo2, hi2 = wilson_ci(k_span, n)
            row.update({
                f'{side}_n': n,
                f'{side}_frac_tauz_lt_099': k_frag / n,
                f'{side}_tauz_ci': (round(lo1, 4), round(hi1, 4)),
                f'{side}_frac_span': k_span / n,
                f'{side}_span_ci': (round(lo2, 4), round(hi2, 4)),
                f'{side}_largest_median': float(np.median(lf)),
                f'{side}_largest_iqr': (float(np.percentile(lf, 25)),
                                        float(np.percentile(lf, 75))),
            })
        comp_rows.append(row)

    json.dump({'w1_sensitivity': w1_rows, 'compartmentalization': comp_rows},
              open(out / 'posthoc_geobody.json', 'w'), indent=2, default=str)

    # markdown renders
    md = ['## Geobody W1 vs minimum body size (post-hoc)', '',
          '| environment | ' + ' | '.join(
              f'W1 min≥{k} (band)' for k in SIZE_FILTERS) + ' |',
          '|' + '---|' * (len(SIZE_FILTERS) + 1)]
    for r in w1_rows:
        cells = []
        for k in SIZE_FILTERS:
            b = r[f'band_min{k}']
            band_s = f'{b:.4f}' if np.isfinite(b) else '—'
            cells.append(f"{r[f'w1_min{k}']:.4f} ({r[f'verdict_min{k}']}; {band_s})")
        md.append(f"| {r['environment']} | " + ' | '.join(cells) + ' |')

    md += ['', '## Per-volume compartmentalization: reference vs ResFlow (post-hoc)', '',
           '| environment | % vols τ_z<0.99 ref | gen | % vols spanning z ref | gen '
           '| largest-frac median [IQR] ref | gen |',
           '|' + '---|' * 7]
    for r in comp_rows:
        def pct(side, key, ci):
            lo, hi = r[f'{side}_{ci}']
            return f"{r[f'{side}_{key}'] * 100:.1f}% [{float(lo) * 100:.1f}, {float(hi) * 100:.1f}]"
        def lf(side):
            lo, hi = r[f'{side}_largest_iqr']
            return f"{r[f'{side}_largest_median']:.3f} [{lo:.3f}, {hi:.3f}]"
        md.append(
            f"| {r['environment']} | {pct('ref', 'frac_tauz_lt_099', 'tauz_ci')} "
            f"| {pct('gen', 'frac_tauz_lt_099', 'tauz_ci')} "
            f"| {pct('ref', 'frac_span', 'span_ci')} | {pct('gen', 'frac_span', 'span_ci')} "
            f"| {lf('ref')} | {lf('gen')} |")
    (out / 'posthoc_geobody.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
