"""Ensemble summaries, master-table cells, split-half null band, bootstrap.

All estimator choices are frozen in EVAL.md §5-6. The same `compare`
estimator is used for gen-vs-ref and for half-vs-half (the null band), so the
band is directly comparable cell by cell.
"""
import numpy as np

from . import MAX_LAGS, SPLIT_HALF_SEED, BOOTSTRAP_SEED
from . import metrics

CELLS = ['abs_dntg', 'variogram_mae', 'connectivity_mae',
         'geobody_w1', 'abs_dlargest_frac']


def env_summary(vols, max_lags=MAX_LAGS):
    """All ensemble statistics for one volume stack, in a single pass.

    Keeps the raw pooled counts alongside the curves so summaries of disjoint
    stacks can be combined exactly (`merge_summaries`) for the pooled row.
    """
    gacc = {a: [np.zeros(L, np.int64), np.zeros(L, np.int64)]
            for a, L in enumerate(max_lags)}
    tacc = {a: [np.zeros(L, np.int64), np.zeros(L, np.int64)]
            for a, L in enumerate(max_lags)}
    sizes_all, fracs = [], []
    for v in vols:
        for a, (sq, n) in metrics.variogram_sums(v, max_lags).items():
            gacc[a][0] += sq
            gacc[a][1] += n
        labels, _ = metrics.label_geobodies(v)
        for a, (same, pairs) in metrics.connectivity_counts(
                v, max_lags, labels=labels).items():
            tacc[a][0] += same
            tacc[a][1] += pairs
        s = metrics.geobody_sizes(v, labels=labels)
        sizes_all.append(s)
        total = int(s.sum())
        fracs.append(float(s[0]) / total if total > 0 else 0.0)
    return _finalize_summary(len(vols), metrics.ensemble_ntg(vols),
                             gacc, tacc,
                             np.concatenate(sizes_all) if sizes_all
                             else np.empty(0, np.int64),
                             np.asarray(fracs, np.float64))


def _finalize_summary(n, ntg, gacc, tacc, sizes, fracs):
    with np.errstate(invalid='ignore'):
        gamma = {a: 0.5 * gacc[a][0] / gacc[a][1] for a in gacc}
        tau = {a: np.where(tacc[a][1] > 0,
                           tacc[a][0] / np.maximum(tacc[a][1], 1), np.nan)
               for a in tacc}
    return {'n': n, 'ntg': ntg, 'gamma': gamma, 'tau': tau,
            'gamma_counts': gacc, 'tau_counts': tacc,
            'sizes': sizes, 'largest_frac': fracs}


def merge_summaries(summaries):
    """Exact pooled summary of disjoint stacks (sums the raw counts)."""
    first = summaries[0]
    axes = list(first['gamma_counts'])
    gacc = {a: [sum(s['gamma_counts'][a][0] for s in summaries),
                sum(s['gamma_counts'][a][1] for s in summaries)] for a in axes}
    tacc = {a: [sum(s['tau_counts'][a][0] for s in summaries),
                sum(s['tau_counts'][a][1] for s in summaries)] for a in axes}
    return _finalize_summary(
        sum(s['n'] for s in summaries),
        np.concatenate([s['ntg'] for s in summaries]),
        gacc, tacc,
        np.concatenate([s['sizes'] for s in summaries]),
        np.concatenate([s['largest_frac'] for s in summaries]))


def compare(ref, gen, sill):
    """Master-table cells between two summaries (EVAL.md §5)."""
    return {
        'abs_dntg': float(abs(gen['ntg'].mean() - ref['ntg'].mean())),
        'variogram_mae': metrics.variogram_mae(gen['gamma'], ref['gamma'], sill),
        'connectivity_mae': metrics.connectivity_mae(gen['tau'], ref['tau']),
        'geobody_w1': metrics.geobody_w1(gen['sizes'], ref['sizes']),
        'abs_dlargest_frac': float(abs(gen['largest_frac'].mean()
                                       - ref['largest_frac'].mean())),
    }


def reference_sill(ref_summary):
    """Frozen normalization: p(1-p) at the reference-ensemble mean NTG."""
    p = float(ref_summary['ntg'].mean())
    return p * (1.0 - p)


def split_half(vols, rng, max_lags=MAX_LAGS):
    """Split one reference stack into two random halves; return (cells, halves).

    The half-vs-half discrepancy under the same `compare` estimator is the
    engine's Monte-Carlo noise band. Sill comes from the FULL reference stack
    so band and score share one normalization.
    """
    n = len(vols)
    perm = rng.permutation(n)
    a, b = vols[perm[:n // 2]], vols[perm[n // 2:]]
    sa, sb = env_summary(a, max_lags), env_summary(b, max_lags)
    p = float(np.asarray(vols).mean())
    return compare(sa, sb, p * (1.0 - p)), (sa, sb)


def bootstrap_mean_ci(values, n_boot=1000, seed=BOOTSTRAP_SEED, alpha=0.05):
    """(mean, lo, hi): percentile bootstrap CI for the mean over volumes."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (float(values.mean()),
            float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def verdict(value, band):
    """EVAL.md acceptance: inside (<= band), near (<= 2x band), outside."""
    if not np.isfinite(value) or not np.isfinite(band):
        return 'n/a'
    if value <= band:
        return 'inside'
    if value <= 2.0 * band:
        return 'near'
    return 'outside'


def pooled_stack(env_vols):
    """Concatenate {env: vols} into one stack in canonical dict order."""
    return np.concatenate(list(env_vols.values()))
