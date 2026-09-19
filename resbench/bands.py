"""The band: how much ResMill disagrees with itself.

Run ResMill twice with different seeds and you get two different reservoirs.
Measure anything on them and the answers will not match exactly. That leftover
disagreement is pure randomness and it is the smallest gap anyone could ever
achieve, so it is the natural unit for every check.

Measured by splitting the reference into two halves and comparing the halves
to each other with the same estimator the model is scored by. Dividing each
check by its own band is what makes bits, cell counts and distances between
histograms add up to one number.

`net_to_gross` is the one exception: ResMill is never asked to reproduce a
realized sand fraction, so it has no self-disagreement to measure and uses a
fixed tolerance instead (see that module).
"""
import numpy as np

SPLIT_SEED = 20260918


def split_halves(n, seed=SPLIT_SEED):
    """A reproducible half/half partition of n items."""
    idx = np.random.default_rng(seed).permutation(n)
    return idx[: n // 2], idx[n // 2: 2 * (n // 2)]


def band_for(check, ref_vols, ctx=None, seed=SPLIT_SEED):
    """Split-half distance for one check, per sub-part."""
    fixed = getattr(check, 'TOLERANCE', None)
    if fixed is not None:
        return {k: float(fixed) for k in _part_names(check, ref_vols, ctx)}
    a, b = split_halves(len(ref_vols), seed)
    sa = check.summarize(np.asarray(ref_vols)[a], ctx)
    sb = check.summarize(np.asarray(ref_vols)[b], ctx)
    return {k: float(v) for k, v in check.compare(sa, sb)['parts'].items()}


def band_for_repeats(check, ref_groups, seed=SPLIT_SEED):
    """Split-half band for a repeats-based check.

    `ref_groups` is {condition_id: (volumes, ctx)}. Each condition's repeats
    are split in half, so the band is ResMill's own ensemble disagreeing with
    itself at the same condition.
    """
    fixed = getattr(check, 'TOLERANCE', None)
    parts_a, parts_b = [], []
    for cid, (vols, ctx) in sorted(ref_groups.items()):
        a, b = split_halves(len(vols), seed)
        parts_a.append(check.summarize(np.asarray(vols)[a], ctx))
        parts_b.append(check.summarize(np.asarray(vols)[b], ctx))
    if not parts_a:
        return {}
    sa, sb = check.merge(parts_a), check.merge(parts_b)
    if fixed is not None:
        return {k: float(fixed) for k in check.compare(sa, sb)['parts']}
    return {k: float(v) for k, v in check.compare(sa, sb)['parts'].items()}


def _part_names(check, vols, ctx):
    s = check.summarize(np.asarray(vols)[:2], ctx)
    return list(check.compare(s, s)['parts'])
