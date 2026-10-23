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
        # Use the declared part names: calling summarize here would hand a
        # 2-volume slice a full-length ctx and fail the alignment check.
        return {k: float(fixed) for k in _part_names(check, ref_vols, ctx)}
    a, b = split_halves(len(ref_vols), seed)
    sa = check.summarize(np.asarray(ref_vols)[a], ctx)
    sb = check.summarize(np.asarray(ref_vols)[b], ctx)
    return {k: float(v) for k, v in check.compare(sa, sb)['parts'].items()}


def _part_names(check, vols, ctx):
    declared = getattr(check, 'PARTS', None)
    if declared is not None:
        return list(declared)
    s = check.summarize(np.asarray(vols)[:2], ctx)
    return list(check.compare(s, s)['parts'])


# ---------------------------------------------------------------------------------------------
# Expected band and equal-size comparison
#
# The band is averaged over many random half/half partitions of the reference, so no single split
# can land far in either tail. The model is measured at the same size: a random subset of the
# model, of half the reference's size, against a random half of the reference, averaged over as
# many draws. A model that is ResMill therefore scores s = 1 on average, which is what "1 means
# indistinguishable" claims.
# ---------------------------------------------------------------------------------------------
N_SPLITS = 500            # partitions / draws for sample-based checks (per-volume summaries, cheap)
N_SPLITS_REPEATS = 200    # repeats-based checks rebuild an ensemble map per draw, so fewer
BAND_SEED, MODEL_SEED = SPLIT_SEED, SPLIT_SEED + 1


def _mean_parts(dists):
    keys = dists[0].keys()
    return {k: float(np.nanmean([d[k] for d in dists])) if any(np.isfinite(d[k]) for d in dists)
            else float('nan') for k in keys}


def expected_band(check, ref_summaries, k=N_SPLITS, seed=BAND_SEED):
    """Mean split-half distance over k random half/half partitions of per-volume summaries."""
    n = len(ref_summaries); h = n // 2
    out = []
    for i in range(k):
        p = np.random.default_rng([seed, i]).permutation(n)
        a = check.merge([ref_summaries[j] for j in p[:h]])
        b = check.merge([ref_summaries[j] for j in p[h:2 * h]])
        out.append(check.compare(a, b)['parts'])
    return _mean_parts(out)


def matched_distance(check, model_summaries, ref_summaries, k=N_SPLITS, seed=MODEL_SEED):
    """Mean distance between a random subset of the model (half the reference's size) and a
    random half of the reference, over k draws: the same sample sizes as the band."""
    h = len(ref_summaries) // 2
    hm = min(h, len(model_summaries))
    out = []
    for i in range(k):
        rng = np.random.default_rng([seed, i])
        pm = rng.permutation(len(model_summaries))[:hm]
        pr = rng.permutation(len(ref_summaries))[:h]
        out.append(check.compare(check.merge([model_summaries[j] for j in pm]),
                                 check.merge([ref_summaries[j] for j in pr]))['parts'])
    return _mean_parts(out)


def expected_band_repeats(check, ref_groups, k=N_SPLITS_REPEATS, seed=BAND_SEED):
    """Repeats-based band: per condition, ResMill's runs split half/half; mean over k partitions."""
    out = []
    for i in range(k):
        pa, pb = [], []
        for c, (cid, (vols, ctx)) in enumerate(sorted(ref_groups.items())):
            v = np.asarray(vols); n = len(v); h = n // 2
            p = np.random.default_rng([seed, i, c]).permutation(n)
            pa.append(check.summarize(v[p[:h]], ctx)); pb.append(check.summarize(v[p[h:2 * h]], ctx))
        out.append(check.compare(check.merge(pa), check.merge(pb))['parts'])
    return _mean_parts(out)


def run_index(ids, n):
    """Position of the model run tied to each of ResMill's n runs, from ids '<condition>|<k>', when the model has
    exactly one run per ResMill run (the repeat inputs); otherwise None."""
    try:
        ks = [int(str(i).rsplit('|', 1)[1]) for i in ids]
    except (IndexError, ValueError):
        return None
    if sorted(ks) != list(range(n)):
        return None
    pos = np.empty(n, dtype=int)
    pos[ks] = np.arange(n)
    return pos


def matched_distance_repeats(check, model_groups, ref_groups, k=N_SPLITS_REPEATS, seed=MODEL_SEED, model_ids=None):
    """Repeats-based model distance: per condition, the model's runs against half of ResMill's runs, at half
    the reference's size; mean over k draws.

    Paired (one model run per ResMill run, each given that run's realised sand fraction): ResMill's runs are split
    in half, and the model's runs tied to one half are compared with ResMill's other half, so no draw sets a model
    run against the ResMill run whose sand fraction it was given. Otherwise a random subset of the model's runs
    is compared with a random half of ResMill's."""
    out = []
    for i in range(k):
        pm_, pr_ = [], []
        for c, cid in enumerate(sorted(ref_groups)):
            rv, ctx = ref_groups[cid]; mv = model_groups[cid]
            rv, mv = np.asarray(rv), np.asarray(mv)
            h = len(rv) // 2; hm = min(h, len(mv))
            rng = np.random.default_rng([seed, i, c])
            pos = run_index(model_ids[cid], len(rv)) if model_ids is not None else None
            if pos is not None:
                p = rng.permutation(len(rv))
                pm_.append(check.summarize(mv[pos[p[:h]]], ctx))
                pr_.append(check.summarize(rv[p[h:2 * h]], ctx))
                continue
            pm_.append(check.summarize(mv[rng.permutation(len(mv))[:hm]], ctx))
            pr_.append(check.summarize(rv[rng.permutation(len(rv))[:h]], ctx))
        out.append(check.compare(check.merge(pm_), check.merge(pr_))['parts'])
    return _mean_parts(out)
