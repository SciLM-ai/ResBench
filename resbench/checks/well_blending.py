"""Does the model blend the borehole into the surrounding rock the way ResMill
does, or leave a hard seam around it and stay collapsed further out?

This is the only check that measures conditioning. Well reproduction error is
exactly zero for everybody, because every model pastes the true values into
the well cells after generating, so nothing else distinguishes a model that
conditioned from one that merely overwrote.

Take the same entropy map as `variety`, built from repeats of one published
well. With a well present the map has real structure. Sorting ResMill's cells
by distance from the borehole gives mean entropy 0.00 at the well, then 0.13,
0.22, 0.39, 0.55, 0.81 and 0.96 at distances 1, 2, 4, 6, 10 and 20: the well
pins the rock beside it and its influence fades outward. That decay curve is
what gets compared, and it catches two failures at once -- a hard artificial
boundary around the conditioned column, and a model that never recovers full
variety away from the well.

The SIGNED near-field offset is reported beside the score. Negative means the
model is more certain near the borehole than reality warrants: confident,
wrong, and right where somebody is about to drill. Positive means it pasted
the values in and left the surrounding rock as vague as if there were no well.
"""
import numpy as np
from .. import metrics as M

NAME, TASKS, NEEDS = 'well_blending', ('well_conditioned',), 'repeats'
MAX_DIST, NEAR_FIELD = 20, 6


def _profile(p_map, xy):
    """Mean entropy in each ring of constant distance from the borehole."""
    H = M.entropy_map_from_p(p_map)
    d = M.chebyshev_distance_map(H.shape, xy)
    return np.array([H[d == k].mean() if (d == k).any() else np.nan
                     for k in range(MAX_DIST + 1)])


def summarize(vols, ctx=None):
    ctx = ctx or {}
    cid = str(ctx.get('condition_id', '0'))
    xy = tuple(ctx.get('well_xy', (0, 0)))
    return {'maps': {cid: (M.prob_map(vols), xy)}}


def merge(parts):
    out = {}
    for p in parts:
        out.update(p['maps'])
    return {'maps': out}


def compare(model, ref):
    shared = sorted(set(model['maps']) & set(ref['maps']))
    if not shared:
        return {'parts': {'profile_mae': float('nan')}, 'near_field_offset': float('nan')}
    maes, offs = [], []
    for c in shared:
        pm, xy = model['maps'][c]
        pr, _ = ref['maps'][c]
        a, b = _profile(pm, xy), _profile(pr, xy)
        maes.append(float(np.nanmean(np.abs(a - b))))
        offs.append(float(np.nanmean((a - b)[:NEAR_FIELD + 1])))
    return {'parts': {'profile_mae': float(np.mean(maes))},
            'near_field_offset': float(np.mean(offs))}
