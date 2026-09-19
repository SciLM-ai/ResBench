"""Is the sand one connected reservoir or many isolated pockets?

Two numbers from the same flood-fill as `connectivity`, both about the whole
volume rather than any distance.

  gamma  P(two sand cells drawn anywhere land in the same body),
         sum(n^2) / (sum n)^2. All the sand in one body gives 1; ten equal
         bodies give about 0.1. The n^2 deliberately weights toward the
         largest bodies, because that is the fraction of the sand one well
         reaches. `body_size` takes logs and weights every body equally, so
         the two see different failures and move in opposite directions.

  euler  pieces - tunnels + enclosed cavities, per million cells so volumes of
         different sizes compare. It separates rocks gamma cannot: a body
         riddled with shale tunnels and a solid one of the same size share a
         gamma but drain very differently.
"""
import numpy as np
from .. import metrics as M
from ._base import ALL_TASKS

NAME, TASKS, NEEDS = 'compartments', ALL_TASKS, 'samples'


def summarize(vols, ctx=None):
    g, e = [], []
    for vol in np.asarray(vols):
        labels, _ = M.label_geobodies(vol)
        g.append(M.gamma_global(M.geobody_sizes(vol, labels=labels)))
        e.append(M.euler_characteristic(vol) / (vol.size / 1e6))
    return {'gamma_sum': float(np.sum(g)), 'euler_sum': float(np.sum(e)),
            'n': int(len(g))}


def merge(parts):
    return {'gamma_sum': float(sum(p['gamma_sum'] for p in parts)),
            'euler_sum': float(sum(p['euler_sum'] for p in parts)),
            'n': int(sum(p['n'] for p in parts))}


def compare(model, ref):
    def mean(s, k):
        return s[k] / max(s['n'], 1)
    return {'parts': {
        'gamma': abs(mean(model, 'gamma_sum') - mean(ref, 'gamma_sum')),
        'euler_per_1e6': abs(mean(model, 'euler_sum') - mean(ref, 'euler_sum')),
    }}
