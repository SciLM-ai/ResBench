"""How long and wide are the bodies, and which way do they point?

For each axis and separation h, count the cell pairs that DISAGREE (one sand,
one shale), divide by the pair count and halve: that is gamma(h). h runs to
half the axis, so a 64x64x32 volume gives 32 + 32 + 16 = 80 numbers.

Counts are summed over the ensemble before dividing, so every volume is
weighted by how many pairs it actually contributes.
"""
import numpy as np
from .. import metrics as M
from ._base import ALL_TASKS, sum_merge

NAME, TASKS, NEEDS = 'variogram', ALL_TASKS, 'samples'


def summarize(vols, ctx=None):
    v = np.asarray(vols)
    lags = M.max_lags_for(v.shape[1:])
    sq = [np.zeros(L, np.int64) for L in lags]
    npair = [np.zeros(L, np.int64) for L in lags]
    ntg_sum, n = 0.0, 0
    for vol in v:
        out = M.variogram_sums(vol, max_lags=lags)
        for a in range(3):
            sq[a] += out[a][0]; npair[a] += out[a][1]
        ntg_sum += float(vol.mean()); n += 1
    return {'sq': np.concatenate(sq), 'npair': np.concatenate(npair),
            'lags': np.asarray(lags), 'ntg_sum': ntg_sum, 'n': n}


def merge(parts):
    out = sum_merge(parts, ['sq', 'npair'])
    out['lags'] = np.asarray(parts[0]['lags'])
    out['ntg_sum'] = float(sum(p['ntg_sum'] for p in parts))
    out['n'] = int(sum(p['n'] for p in parts))
    return out


def _gamma(s):
    return 0.5 * s['sq'] / np.maximum(s['npair'], 1)


def compare(model, ref):
    # Normalize by the sill p(1-p) of the REFERENCE, so the same relative
    # error does not look three times worse in a sand-rich environment.
    p = ref['ntg_sum'] / max(ref['n'], 1)
    sill = max(p * (1.0 - p), 1e-12)
    mae = float(np.abs(_gamma(model) - _gamma(ref)).mean() / sill)
    return {'parts': {'gamma_mae': mae}}
