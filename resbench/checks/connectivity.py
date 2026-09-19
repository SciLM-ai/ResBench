"""Can you actually walk from one sand cell to another through sand?

Two sand cells count as touching only if they share a whole FACE. Edge and
corner contact do not count, because fluid cannot flow through a contact with
no area. Flood-fill under that rule and every sand cell gets a body number;
two cells are connected when the numbers match, meaning some unbroken path
links them however long and winding.

Then: among all pairs a given distance apart that are BOTH sand, what fraction
sit in the same body? At distance 1 that is exactly 1 by construction.

The two tallies are summed over the ensemble and divided once at the end,
never averaged as per-volume fractions. Both-sand pairs scale roughly as p^2,
so a volume at 60% sand offers about nine times as many as one at 20%, and
averaging fractions would give the sand-poor volume the same say.
"""
import numpy as np
from .. import metrics as M
from ._base import ALL_TASKS, sum_merge

NAME, TASKS, NEEDS = 'connectivity', ALL_TASKS, 'samples'


def summarize(vols, ctx=None):
    v = np.asarray(vols)
    lags = M.max_lags_for(v.shape[1:])
    same = [np.zeros(L, np.int64) for L in lags]
    pairs = [np.zeros(L, np.int64) for L in lags]
    for vol in v:
        out = M.connectivity_counts(vol, max_lags=lags)
        for a in range(3):
            same[a] += out[a][0]; pairs[a] += out[a][1]
    return {'same': np.concatenate(same), 'pairs': np.concatenate(pairs),
            'lags': np.asarray(lags)}


def merge(parts):
    out = sum_merge(parts, ['same', 'pairs'])
    out['lags'] = np.asarray(parts[0]['lags'])
    return out


def _tau(s):
    return s['same'] / np.maximum(s['pairs'], 1)


def compare(model, ref):
    # No normalization: tau is already a probability.
    return {'parts': {'tau_mae': float(np.abs(_tau(model) - _tau(ref)).mean())}}
