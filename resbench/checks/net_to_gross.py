"""Does the model honor the sand fraction it was given?

The sand fraction handed to the model is the reference volume's REALIZED
value, not the target ResMill was originally given, so it is a number the
model is expected to reproduce. Reading it back off the output is one line of
arithmetic, which makes this the only check that needs no reference data: the
target is the condition itself and a perfect score is zero.

That also means there is no band. ResMill is never asked to reproduce a
realized sand fraction, so it has no self-disagreement to measure here, and
the divisor below is a chosen tolerance rather than a measurement.
"""
import numpy as np
from ._base import ALL_TASKS

NAME, TASKS, NEEDS = 'net_to_gross', ALL_TASKS, 'samples'
TOLERANCE = 0.01          # one percentage point of sand; a choice, not a band


def summarize(vols, ctx=None):
    v = np.asarray(vols)
    ntg = v.reshape(len(v), -1).mean(axis=1).astype(np.float64)
    target = np.asarray((ctx or {}).get('target_ntg', ntg), np.float64)
    return {'err_sum': float(np.abs(ntg - target).sum()), 'n': int(len(v))}


def merge(parts):
    return {'err_sum': float(sum(p['err_sum'] for p in parts)),
            'n': int(sum(p['n'] for p in parts))}


def compare(model, ref):
    mae = model['err_sum'] / max(model['n'], 1)
    return {'parts': {'ntg_error': float(mae)}}
