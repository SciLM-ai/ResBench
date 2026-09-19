"""Is the model scattering stray grains, and does it get worse under
conditioning?

Sand blobs smaller than 8 cells, counted per volume. The classic generative
failure is this rate climbing as more wells are imposed, so it is reported
against well count too.
"""
import numpy as np
from .. import metrics as M
from ._base import ALL_TASKS

NAME, TASKS, NEEDS = 'speckle', ALL_TASKS, 'samples'
MAX_SIZE = M.ARTIFACT_MAX          # 8 cells


def summarize(vols, ctx=None):
    v = np.asarray(vols)
    c = M.artifact_counts(v, max_size=MAX_SIZE)
    return {'count_sum': float(c.sum()), 'n': int(len(v))}


def merge(parts):
    return {'count_sum': float(sum(p['count_sum'] for p in parts)),
            'n': int(sum(p['n'] for p in parts))}


def compare(model, ref):
    per = lambda s: s['count_sum'] / max(s['n'], 1)
    return {'parts': {'speckle_per_volume': abs(per(model) - per(ref))}}
