"""Are the beds the right thickness?

Read straight down each of the map locations, exactly what a vertical well
logs: every unbroken run of sand in that column is one bed. A 64x64x32 volume
has 4,096 columns, so an environment yields a few million beds.

These are INTERCEPT thicknesses, what a well sees, not the thicknesses of the
bodies: a vertical line hits most lobes near their feathered edge, so the
distribution decays from 1 cell rather than peaking at a characteristic value,
and stacked lobes read as one bed. A bed running into the top or bottom of the
volume is kept at its truncated length, identically on both sides.
"""
import numpy as np
from .. import metrics as M
from ._base import ALL_TASKS, sum_merge, hist_w1

NAME, TASKS, NEEDS = 'bed_thickness', ALL_TASKS, 'samples'
MAX_RUN = 512


def summarize(vols, ctx=None):
    h = np.zeros(MAX_RUN + 1, np.int64)
    for vol in np.asarray(vols):
        r = M.run_lengths(vol, axis=2)
        if r.size:
            h += np.bincount(np.clip(r, 0, MAX_RUN), minlength=MAX_RUN + 1)
    return {'hist': h}


def merge(parts):
    return sum_merge(parts, ['hist'])


def compare(model, ref):
    edges = np.arange(MAX_RUN + 2, dtype=float)
    return {'parts': {'bed_w1': hist_w1(model['hist'], ref['hist'], edges)}}
