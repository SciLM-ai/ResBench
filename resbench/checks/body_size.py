"""Are the sand blobs the right size?

Same flood-fill as `connectivity`: each blob's size is its cell count. Pool
every blob from every volume into one distribution -- a lobe volume holds
about 49 blobs, so 512 volumes give roughly 25,000 sizes, running from 1 cell
to tens of thousands.

Sizes are compared in log10. A linear comparison is a distance in cells, so a
4,000-cell error on the largest body would drown out every error among the
thousands of one- and two-cell bodies. Logs turn "how many cells off" into
"what factor off", so a body twice too big counts the same at 10 cells as at
10,000.
"""
import numpy as np
from .. import metrics as M
from ._base import ALL_TASKS, sum_merge, hist_w1

NAME, TASKS, NEEDS = 'body_size', ALL_TASKS, 'samples'
LOG_MIN, LOG_MAX, NBINS = 0.0, 8.0, 800          # 0.01 dex bins
EDGES = np.linspace(LOG_MIN, LOG_MAX, NBINS + 1)


def summarize(vols, ctx=None):
    h = np.zeros(NBINS, np.int64)
    for vol in np.asarray(vols):
        s = M.geobody_sizes(vol)
        if len(s):
            h += np.histogram(np.log10(np.asarray(s, float)), bins=EDGES)[0]
    return {'hist': h}


def merge(parts):
    return sum_merge(parts, ['hist'])


def compare(model, ref):
    return {'parts': {'size_w1': hist_w1(model['hist'], ref['hist'], EDGES)}}
