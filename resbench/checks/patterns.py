"""Is the local texture right, even where the variogram is?

Every 2x2x2 block of cells is one of 256 possible fillings: read its 8 cells
in a fixed order and the block IS an 8-bit number. Not an eight-dimensional
comparison, one label out of 256. Blocks are taken at every offset, so a
64x64x32 volume gives 63*63*31 = 123,039 of them.

Compared with Jensen-Shannon rather than KL: KL is asymmetric, so the answer
would depend on argument order, and it is infinite as soon as one side holds a
pattern the other never produced, which with 256 bins always happens.
"""
import numpy as np
from .. import metrics as M
from ._base import ALL_TASKS, sum_merge

NAME, TASKS, NEEDS = 'patterns', ALL_TASKS, 'samples'


def summarize(vols, ctx=None):
    h = np.zeros(256, np.int64)
    for vol in np.asarray(vols):
        h += M.mps_hist(vol)
    return {'hist': h}


def merge(parts):
    return sum_merge(parts, ['hist'])


def compare(model, ref):
    return {'parts': {'pattern_jsd': M.jsd_bits(model['hist'], ref['hist'])}}
