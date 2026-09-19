"""Of the cells the model calls sand 70% of the time, are 70% actually sand?

Same repeats and the same probability map as `variety`, asking a different
question of it. Sort every off-well cell into ten bins by the model's stated
probability; for each bin compare the model's average claim against ResMill's
actual sand frequency in those same cells; average the gaps weighted by bin
population.

This is NOT `variety` again. Entropy is symmetric -- H(0.3) = H(0.7) = 0.8813
bits -- so a cell where the model says 0.3 and ResMill says 0.7 scores ZERO
entropy error. A model that inverted every probability in the volume would
pass `variety` perfectly; measured on 200,000 cells it scores 0.000 bits there
and 0.375 here. `variety` asks how much uncertainty there is, calibration asks
whether it points the right way.

Well cells are excluded: hard replacement makes them trivially calibrated.
"""
import numpy as np
from .. import metrics as M

NAME, TASKS, NEEDS = 'calibration', ('unconditional', 'well_conditioned'), 'repeats'
BINS = 10


def summarize(vols, ctx=None):
    ctx = ctx or {}
    cid = str(ctx.get('condition_id', '0'))
    mask = ctx.get('well_mask')
    return {'maps': {cid: (M.prob_map(vols),
                           None if mask is None else np.asarray(mask) > 0)}}


def merge(parts):
    out = {}
    for p in parts:
        out.update(p['maps'])
    return {'maps': out}


def compare(model, ref):
    shared = sorted(set(model['maps']) & set(ref['maps']))
    if not shared:
        return {'parts': {'ece': float('nan')}}
    vals = []
    for c in shared:
        pm, mask = model['maps'][c]
        pr, _ = ref['maps'][c]
        vals.append(M.expected_calibration_error(pm, pr, mask=mask, bins=BINS))
    return {'parts': {'ece': float(np.mean(vals))}}
