"""Given identical inputs, does the model vary as much as ResMill does?

A model that returns the same volume every time for the same parameters
passes every structural check and is useless, because the whole point of
stochastic reservoir modeling is a range of outcomes to drill against.
Nothing else catches it: the other checks pool across 512 DIFFERENT
conditions, and a model that is deterministic per condition but responsive
across them looks fine in all of them.

Run the model 128 times on one fixed input, count how often each cell came out
sand, and turn that into entropy in bits: a cell that is the same rock every
time scores 0, a coin-flip cell scores 1.

Without a well that map is nearly flat. Measured on 256 ResMill runs of one
lobe parameter vector, mean entropy is 0.98 with a standard deviation of 0.02
and not one cell in 131,072 is reliably sand or shale. Nothing is pinned down,
so the useful signal is the overall LEVEL, not any structure: ResMill sits near
the maximum and a collapsing model falls away from it. The structure only
appears once a well is present, which is `well_blending`'s job.
"""
import numpy as np
from .. import metrics as M

NAME, TASKS, NEEDS = 'variety', ('unconditional',), 'repeats'


def summarize(vols, ctx=None):
    cid = str((ctx or {}).get('condition_id', '0'))
    return {'maps': {cid: M.prob_map(vols)}}


def merge(parts):
    out = {}
    for p in parts:
        out.update(p['maps'])
    return {'maps': out}


def compare(model, ref):
    shared = sorted(set(model['maps']) & set(ref['maps']))
    if not shared:
        return {'parts': {'entropy_mae': float('nan')}}
    vals = [float(np.abs(M.entropy_map_from_p(model['maps'][c])
                         - M.entropy_map_from_p(ref['maps'][c])).mean())
            for c in shared]
    return {'parts': {'entropy_mae': float(np.mean(vals))}}
