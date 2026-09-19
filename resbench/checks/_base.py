"""Shared shape for every check.

A check module exports:
    NAME     str
    TASKS    which of unconditional / well_conditioned / field_scale it runs in
    NEEDS    'samples' (one volume per condition) or 'repeats' (many per condition)
    summarize(vols, ctx) -> dict      mergeable summary of one stack
    merge(parts)         -> dict      combine summaries of disjoint stacks
    compare(model, ref)  -> dict      {'parts': {sub_name: distance}}

`compare` returns one distance per sub-quantity. Most checks have a single
part; `compartments` has two (gamma, euler) and `chord_lengths` has two
(length distribution, elongation ratio). The scorer bands each part on its own
and averages them, so a check with two parts still contributes one number.

Summaries merge across shards, so nothing ever needs a whole ensemble in
memory: `summarize` each file, `merge` the results, `compare` at the end.
"""
import numpy as np

ALL_TASKS = ('unconditional', 'well_conditioned', 'field_scale')
NATIVE_TASKS = ('unconditional', 'well_conditioned')


def sum_merge(parts, keys):
    """Merge by adding counts -- the right rule whenever a summary is a tally."""
    out = {}
    for k in keys:
        out[k] = np.sum([np.asarray(p[k]) for p in parts], axis=0)
    return out


def hist_w1(h_a, h_b, edges):
    """Wasserstein-1 between two distributions held as histograms.

    Histograms keep summaries mergeable and bounded in size; with fine bins
    this equals W1 on the raw samples to within a bin width.
    """
    a = np.asarray(h_a, float); b = np.asarray(h_b, float)
    if a.sum() == 0 or b.sum() == 0:
        return float('nan')
    ca, cb = np.cumsum(a / a.sum()), np.cumsum(b / b.sum())
    centers = 0.5 * (np.asarray(edges[:-1], float) + np.asarray(edges[1:], float))
    return float(np.sum(np.abs(ca - cb)[:-1] * np.diff(centers)))
