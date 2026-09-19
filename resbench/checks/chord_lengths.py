"""How wide are the bodies where blobs are ambiguous, and how elongated?

ResMill builds a lobe, builds the next on top, and keeps going. Where two
lobes touch the sand between them is continuous, and ResMill never draws a
line marking where one stopped and the next began, because in real rock there
is none. Flood-fill therefore returns three merged lobes as one blob, so
`body_size` is measuring welded clumps. That is the geology, not a bug.

A chord sidesteps it: draw a line, measure each unbroken stretch of sand, with
no decision about where anything ends. It is `bed_thickness` turned on its
side, with three differences. The lines run horizontally within each depth
slice. They follow the direction the bodies point, rarely a grid axis, so the
volume is sampled between cell centers and rounded -- with nearest-neighbor
sampling a true 2.00 elongation measures as 1.58 and gets WORSE with a finer
step. And a chord running off the end of its line is dropped rather than
counted short, because its true length is unknowable.

The elongation ratio, mean chord along the azimuth over mean chord across it,
is the body's shape and is why this check exists.
"""
import numpy as np
from .. import anisotropy as A
from ._base import ALL_TASKS, sum_merge, hist_w1

NAME, TASKS, NEEDS = 'chord_lengths', ALL_TASKS, 'samples'
LOG_MIN, LOG_MAX, NBINS = 0.0, 4.0, 400
EDGES = np.linspace(LOG_MIN, LOG_MAX, NBINS + 1)


def _hist(chords):
    c = np.asarray(chords, float)
    c = c[c > 0]
    if not c.size:
        return np.zeros(NBINS, np.int64)
    return np.histogram(np.log10(c), bins=EDGES)[0]


def summarize(vols, ctx=None):
    az = float((ctx or {}).get('azimuth', 0.0))
    v = np.asarray(vols)
    along = A.ensemble_chords(v, az)
    across = A.ensemble_chords(v, az + 90.0)
    ma = float(np.mean(along)) if len(along) else float('nan')
    mp = float(np.mean(across)) if len(across) else float('nan')
    return {'hist': _hist(along), 'along_sum': ma * len(along),
            'along_n': len(along), 'across_sum': mp * len(across),
            'across_n': len(across)}


def merge(parts):
    out = sum_merge(parts, ['hist'])
    for k in ('along_sum', 'along_n', 'across_sum', 'across_n'):
        out[k] = float(sum(p[k] for p in parts))
    return out


def _ratio(s):
    a = s['along_sum'] / max(s['along_n'], 1)
    b = s['across_sum'] / max(s['across_n'], 1)
    return a / b if b > 0 else float('nan')


def compare(model, ref):
    return {'parts': {
        'chord_w1': hist_w1(model['hist'], ref['hist'], EDGES),
        'elongation': abs(_ratio(model) - _ratio(ref)),
    }}
