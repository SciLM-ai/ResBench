"""Turn distances into one number.

    band  = check( half of ResMill , the other half )
    s     = check( your model , ResMill )  /  band
    score = average of s over every check, task and environment

s = 1 means your model is as close to ResMill as ResMill is to itself. Lower
is better and 1.0 is the floor, not zero: you cannot beat ResMill at being
ResMill.

The aggregation is a plain AVERAGE, not a median. Measured on 47 scored
checkpoints the two rank models at a correlation of -0.63 -- the order does
not shift, it reverses, and it reverses the wrong way: the median's top model
misses one check by 5.9x its band and another by 8.0x while the model that is
good everywhere lands at rank 16. A median rewards passing a bare majority and
blowing the rest, and every check is a real way for a reservoir to be wrong.

A check with two sub-parts (compartments, chord_lengths) bands each part on
its own and averages them, so it still contributes exactly one number.
"""
import numpy as np

MATCHED, CLOSE = 1.0, 2.0


def verdict(s):
    if not np.isfinite(s):
        return 'n/a'
    return 'matched' if s <= MATCHED else ('close' if s <= CLOSE else 'distinguishable')


def s_for_check(distances, bands):
    """Mean of value/band over a check's parts."""
    vals = []
    for k, v in distances.items():
        b = bands.get(k)
        if b is None or not np.isfinite(v):
            continue
        vals.append(v / b if b > 0 else (0.0 if v == 0 else np.inf))
    return float(np.mean(vals)) if vals else float('nan')


def aggregate(rows):
    """rows: list of {task, environment, check, s}. Returns the score tree."""
    rows = [r for r in rows if np.isfinite(r['s'])]
    out = {'tasks': {}, 'overall': float('nan'), 'matched': [0, 0]}
    per_task = {}
    for r in rows:
        per_task.setdefault(r['task'], []).append(r)
    for task, rs in per_task.items():
        by_check = {}
        for r in rs:
            by_check.setdefault(r['check'], []).append(r['s'])
        means = {c: float(np.mean(v)) for c, v in by_check.items()}
        inside = sum(1 for v in means.values() if v <= MATCHED)
        worst = max(means, key=means.get) if means else None
        out['tasks'][task] = {
            'score': float(np.mean(list(means.values()))) if means else float('nan'),
            'matched': [inside, len(means)],
            'weakest': worst,
            'weakest_s': means.get(worst, float('nan')),
            'checks': means,
        }
        out['matched'][0] += inside
        out['matched'][1] += len(means)
    if out['tasks']:
        out['overall'] = float(np.mean([t['score'] for t in out['tasks'].values()]))
    return out


def format_table(tree, header=''):
    lines = [header, ''] if header else []
    lines.append(f"{'':<22}{'score':>7}{'matched':>10}   weakest")
    for task in ('unconditional', 'well_conditioned', 'field_scale'):
        t = tree['tasks'].get(task)
        if not t:
            continue
        m = f"{t['matched'][0]}/{t['matched'][1]}"
        lines.append(f"  {task:<20}{t['score']:>7.2f}{m:>10}   "
                     f"{t['weakest'] or '':<16}{t['weakest_s']:.1f}")
    lines.append('')
    m = f"{tree['matched'][0]}/{tree['matched'][1]}"
    lines.append(f"  {'overall':<20}{tree['overall']:>7.2f}{m:>10}")
    return '\n'.join(lines)
