"""Volume-directory loading (EVAL.md §8 input contract)."""
from pathlib import Path

import numpy as np

from . import LAYER_TYPES, env_slug


def load_volume_dir(path, key='volumes'):
    """Load one environment directory of `*_*.npz` shards.

    Returns (ids, array) with rows concatenated in sorted-filename order.
    """
    files = sorted(Path(path).glob('*.npz'))
    if not files:
        raise FileNotFoundError(f'no .npz shards under {path}')
    ids, arrs = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        ids += [str(i) for i in d['ids']]
        arrs.append(d[key])
    return np.array(ids), np.concatenate(arrs)


def load_ensemble(root, key='volumes'):
    """Load an ensemble root containing one subdir per environment slug.

    Returns {layer_type: (ids, volumes)} in canonical environment order,
    restricted to environments present under root.
    """
    root = Path(root)
    out = {}
    for lt in LAYER_TYPES:
        d = root / env_slug(lt)
        if d.is_dir():
            out[lt] = load_volume_dir(d, key=key)
    if not out:
        raise FileNotFoundError(f'no environment subdirs under {root}')
    return out


def align(ids_a, vols_a, ids_b, vols_b):
    """Row-align two (ids, volumes) pairs on shared ids (order of a).

    Ids may carry a '#k' sample suffix on the prediction side; the join key
    strips it.
    """
    def base(i):
        return i.split('#')[0]

    idx_b = {base(i): j for j, i in enumerate(ids_b)}
    keep = [j for j, i in enumerate(ids_a) if base(i) in idx_b]
    sel_b = [idx_b[base(ids_a[j])] for j in keep]
    return vols_a[keep], vols_b[sel_b], [ids_a[j] for j in keep]
