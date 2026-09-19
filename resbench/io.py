"""Reading submissions and references off disk.

A stack of volumes is an .npz holding `ids` and `volumes`. Nothing here loads
a checkpoint or imports model code: ResBench only ever sees saved arrays,
which is what lets any framework be scored.

Large ensembles may be split across several .npz files in a directory. Every
check summarizes each file and merges, so a directory is never held in memory
all at once.
"""
import json
from pathlib import Path

import numpy as np

ENVIRONMENTS = ('lobe', 'channel:PV_SHOESTRING', 'channel:CB_LABYRINTH',
                'channel:CB_JIGSAW', 'channel:SH_DISTAL', 'channel:SH_PROXIMAL',
                'channel:MEANDER_OXBOW', 'delta')
SLUG = {e: e.replace(':', '_') for e in ENVIRONMENTS}
TASKS = ('unconditional', 'well_conditioned', 'field_scale')

# Field extent per environment. Channels are elongated along flow because that
# is how a channel belt extends; a square box clips every channel at the same
# length. Cell size is each environment's own and never changes.
FIELD_EXTENT = {e: ((512, 512, 32) if e in ('lobe', 'delta') else (512, 64, 32))
                for e in ENVIRONMENTS}
NATIVE_SHAPE = (64, 64, 32)


class SubmissionError(Exception):
    """Something about the submission is wrong, with a message saying what."""


def load_stack(path):
    z = np.load(path, allow_pickle=True)
    if 'volumes' not in z:
        raise SubmissionError(f'{path}: no "volumes" array')
    vols = z['volumes']
    ids = z['ids'] if 'ids' in z else np.arange(len(vols)).astype(str)
    if len(ids) != len(vols):
        raise SubmissionError(f'{path}: {len(ids)} ids for {len(vols)} volumes')
    return np.asarray(ids, dtype=object), np.asarray(vols)


def iter_shards(directory):
    """Every .npz in a directory, sorted, as (ids, volumes)."""
    d = Path(directory)
    files = sorted(d.glob('*.npz'))
    if not files:
        raise SubmissionError(f'{d}: no .npz files')
    for f in files:
        yield load_stack(f)


def part_dir(root, task, env, kind):
    return Path(root) / task / SLUG[env] / kind


def check_shape(vols, expect, where):
    got = tuple(np.asarray(vols).shape[1:])
    if got != tuple(expect):
        raise SubmissionError(f'{where}: volumes are {got}, expected {tuple(expect)}')


def check_binary(vols, where):
    u = np.unique(np.asarray(vols)[:1])
    if not set(np.asarray(u).ravel().tolist()) <= {0, 1}:
        raise SubmissionError(f'{where}: values must be 0 (shale) and 1 (sand), '
                              f'found {sorted(set(u.ravel().tolist()))[:5]}')


def read_meta(root):
    p = Path(root) / 'submission.yaml'
    if not p.exists():
        return {}
    # A deliberately tiny reader: one `key: value` per line. Avoids adding a
    # YAML dependency for a file this simple.
    out = {}
    for line in p.read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line or ':' not in line:
            continue
        k, v = line.split(':', 1)
        out[k.strip()] = v.strip().strip('"\'')
    return out


def load_manifest(reference_root, env):
    """Rows of `reference/manifest.csv` for one environment."""
    p = Path(reference_root) / 'manifest.csv'
    if not p.exists():
        return None
    import csv
    with open(p, newline='') as fh:
        return [r for r in csv.DictReader(fh) if r.get('environment') == env]


def load_targets(reference_root, env):
    """{volume id -> conditioned sand fraction} for one environment.

    The target is the reference volume's REALIZED sand fraction, the `ntg`
    column, because that is the number handed to the model as a condition.
    ResMill's own input is preserved separately as `requested_ntg` and is NOT
    what the model sees; the two differ by about 0.02 on average, which is
    larger than any model's error, so using the wrong one would measure the
    engine's targeting error instead of the model's.
    """
    rows = load_manifest(reference_root, env)
    if rows is None:
        return None
    out = {}
    for r in rows:
        vid, ntg = r.get('id'), r.get('ntg')
        if vid is not None and ntg not in (None, ''):
            out[str(vid)] = float(ntg)
    return out


def targets_for(ids, table, where):
    """Line a shard's ids up with their targets, in the shard's own order."""
    missing = [str(i) for i in ids if str(i) not in table]
    if missing:
        raise SubmissionError(
            f'{where}: no conditioned sand fraction for {len(missing)} ids '
            f'(first: {missing[:3]}). The submission and the reference must '
            f'key into the same manifest.')
    return np.array([table[str(i)] for i in ids], dtype=np.float64)
