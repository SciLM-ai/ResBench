"""One manifest.csv, written by several builders.

Each builder owns one `task` value and replaces only its own rows, so
building the field reference never discards the unconditional rows and
vice versa. Columns are the union across tasks; a task leaves blank the
columns it has no use for. Ids must be unique across the whole file, because
`io.load_targets` looks volumes up by id alone.
"""
import csv
from pathlib import Path

COLUMNS = ['task', 'environment', 'id', 'row_index', 'source_id', 'shard_dir',
           'sample_idx', 'ntg', 'ntg_source_cube', 'requested_ntg', 'azimuth',
           'width_cells', 'depth_cells', 'asp', 'mCHsinu', 'mFFCHprop',
           'probAvulInside', 'trunk_length_fraction', 'noise_seed',
           'well_x', 'well_y', 'pattern']


def read(ref):
    p = Path(ref) / 'manifest.csv'
    if not p.exists():
        return []
    with open(p, newline='') as fh:
        return list(csv.DictReader(fh))


def upsert(ref, task, rows):
    """Replace all rows of `task` with `rows`; keep every other task's rows."""
    keep = [r for r in read(ref) if r.get('task') != task]
    new = []
    for r in rows:
        r = dict(r); r['task'] = task
        new.append({c: ('' if r.get(c) is None else r.get(c, '')) for c in COLUMNS})
    allrows = keep + new
    ids = [r['id'] for r in allrows]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise SystemExit(f'manifest ids must be unique; duplicated: {sorted(dup)[:5]}')
    Path(ref).mkdir(parents=True, exist_ok=True)
    with open(Path(ref) / 'manifest.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in allrows:
            w.writerow({c: r.get(c, '') for c in COLUMNS})
    return len(new), len(keep)
