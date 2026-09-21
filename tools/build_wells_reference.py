"""Put the well ensembles into the v1 reference and write their manifest rows.

Each environment has five published wells: a vertical borehole pattern at a
map location (x, y), and an ensemble of ResMill realizations from ONE
parameter row whose column at (x, y) matches the pattern exactly. This
converts them from the old `references/well_conditional/<slug>_well<i>.npz`
layout into `REF/repeats/<slug>/well<i>.npz` and adds task=repeats_well rows
to REF/manifest.csv, with `source_id` naming the parameter row the ensemble
was drawn from, so a submitter knows exactly what to condition on.

Nothing is trusted from file names. For every ensemble the script requires:
  * pattern == volume[x, y, :] for every member,
  * no duplicate members,
  * well_mask is the one-hot column at (x, y),
  * at least --min-n members unless --allow-short,
  * every batch manifest of the environment's draw pool names the SAME
    source row, which must exist in the unconditional manifest.

    python tools/build_wells_reference.py --ref REF [--envs a,b] [--wells-dir DIR]
"""
import argparse, glob, hashlib, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resbench.io import ENVIRONMENTS, SLUG, CONDITIONS  # noqa: E402
from tools._manifest import read, upsert                 # noqa: E402

PUBLISHED = Path('/work/08405/ilgar/vista/codes/ResBench-public/references/well_conditional')
POOLS = Path('/scratch/08405/ilgar/resbench_eval/deep_draws')
MIN_N = 50


def source_row_index(env):
    """The one parameter row every batch of this environment's pool used."""
    rows = set()
    for j in glob.glob(str(POOLS / SLUG[env] / '**' / 'resmill_ref_manifest*.json'), recursive=True):
        m = json.load(open(j))
        if m.get('envs') != env:
            raise SystemExit(f'{j}: batch is for {m.get("envs")!r}, not {env!r}')
        a, b = (int(x) for x in m['rows'].split(':'))
        if b != a + 1:
            raise SystemExit(f'{j}: batch covers rows {m["rows"]}, expected a single row')
        rows.add(a)
    if len(rows) != 1:
        raise SystemExit(f'{env}: pool batches disagree on the source row: {sorted(rows)}')
    return rows.pop()


def hdigest(v):
    return hashlib.blake2b(np.ascontiguousarray(v).tobytes(), digest_size=16).digest()


def check_ensemble(name, vols, pattern, mask, xy, min_n, allow_short):
    x, y = int(xy[0]), int(xy[1])
    if vols.ndim != 4 or tuple(vols.shape[1:]) != (64, 64, 32):
        raise SystemExit(f'{name}: volumes are {vols.shape}')
    if not (vols[:, x, y, :] == pattern).all():
        raise SystemExit(f'{name}: a member does not honour the pattern at ({x},{y})')
    if len({hdigest(v) for v in vols}) != len(vols):
        raise SystemExit(f'{name}: duplicate members')
    expect = np.zeros((64, 64, 32), np.uint8); expect[x, y, :] = 1
    if not np.array_equal(np.asarray(mask).astype(np.uint8), expect):
        raise SystemExit(f'{name}: well_mask is not the one-hot column at ({x},{y})')
    if len(vols) < min_n and not allow_short:
        raise SystemExit(f'{name}: {len(vols)} members < {min_n}; pass --allow-short or top up the pool')
    return len(vols) >= min_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--envs', default=','.join(ENVIRONMENTS))
    ap.add_argument('--wells-dir', default=None,
                    help='directory of mined wells (<slug>_well<i>.npz from tools/mine_wells.py); an '
                         'environment whose files are there is read from it, the rest from the published wells')
    ap.add_argument('--min-n', type=int, default=MIN_N)
    ap.add_argument('--allow-short', action='store_true')
    a = ap.parse_args()
    ref = Path(a.ref)
    uncond = {(r['environment'], int(r['row_index'])): r
              for r in read(ref) if r['task'] == 'unconditional'}
    if not uncond:
        raise SystemExit('no unconditional rows in the manifest; run build_unconditional_reference.py first')

    rows, kept = [], [r for r in read(ref) if r['task'] == 'repeats_well']
    kept_envs = {r['environment'] for r in kept}
    for env in (e for e in a.envs.split(',') if e):
        mined = a.wells_dir and (Path(a.wells_dir) / f'{SLUG[env]}_well1.npz').exists()
        ri = None
        d = ref / 'repeats' / SLUG[env]; d.mkdir(parents=True, exist_ok=True)
        for i, cond in enumerate(CONDITIONS['well_conditioned'], start=1):
            f = (Path(a.wells_dir) / f'{SLUG[env]}_well{i}.npz') if mined else PUBLISHED / f'{SLUG[env]}_well{i}.npz'
            z = np.load(f)
            vols, pat, mask, xy = z['volumes'].astype(np.int8), z['pattern'].astype(np.int8), z['well_mask'], z['well_xy']
            # the source row: recorded in a mined file, or the one row every batch of the published pool used
            ri_f = int(z['cond_row_index']) if 'cond_row_index' in z.files else source_row_index(env)
            if ri is None:
                ri = ri_f
                src = uncond.get((env, ri))
                if src is None:
                    raise SystemExit(f'{env}: source row {ri} is not in the unconditional manifest')
            elif ri_f != ri:
                raise SystemExit(f'{f.name}: source row {ri_f}, but well1 used row {ri}')
            ok = check_ensemble(f.name, vols, pat, mask, xy, a.min_n, a.allow_short)
            np.savez_compressed(d / f'{cond}.npz',
                                ids=np.array([f'{cond}|{k}' for k in range(len(vols))], dtype=object),
                                volumes=vols, pattern=pat, well_mask=mask.astype(np.uint8),
                                well_xy=np.asarray(xy, dtype=np.int64), source_id=np.array(src['id']))
            row = {k: src[k] for k in ('environment', 'row_index', 'shard_dir', 'sample_idx', 'ntg',
                                      'requested_ntg', 'azimuth', 'width_cells', 'depth_cells', 'asp',
                                      'mCHsinu', 'mFFCHprop', 'probAvulInside', 'trunk_length_fraction')}
            row.update({'id': f'{env}|{cond}', 'source_id': src['id'],
                        'well_x': int(xy[0]), 'well_y': int(xy[1]),
                        'pattern': ''.join(str(int(c)) for c in pat)})
            rows.append(row)
            print(f"{env:<24} {cond}  ({int(xy[0]):>2},{int(xy[1]):>2})  n={len(vols):>3}  "
                  f"{'ok' if ok else 'SHORT'}   source row {ri} = {src['id']}")
        kept_envs.discard(env)
    # keep other environments' rows that this run did not touch
    rows += [r for r in kept if r['environment'] in kept_envs]
    n_new, n_kept = upsert(ref, 'repeats_well', rows)
    print(f'manifest.csv: {n_new} repeats_well rows, {n_kept} rows of other tasks kept')


if __name__ == '__main__':
    main()
