"""Assemble the ResBench v1 reference directory and write its manifest.

    python tools/build_reference.py --out REF \\
        --fields lobe=/path/to/lobe_dir  \\
        --fields all=/path/to/dir_with_one_subdir_per_env

Each source is a directory holding `fields.npz` (ids + volumes) and
`manifest.json`, as written by tools/gen_field_reference.py, or a parent
directory of such per-environment subdirectories (`all=`). Later `--fields`
entries override earlier ones for the same environment.

Writes REF/fields/<slug>/fields.npz (one shard per environment; `band_for`
reads only the first shard) and REF/manifest.csv with one row per field:

    environment, id, ntg, ntg_source_cube, requested_ntg, azimuth

`ntg` is the reference FIELD's realized sand fraction, which is the number a
submitter must condition on. It differs from the source 64-cube's realized
value (`ntg_source_cube`) by ~0.02 on average -- twice net_to_gross's
tolerance -- and from ResMill's `requested_ntg` input as well, so publishing
it is what makes the check measure the model rather than the engine.
"""
import argparse, csv, json, shutil, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.gen_field_reference import ENVS, SEED_BASE, pick_rows, extent_for  # noqa: E402

SLUG = {e: e.replace(':', '_') for e in ENVS}
UNSLUG = {v: k for k, v in SLUG.items()}


def read_source(path):
    """{slug: (fields.npz path, manifest dict)} from a per-env dir or a parent."""
    p = Path(path)
    if (p / 'fields.npz').exists():
        return {p.name: (p / 'fields.npz', json.loads((p / 'manifest.json').read_text()))}
    out = {}
    for d in sorted(p.iterdir()):
        if (d / 'fields.npz').exists():
            out[d.name] = (d / 'fields.npz', json.loads((d / 'manifest.json').read_text()))
    return out


def manifest_rows(slug, npz, expect):
    env = UNSLUG[slug]
    z = np.load(npz, allow_pickle=True)
    ids, vols = [str(i) for i in z['ids']], z['volumes']
    base = SEED_BASE + 1000 * ENVS.index(env)
    ks = [int(i.rsplit('|', 1)[1]) - base for i in ids]
    if len(set(ks)) != len(ks) or min(ks) < 0 or max(ks) >= expect:
        raise SystemExit(f'{slug}: ids must be distinct seeds in '
                         f'{base}..{base + expect - 1}, got {ids[:3]}')
    # Field k pairs with pick_rows(env, EXPECT)[k]: the row selection is fixed
    # by the n the generator was run with, not by how many fields happen to be
    # present. A 7-field partial keyed into pick_rows(env, 7) would be
    # silently matched to the wrong rows.
    rows = pick_rows(env, expect)
    out = []
    for i, k in enumerate(ks):
        r = rows[k]
        req = r.get('NTGtarget')
        if req is None:
            req = r.get('requested_ntg')
        out.append({
            'environment': env,
            'id': ids[i],
            'ntg': float(vols[i].mean()),                       # the condition
            'ntg_source_cube': float(r['ntg']),                 # the 64-cube's realized value
            'requested_ntg': '' if req is None else float(req), # ResMill's own input
            'azimuth': 0.0 if env.startswith('channel:') else float(r.get('azimuth', 0.0)),
        })
    ex = extent_for(env)
    if tuple(vols.shape[1:]) != (ex[0], ex[1], 32):
        raise SystemExit(f'{slug}: fields are {vols.shape[1:]}, spec says {(ex[0], ex[1], 32)}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--fields', action='append', default=[],
                    help='<slug>=<dir> or all=<parent dir>; repeatable')
    ap.add_argument('--expect', type=int, default=32,
                    help='fields per environment the reference is supposed to have')
    ap.add_argument('--allow-partial', action='store_true',
                    help='assemble environments with fewer than --expect fields anyway')
    a = ap.parse_args()
    out = Path(a.out)
    sources = {}
    for spec in a.fields:
        key, _, path = spec.partition('=')
        found = read_source(path)
        if key != 'all':
            found = {k: v for k, v in found.items() if k == key} or \
                    ({key: found[Path(path).name]} if Path(path).name in found else {})
        sources.update(found)
    if not sources:
        raise SystemExit('no fields found')

    rows = []
    for slug, (npz, man) in sorted(sources.items(), key=lambda kv: ENVS.index(UNSLUG[kv[0]])):
        dst = out / 'fields' / slug
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(npz, dst / 'fields.npz')
        (dst / 'manifest.json').write_text(json.dumps(man, indent=2))
        r = manifest_rows(slug, npz, a.expect)
        # Completeness comes from the count, never from a flag: a generator
        # interrupted mid-run writes checkpoints whose manifest.json may not
        # carry one.
        complete = len(r) >= a.expect
        if not complete and not a.allow_partial:
            raise SystemExit(f'{slug}: {len(r)} fields, expected {a.expect}; '
                             f'pass --allow-partial to assemble it anyway')
        rows += r
        print(f"{slug:<26} {len(r):>3} fields  {'complete' if complete else 'PARTIAL '}"
              f"  mean ntg {np.mean([x['ntg'] for x in r]):.4f}  "
              f"(source cube {np.mean([x['ntg_source_cube'] for x in r]):.4f})")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'manifest.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['environment', 'id', 'ntg', 'ntg_source_cube',
                                           'requested_ntg', 'azimuth'])
        w.writeheader(); w.writerows(rows)
    print(f'wrote {out / "manifest.csv"}: {len(rows)} rows, {len(sources)} environments')


if __name__ == '__main__':
    main()
