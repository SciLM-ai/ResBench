"""Write a valid ResBench submission for one environment, filled with random
volumes, so you can see the layout and run `resbench validate` on it.

    python example/make_example.py OUT_DIR --reference REF [--env lobe]

The ids and the well patterns are read from the reference, which is the only
way to get them right: samples share the reference volumes' ids, repeat
conditions are cond0..cond4 and well1..well5, and fields carry the manifest's
field ids. Everything else is noise, so scoring it is meaningless -- the point
is that `validate` passes and every file is where the harness looks for it.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resbench import io  # noqa: E402


def blobs(n, shape, rng, p=0.4):
    from scipy import ndimage
    v = rng.random((n,) + shape)
    return (ndimage.uniform_filter(v, size=(1, 3, 3, 3)) < p).astype(np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--reference', required=True)
    ap.add_argument('--env', default='lobe'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    ref, out, env = Path(a.reference), Path(a.out), a.env
    slug, rng = io.SLUG[env], np.random.default_rng(a.seed)
    rows = [r for r in csv.DictReader(open(ref / 'manifest.csv', newline='')) if r['environment'] == env]
    vol_ids = [r['id'] for r in rows if r['task'] == 'unconditional']
    field_ids = [r['id'] for r in rows if r['task'] == 'field_scale']
    wells = {r['id'].split('|')[1]: r for r in rows if r['task'] == 'repeats_well'}

    for task in ('unconditional', 'well_conditioned'):
        d = out / task / slug / 'samples'; d.mkdir(parents=True, exist_ok=True)
        v = blobs(len(vol_ids), io.NATIVE_SHAPE, rng)
        if task == 'well_conditioned':
            # honor each reference volume's borehole exactly, as the rules require
            refv = np.load(ref / 'volumes' / slug / 'volumes.npz', allow_pickle=True)
            byid = dict(zip([str(i) for i in refv['ids']], refv['volumes']))
            for k, r in enumerate(r for r in rows if r['task'] == 'unconditional'):
                x, y = int(r['well_x']), int(r['well_y'])
                v[k, x, y, :] = byid[r['id']][x, y, :]
        np.savez_compressed(d / 'shard00.npz', ids=np.array(vol_ids, dtype=object), volumes=v)
    d = out / 'unconditional' / slug / 'repeats'; d.mkdir(parents=True, exist_ok=True)
    for cond in io.CONDITIONS['unconditional']:
        np.savez_compressed(d / f'{cond}.npz', ids=np.array([f'{cond}|{k}' for k in range(io.K_REPEATS)], dtype=object),
                            volumes=blobs(io.K_REPEATS, io.NATIVE_SHAPE, rng))
    d = out / 'well_conditioned' / slug / 'repeats'; d.mkdir(parents=True, exist_ok=True)
    for cond in io.CONDITIONS['well_conditioned']:
        w = wells[cond]; x, y = int(w['well_x']), int(w['well_y'])
        pat = np.array([int(c) for c in w['pattern']], np.int8)
        v = blobs(io.K_REPEATS, io.NATIVE_SHAPE, rng); v[:, x, y, :] = pat
        np.savez_compressed(d / f'{cond}.npz', ids=np.array([f'{cond}|{k}' for k in range(io.K_REPEATS)], dtype=object), volumes=v)
    d = out / 'field_scale' / slug / 'fields'; d.mkdir(parents=True, exist_ok=True)
    ext = io.FIELD_EXTENT[env]
    for s in range(0, len(field_ids), 4):
        ids = field_ids[s:s + 4]
        np.savez_compressed(d / f'shard{s // 4:02d}.npz', ids=np.array(ids, dtype=object),
                            volumes=blobs(len(ids), ext, rng))
    (out / 'submission.yaml').write_text('model: example\nsampler: none\nnfe: 0\ntraining_set: none\n')
    print(f'wrote {out}: run  resbench validate {out} --envs {env}')


if __name__ == '__main__':
    main()
