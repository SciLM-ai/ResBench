"""Relabel a submission's field ids from row index to manifest id.

Generators that predate the manifest wrote ids as `field|<slug>|<i>` with
i = row index into pick_rows(env, n). The reference and manifest use
`field|<slug>|<SEED_BASE + 1000*ENVS.index(env) + i>`. This rewrites the
ids in place, or into a copy with --out, and refuses to run twice.

    python tools/relabel_submission.py SUBMISSION_DIR [--out DIR]
"""
import argparse, shutil, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.gen_field_reference import ENVS, SEED_BASE  # noqa: E402

SLUG = {e.replace(':', '_'): e for e in ENVS}


def relabel(path):
    z = np.load(path, allow_pickle=True)
    ids = [str(i) for i in z['ids']]
    new = []
    for i in ids:
        head, slug, tail = i.split('|')
        env = SLUG[slug]
        base = SEED_BASE + 1000 * ENVS.index(env)
        k = int(tail)
        if k >= base:                        # already a manifest id
            return 0
        new.append(f'{head}|{slug}|{base + k}')
    np.savez_compressed(path, ids=np.array(new, dtype=object), volumes=z['volumes'])
    return len(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('submission')
    ap.add_argument('--out', help='write a relabelled copy here instead of editing in place')
    a = ap.parse_args()
    src = Path(a.submission)
    dst = Path(a.out) if a.out else src
    if a.out:
        if dst.exists():
            raise SystemExit(f'{dst} exists')
        shutil.copytree(src, dst)
    n_files = n_ids = 0
    for f in sorted(dst.rglob('*.npz')):
        n = relabel(f); n_files += 1; n_ids += n
    print(f'{dst}: {n_files} files, {n_ids} ids relabelled'
          + ('' if n_ids else ' (already manifest ids; nothing changed)'))


if __name__ == '__main__':
    main()
