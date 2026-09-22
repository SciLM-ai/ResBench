"""Score the reference against itself.

Builds a submission out of REF (every reference volume, field and repeat
ensemble, in the submission layout) and runs `resbench validate` and
`resbench score` on it. Every samples-based check must come out at 0; the
repeats-based checks compare the first 128 members with the whole ensemble
and land well below 1 (well ensembles shorter than 128 are cycled). Anything
else means the reference, the manifest and the checks disagree.

    python tools/reference_self_check.py REF OUT_DIR
"""
import subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resbench import io  # noqa: E402


def build(ref, sub):
    envs = [e for e in io.ENVIRONMENTS if (ref / 'volumes' / io.SLUG[e] / 'volumes.npz').exists()]
    for env in envs:
        slug = io.SLUG[env]
        z = np.load(ref / 'volumes' / slug / 'volumes.npz', allow_pickle=True)
        for task in ('unconditional', 'well_conditioned'):
            d = sub / task / slug / 'samples'; d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / 'all.npz', ids=z['ids'], volumes=z['volumes'])
        for task, conds in io.CONDITIONS.items():
            for cond in conds:
                f = ref / 'repeats' / slug / f'{cond}.npz'
                if not f.exists():
                    print(f'  no reference repeats {slug}/{cond}; that part is skipped')
                    continue
                v = np.load(f, allow_pickle=True)['volumes']
                d = sub / task / slug / 'repeats'; d.mkdir(parents=True, exist_ok=True)
                idx = np.arange(io.K_REPEATS) % len(v)
                np.savez_compressed(d / f'{cond}.npz', volumes=v[idx],
                                    ids=np.array([f'{cond}|{k}' for k in range(io.K_REPEATS)], dtype=object))
        f = ref / 'fields' / slug / 'fields.npz'
        if f.exists():
            zf = np.load(f, allow_pickle=True)
            d = sub / 'field_scale' / slug / 'fields'; d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / 'all.npz', ids=zf['ids'], volumes=zf['volumes'])
        else:
            print(f'  no reference fields for {env}; that part is skipped')
    print(f'self-submission built for {len(envs)} environments at {sub}', flush=True)


def main():
    ref, sub = Path(sys.argv[1]), Path(sys.argv[2])
    build(ref, sub)
    here = Path(__file__).resolve().parents[1]
    for args in (['validate', str(sub)], ['score', str(sub), '--reference', str(ref), '--out', str(sub / 'results.json')]):
        print(f'\n$ resbench {" ".join(args)}', flush=True)
        r = subprocess.run([sys.executable, '-m', 'resbench.cli'] + args, cwd=here, text=True, capture_output=True)
        print(r.stdout); print(r.stderr[-4000:], file=sys.stderr)


if __name__ == '__main__':
    main()
