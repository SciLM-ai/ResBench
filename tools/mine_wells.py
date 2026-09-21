"""Mine the five well conditions of one environment with the checked-out ResMill.

The rule is the published one (ResBench-public EVAL.md, C.2). A candidate is a
column of the environment's reference volume for the source row, at an
interior location x, y in [8, 56], that is

    informative     at least two sand bodies in the column, separated by mud;
    representative  |column sand fraction - environment mean ntg| <= 0.15;
    estimable       at least --min-n distinct ResMill runs of the source row's
                    parameters reproduce the column exactly.

The five candidates with the most exact matches, distinct patterns, are the
wells. Pass 1 runs --draws fresh seeds of the source row and records which
candidate columns every run reproduces; pass 2 regenerates only the runs that
matched a chosen column (up to --cap) and writes

    OUT/<slug>_well<i>.npz   volumes, pattern, well_mask, well_xy, cond_row_index, seeds

which is what build_wells_reference.py --wells-dir reads. Seeds come from
default_rng([WELL_POOL_SEED, env_index, row_index]); the source row's own seed
is excluded so the reference volume never sits in its own ensemble.

Pass 1 is saved to OUT/<slug>_pool.npz (seeds, volume hashes, packed match
masks) and resumed: running again with a larger --draws only runs the seeds
not drawn yet, so a pool can be grown across allocations. MEANDER_OXBOW and
delta cost about 100 s a run, the other channels 2 to 15 s.

    python tools/mine_wells.py --ref REF --env channel:PV_SHOESTRING --row-index 13 \\
        --draws 60000 --out WELLS [--jobs 48]
"""
import argparse, csv, hashlib, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resbench.io import ENVIRONMENTS, SLUG                                   # noqa: E402
from tools.gen_repeats_reference import native_kwargs, load_row, CONFIG_DIR, CONFIG_FOR_ENV  # noqa: E402

WELL_POOL_SEED = 20260922
INTERIOR = (8, 56)          # inclusive, the published mining window
MIN_N = 50
CAP = 256
NTG_TOL = 0.15
LO, HI = INTERIOR[0], INTERIOR[1] + 1
_REF = None                 # reference volume, set once per worker


def hdigest(v):
    return hashlib.blake2b(np.ascontiguousarray(v).tobytes(), digest_size=16).digest()


def informative(col):
    """At least two sand bodies separated by mud."""
    c = np.asarray(col, np.int8)
    return int(((c[1:] == 1) & (c[:-1] == 0)).sum() + (c[0] == 1)) >= 2


def _init(ref):
    global _REF
    _REF = ref


def _run(task):
    env, row, seed = task
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from resmill.dataset.generate import generate_sample
    grid = json.loads((CONFIG_DIR / CONFIG_FOR_ENV[env]).read_text())['grid']
    f, _, _, _, _ = generate_sample({'layer_type': env.split(':')[0],
                                     'params': native_kwargs(row, env), 'seed': int(seed)}, grid)
    return np.asarray(f, np.int8)


def _pass1(task):
    f = _run(task)
    hit = (f[LO:HI, LO:HI, :] == _REF[LO:HI, LO:HI, :]).all(axis=2)
    return int(task[2]), hdigest(f), np.packbits(hit.ravel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--env', required=True, choices=ENVIRONMENTS)
    ap.add_argument('--row-index', type=int, required=True, help='unconditional row of the source volume')
    ap.add_argument('--draws', type=int, default=60000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=48)
    ap.add_argument('--min-n', type=int, default=MIN_N)
    ap.add_argument('--cap', type=int, default=CAP)
    a = ap.parse_args()
    ref, out, env = Path(a.ref), Path(a.out), a.env
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'STOP').exists():
        raise SystemExit(f'{out}/STOP exists: not mining {env}')

    rows = [r for r in csv.DictReader(open(ref / 'manifest.csv', newline=''))
            if r['task'] == 'unconditional' and r['environment'] == env]
    src = next(r for r in rows if int(r['row_index']) == a.row_index)
    env_ntg = float(np.mean([float(r['ntg']) for r in rows]))
    z = np.load(ref / 'volumes' / SLUG[env] / 'volumes.npz', allow_pickle=True)
    vol = z['volumes'][[str(i) for i in z['ids']].index(src['id'])].astype(np.int8)
    row = load_row(src['shard_dir'], src['sample_idx'])
    print(f'{env}: source row {a.row_index} = {src["id"]}  volume ntg {vol.mean():.3f}  '
          f'environment mean ntg {env_ntg:.3f}', flush=True)

    # candidate columns of the reference volume
    cand = np.zeros((HI - LO, HI - LO), bool)
    for x in range(LO, HI):
        for y in range(LO, HI):
            col = vol[x, y, :]
            cand[x - LO, y - LO] = informative(col) and abs(col.mean() - env_ntg) <= NTG_TOL
    print(f'{int(cand.sum())} informative, representative candidate columns of {cand.size}', flush=True)

    rng = np.random.default_rng([WELL_POOL_SEED, ENVIRONMENTS.index(env), a.row_index])
    seeds = []
    for s in rng.integers(1, 2**31 - 1, size=a.draws + 64):
        if int(s) != int(row['seed']) and int(s) not in seeds:
            seeds.append(int(s))
        if len(seeds) == a.draws:
            break
    # pass 1: which candidate columns does every run reproduce; resumable
    pool_file = out / f'{SLUG[env]}_pool.npz'
    done = {}                       # seed -> (hash, packed mask)
    if pool_file.exists():
        z0 = np.load(pool_file, allow_pickle=True)
        if str(z0['source_id']) != src['id']:
            raise SystemExit(f'{pool_file}: pool of {z0["source_id"]}, not {src["id"]}')
        done = {int(s): (bytes(h), m) for s, h, m in zip(z0['seeds'], z0['hashes'], z0['masks'])}
        print(f'resuming: {len(done)} runs already in {pool_file.name}', flush=True)
    tasks = [(env, row, s) for s in seeds if s not in done]
    t0 = time.time()
    if tasks:
        with Pool(a.jobs, initializer=_init, initargs=(vol,)) as pool:
            for k, (seed, h, packed) in enumerate(pool.imap_unordered(_pass1, tasks, chunksize=8), 1):
                done[seed] = (h, packed)
                if k % 2000 == 0 or k == len(tasks):
                    np.savez_compressed(pool_file, source_id=np.array(src['id']),
                                        seeds=np.array(list(done), np.int64),
                                        hashes=np.array([v[0] for v in done.values()], dtype='S16'),
                                        masks=np.stack([v[1] for v in done.values()]))
                    print(f'  {k}/{len(tasks)} new runs ({len(done)} in pool)  {time.time() - t0:.0f}s', flush=True)
    counts = np.zeros((HI - LO, HI - LO), np.int64)
    hits, seen, dup = {}, set(), 0
    for seed in seeds:                 # seed order, so a grown pool extends the same ensembles
        if seed not in done:
            continue
        h, packed = done[seed]
        if h in seen:
            dup += 1
            continue
        seen.add(h)
        m = np.unpackbits(packed)[:cand.size].reshape(cand.shape).astype(bool) & cand
        if m.any():
            hits[seed] = m
            counts += m
    best = np.sort(counts[cand])[::-1][:5]
    print(f'pool: {len(seen)} distinct runs, {dup} duplicates, top-5 match counts {best.tolist()}', flush=True)

    # choose five: most matches first, distinct patterns
    order = sorted(zip(*np.nonzero(cand)), key=lambda xy: -counts[xy])
    chosen, pats = [], set()
    for (i, j) in order:
        if counts[i, j] < a.min_n:
            break
        pat = vol[i + LO, j + LO, :].tobytes()
        if pat in pats:
            continue
        pats.add(pat)
        chosen.append((i + LO, j + LO, int(counts[i, j])))
        if len(chosen) == 5:
            break
    for x, y, n in chosen:
        print(f'  well ({x:>2},{y:>2})  pattern {"".join(map(str, vol[x, y, :]))}  '
              f'column ntg {vol[x, y, :].mean():.2f}  {n} exact matches', flush=True)
    if len(chosen) < 5:
        print(f'SHORT: only {len(chosen)} columns reach {a.min_n} matches in {len(seen)} distinct runs; '
              f'raise --draws', flush=True)

    # pass 2: regenerate the matching runs of each chosen column
    with Pool(a.jobs) as pool:
        for i, (x, y, n) in enumerate(chosen, start=1):
            want = [s for s in seeds if s in hits and hits[s][x - LO, y - LO]][:a.cap]
            vols = pool.map(_run, [(env, row, s) for s in want], chunksize=4)
            keep, hs = [], set()
            for s, f in zip(want, vols):
                if not np.array_equal(f[x, y, :], vol[x, y, :]):
                    raise SystemExit(f'seed {s} no longer reproduces the column at ({x},{y}); '
                                     f'ResMill is not deterministic')
                h = hdigest(f)
                if h not in hs:
                    hs.add(h); keep.append((s, f))
            mask = np.zeros((64, 64, 32), np.uint8); mask[x, y, :] = 1
            np.savez_compressed(out / f'{SLUG[env]}_well{i}.npz',
                                volumes=np.stack([f for _, f in keep]), pattern=vol[x, y, :].copy(),
                                well_mask=mask, well_xy=np.array([x, y], np.int64),
                                cond_row_index=np.int64(a.row_index),
                                seeds=np.array([s for s, _ in keep], np.int64))
            print(f'WROTE {SLUG[env]}_well{i}.npz  ({x},{y})  {len(keep)} members  {time.time() - t0:.0f}s', flush=True)
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
