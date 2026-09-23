"""Mine the five well conditions of one environment with the checked-out ResMill.

The rule is the published one (ResBench-public EVAL.md, C.2). A well is a
location x, y in [8, 56] together with a 32-cell facies column that ResMill
produces there under the source row's parameters. Candidates are every
(location, column) pair seen in a pool of fresh runs of that row; a candidate
is

    informative     at least two sand bodies in the column, separated by mud;
    representative  |column sand fraction - environment mean ntg| <= 0.15;
    estimable       at least --min-n distinct runs of the pool reproduce the
                    column exactly.

The five candidates with the most exact matches, distinct locations and
distinct columns, are the wells. Pass 1 runs --draws fresh seeds of the source
row and records the column at every interior location (a 32-bit key each);
pass 2 regenerates only the runs that matched a chosen well (up to --cap) and
writes

    OUT/<slug>_well<i>.npz   volumes, pattern, well_mask, well_xy, cond_row_index, seeds

which is what build_wells_reference.py --wells-dir reads. Seeds come from
default_rng([WELL_POOL_SEED, env_index, row_index]); the source row's own seed
is excluded.

A run is one WINDOW of a fresh 128 x 128 x 64 ResMill volume, cut exactly as
the dataset cuts its samples (resmill.dataset.windows). Every volume yields
--windows of them (default 8, window 0 being the dataset's own draw, the others
further seeded draws of the same volume), so the pool costs one volume per
eight windows. Windows of one volume share their geology: an ensemble may hold
several members from one volume and its effective size is smaller than its
count. The pool file records (seed, window) per row and a member is
(seed, window) too, so everything regenerates.

Pass 1 is saved to OUT/<slug>_pool.npz (seeds, volume hashes, column keys) and
resumed: running again with a larger --draws only runs the seeds not drawn
yet, so a pool can be grown across allocations. MEANDER_OXBOW and delta cost
about 100 s a run, SH and CB_JIGSAW about 30 s, CB_LABYRINTH 15 s, PV 2 s.

    python tools/mine_wells.py --ref REF --env channel:PV_SHOESTRING --row-index 13 \\
        --draws 60000 --out WELLS [--jobs 48]
"""
import argparse, csv, hashlib, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resbench.io import ENVIRONMENTS, SLUG                                   # noqa: E402
from tools.gen_repeats_reference import native_kwargs, load_row, engine_grid  # noqa: E402

WELL_POOL_SEED = 20260922
INTERIOR = (8, 56)          # inclusive, the published mining window
MIN_N = 50
CAP = 256
NTG_TOL = 0.15
LO, HI = INTERIOR[0], INTERIOR[1] + 1
NZ = 32


def hdigest(v):
    return hashlib.blake2b(np.ascontiguousarray(v).tobytes(), digest_size=16).digest()


def column_keys(f):
    """One uint32 per interior location: the 32-cell column, top cell first."""
    cols = np.ascontiguousarray(f[LO:HI, LO:HI, :]).reshape(-1, NZ).astype(np.uint8)
    return np.packbits(cols, axis=1).view('>u4').ravel().astype(np.uint32)


def key_to_column(key):
    return np.unpackbits(np.array([key], dtype='>u4').view(np.uint8))[:NZ].astype(np.int8)


def informative(col):
    """At least two sand bodies separated by mud."""
    c = np.asarray(col, np.int8)
    return int(((c[1:] == 1) & (c[:-1] == 0)).sum() + (c[0] == 1)) >= 2


def _volume(env, row, seed):
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from resmill.dataset.generate import generate_sample
    f, _, _, _, _ = generate_sample({'layer_type': env.split(':')[0],
                                     'params': native_kwargs(row, env), 'seed': int(seed)}, engine_grid(env))
    return np.asarray(f, np.int8)


def _windows(vol, seed, ks):
    from resmill.dataset.windows import dataset_window
    return [(int(k), dataset_window(vol, int(seed), k=int(k))[1]) for k in ks]


def _pass1(task):
    env, row, seed, ks = task
    vol = _volume(env, row, seed)
    return [(int(seed), k, hdigest(w), column_keys(w)) for k, w in _windows(vol, seed, ks)]


def _pass2(task):
    env, row, seed, ks = task
    vol = _volume(env, row, seed)
    return [(int(seed), k, w) for k, w in _windows(vol, seed, ks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--env', required=True, choices=ENVIRONMENTS)
    ap.add_argument('--row-index', type=int, required=True, help='unconditional row whose parameters the pool runs')
    ap.add_argument('--draws', type=int, default=60000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=48)
    ap.add_argument('--min-n', type=int, default=MIN_N)
    ap.add_argument('--cap', type=int, default=CAP)
    ap.add_argument('--windows', type=int, default=8, help='windows per volume in the pool (window 0 is the dataset draw)')
    a = ap.parse_args()
    ref, out, env = Path(a.ref), Path(a.out), a.env
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'STOP').exists():
        raise SystemExit(f'{out}/STOP exists: not mining {env}')

    rows = [r for r in csv.DictReader(open(ref / 'manifest.csv', newline=''))
            if r['task'] == 'unconditional' and r['environment'] == env]
    src = next(r for r in rows if int(r['row_index']) == a.row_index)
    env_ntg = float(np.mean([float(r['ntg']) for r in rows]))
    row = load_row(src['shard_dir'], src['sample_idx'])
    print(f'{env}: source row {a.row_index} = {src["id"]}  row ntg {float(src["ntg"]):.3f}  '
          f'environment mean ntg {env_ntg:.3f}', flush=True)

    rng = np.random.default_rng([WELL_POOL_SEED, ENVIRONMENTS.index(env), a.row_index])
    seeds = []
    for s in rng.integers(1, 2**31 - 1, size=a.draws + 64):
        if int(s) != int(row['seed']) and int(s) not in seeds:
            seeds.append(int(s))
        if len(seeds) == a.draws:
            break

    # pass 1: the column at every interior location of every run; resumable
    pool_file = out / f'{SLUG[env]}_pool.npz'
    done = {}                       # (seed, window) -> (hash, keys)
    if pool_file.exists():
        z0 = np.load(pool_file, allow_pickle=True)
        if str(z0['source_id']) != src['id']:
            raise SystemExit(f'{pool_file}: pool of {z0["source_id"]}, not {src["id"]}')
        wins = z0['windows'] if 'windows' in z0.files else np.zeros(len(z0['seeds']), np.int64)
        done = {(int(s), int(w)): (bytes(h), k) for s, w, h, k in zip(z0['seeds'], wins, z0['hashes'], z0['keys'])}
        print(f'resuming: {len(done)} windows of {len({s for s, _ in done})} volumes already in {pool_file.name}', flush=True)

    def save():
        items = list(done.items())
        np.savez_compressed(pool_file, source_id=np.array(src['id']),
                            seeds=np.array([s for (s, _), _ in items], np.int64),
                            windows=np.array([w for (_, w), _ in items], np.int64),
                            hashes=np.array([v[0] for _, v in items], dtype='S16'),
                            keys=np.stack([v[1] for _, v in items]))

    # every volume of the pool gets windows 0..W-1; a grown or older pool only runs what is missing
    tasks = []
    for s in seeds:
        ks = [k for k in range(a.windows) if (s, k) not in done]
        if ks:
            tasks.append((env, row, s, ks))
    t0 = time.time()
    if tasks:
        with Pool(a.jobs) as pool:
            for n, results in enumerate(pool.imap_unordered(_pass1, tasks, chunksize=4), 1):
                for seed, k, h, keys in results:
                    done[(seed, k)] = (h, keys)
                if n % 1000 == 0 or n == len(tasks):
                    save()
                    print(f'  {n}/{len(tasks)} volumes ({len(done)} windows in pool)  {time.time() - t0:.0f}s', flush=True)

    # distinct windows, in (seed, window) order so a grown pool extends the same ensembles
    order, seen, dup = [], set(), 0
    for s in seeds:
        for k in range(a.windows):
            if (s, k) in done:
                if done[(s, k)][0] in seen:
                    dup += 1
                    continue
                seen.add(done[(s, k)][0]); order.append((s, k))
    K = np.stack([done[sk][1] for sk in order])            # (windows, locations)
    print(f'pool: {len(order)} distinct windows of {len({s for s, _ in order})} volumes, {dup} duplicates', flush=True)

    # candidates: (location, column) with count >= min_n, informative, representative
    cands = []
    for p in range(K.shape[1]):
        keys, counts = np.unique(K[:, p], return_counts=True)
        for key, n in zip(keys, counts):
            if n < a.min_n:
                continue
            col = key_to_column(key)
            if informative(col) and abs(col.mean() - env_ntg) <= NTG_TOL:
                cands.append((int(n), p, int(key)))
    cands.sort(key=lambda c: (-c[0], c[1]))
    chosen, used_p, used_k = [], set(), set()
    for n, p, key in cands:
        if p in used_p or key in used_k:
            continue
        used_p.add(p); used_k.add(key)
        chosen.append((n, p, key))
        if len(chosen) == 5:
            break
    print(f'{len(cands)} candidate (location, column) pairs with >= {a.min_n} matches', flush=True)
    for n, p, key in chosen:
        x, y = LO + p // (HI - LO), LO + p % (HI - LO)
        col = key_to_column(key)
        print(f'  well ({x:>2},{y:>2})  column {"".join(map(str, col))}  sand {col.mean():.2f}  {n} exact matches', flush=True)
    if len(chosen) < 5:
        print(f'SHORT: only {len(chosen)} wells reach {a.min_n} matches in {len(order)} distinct windows; raise --draws', flush=True)

    # pass 2: regenerate the matching runs of each chosen well
    with Pool(a.jobs) as pool:
        for i, (n, p, key) in enumerate(chosen, start=1):
            x, y = LO + p // (HI - LO), LO + p % (HI - LO)
            col = key_to_column(key)
            want = [sk for sk, kk in zip(order, K[:, p]) if int(kk) == key][:a.cap]
            by_seed = {}
            for s, k in want:
                by_seed.setdefault(s, []).append(k)
            got = {}
            for results in pool.imap_unordered(_pass2, [(env, row, s, ks) for s, ks in by_seed.items()], chunksize=2):
                for s, k, w in results:
                    got[(s, k)] = w
            keep = []
            for s, k in want:
                f = got[(s, k)]
                if not np.array_equal(f[x, y, :], col):
                    raise SystemExit(f'seed {s} window {k} no longer reproduces the column at ({x},{y}); '
                                     f'ResMill is not deterministic')
                keep.append((s, k, f))
            mask = np.zeros((64, 64, NZ), np.uint8); mask[x, y, :] = 1
            np.savez_compressed(out / f'{SLUG[env]}_well{i}.npz',
                                volumes=np.stack([f for _, _, f in keep]), pattern=col.copy(),
                                well_mask=mask, well_xy=np.array([x, y], np.int64),
                                cond_row_index=np.int64(a.row_index),
                                seeds=np.array([s for s, _, _ in keep], np.int64),
                                windows=np.array([k for _, k, _ in keep], np.int64))
            print(f'WROTE {SLUG[env]}_well{i}.npz  ({x},{y})  {len(keep)} members from {len(by_seed)} volumes  {time.time() - t0:.0f}s', flush=True)
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
