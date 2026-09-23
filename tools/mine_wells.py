"""Mine the five well conditions of one environment with the checked-out ResMill.

A well is the CENTRE column (32, 32) of a 64 x 64 x 32 cube together with a
32-cell facies column that ResMill produces there under the source row's
parameters. A candidate column is

    informative     at least two sand bodies in the column, separated by mud;
    representative  |column sand fraction - environment mean ntg| <= 0.15;
    estimable       at least --min-n distinct volumes of the pool contain it.

The five most frequent candidates, distinct columns, are the wells; the
matching volumes give the ensembles.

What a match is, stated plainly. The dataset stores one 64 x 64 x 32 window of
every 128 x 128 x 64 ResMill volume at a seeded random origin (x0, y0 in
0..64, z0 in 1..31; bases 0 and 32 never occur, so no window holds the
simulated floor or roof). A cube's centre column is therefore the volume
column at (X, Y) = (x0 + 32, y0 + 32), z0..z0 + 31: the 65 x 65 x 31 columns
with X, Y in 32..96 and z0 in 1..31 are every centre column a window of that
volume can have. The pool runs --draws fresh volumes of the source row, records
all 65 x 65 x 31 column keys of each, and a volume matches a column when the
column occurs anywhere in that block. The member is the window at the first
occurrence (origin X - 32, Y - 32, z0), one per volume, so the members are
windows of distinct volumes with the well at their centre. This is rejection
sampling from the conditional distribution a model is asked for, with the
origin marginalised exactly instead of by one random draw per volume.

Pass 1 generates the volumes and stores their key blocks in OUT/<slug>_keys.u32
(a memmap, 0.5 MB a volume; resumed when the seeds match); pass 2 regenerates
the matching volumes and writes

    OUT/<slug>_well<i>.npz   volumes, pattern, well_mask, well_xy, cond_row_index, seeds, origins

which is what build_wells_reference.py --wells-dir reads. Seeds come from
default_rng([WELL_POOL_SEED, env_index, row_index]); the source row's own seed
is excluded.

    python tools/mine_wells.py --ref REF --env channel:PV_SHOESTRING --row-index 13 \\
        --draws 2000 --out WELLS [--jobs 140]
"""
import argparse, csv, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resbench.io import ENVIRONMENTS, SLUG                                   # noqa: E402
from tools.gen_repeats_reference import native_kwargs, load_row, engine_grid  # noqa: E402

WELL_POOL_SEED = 20260922
MIN_N = 50
CAP = 256
NTG_TOL = 0.15
NZ = 32
WIN = (64, 64, 32)
WELL_XY = (32, 32)          # the well is the cube's centre column
XMAX = 64                   # largest window origin in x and y on the 128 grid
Z0 = (1, 31)                # window base range, as resmill.dataset.windows
XR = (WELL_XY[0], WELL_XY[0] + XMAX)   # volume X, Y a window centre can have: 32..96
NX_KEYS = XR[1] - XR[0] + 1            # 65
NZ0 = Z0[1] - Z0[0] + 1                # 31


def volume_keys(f):
    """uint32 column key of every window-centre column (X, Y, z0): cell z0 of the window at
    bit 31, matching ``column_keys`` of a window (packbits, big-endian)."""
    sub = (np.asarray(f[XR[0]:XR[1] + 1, XR[0]:XR[1] + 1, :]) > 0)
    col64 = np.packbits(sub, axis=2).copy().view('>u8').reshape(NX_KEYS, NX_KEYS).astype(np.uint64)
    out = np.empty((NX_KEYS, NX_KEYS, NZ0), np.uint32)
    for i, z0 in enumerate(range(Z0[0], Z0[1] + 1)):
        out[:, :, i] = ((col64 >> np.uint64(32 - z0)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    return out


def popcount32(a):
    a = np.asarray(a, np.uint32)
    return np.unpackbits(a.view(np.uint8).reshape(-1, 4), axis=1).sum(1).reshape(a.shape)


def candidate_mask(keys, env_ntg, tol=NTG_TOL):
    """informative (>= 2 sand bodies) and representative (sand fraction within tol of env_ntg)."""
    k = np.asarray(keys, np.uint32)
    starts = k & ~(k >> np.uint32(1))
    return (popcount32(starts) >= 2) & (np.abs(popcount32(k) / NZ - env_ntg) <= tol)


def key_to_column(key):
    return np.unpackbits(np.array([key], dtype='>u4').view(np.uint8))[:NZ].astype(np.int8)


def _volume(env, row, seed):
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from resmill.dataset.generate import generate_sample
    f, _, _, _, _ = generate_sample({'layer_type': env.split(':')[0],
                                     'params': native_kwargs(row, env), 'seed': int(seed)}, engine_grid(env))
    return np.asarray(f, np.int8)


def _pass1(task):
    env, row, seed, slot, keyfile, n_slots, env_ntg = task
    f = _volume(env, row, seed)
    keys = volume_keys(f)
    mm = np.memmap(keyfile, dtype=np.uint32, mode='r+', shape=(n_slots, NX_KEYS, NX_KEYS, NZ0))
    mm[slot] = keys; mm.flush(); del mm
    cand = np.unique(keys[candidate_mask(keys, env_ntg)])
    return slot, int(seed), cand


def _pass2(task):
    env, row, seed, origin = task
    f = _volume(env, row, seed)
    x0, y0, z0 = origin
    return int(seed), np.ascontiguousarray(f[x0:x0 + WIN[0], y0:y0 + WIN[1], z0:z0 + WIN[2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--env', required=True, choices=ENVIRONMENTS)
    ap.add_argument('--row-index', type=int, required=True, help='unconditional row whose parameters the pool runs')
    ap.add_argument('--draws', type=int, default=2000, help='volumes in the pool')
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
    row = load_row(src['shard_dir'], src['sample_idx'])
    print(f'{env}: source row {a.row_index} = {src["id"]}  row ntg {float(src["ntg"]):.3f}  '
          f'environment mean ntg {env_ntg:.3f}', flush=True)

    ei = ENVIRONMENTS.index(env)
    rng = np.random.default_rng([WELL_POOL_SEED, ei, a.row_index])
    seeds = []
    for s in rng.integers(1, 2**31 - 1, size=a.draws + 64):
        if int(s) != int(row['seed']) and int(s) not in seeds:
            seeds.append(int(s))
        if len(seeds) == a.draws:
            break

    # pass 1: every reachable column key of every volume, in a memmap; resumable
    slug = SLUG[env]
    keyfile, metafile = out / f'{slug}_keys.u32', out / f'{slug}_keys.json'
    done = {}
    if metafile.exists():
        meta = json.load(open(metafile))
        if meta['source_id'] == src['id'] and meta['seeds'][:len(seeds)] == seeds[:len(meta['seeds'])]:
            done = {int(s): i for i, s in enumerate(meta['seeds'])}
            print(f'resuming: {len(done)} volumes already in {keyfile.name}', flush=True)
    n_slots = max(len(seeds), len(done))
    mode = 'r+' if keyfile.exists() and done else 'w+'
    mm = np.memmap(keyfile, dtype=np.uint32, mode=mode, shape=(n_slots, NX_KEYS, NX_KEYS, NZ0)); del mm
    slot_of = dict(done)
    for s in seeds:
        if s not in slot_of:
            slot_of[s] = len(slot_of)
    order = sorted(slot_of, key=slot_of.get)

    def save_meta():
        json.dump({'source_id': src['id'], 'seeds': order, 'n_slots': n_slots}, open(metafile, 'w'))

    tasks = [(env, row, s, slot_of[s], str(keyfile), n_slots, env_ntg) for s in seeds if s not in done]
    cand_lists, t0 = {}, time.time()
    # candidate keys of the volumes already on disk
    if done:
        mm = np.memmap(keyfile, dtype=np.uint32, mode='r', shape=(n_slots, NX_KEYS, NX_KEYS, NZ0))
        for s, i in done.items():
            k = np.asarray(mm[i]); cand_lists[s] = np.unique(k[candidate_mask(k, env_ntg)])
        del mm
    if tasks:
        with Pool(a.jobs) as pool:
            for n, (slot, seed, cand) in enumerate(pool.imap_unordered(_pass1, tasks, chunksize=1), 1):
                cand_lists[seed] = cand; done[seed] = slot
                if n % 200 == 0 or n == len(tasks):
                    save_meta()
                    print(f'  {n}/{len(tasks)} volumes  {time.time() - t0:.0f}s', flush=True)
    save_meta()
    seeds = [s for s in seeds if s in done]

    # candidates: distinct columns ranked by the number of volumes that contain them anywhere reachable
    allc = np.concatenate([cand_lists[s] for s in seeds])
    ukeys, counts = np.unique(allc, return_counts=True)
    rank = np.argsort(-counts, kind='stable')
    print(f'pool: {len(seeds)} volumes, {len(ukeys):,} distinct informative+representative columns; '
          f'top counts {counts[rank[:5]].tolist()} volumes', flush=True)

    mm = np.memmap(keyfile, dtype=np.uint32, mode='r', shape=(n_slots, NX_KEYS, NX_KEYS, NZ0))
    slots = np.array([done[s] for s in seeds])

    def members_of(key):
        """per volume: the window at the first occurrence of the column, (seed, origin)"""
        members = []
        for s, i in zip(seeds, slots):
            hit = np.argwhere(mm[i] == key)
            if hit.size:
                dx, dy, dz = hit[0]
                members.append((s, (int(dx), int(dy), int(dz) + Z0[0])))     # origin = (X - 32, Y - 32, z0)
        return members

    x, y = WELL_XY
    chosen, used_k = [], set()
    for j in rank:
        if counts[j] < a.min_n:
            break
        key = int(ukeys[j])
        if key in used_k:
            continue
        i = len(chosen) + 1
        members = members_of(key)
        assert len(members) == int(counts[j]), (len(members), int(counts[j]))
        used_k.add(key); chosen.append((key, x, y, members))
        col = key_to_column(key)
        print(f'  well {i} ({x:>2},{y:>2})  column {"".join(map(str, col))}  sand {col.mean():.2f}  '
              f'{len(members)} volumes contain it', flush=True)
        if len(chosen) == 5:
            break
    if len(chosen) < 5:
        print(f'SHORT: only {len(chosen)} wells reach {a.min_n} volumes in a pool of {len(seeds)}; raise --draws', flush=True)

    # pass 2: regenerate the matching volumes, cut the window that puts the column at (x, y)
    with Pool(a.jobs) as pool:
        for i, (key, x, y, members) in enumerate(chosen, start=1):
            col = key_to_column(key)
            members = members[:a.cap]
            got = dict(pool.imap_unordered(_pass2, [(env, row, s, o) for s, o in members], chunksize=2))
            vols = []
            for s, o in members:
                w = got[s]
                if not np.array_equal(w[x, y, :], col):
                    raise SystemExit(f'seed {s} origin {o} no longer shows the column at ({x},{y}); ResMill is not deterministic')
                vols.append(w)
            mask = np.zeros(WIN, np.uint8); mask[x, y, :] = 1
            np.savez_compressed(out / f'{slug}_well{i}.npz',
                                volumes=np.stack(vols), pattern=col.copy(), well_mask=mask,
                                well_xy=np.array([x, y], np.int64), cond_row_index=np.int64(a.row_index),
                                seeds=np.array([s for s, _ in members], np.int64),
                                origins=np.array([o for _, o in members], np.int64))
            print(f'WROTE {slug}_well{i}.npz  ({x},{y})  {len(vols)} members, one window per volume  {time.time() - t0:.0f}s', flush=True)
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
