"""Generate the repeat ensembles for the `unconditional` task: many ResMill
runs of ONE input, five conditions per environment.

Reads the task=repeats_unconditional rows of REF/manifest.csv (written by
build_unconditional_reference.py), regenerates each condition's source row
with N fresh ResMill seeds at NATIVE extent -- the environment's own grid,
no azimuth override, no event scaling -- and writes
REF/repeats/<slug>/cond<i>.npz holding `ids`, `volumes` and `seeds`.

Seeds come from default_rng([SEED_BASE, env_index, cond_index]), the scheme
the frozen protocol's generator used, with the source row's own dataset seed
excluded so the ensemble never contains the reference volume itself. Each
condition is self-checked for determinism: its first member is regenerated
and must match bit for bit. Conditions whose file already exists are
skipped, so an interrupted run resumes.

    python tools/gen_repeats_reference.py --ref REF [--n 256] [--jobs 40]
"""
import argparse, csv, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resbench.io import ENVIRONMENTS, SLUG                                   # noqa: E402
from tools.gen_field_reference import (RESMILL_REPO, CONFIG_DIR, CONFIG_FOR_ENV,  # noqa: E402
                                       ENGINE_IGNORE, DATA_DIR)
sys.path.insert(0, str(RESMILL_REPO))

SEED_BASE = 20260921
N_DEFAULT = 256


def native_kwargs(row, env):
    """Engine kwargs for one dataset row, exactly as the frozen protocol
    rebuilt them: drop meta/derived columns, restore the NTG target under the
    engine's own key, keep lobe's poro/perm inputs."""
    kw = {k: v for k, v in row.items()
          if k not in ENGINE_IGNORE and v is not None
          and not (isinstance(v, float) and np.isnan(v))}
    if 'NTGtarget' not in row or row.get('NTGtarget') is None:
        kw['ntg'] = row['requested_ntg']
    if env == 'lobe':
        kw['poro_ave'], kw['perm_ave'] = row['poro_ave'], row['perm_ave']
    return kw


def _one(task):
    env, cond, row, seed = task
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from resmill.dataset.generate import generate_sample
    grid = json.loads((CONFIG_DIR / CONFIG_FOR_ENV[env]).read_text())['grid']
    facies, _, _, _, _ = generate_sample(
        {'layer_type': env.split(':')[0], 'params': native_kwargs(row, env),
         'seed': int(seed)}, grid)
    f = np.asarray(facies, np.int8)
    assert f.shape == (64, 64, 32), f.shape
    return env, cond, int(seed), f


def load_row(shard_dir, sample_idx):
    t = pq.read_table(DATA_DIR / shard_dir / 'params.parquet')
    return {c: t[c][int(sample_idx)].as_py() for c in t.column_names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--n', type=int, default=N_DEFAULT)
    ap.add_argument('--jobs', type=int, default=40)
    ap.add_argument('--envs', default=','.join(ENVIRONMENTS))
    a = ap.parse_args()
    ref = Path(a.ref)
    want_envs = [e for e in a.envs.split(',') if e]

    conds = [r for r in csv.DictReader(open(ref / 'manifest.csv', newline=''))
             if r['task'] == 'repeats_unconditional' and r['environment'] in want_envs]
    if not conds:
        raise SystemExit('no repeats_unconditional rows in the manifest; run build_unconditional_reference.py first')

    plan = {}          # (env, cond) -> (row, seeds, outpath)
    for r in conds:
        env, cond = r['environment'], r['id'].split('|', 1)[1]
        out = ref / 'repeats' / SLUG[env] / f'{cond}.npz'
        if out.exists():
            print(f'skip {env} {cond}: exists', flush=True); continue
        row = load_row(r['shard_dir'], r['sample_idx'])
        ci = int(cond[len('cond'):])
        rng = np.random.default_rng([SEED_BASE, ENVIRONMENTS.index(env), ci])
        seeds = rng.integers(1, 2**31 - 1, size=a.n + 16)
        seeds = [int(s) for s in seeds if int(s) != int(row['seed'])][:a.n]
        assert len(set(seeds)) == a.n, 'seed collision inside one condition'
        plan[(env, cond)] = (row, seeds, out, r['source_id'])

    tasks = [(env, cond, row, s) for (env, cond), (row, seeds, _, _) in plan.items() for s in seeds]
    print(f'{len(plan)} conditions, {len(tasks)} ResMill runs over {a.jobs} workers', flush=True)
    assert len({s for (_, seeds, _, _) in plan.values() for s in seeds}) == len(tasks), \
        'a seed is shared between conditions; the file layout assumes they are disjoint'
    got = {k: {} for k in plan}
    t0 = time.time(); done = 0
    with Pool(a.jobs) as pool:
        for env, cond, seed, f in pool.imap_unordered(_one, tasks, chunksize=4):
            key = (env, cond)                      # carried with the task, never inferred
            got[key][seed] = f
            done += 1
            if done % 256 == 0:
                print(f'  {done}/{len(tasks)}  {time.time()-t0:.0f}s', flush=True)
            if len(got[key]) == len(plan[key][1]):
                row, seeds, out, source_id = plan[key]
                vols = np.stack([got[key][s] for s in seeds])
                # determinism self-check: the first member must regenerate exactly
                _, _, _, again = _one((env, cond, row, seeds[0]))
                if not np.array_equal(again, vols[0]):
                    raise SystemExit(f'{key}: ResMill is not deterministic for seed {seeds[0]}')
                out.parent.mkdir(parents=True, exist_ok=True)
                cond = key[1]
                np.savez_compressed(out, ids=np.array([f'{cond}|{k}' for k in range(len(seeds))], dtype=object),
                                    volumes=vols, seeds=np.array(seeds, dtype=np.int64),
                                    source_id=np.array(source_id))
                print(f'WROTE {env:<24} {cond}  {vols.shape}  mean ntg {vols.mean():.4f}  '
                      f'(source volume ntg {float(row["ntg"]):.4f})  {time.time()-t0:.0f}s', flush=True)
                got.pop(key)
    print(f'done in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
