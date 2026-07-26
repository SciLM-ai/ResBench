"""ResBench CLI: score a prediction ensemble against a reference ensemble.

    python -m resbench.run --pred-dir ENSEMBLE_A --ref-dir REFERENCE \
        --out results/metrics.parquet \
        [--well-pred-dir ENSEMBLE_B --mask-dir MASKS --manifest manifest.csv] \
        [--figures] [--fig-dir results]

Computes the EVAL.md master table (one row per environment + pooled), the
split-half null band, verdict annotations, bootstrap CIs, optional well
exactitude (pooled and by well count), and optional figures. Everything is
deterministic (seeds in resbench/__init__.py, frozen in EVAL.md §6).
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from . import LAYER_TYPES, MAX_LAGS, SPLIT_HALF_SEED, BOOTSTRAP_SEED
from . import io, metrics, stats

_G = {}   # fork-shared volume stacks: {env: (ref_vols, gen_vols)}

CELL_LABELS = {
    'abs_dntg': '|dNTG|',
    'variogram_mae': 'variogram MAE/sill',
    'connectivity_mae': 'connectivity MAE',
    'geobody_w1': 'geobody W1 (log10)',
    'abs_dlargest_frac': '|d largest frac|',
}


def _env_worker(args):
    """Summaries + band for one environment (runs in a forked worker)."""
    lt, env_idx = args
    ref_vols, gen_vols = _G[lt]
    ref = stats.env_summary(ref_vols, MAX_LAGS)
    gen = stats.env_summary(gen_vols, MAX_LAGS)
    sill = stats.reference_sill(ref)
    cells = stats.compare(ref, gen, sill)
    rng = np.random.default_rng([SPLIT_HALF_SEED, env_idx])
    band, halves = stats.split_half(ref_vols, rng, MAX_LAGS)
    return lt, {'ref': ref, 'gen': gen, 'cells': cells, 'band': band,
                'halves': halves, 'sill': sill}


def exactitude_report(well_pred_dir, ref_dir, mask_dir, manifest_path):
    """{env: {'pooled': %, 'by_config': {...}}} plus overall pooled."""
    mf = pd.read_csv(manifest_path, keep_default_na=False)
    cfg_by_id = dict(zip(mf['row_id'], mf['well_config']))
    preds = io.load_ensemble(well_pred_dir)
    refs = io.load_ensemble(ref_dir)
    masks = io.load_ensemble(mask_dir, key='masks')
    out = {}
    tot_bad = tot_n = 0
    cfg_tot = {}
    for lt in preds:
        pid, pv = preds[lt]
        rid, rv = refs[lt]
        mid, mv = masks[lt]
        pv_r, rv_r, ids = io.align(pid, pv, rid, rv)
        pv_m, mv_m, ids_m = io.align(pid, pv, mid, mv)
        assert ids == ids_m, f'{lt}: mask/ref id mismatch'
        env_bad = env_n = 0
        by_cfg = {}
        for g, r, m, i in zip(pv_r, rv_r, mv_m, ids):
            bad, n = metrics.exactitude_counts(g, r, m)
            env_bad += bad
            env_n += n
            cfg = cfg_by_id.get(i.split('#')[0], '?')
            b, t = by_cfg.get(cfg, (0, 0))
            by_cfg[cfg] = (b + bad, t + n)
            b, t = cfg_tot.get(cfg, (0, 0))
            cfg_tot[cfg] = (b + bad, t + n)
        out[lt] = {
            'pooled_pct': 100.0 * env_bad / env_n,
            'by_config_pct': {c: 100.0 * b / t for c, (b, t) in sorted(by_cfg.items())},
            'n_mask_voxels': env_n,
        }
        tot_bad += env_bad
        tot_n += env_n
    out['__pooled__'] = {
        'pooled_pct': 100.0 * tot_bad / tot_n,
        'by_config_pct': {c: 100.0 * b / t for c, (b, t) in sorted(cfg_tot.items())},
        'n_mask_voxels': tot_n,
    }
    return out


def render_markdown(table, well=None):
    lines = ['| environment | ' + ' | '.join(CELL_LABELS.values())
             + ' | well mismatch % |',
             '|' + '---|' * (len(CELL_LABELS) + 2)]
    for _, r in table.iterrows():
        cells = []
        for c in CELL_LABELS:
            cells.append(f"{r[c]:.4f} ({r[f'{c}_verdict']}; band {r[f'{c}_band']:.4f})")
        wm = f"{r['well_mismatch_pct']:.3f} ({r['well_verdict']})" \
            if np.isfinite(r.get('well_mismatch_pct', np.nan)) else 'n/a'
        lines.append(f"| {r['environment']} | " + ' | '.join(cells) + f' | {wm} |')
    if well and '__pooled__' in well:
        by = well['__pooled__']['by_config_pct']
        lines.append('')
        lines.append('Well mismatch by config (pooled): '
                     + ', '.join(f'{c}: {v:.3f}%' for c, v in by.items()))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-dir', required=True)
    ap.add_argument('--ref-dir', required=True)
    ap.add_argument('--out', required=True, help='master table parquet path')
    ap.add_argument('--well-pred-dir')
    ap.add_argument('--mask-dir')
    ap.add_argument('--manifest')
    ap.add_argument('--figures', action='store_true')
    ap.add_argument('--fig-dir', default=None)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    preds = io.load_ensemble(args.pred_dir)
    refs = io.load_ensemble(args.ref_dir)
    envs = [lt for lt in LAYER_TYPES if lt in preds and lt in refs]

    for lt in envs:
        pid, pv = preds[lt]
        rid, rv = refs[lt]
        pv_al, rv_al, _ = io.align(pid, pv, rid, rv)
        assert len(pv_al) > 0, f'{lt}: no aligned rows'
        _G[lt] = (rv_al, pv_al)

    with mp.get_context('fork').Pool(min(args.workers, len(envs))) as pool:
        results = dict(pool.map(_env_worker,
                                [(lt, i) for i, lt in enumerate(envs)]))

    # Pooled row: exact count-pooling across environments; band pools the
    # per-environment halves (stratification preserved).
    pooled_ref = stats.merge_summaries([results[lt]['ref'] for lt in envs])
    pooled_gen = stats.merge_summaries([results[lt]['gen'] for lt in envs])
    pooled_sill = stats.reference_sill(pooled_ref)
    pooled_cells = stats.compare(pooled_ref, pooled_gen, pooled_sill)
    pooled_band = stats.compare(
        stats.merge_summaries([results[lt]['halves'][0] for lt in envs]),
        stats.merge_summaries([results[lt]['halves'][1] for lt in envs]),
        pooled_sill)

    well = None
    if args.well_pred_dir:
        assert args.mask_dir and args.manifest, \
            '--well-pred-dir needs --mask-dir and --manifest'
        well = exactitude_report(args.well_pred_dir, args.ref_dir,
                                 args.mask_dir, args.manifest)

    rows = []
    sanity_warnings = {}
    for i, lt in enumerate(envs + ['__pooled__']):
        if lt == '__pooled__':
            ref, gen = pooled_ref, pooled_gen
            cells, band = pooled_cells, pooled_band
        else:
            r = results[lt]
            ref, gen = r['ref'], r['gen']
            cells, band = r['cells'], r['band']
        row = {'environment': 'pooled' if lt == '__pooled__' else lt,
               'n_ref': ref['n'], 'n_gen': gen['n']}
        for c in stats.CELLS:
            row[c] = cells[c]
            row[f'{c}_band'] = band[c]
            row[f'{c}_verdict'] = stats.verdict(cells[c], band[c])
        if well and lt in well:
            row['well_mismatch_pct'] = well[lt]['pooled_pct']
            row['well_verdict'] = ('inside' if row['well_mismatch_pct'] <= 1.0
                                   else 'near' if row['well_mismatch_pct'] <= 2.0
                                   else 'outside')
            for c, v in well[lt]['by_config_pct'].items():
                row[f'well_{c}_pct'] = v
        else:
            row['well_mismatch_pct'] = np.nan
            row['well_verdict'] = 'n/a'
        rows.append(row)
        sanity_warnings[row['environment']] = (
            metrics.sanity_check(float(ref['ntg'].mean()), gen['gamma'], gen['tau'])
            + metrics.sanity_check(float(ref['ntg'].mean()), ref['gamma'], ref['tau']))

    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    md = render_markdown(table, well)
    out.with_suffix('.md').write_text(md + '\n')
    for env, w in sanity_warnings.items():
        for msg in w:
            print(f'SANITY [{env}]: {msg}')
    print(md)

    # Figure report + curve archive (bootstrap CIs per EVAL.md §6).
    report = {}
    for i, lt in enumerate(envs):
        r = results[lt]
        report[lt] = {
            'ref': {'gamma': r['ref']['gamma'], 'tau': r['ref']['tau'],
                    'sizes': r['ref']['sizes'],
                    'ntg_mean': float(r['ref']['ntg'].mean())},
            'gen': {'gamma': r['gen']['gamma'], 'tau': r['gen']['tau'],
                    'sizes': r['gen']['sizes'],
                    'ntg_mean': float(r['gen']['ntg'].mean())},
            'halves': tuple({'gamma': h['gamma'], 'tau': h['tau']}
                            for h in r['halves']),
            'ref_ntg_ci': stats.bootstrap_mean_ci(
                r['ref']['ntg'], seed=[BOOTSTRAP_SEED, i, 0]),
            'gen_ntg_ci': stats.bootstrap_mean_ci(
                r['gen']['ntg'], seed=[BOOTSTRAP_SEED, i, 1]),
            'ref_largest_ci': stats.bootstrap_mean_ci(
                r['ref']['largest_frac'], seed=[BOOTSTRAP_SEED, i, 2]),
            'gen_largest_ci': stats.bootstrap_mean_ci(
                r['gen']['largest_frac'], seed=[BOOTSTRAP_SEED, i, 3]),
        }
    np.save(out.parent / 'report.npy', np.array([report], dtype=object),
            allow_pickle=True)
    if well:
        (out.parent / 'well_exactitude.json').write_text(
            json.dumps(well, indent=2))

    if args.figures:
        from . import figures
        fig_dir = Path(args.fig_dir or out.parent)
        fig_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        paths += figures.ntg_parity(report, fig_dir / 'ntg_parity')
        paths += figures.variogram_grid(report, fig_dir / 'variogram_overlays')
        paths += figures.connectivity_geobody_grid(
            report, fig_dir / 'connectivity_geobody')
        print('figures:', *paths, sep='\n  ')


if __name__ == '__main__':
    main()
