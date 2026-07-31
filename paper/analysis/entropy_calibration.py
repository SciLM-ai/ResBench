"""Conditional-entropy calibration analysis (EVAL.md Addendum A.5).

Inputs (per condition, 32 total = 8 envs x 4):
  --model-dir  <slug>/cond_rXXXX_<cfg>.npz : volumes (K,X,Y,Z) int8, mask, ref_id
  --uncond-dir <slug>/cond_rXXXX_<cfg>.npz : volumes (N,X,Y,Z) int8 unconditional-
               under-params ResMill realizations (+ seeds)
  --ref-dir    reference volume dirs (well values for the acceptance test)
  --times-json optional {slug_rXXXX: seconds_per_realization} measured engine
               cost, for the rejection feasibility projection
  --reject-dir optional dir of accepted-realization npz files from targeted
               rejection (key volumes), same naming as model-dir conditions

Outputs: entropy_calibration.{md,json}, curves entropy_vs_distance.{pdf,png},
maps entropy_maps.{pdf,png}, per-condition acceptance/cost table.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resbench import LAYER_TYPES, env_slug   # noqa: E402
from resbench import io                       # noqa: E402

BIN_W = 2
D_MAX = 24
SPLIT_SEED = 20260731
MIN_ACCEPTED = 20      # minimum accepted realizations for a near-field curve
TARGET_ACCEPTED = 200  # protocol target (A.4)


def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return np.where((p <= eps) | (p >= 1 - eps), 0.0, h)


def chebyshev_dist(mask):
    """Chebyshev distance to the nearest well voxel."""
    return ndimage.distance_transform_cdt(1 - mask, metric='chessboard')


def bin_means(field, dist):
    """Mean of `field` in distance bins [1,2], [3,4], ... up to D_MAX."""
    edges = np.arange(1, D_MAX + 1, BIN_W)
    out = np.full(len(edges), np.nan)
    for i, lo in enumerate(edges):
        m = (dist >= lo) & (dist < lo + BIN_W)
        if m.any():
            out[i] = float(field[m].mean())
    return edges, out


def load_conditions(root):
    out = {}
    for lt in LAYER_TYPES:
        d = Path(root) / env_slug(lt)
        if d.is_dir():
            for f in sorted(d.glob('cond_*.npz')):
                out[(lt, f.stem)] = f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--uncond-dir', required=True)
    ap.add_argument('--ref-dir', required=True)
    ap.add_argument('--reject-dir')
    ap.add_argument('--times-json')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    times = json.load(open(args.times_json)) if args.times_json else {}
    model_conds = load_conditions(args.model_dir)
    uncond_conds = load_conditions(args.uncond_dir)
    ref_vols = {lt: dict(zip(*io.load_ensemble(args.ref_dir)[lt]))
                for lt in {lt for lt, _ in model_conds}}

    rows, curves = [], {}
    for ci, ((lt, stem), mf) in enumerate(sorted(model_conds.items())):
        md = np.load(mf, allow_pickle=True)
        vols_m, mask = md['volumes'], md['mask'].astype(np.uint8)
        ref_id = str(md['ref_id'])
        ref = ref_vols[lt][ref_id]
        wells = mask > 0
        dist = chebyshev_dist(mask)

        p_m = vols_m.mean(axis=0)
        H_m = entropy(p_m)
        assert np.all(H_m[wells] == 0.0), 'H!=0 at well cells (by construction)'

        ud = np.load(uncond_conds[(lt, stem)], allow_pickle=True)
        vols_u = ud['volumes']
        H_u = entropy(vols_u.mean(axis=0))
        # split-half band of the unconditional level, per distance bin
        perm = np.random.default_rng([SPLIT_SEED, ci]).permutation(len(vols_u))
        H_a = entropy(vols_u[perm[:len(vols_u) // 2]].mean(axis=0))
        H_b = entropy(vols_u[perm[len(vols_u) // 2:]].mean(axis=0))

        # acceptance test on the unconditional draws (A.4 probe)
        w_ref = ref[wells]
        acc_mask = np.array([np.array_equal(v[wells], w_ref) for v in vols_u])
        n_acc = int(acc_mask.sum())
        sec = times.get(f'{env_slug(lt)}_{stem}', np.nan)
        acc_rate = n_acc / len(vols_u)
        proj_draws = (TARGET_ACCEPTED / acc_rate) if n_acc else np.inf
        proj_core_h = proj_draws * sec / 3600 if np.isfinite(sec) else np.nan

        # near-field reference: targeted rejection file if present, else the
        # accepted subset of the unconditional draws (if enough)
        H_r, n_r = None, 0
        if args.reject_dir:
            rf = Path(args.reject_dir) / env_slug(lt) / f'{stem}.npz'
            if rf.exists():
                va = np.load(rf, allow_pickle=True)['volumes']
                H_r, n_r = entropy(va.mean(axis=0)), len(va)
        if H_r is None and n_acc >= MIN_ACCEPTED:
            H_r, n_r = entropy(vols_u[acc_mask].mean(axis=0)), n_acc

        edges, c_m = bin_means(H_m, dist)
        _, c_u = bin_means(H_u, dist)
        _, band = bin_means(np.abs(H_a - H_b), dist)
        _, c_r = bin_means(H_r, dist) if H_r is not None else (edges, None)

        # distance beyond which the model curve stays within the band of the
        # unconditional level
        conv_d = None
        for i in range(len(edges)):
            tail_ok = all(
                not np.isfinite(c_m[j]) or abs(c_m[j] - c_u[j]) <= band[j]
                for j in range(i, len(edges)))
            if tail_ok:
                conv_d = int(edges[i])
                break

        calib_err = (np.nanmean(np.abs(c_m - c_r)) if c_r is not None
                     else np.nan)
        near = slice(0, 3)   # bins within Chebyshev distance <= 6
        under_disp = (bool(np.nanmean(c_m[near] - c_r[near]) < 0)
                      if c_r is not None else None)

        curves.setdefault(lt, []).append(
            {'stem': stem, 'edges': edges.tolist(),
             'model': c_m.tolist(), 'uncond': c_u.tolist(),
             'band': band.tolist(),
             'reject': (c_r.tolist() if c_r is not None else None),
             'n_reject': n_r})
        rows.append({
            'environment': lt, 'condition': stem, 'K_model': len(vols_m),
            'N_uncond': len(vols_u), 'acc_rate': acc_rate, 'n_accepted': n_acc,
            'sec_per_draw': None if not np.isfinite(sec) else sec,
            'proj_core_h_200acc': None if not np.isfinite(proj_core_h)
            else round(proj_core_h, 1),
            'n_reject_used': n_r,
            'calib_err_bits': None if not np.isfinite(calib_err)
            else round(float(calib_err), 4),
            'conv_dist': conv_d,
            'under_dispersed_nearfield': under_disp,
        })
        print(f"{lt} {stem}: acc={acc_rate * 100:.1f}% n_acc={n_acc} "
              f"conv_d={conv_d} calib_err={calib_err if np.isfinite(calib_err) else float('nan'):.4f} "
              f"under_disp={under_disp}", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({'rows': rows, 'curves': curves},
              open(out / 'entropy_calibration.json', 'w'), indent=2)

    # ---- markdown ------------------------------------------------------
    md = ['## Conditional-entropy calibration (EVAL.md Addendum A)', '',
          '| environment | condition | acc. rate | n acc | proj. core-h '
          'for 200 acc | n ref used | calib err (bits) | conv. dist | '
          'under-dispersed? |', '|' + '---|' * 9]
    for r in rows:
        md.append(
            f"| {r['environment']} | {r['condition']} "
            f"| {r['acc_rate'] * 100:.1f}% | {r['n_accepted']} "
            f"| {r['proj_core_h_200acc'] if r['proj_core_h_200acc'] is not None else '—'} "
            f"| {r['n_reject_used'] or '—'} "
            f"| {r['calib_err_bits'] if r['calib_err_bits'] is not None else '—'} "
            f"| {r['conv_dist'] if r['conv_dist'] is not None else '>24'} "
            f"| {'YES' if r['under_dispersed_nearfield'] else ('no' if r['under_dispersed_nearfield'] is not None else '—')} |")
    (out / 'entropy_calibration.md').write_text('\n'.join(md) + '\n')

    # ---- figures -------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from resbench.figures import AXIS_COLORS, REF_GRAY, SHORT
    BLUE, ORANGE = AXIS_COLORS[0], AXIS_COLORS[1]

    fig, axes = plt.subplots(2, 4, figsize=(11, 5), sharex=True, sharey=True)
    for i, lt in enumerate([lt for lt in LAYER_TYPES if lt in curves]):
        ax = axes.flat[i]
        cs = curves[lt]
        edges = np.array(cs[0]['edges'])
        for arr, color, ls, lw in (
                ('uncond', REF_GRAY, ':', 1.2), ('model', BLUE, '-', 1.4),
                ('reject', ORANGE, (0, (4, 2)), 1.4)):
            vals = [c[arr] for c in cs if c[arr] is not None]
            if not vals:
                continue
            m = np.nanmean(np.array(vals, dtype=float), axis=0)
            ax.plot(edges, m, c=color, ls=ls, lw=lw)
        band = np.nanmean(np.array([c['band'] for c in cs], dtype=float), axis=0)
        unc = np.nanmean(np.array([c['uncond'] for c in cs], dtype=float), axis=0)
        ax.fill_between(edges, unc - band, unc + band, color=REF_GRAY,
                        alpha=0.18, lw=0)
        ax.set_title(SHORT[lt])
        if i >= 4:
            ax.set_xlabel('Chebyshev distance to nearest well')
        if i % 4 == 0:
            ax.set_ylabel('mean entropy (bits)')
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], c=BLUE, lw=1.4, label='ResFlow conditional'),
        Line2D([], [], c=ORANGE, lw=1.4, ls=(0, (4, 2)),
               label='ResMill conditional (rejection)'),
        Line2D([], [], c=REF_GRAY, lw=1.2, ls=':',
               label='ResMill unconditional-under-params (± split-half band)')],
        ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.suptitle('Conditional entropy vs distance to wells '
                 '(mean over 4 conditions/env)', y=1.0)
    fig.tight_layout()
    fig.savefig(out / 'entropy_vs_distance.pdf', bbox_inches='tight')
    fig.savefig(out / 'entropy_vs_distance.png', bbox_inches='tight', dpi=250)
    plt.close(fig)

    # representative condition maps: highest-acceptance 1-well condition
    best = max((r for r in rows if '1well' in r['condition']
                and r['n_accepted'] >= MIN_ACCEPTED),
               key=lambda r: r['n_accepted'], default=None)
    if best is not None:
        lt, stem = best['environment'], best['condition']
        md_ = np.load(model_conds[(lt, stem)], allow_pickle=True)
        ud_ = np.load(uncond_conds[(lt, stem)], allow_pickle=True)
        mask = md_['mask'].astype(np.uint8)
        wells = mask > 0
        ref = ref_vols[lt][str(md_['ref_id'])]
        w_ref = ref[wells]
        # Prefer the targeted-rejection ensemble; fall back to the accepted
        # subset of the unconditional probe draws.
        rej_f = (Path(args.reject_dir) / env_slug(lt) / f'{stem}.npz'
                 if args.reject_dir else None)
        if rej_f is not None and rej_f.exists():
            acc_vols = np.load(rej_f, allow_pickle=True)['volumes']
        else:
            acc = np.array([np.array_equal(v[wells], w_ref)
                            for v in ud_['volumes']])
            acc_vols = ud_['volumes'][acc]
        H_m = entropy(md_['volumes'].mean(axis=0))
        H_r = entropy(acc_vols.mean(axis=0))
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
        for col, (H, name) in enumerate(
                ((H_m, f'ResFlow (K={best["K_model"]})'),
                 (H_r, f'ResMill rejection (n={len(acc_vols)})'))):
            im0 = axes[0, col].imshow(H[:, :, 18].T, origin='lower',
                                      vmin=0, vmax=1, cmap='magma')
            axes[0, col].set_title(f'{name}\nXY @ z=18', fontsize=8.5)
            axes[1, col].imshow(H[:, 32, :].T, origin='lower',
                                vmin=0, vmax=1, cmap='magma', aspect='auto')
            axes[1, col].set_title('XZ @ y=32', fontsize=8.5)
            wx = np.where(mask[:, 32, :].max(axis=1) > 0)[0]
            for x in wx:
                axes[1, col].axvline(x, c='white', lw=0.6, alpha=0.7)
        fig.colorbar(im0, ax=axes, shrink=0.8, label='entropy (bits)')
        fig.suptitle(f'Voxelwise conditional entropy — {SHORT[lt]}, {stem}',
                     y=0.98)
        fig.savefig(out / 'entropy_maps.pdf', bbox_inches='tight')
        fig.savefig(out / 'entropy_maps.png', bbox_inches='tight', dpi=250)
    print('done ->', out / 'entropy_calibration.md')


if __name__ == '__main__':
    main()
