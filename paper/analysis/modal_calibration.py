"""Addendum C: well-conditional entropy calibration on modal patterns.

Per environment: engine conditional ensemble = existing mined draws +
targeted top-up (all matching the modal 1-well pattern), vs model K = 128
samples conditioned on the same (parameters, pattern). Metrics per A.5 /
C.4: entropy vs Chebyshev distance to the well column, mean |dH| overall
and near-field signed offset (d <= 6), split-half band of the engine
conditional ensemble as yardstick.

Also assembles the published benchmark asset
(results/well_conditional_reference/): merged accepted volumes + pattern +
well mask + parameter pointer per environment.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resbench import LAYER_TYPES, env_slug   # noqa: E402

SPLIT_SEED = 20260804
BIN_W, D_MAX = 2, 24


def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return np.where((p <= eps) | (p >= 1 - eps), 0.0, h)


def bin_means(field, dist):
    edges = np.arange(1, D_MAX + 1, BIN_W)
    out = np.full(len(edges), np.nan)
    for i, lo in enumerate(edges):
        m = (dist >= lo) & (dist < lo + BIN_W)
        if m.any():
            out[i] = float(field[m].mean())
    return edges, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modal-json', required=True)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--existing-dir', required=True)
    ap.add_argument('--topup-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--asset-dir', required=True)
    ap.add_argument('--tag', default='modal', choices=['modal', 'mixed'])
    args = ap.parse_args()

    conds = {c['environment']: c
             for c in json.loads(Path(args.modal_json).read_text())}
    rows, curves = [], {}
    for ei, lt in enumerate([lt for lt in LAYER_TYPES if lt in conds]):
        slug = env_slug(lt)
        rec = conds[lt]
        md = np.load(Path(args.model_dir) / f'{slug}_{args.tag}.npz',
                     allow_pickle=True)
        vols_m, mask = md['volumes'], md['mask'].astype(np.uint8)
        pat = np.array(rec['pattern'], np.int8)
        assert np.array_equal(np.asarray(md['pattern']), pat)

        parts = [np.load(Path(args.existing_dir) / f'{slug}_{args.tag}_existing.npz',
                         allow_pickle=True)['volumes']]
        tf = Path(args.topup_dir) / slug / f'{slug}_{args.tag}.npz'
        if tf.exists():
            tv = np.load(tf, allow_pickle=True)['volumes']
            if len(tv):
                parts.append(tv)
        vr = np.concatenate(parts)
        # every reference member must actually carry the pattern
        assert (vr[:, 32, 32, :] == pat).all(), f'{lt}: non-matching member'

        H_m = entropy(vols_m.mean(0))
        H_r = entropy(vr.mean(0))
        wells = mask > 0
        assert np.all(H_m[wells] == 0.0)
        dist = ndimage.distance_transform_cdt(1 - mask, metric='chessboard')

        perm = np.random.default_rng([SPLIT_SEED, ei]).permutation(len(vr))
        H_a = entropy(vr[perm[:len(vr) // 2]].mean(0))
        H_b = entropy(vr[perm[len(vr) // 2:]].mean(0))

        edges, c_m = bin_means(H_m, dist)
        _, c_r = bin_means(H_r, dist)
        _, band = bin_means(np.abs(H_a - H_b), dist)
        near = slice(0, 3)  # Chebyshev distance <= 6
        rec_out = {
            'environment': lt, 'n_ref': int(len(vr)),
            'n_existing': int(rec['n_existing']),
            'pattern_ntg': float(pat.mean()),
            'mae_bits': float(np.abs(H_m - H_r).mean()),
            'band_bits': float(np.abs(H_a - H_b).mean()),
            'near_signed': float(np.nanmean(c_m[near] - c_r[near])),
            'under_dispersed_nearfield': bool(
                np.nanmean(c_m[near] - c_r[near]) < 0),
        }
        rec_out['verdict'] = ('inside' if rec_out['mae_bits'] <= rec_out['band_bits']
                              else 'near' if rec_out['mae_bits'] <= 2 * rec_out['band_bits']
                              else 'outside')
        rows.append(rec_out)
        curves[lt] = {'edges': edges.tolist(), 'model': c_m.tolist(),
                      'ref': c_r.tolist(), 'band': band.tolist()}
        print(f"{lt}: n_ref={len(vr)} mae={rec_out['mae_bits']:.4f} "
              f"band={rec_out['band_bits']:.4f} near_signed="
              f"{rec_out['near_signed']:+.4f} {rec_out['verdict']}", flush=True)

        d = Path(args.asset_dir)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            d / f'{slug}_{args.tag}.npz', volumes=vr.astype(np.int8), pattern=pat,
            well_mask=mask, ref_id=rec['row_id'],
            params_row_index=rec['row_index'])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({'rows': rows, 'curves': curves},
              open(out / f'{args.tag}_calibration.json', 'w'), indent=2)

    md_l = ['## Well-conditional calibration on modal patterns '
            '(EVAL.md Addendum C)', '',
            '| environment | well data | n ref | MAE (bits) | band | '
            'near-field signed | verdict |', '|' + '---|' * 7]
    for r in rows:
        wd = ('dry (all shale)' if r['pattern_ntg'] == 0
              else 'full sand' if r['pattern_ntg'] == 1
              else f"NTG {r['pattern_ntg']:.2f}")
        md_l.append(f"| {r['environment']} | {wd} | {r['n_ref']} "
                    f"| {r['mae_bits']:.4f} | {r['band_bits']:.4f} "
                    f"| {r['near_signed']:+.4f} | {r['verdict']}"
                    f"{' (under-disp.)' if r['under_dispersed_nearfield'] else ''} |")
    (out / f'{args.tag}_calibration.md').write_text('\n'.join(md_l) + '\n')
    print('\n'.join(md_l))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from resbench.figures import AXIS_COLORS, REF_GRAY, SHORT
    BLUE, ORANGE = AXIS_COLORS[0], AXIS_COLORS[1]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5), sharex=True, sharey=True)
    for i, lt in enumerate(curves):
        ax = axes.flat[i]
        c = curves[lt]
        e = np.array(c['edges'])
        ref = np.array(c['ref'], float)
        band = np.array(c['band'], float)
        ax.fill_between(e, ref - band, ref + band, color=ORANGE, alpha=0.18, lw=0)
        ax.plot(e, ref, c=ORANGE, ls=(0, (4, 2)), lw=1.4)
        ax.plot(e, np.array(c['model'], float), c=BLUE, lw=1.4)
        ax.set_title(SHORT[lt])
        if i >= 4:
            ax.set_xlabel('Chebyshev distance to well')
        if i % 4 == 0:
            ax.set_ylabel('mean entropy (bits)')
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], c=BLUE, lw=1.4, label='ResFlow conditional (K=128)'),
        Line2D([], [], c=ORANGE, lw=1.4, ls=(0, (4, 2)),
               label='ResMill conditional (modal rejection, ± split-half band)')],
        ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.suptitle('Well-conditional entropy calibration, modal patterns '
                 '(all 8 environments)', y=1.0)
    fig.tight_layout()
    fig.savefig(out / f'{args.tag}_calibration.pdf', bbox_inches='tight')
    fig.savefig(out / f'{args.tag}_calibration.png', bbox_inches='tight', dpi=250)
    print('done ->', out / f'{args.tag}_calibration.md')


if __name__ == '__main__':
    main()
