"""CFG sensitivity sweep scoring (post-hoc, item 4; no master-table rescoring).

For lobe and channel:PV_SHOESTRING at CFG {1.0, 1.5, 2.0, 3.0} (128 samples
each, manifest rows 0-127), reports vs the frozen reference statistics:
|dNTG|, signed gamma_z plateau deviation (mean over the last quarter of
z lags), largest-body-fraction delta, tau MAE (lobe), and per-volume
compartmentalization frequency (% volumes with tau_z < 0.99).

Outputs: cfg_sweep.md + cfg_sweep.json + cfg_sweep.{pdf,png} line plots.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resbench import MAX_LAGS                                  # noqa: E402
from resbench import io, metrics                               # noqa: E402
from resbench.stats import env_summary                         # noqa: E402

SWEEP_ENVS = ['lobe', 'channel:PV_SHOESTRING']
CFG_SCALES = [1.0, 1.5, 2.0, 3.0]
TAU_Z_THRESHOLD = 0.99


def comp_freq(vols):
    n = 0
    for v in vols:
        labels, _ = metrics.label_geobodies(v)
        same, pairs = metrics.connectivity_counts(v, (1, 1, 16), labels=labels)[2]
        if pairs.sum() and same.sum() / pairs.sum() < TAU_Z_THRESHOLD:
            n += 1
    return n / len(vols)


def plateau(gamma_z):
    return float(gamma_z[-max(1, len(gamma_z) // 4):].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep-dir', required=True)
    ap.add_argument('--ref-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    rows = []
    comp_ref = {}
    for lt in SWEEP_ENVS:
        slug = lt.replace(':', '_')
        rid, rv = io.load_volume_dir(Path(args.ref_dir) / slug)
        # Reference statistics from the MATCHED 128 rows (same ids as the
        # sweep), so deltas are free of row-mix mismatch vs the 512-row run.
        gid0, gv0 = io.load_volume_dir(
            Path(args.sweep_dir) / f'cfg_{CFG_SCALES[0]:.1f}' / slug)
        gv0_al, rv_sub, _ = io.align(gid0, gv0, rid, rv)
        ref = env_summary(rv_sub, MAX_LAGS)
        comp_ref[lt] = comp_freq(rv_sub)
        for cfg in CFG_SCALES:
            gid, gv = io.load_volume_dir(
                Path(args.sweep_dir) / f'cfg_{cfg:.1f}' / slug)
            gv, _, _ = io.align(gid, gv, rid, rv)
            gs = env_summary(gv, MAX_LAGS)
            row = {
                'environment': lt, 'cfg': cfg, 'n': len(gv),
                'dntg': float(gs['ntg'].mean() - ref['ntg'].mean()),
                'gamma_z_plateau_dev': plateau(gs['gamma'][2]) - plateau(ref['gamma'][2]),
                'largest_frac': float(gs['largest_frac'].mean()),
                'ref_largest_frac': float(ref['largest_frac'].mean()),
                'dlargest_frac': float(gs['largest_frac'].mean()
                                       - ref['largest_frac'].mean()),
                'comp_freq': comp_freq(gv),
                'comp_freq_ref': comp_ref[lt],
            }
            if lt == 'lobe':
                row['tau_mae'] = float(np.concatenate(
                    [np.abs(gs['tau'][a] - ref['tau'][a]) for a in (0, 1, 2)]).mean())
            rows.append(row)
            print(f"{lt} cfg={cfg}: dNTG={row['dntg']:+.4f} "
                  f"gz_plateau={row['gamma_z_plateau_dev']:+.4f} "
                  f"largest={row['largest_frac']:.3f} comp={row['comp_freq'] * 100:.1f}%"
                  f" (ref {comp_ref[lt] * 100:.1f}%)", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / 'cfg_sweep.json', 'w'), indent=2)

    md = ['## CFG sensitivity sweep (post-hoc; 128 samples/point, fresh seeds)', '',
          '| environment | CFG | ΔNTG | γ_z plateau dev | largest frac (ref) '
          '| comp. freq (ref) | τ MAE |',
          '|' + '---|' * 7]
    for r in rows:
        ref_lf = r['ref_largest_frac']
        md.append(
            f"| {r['environment']} | {r['cfg']:.1f} | {r['dntg']:+.4f} "
            f"| {r['gamma_z_plateau_dev']:+.4f} "
            f"| {r['largest_frac']:.3f} ({ref_lf:.3f}) "
            f"| {r['comp_freq'] * 100:.1f}% ({r['comp_freq_ref'] * 100:.1f}%) "
            f"| {r.get('tau_mae', float('nan')):.4f} |")
    (out / 'cfg_sweep.md').write_text('\n'.join(md) + '\n')

    # line plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from resbench.figures import AXIS_COLORS, REF_GRAY, SHORT
    panels = [('dntg', r'$\Delta$NTG', True),
              ('gamma_z_plateau_dev', r'$\gamma_z$ plateau dev', True),
              ('dlargest_frac', r'$\Delta$ largest frac', True),
              ('comp_freq', 'compartmentalized vol. frac', False)]
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9))
    for ax, (key, label, zero_line) in zip(axes, panels):
        for ei, lt in enumerate(SWEEP_ENVS):
            vals = [r[key] for r in rows if r['environment'] == lt]
            ax.plot(CFG_SCALES, vals, 'o-', c=list(AXIS_COLORS.values())[ei],
                    lw=1.4, ms=4, label=SHORT[lt])
            if key == 'comp_freq':
                ax.axhline(comp_ref[lt], c=list(AXIS_COLORS.values())[ei],
                           lw=0.8, ls=':')
        if zero_line:
            ax.axhline(0, c=REF_GRAY, lw=0.8, ls=':')
        ax.set_xlabel('CFG scale')
        ax.set_title(label, fontsize=9)
        ax.set_xticks(CFG_SCALES)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle('CFG sensitivity (post-hoc; dotted = reference level)', y=1.04)
    fig.tight_layout()
    fig.savefig(out / 'cfg_sweep.pdf', bbox_inches='tight')
    fig.savefig(out / 'cfg_sweep.png', bbox_inches='tight', dpi=250)
    print('done ->', out / 'cfg_sweep.md')


if __name__ == '__main__':
    main()
