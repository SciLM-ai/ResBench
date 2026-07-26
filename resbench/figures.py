"""Publication figures (matplotlib, PDF + PNG). EVAL.md Phase 8.

Encoding rules: axis identity (x/y/z) carries color (colorblind-validated
palette, fixed order); data source carries linestyle (reference solid,
generated dashed); the split-half band is a shaded region. Reference vs
generated in the CDF/parity panels: reference neutral gray, generated blue.
"""
import string

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

# Validated categorical palette (dataviz reference, light mode, slots 1-3).
AXIS_COLORS = {0: '#2a78d6', 1: '#eb6834', 2: '#1baf7a'}   # x, y, z
AXIS_NAMES = {0: 'x', 1: 'y', 2: 'z'}
REF_GRAY = '#52514e'
GEN_BLUE = '#2a78d6'
DIAG_GRAY = '#b0afa9'

SHORT = {
    'lobe': 'lobe',
    'channel:PV_SHOESTRING': 'PV shoestring',
    'channel:CB_LABYRINTH': 'CB labyrinth',
    'channel:CB_JIGSAW': 'CB jigsaw',
    'channel:SH_DISTAL': 'SH distal',
    'channel:SH_PROXIMAL': 'SH proximal',
    'channel:MEANDER_OXBOW': 'meander-oxbow',
    'delta': 'delta',
}

plt.rcParams.update({
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
    'legend.fontsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#e8e7e3', 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'figure.dpi': 110, 'savefig.dpi': 250,
})


def _save(fig, out_base):
    fig.savefig(f'{out_base}.pdf', bbox_inches='tight')
    fig.savefig(f'{out_base}.png', bbox_inches='tight')
    plt.close(fig)
    return [f'{out_base}.pdf', f'{out_base}.png']


def ntg_parity(report, out_base):
    """Generated vs reference mean NTG per environment, bootstrap CIs, y=x."""
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    lo = min(min(r['ref_ntg_ci'][1] for r in report.values()),
             min(r['gen_ntg_ci'][1] for r in report.values())) - 0.03
    hi = max(max(r['ref_ntg_ci'][2] for r in report.values()),
             max(r['gen_ntg_ci'][2] for r in report.values())) + 0.03
    ax.plot([lo, hi], [lo, hi], ls='--', lw=1, c=DIAG_GRAY, zorder=1)
    for lt, r in report.items():
        rm, rl, rh = r['ref_ntg_ci']
        gm, gl, gh = r['gen_ntg_ci']
        ax.errorbar(rm, gm, xerr=[[rm - rl], [rh - rm]],
                    yerr=[[gm - gl], [gh - gm]],
                    fmt='o', ms=5, c=GEN_BLUE, ecolor=REF_GRAY,
                    elinewidth=1, capsize=2, zorder=3)
        ax.annotate(SHORT[lt], (rm, gm), textcoords='offset points',
                    xytext=(5, 4), fontsize=7, color='#0b0b0b')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel('reference mean NTG (95% bootstrap CI)')
    ax.set_ylabel('generated mean NTG (95% bootstrap CI)')
    ax.set_title('Facies-proportion recovery per environment')
    return _save(fig, out_base)


def _curve_panel(ax, r, kind):
    """One environment's overlay panel: kind in {'gamma', 'tau'}."""
    ha, hb = r['halves']
    for a in (0, 1, 2):
        c = AXIS_COLORS[a]
        lags = np.arange(1, len(r['ref'][kind][a]) + 1)
        band_lo = np.minimum(ha[kind][a], hb[kind][a])
        band_hi = np.maximum(ha[kind][a], hb[kind][a])
        ax.fill_between(lags, band_lo, band_hi, color=c, alpha=0.18, lw=0)
        ax.plot(lags, r['ref'][kind][a], c=c, lw=1.4)
        ax.plot(lags, r['gen'][kind][a], c=c, lw=1.4, ls=(0, (4, 2)))


def _legend_handles():
    from matplotlib.lines import Line2D
    h = [Line2D([], [], c=AXIS_COLORS[a], lw=1.4, label=f'{AXIS_NAMES[a]} axis')
         for a in (0, 1, 2)]
    h += [Line2D([], [], c=REF_GRAY, lw=1.4, label='reference'),
          Line2D([], [], c=REF_GRAY, lw=1.4, ls=(0, (4, 2)), label='ResFlow'),
          plt.Rectangle((0, 0), 1, 1, fc=REF_GRAY, alpha=0.18, lw=0,
                        label='split-half band')]
    return h


def variogram_grid(report, out_base):
    """2x4 indicator-variogram overlays, one panel per environment."""
    fig, axes = plt.subplots(2, 4, figsize=(11, 5), sharex=True)
    for i, (lt, r) in enumerate(report.items()):
        ax = axes.flat[i]
        _curve_panel(ax, r, 'gamma')
        p = float(np.mean(r['ref']['ntg_mean']))
        ax.axhline(p * (1 - p), c=DIAG_GRAY, lw=0.8, ls=':')
        ax.set_title(f'({string.ascii_lowercase[i]}) {SHORT[lt]}')
        if i % 4 == 0:
            ax.set_ylabel(r'$\gamma(h)$')
        if i >= 4:
            ax.set_xlabel('lag $h$ (voxels)')
    for j in range(len(report), 8):
        axes.flat[j].set_visible(False)
    fig.legend(handles=_legend_handles(), ncol=6, loc='lower center',
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle('Indicator variograms: reference vs ResFlow '
                 '(dotted: sill $p(1-p)$)', y=1.0)
    fig.tight_layout()
    return _save(fig, out_base)


def connectivity_geobody_grid(report, out_base):
    """4x4 grid: rows 1-2 connectivity tau(h); rows 3-4 geobody-size CDFs."""
    fig, axes = plt.subplots(4, 4, figsize=(11, 10))
    items = list(report.items())
    for i, (lt, r) in enumerate(items):
        ax = axes.flat[i]
        _curve_panel(ax, r, 'tau')
        ax.set_ylim(0, 1.05)
        ax.set_title(f'({string.ascii_lowercase[i]}) {SHORT[lt]}')
        if i % 4 == 0:
            ax.set_ylabel(r'$\tau(h)$')
        if i >= 4:
            ax.set_xlabel('lag $h$ (voxels)')
    for i, (lt, r) in enumerate(items):
        ax = axes.flat[8 + i]
        for sizes, color, ls in ((r['ref']['sizes'], REF_GRAY, '-'),
                                 (r['gen']['sizes'], GEN_BLUE, (0, (4, 2)))):
            s = np.sort(np.log10(sizes))
            ax.plot(s, np.arange(1, len(s) + 1) / len(s), c=color, ls=ls, lw=1.4)
        ax.set_ylim(0, 1.02)
        ax.set_title(f'({string.ascii_lowercase[8 + i]}) {SHORT[lt]}')
        if i % 4 == 0:
            ax.set_ylabel('CDF')
        ax.set_xlabel(r'$\log_{10}$ geobody size (voxels)')
    for j in list(range(len(report), 8)) + list(range(8 + len(report), 16)):
        axes.flat[j].set_visible(False)
    from matplotlib.lines import Line2D
    handles = _legend_handles() + [
        Line2D([], [], c=REF_GRAY, lw=1.4, label='reference CDF'),
        Line2D([], [], c=GEN_BLUE, lw=1.4, ls=(0, (4, 2)), label='ResFlow CDF')]
    fig.legend(handles=handles, ncol=8, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle('Connectivity functions (top) and geobody-size CDFs (bottom)',
                 y=1.0)
    fig.tight_layout()
    return _save(fig, out_base)
