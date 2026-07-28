"""Foundation-vs-specialist comparison assets (EVAL.md Addendum E, E.5-E.6).

Not a re-scoring: the master table rows and split-half bands are read
verbatim from the frozen results/metrics.parquet; specialist cell values
come from their own resbench.run outputs, with verdicts recomputed against
the FROZEN master band (E.5 — the specialist run's own band can differ for
environments whose index in the filtered env list differs from the master
run's; absolute cell values are band-independent).

Outputs to results/specialists/:
  comparison.parquet / comparison.md    3 rows per env (band / foundation / specialist)
  variogram_overlay_<tag>.{pdf,png}     x/y/z panels, 3 sources
  conn_geobody_overlay_<tag>.{pdf,png}  tau x/y/z + geobody-size CDF, 3 sources
  ntg_hist_<tag>.{pdf,png}              per-volume NTG histograms, 3 sources
  compartmentalization.md / .json       reference + foundation + specialist rows

Usage: python analysis/build_specialists_comparison.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RB))
from resbench import env_slug                              # noqa: E402
from resbench import io as rio                             # noqa: E402
from resbench.stats import CELLS, verdict                  # noqa: E402
from resbench import figures as F                          # noqa: E402
import matplotlib.pyplot as plt                            # noqa: E402

EVAL = Path('/work/08405/ilgar/vista/resbench_eval')
OUT = RB / 'results' / 'specialists'
ENVS = [('channel:PV_SHOESTRING', 'pv_shoestring'), ('lobe', 'lobe')]

CELL_LABELS = {
    'abs_dntg': '|dNTG|',
    'variogram_mae': 'variogram MAE/sill',
    'connectivity_mae': 'connectivity MAE',
    'geobody_w1': 'geobody W1 (log10)',
    'abs_dlargest_frac': '|d largest frac|',
}
SRC_COLORS = {'foundation': F.GEN_BLUE, 'specialist': '#1baf7a'}
SRC_LS = {'foundation': '--', 'specialist': '-.'}


def load_master():
    return pd.read_parquet(RB / 'results' / 'metrics.parquet')


def comparison_table(master):
    rows = []
    for env, tag in ENVS:
        m = master[master['environment'] == env].iloc[0]
        s = pd.read_parquet(OUT / f'{tag}_metrics.parquet')
        s = s[s['environment'] == env].iloc[0]
        band = {c: float(m[f'{c}_band']) for c in CELLS}
        # sanity: when the env index matches the master run (lobe = 0), the
        # specialist run's own band must reproduce the frozen band exactly
        own_band = {c: float(s[f'{c}_band']) for c in CELLS}
        band_match = all(np.isclose(own_band[c], band[c], rtol=1e-12) for c in CELLS)
        print(f'{env}: specialist-run band == frozen band: {band_match} '
              f'(expected {"True" if env == "lobe" else "False (index differs; frozen band used)"})')

        rows.append({'environment': env, 'source': 'split-half band',
                     **{c: band[c] for c in CELLS},
                     **{f'{c}_verdict': '' for c in CELLS},
                     'well_mismatch': 'n/a'})
        rows.append({'environment': env, 'source': 'foundation',
                     **{c: float(m[c]) for c in CELLS},
                     **{f'{c}_verdict': str(m[f'{c}_verdict']) for c in CELLS},
                     'well_mismatch': f"{m['well_mismatch_pct']:.3f} ({m['well_verdict']})"})
        rows.append({'environment': env, 'source': 'specialist (matched 40 ep)',
                     **{c: float(s[c]) for c in CELLS},
                     **{f'{c}_verdict': verdict(float(s[c]), band[c]) for c in CELLS},
                     'well_mismatch': 'n/a*'})
        ext_path = OUT / f'{tag}_ext_metrics.parquet'
        if ext_path.exists():
            x = pd.read_parquet(ext_path)
            x = x[x['environment'] == env].iloc[0]
            rows.append({'environment': env,
                         'source': 'specialist (ext, union argmin)',
                         **{c: float(x[c]) for c in CELLS},
                         **{f'{c}_verdict': verdict(float(x[c]), band[c])
                            for c in CELLS},
                         'well_mismatch': 'n/a*'})
    return pd.DataFrame(rows)


def render_md(table):
    lines = ['| environment | source | ' + ' | '.join(CELL_LABELS.values())
             + ' | well mismatch % |',
             '|' + '---|' * (len(CELL_LABELS) + 3)]
    for _, r in table.iterrows():
        cells = []
        for c in CELL_LABELS:
            v = f"{r[c]:.4f}"
            if r[f'{c}_verdict']:
                v += f" ({r[f'{c}_verdict']})"
            cells.append(v)
        lines.append(f"| {r['environment']} | {r['source']} | "
                     + ' | '.join(cells) + f" | {r['well_mismatch']} |")
    lines.append('')
    lines.append('\\* Well-conditioned generation out of scope for specialists '
                 '(EVAL.md Addendum E.1): exactitude is guaranteed by hard '
                 'replacement independent of weights.')
    return '\n'.join(lines)


def parity_outcome(table):
    """E.5 criterion: foundation tier >= specialist tier in EVERY column."""
    rank = {'inside': 2, 'near': 1, 'outside': 0}
    out = {}
    for env, _ in ENVS:
        f = table[(table.environment == env) & (table.source == 'foundation')].iloc[0]
        srows = table[(table.environment == env)
                      & (table.source.str.startswith('specialist'))]
        s = srows.iloc[-1]   # ext (union-argmin) row when present, else 40 ep
        per_col = {c: (f[f'{c}_verdict'], s[f'{c}_verdict'],
                       rank[f[f'{c}_verdict']] >= rank[s[f'{c}_verdict']])
                   for c in CELLS}
        out[env] = {'substantiated': all(v[2] for v in per_col.values()),
                    'specialist_source': str(s['source']),
                    'columns': per_col}
    return out


def _band(ax, lags, ha, hb, kind, axis):
    lo = np.minimum(np.asarray(ha[kind][axis]), np.asarray(hb[kind][axis]))
    hi = np.maximum(np.asarray(ha[kind][axis]), np.asarray(hb[kind][axis]))
    ax.fill_between(lags, lo, hi, color=F.REF_GRAY, alpha=0.18, lw=0,
                    label='split-half band')


def overlay_curves(env, tag, mrep, srep):
    """Variogram x/y/z; then tau x/y/z + geobody CDF. 3 sources each,
    plus the extension specialist as a 4th curve when its report exists."""
    r = mrep[env]
    sgen = srep[env]['gen']
    ext_path = OUT / f'{tag}_ext_report.npy'
    xgen = (np.load(ext_path, allow_pickle=True)[0][env]['gen']
            if ext_path.exists() else None)
    ha, hb = r['halves']
    paths = []
    for kind, ylabel, fname in [
            ('gamma', 'directional variogram γ(h)', f'variogram_overlay_{tag}'),
            ('tau', 'connectivity τ(h)', f'conn_overlay_{tag}')]:
        fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.7), sharey=(kind == 'tau'))
        for a, ax in enumerate(axes):
            lags = np.arange(1, len(np.asarray(r['ref'][kind][a])) + 1)
            _band(ax, lags, ha, hb, kind, a)
            ax.plot(lags, np.asarray(r['ref'][kind][a]), c=F.REF_GRAY, lw=1.6,
                    label='reference')
            ax.plot(lags, np.asarray(r['gen'][kind][a]), c=SRC_COLORS['foundation'],
                    ls=SRC_LS['foundation'], lw=1.6, label='foundation')
            ax.plot(lags, np.asarray(sgen[kind][a]), c=SRC_COLORS['specialist'],
                    ls=SRC_LS['specialist'], lw=1.6, label='specialist (40 ep)')
            if xgen is not None:
                ax.plot(lags, np.asarray(xgen[kind][a]), c='#eb6834', ls=':',
                        lw=1.6, label='specialist (ext)')
            ax.set_title(f'{F.SHORT[env]} — {F.AXIS_NAMES[a]}')
            ax.set_xlabel('lag h (voxels)')
            if a == 0:
                ax.set_ylabel(ylabel)
        axes[-1].legend(frameon=False, loc='best')
        fig.tight_layout()
        paths += F._save(fig, OUT / fname)

    # geobody-size CDF (bodies pooled over the ensemble), log10 size
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    series = [
        ('reference', r['ref']['sizes'], F.REF_GRAY, '-'),
        ('foundation', r['gen']['sizes'], SRC_COLORS['foundation'],
         SRC_LS['foundation']),
        ('specialist (40 ep)', sgen['sizes'], SRC_COLORS['specialist'],
         SRC_LS['specialist'])]
    if xgen is not None:
        series.append(('specialist (ext)', xgen['sizes'], '#eb6834', ':'))
    for label, sizes, color, ls in series:
        s = np.sort(np.log10(np.asarray(sizes, dtype=float)))
        ax.plot(s, np.linspace(0, 1, len(s)), c=color, ls=ls, lw=1.6, label=label)
    ax.set_xlabel('log10 geobody size (voxels)')
    ax.set_ylabel('CDF over bodies')
    ax.set_title(f'{F.SHORT[env]} — geobody sizes')
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout()
    paths += F._save(fig, OUT / f'geobody_cdf_{tag}')
    return paths


def ntg_hist(env, tag):
    slug = env_slug(env)
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    bins = np.linspace(0, 1, 41)
    series = [
        ('reference', EVAL / 'reference', F.REF_GRAY, '-'),
        ('foundation', EVAL / 'ensemble_a', SRC_COLORS['foundation'],
         SRC_LS['foundation']),
        ('specialist (40 ep)', EVAL / f'specialist_{tag}' / 'ensemble_a',
         SRC_COLORS['specialist'], SRC_LS['specialist'])]
    if (EVAL / f'specialist_{tag}_ext' / 'ensemble_a').exists():
        series.append(('specialist (ext)',
                       EVAL / f'specialist_{tag}_ext' / 'ensemble_a',
                       '#eb6834', ':'))
    for label, d, color, ls in series:
        _, vols = rio.load_ensemble(str(d))[env]
        ntg = vols.reshape(len(vols), -1).mean(axis=1)
        ax.hist(ntg, bins=bins, density=True, histtype='step', color=color,
                ls=ls, lw=1.6, label=label)
    ax.set_xlabel('per-volume NTG')
    ax.set_ylabel('density')
    ax.set_title(f'{F.SHORT[env]} — per-volume NTG')
    ax.legend(frameon=False, loc='best')
    fig.tight_layout()
    return F._save(fig, OUT / f'ntg_hist_{tag}')


def compartmentalization():
    master = json.load(open(RB / 'results' / 'posthoc' / 'posthoc_geobody.json'))
    mrows = {r['environment']: r for r in master['compartmentalization']}
    lines = ['| environment | source | frac τ_z<0.99 [95% CI] | '
             'floor-to-surface span frac [95% CI] | largest-body frac median [IQR] |',
             '|---|---|---|---|---|']
    blob = {}
    for env, tag in ENVS:
        m = mrows[env]
        sp = json.load(open(OUT / f'posthoc_{tag}' / 'posthoc_geobody.json'))
        s = {r['environment']: r for r in sp['compartmentalization']}[env]
        assert np.isclose(s['ref_frac_tauz_lt_099'], m['ref_frac_tauz_lt_099']), \
            f'{env}: reference mismatch between master and specialist posthoc'
        entries = [('reference', m, 'ref'), ('foundation', m, 'gen'),
                   ('specialist (40 ep)', s, 'gen')]
        ext_dir = OUT / f'posthoc_{tag}_ext'
        if (ext_dir / 'posthoc_geobody.json').exists():
            xp = json.load(open(ext_dir / 'posthoc_geobody.json'))
            x = {r['environment']: r for r in xp['compartmentalization']}[env]
            entries.append(('specialist (ext)', x, 'gen'))
        for label, d, pre in entries:
            lines.append(
                f"| {env} | {label} | "
                f"{d[f'{pre}_frac_tauz_lt_099']:.3f} "
                f"[{d[f'{pre}_tauz_ci'][0]:.3f}, {d[f'{pre}_tauz_ci'][1]:.3f}] | "
                f"{d[f'{pre}_frac_span']:.3f} "
                f"[{d[f'{pre}_span_ci'][0]:.3f}, {d[f'{pre}_span_ci'][1]:.3f}] | "
                f"{d[f'{pre}_largest_median']:.3f} "
                f"[{d[f'{pre}_largest_iqr'][0]:.3f}, {d[f'{pre}_largest_iqr'][1]:.3f}] |")
            blob[f'{env}|{label}'] = {k: d[k] for k in d if k.startswith(pre)}
    (OUT / 'compartmentalization.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'compartmentalization.json').write_text(json.dumps(blob, indent=2))
    return '\n'.join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    master = load_master()
    table = comparison_table(master)
    table.to_parquet(OUT / 'comparison.parquet', index=False)
    md = render_md(table)
    (OUT / 'comparison.md').write_text(md + '\n')
    print(md)

    parity = parity_outcome(table)
    (OUT / 'parity.json').write_text(json.dumps(
        {e: {'substantiated': p['substantiated'],
             'columns': {c: {'foundation': v[0], 'specialist': v[1],
                             'foundation_at_least_as_good': v[2]}
                         for c, v in p['columns'].items()}}
         for e, p in parity.items()}, indent=2))
    for e, p in parity.items():
        print(f"PARITY {e}: {'SUBSTANTIATED' if p['substantiated'] else 'NOT substantiated'}")

    mrep = np.load(RB / 'results' / 'report.npy', allow_pickle=True)[0]
    for env, tag in ENVS:
        srep = np.load(OUT / f'{tag}_report.npy', allow_pickle=True)[0]
        for p in overlay_curves(env, tag, mrep, srep):
            print('fig:', p)
        for p in ntg_hist(env, tag):
            print('fig:', p)
    print(compartmentalization())


if __name__ == '__main__':
    main()
