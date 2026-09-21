"""The standard figure set from a results.json, as PDF.

One page per task: for every check, one bar per environment at its s value,
with the matched (1) and close (2) thresholds drawn across. Plus an overview
heat map of mean s per (task, check). Vector output only.
"""
import json
from pathlib import Path

import numpy as np

TASKS = ('unconditional', 'well_conditioned', 'field_scale')


def make(results_json, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    res = json.load(open(results_json))
    rows = res['rows']
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    written = []
    for task in TASKS:
        rs = [r for r in rows if r['task'] == task]
        if not rs:
            continue
        checks = sorted({r['check'] for r in rs}, key=lambda c: [r['check'] for r in rs].index(c))
        envs = sorted({r['environment'] for r in rs})
        fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(checks) + 2), 4.2), constrained_layout=True)
        w = 0.8 / max(len(envs), 1)
        for j, e in enumerate(envs):
            vals = [next((r['s'] for r in rs if r['check'] == c and r['environment'] == e), np.nan) for c in checks]
            ax.bar(np.arange(len(checks)) + j * w - 0.4 + w / 2, vals, w, label=e.replace('channel:', ''))
        ax.axhline(1, color='k', lw=0.8, ls='--'); ax.axhline(2, color='k', lw=0.8, ls=':')
        ax.set_xticks(range(len(checks))); ax.set_xticklabels(checks, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel('s  (band units; 1 = as close as ResMill to itself)')
        ax.set_title(f'{task}   score {res["score"]["tasks"][task]["score"]:.2f}', loc='left')
        ax.legend(fontsize=7, ncol=2, frameon=False)
        p = out / f'scores_{task}.pdf'; fig.savefig(p); plt.close(fig); written.append(p)
    # overview
    tasks = [t for t in TASKS if any(r['task'] == t for r in rows)]
    checks = []
    for r in rows:
        if r['check'] not in checks:
            checks.append(r['check'])
    grid = np.full((len(tasks), len(checks)), np.nan)
    for i, t in enumerate(tasks):
        for j, c in enumerate(checks):
            v = [r['s'] for r in rows if r['task'] == t and r['check'] == c]
            if v:
                grid[i, j] = np.mean(v)
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(checks) + 2), 1.1 * len(tasks) + 1.6), constrained_layout=True)
    im = ax.imshow(np.clip(grid, 0, 3), cmap='RdYlGn_r', vmin=0, vmax=3, aspect='auto')
    for i in range(len(tasks)):
        for j in range(len(checks)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f'{grid[i, j]:.2f}', ha='center', va='center', fontsize=8)
    ax.set_xticks(range(len(checks))); ax.set_xticklabels(checks, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(len(tasks))); ax.set_yticklabels(tasks)
    ax.set_title(f'mean s per check   overall {res["score"]["overall"]:.2f}', loc='left')
    fig.colorbar(im, ax=ax, label='s (clipped at 3)')
    p = out / 'scores_overview.pdf'; fig.savefig(p); plt.close(fig); written.append(p)
    return written
