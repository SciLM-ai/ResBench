"""resbench — validate and score a submission.

    resbench validate ./my_submission
    resbench score    ./my_submission --reference ./reference
    resbench figures  results.json

`download` reports where the reference data must come from rather than
fetching it: the reference is not hosted yet (see STATUS.md), so pretending
otherwise would just fail confusingly later.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

from . import checks as _checks
from . import bands, io, score

REFERENCE_ENV = 'RESBENCH_REFERENCE'


def _reference_root(arg):
    root = arg or os.environ.get(REFERENCE_ENV)
    if not root:
        raise SystemExit(
            'No reference data. Pass --reference PATH or set '
            f'{REFERENCE_ENV}. See STATUS.md: the reference is not hosted yet.')
    return Path(root)


def cmd_download(a):
    print(f'The ResBench reference is not hosted yet (see STATUS.md).\n'
          f'Point ResBench at a local copy with --reference PATH or '
          f'{REFERENCE_ENV}=PATH.')
    return 1


def cmd_validate(a):
    root = Path(a.submission)
    meta = io.read_meta(root)
    problems, found = [], 0
    for task in io.TASKS:
        for env in io.ENVIRONMENTS:
            shape = io.NATIVE_SHAPE if task != 'field_scale' else io.FIELD_EXTENT[env]
            kinds = ['fields'] if task == 'field_scale' else ['samples', 'repeats']
            for kind in kinds:
                d = io.part_dir(root, task, env, kind)
                if not d.exists():
                    problems.append(f'missing: {d.relative_to(root)}')
                    continue
                try:
                    ids, vols = next(io.iter_shards(d))
                    io.check_shape(vols, shape, str(d.relative_to(root)))
                    io.check_binary(vols, str(d.relative_to(root)))
                    found += 1
                except io.SubmissionError as e:
                    problems.append(str(e))
    print(f'submission: {root}')
    if meta:
        print('  ' + ', '.join(f'{k}={v}' for k, v in list(meta.items())[:6]))
    print(f'  {found} parts readable, {len(problems)} problems')
    for p in problems[:20]:
        print(f'    - {p}')
    if len(problems) > 20:
        print(f'    ... and {len(problems) - 20} more')
    return 1 if problems else 0


def _score_part(mod, model_dir, ref_dir, ctx):
    m = mod.merge([mod.summarize(v, ctx) for _, v in io.iter_shards(model_dir)])
    r = mod.merge([mod.summarize(v, ctx) for _, v in io.iter_shards(ref_dir)])
    d = mod.compare(m, r)
    _, ref_vols = next(io.iter_shards(ref_dir))
    b = bands.band_for(mod, ref_vols, ctx)
    return score.s_for_check(d['parts'], b), d


def cmd_score(a):
    root, ref = Path(a.submission), _reference_root(a.reference)
    rows = []
    for task in io.TASKS:
        for env in io.ENVIRONMENTS:
            kind = 'fields' if task == 'field_scale' else 'samples'
            mdir = io.part_dir(root, task, env, kind)
            rdir = ref / ('fields' if task == 'field_scale' else 'volumes') / io.SLUG[env]
            if not (mdir.exists() and rdir.exists()):
                continue
            ctx = {'azimuth': 0.0}
            for mod in _checks.needing(task, 'samples'):
                try:
                    s, _ = _score_part(mod, mdir, rdir, ctx)
                except (io.SubmissionError, StopIteration) as e:
                    print(f'  skipped {task}/{env}/{mod.NAME}: {e}', file=sys.stderr)
                    continue
                rows.append({'task': task, 'environment': env,
                             'check': mod.NAME, 's': s})
    if not rows:
        raise SystemExit('Nothing scored. Check --reference and the submission layout.')
    tree = score.aggregate(rows)
    meta = io.read_meta(root)
    header = (f"ResBench v1 · {len({r['environment'] for r in rows})} environments"
              + (f" · sampler {meta['sampler']}" if 'sampler' in meta else '')
              + (f" · NFE {meta['nfe']}" if 'nfe' in meta else ''))
    print(score.format_table(tree, header))
    out = Path(a.out or 'results.json')
    out.write_text(json.dumps({'meta': meta, 'rows': rows, 'score': tree}, indent=2))
    print(f'\nwrote {out}')
    return 0


def cmd_figures(a):
    from . import figures as _f
    print(f'figures for {a.results} -> {a.out} (PDF)')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='resbench')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('download'); p.set_defaults(fn=cmd_download)
    p = sub.add_parser('validate'); p.add_argument('submission'); p.set_defaults(fn=cmd_validate)
    p = sub.add_parser('score'); p.add_argument('submission')
    p.add_argument('--reference'); p.add_argument('--out')
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser('figures'); p.add_argument('results')
    p.add_argument('--out', default='figures'); p.set_defaults(fn=cmd_figures)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
