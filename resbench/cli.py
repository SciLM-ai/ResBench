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


def _envs(a):
    if not getattr(a, 'envs', None):
        return list(io.ENVIRONMENTS)
    chosen = [e.strip() for e in a.envs.split(',') if e.strip()]
    bad = [e for e in chosen if e not in io.ENVIRONMENTS]
    if bad:
        raise SystemExit(f'unknown environment(s) {bad}; choose from {list(io.ENVIRONMENTS)}')
    return [e for e in io.ENVIRONMENTS if e in chosen]


def cmd_validate(a):
    root = Path(a.submission)
    meta = io.read_meta(root)
    problems, found = [], 0
    for task in io.TASKS:
        for env in _envs(a):
            shape = io.NATIVE_SHAPE if task != 'field_scale' else io.FIELD_EXTENT[env]
            kinds = ['fields'] if task == 'field_scale' else ['samples', 'repeats']
            for kind in kinds:
                d = io.part_dir(root, task, env, kind)
                if not d.exists():
                    problems.append(f'missing: {d.relative_to(root)}')
                    continue
                try:
                    if kind == 'repeats':
                        for cond, (ids, vols, _) in io.load_repeats(d, task, str(d.relative_to(root))).items():
                            w = f'{d.relative_to(root)}/{cond}'
                            io.check_shape(vols, shape, w)
                            io.check_binary(vols, w)
                            if len(vols) < io.K_REPEATS:
                                raise io.SubmissionError(f'{w}: {len(vols)} runs, need {io.K_REPEATS}')
                    else:
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


def _shard_ctx(ctx, ids, targets, where):
    """Per-shard ctx. Checks never see ids, so anything that varies per volume
    has to be resolved here, in the shard's own order, before summarize."""
    if targets is None:
        return ctx
    c = dict(ctx)
    c['target_ntg'] = io.targets_for(ids, targets, where)
    return c


def _score_part(mod, model_dir, ref_dir, ctx, targets=None):
    m = mod.merge([mod.summarize(v, _shard_ctx(ctx, i, targets, str(model_dir)))
                   for i, v in io.iter_shards(model_dir)])
    r = mod.merge([mod.summarize(v, _shard_ctx(ctx, i, targets, str(ref_dir)))
                   for i, v in io.iter_shards(ref_dir)])
    d = mod.compare(m, r)
    ref_ids, ref_vols = next(io.iter_shards(ref_dir))
    b = bands.band_for(mod, ref_vols,
                       _shard_ctx(ctx, ref_ids, targets, str(ref_dir)))
    return score.s_for_check(d['parts'], b), d


def _score_repeats(mod, model_dir, ref_dir, task, where):
    """Score one repeats-based check: per condition, many runs of one input.

    The band is ResMill's own ensemble split in half at the same condition,
    so `s = 1` means the model's spread is as far from ResMill's as two halves
    of ResMill are from each other.
    """
    model = io.load_repeats(model_dir, task, where=f'{where} (submission)')
    ref = io.load_repeats(ref_dir, task, where=f'{where} (reference)')
    m_parts, r_parts, groups = [], [], {}
    for cond in io.CONDITIONS[task]:
        _, mv, _ = model[cond]
        _, rv, extras = ref[cond]
        if len(mv) < io.K_REPEATS:
            raise io.SubmissionError(f'{where}/{cond}: {len(mv)} runs, need {io.K_REPEATS}')
        ctx = io.repeats_ctx(cond, extras)      # the reference defines the well
        m_parts.append(mod.summarize(mv, ctx))
        r_parts.append(mod.summarize(rv, ctx))
        groups[cond] = (rv, ctx)
    d = mod.compare(mod.merge(m_parts), mod.merge(r_parts))
    b = bands.band_for_repeats(mod, groups)
    return score.s_for_check(d['parts'], b), d


def cmd_score(a):
    root, ref = Path(a.submission), _reference_root(a.reference)
    if not (ref / 'manifest.csv').exists():
        raise SystemExit(f'{ref}: no manifest.csv. net_to_gross scores each volume against '
                         f'the sand fraction it was conditioned on, which only the manifest '
                         f'records; build the reference with tools/build_*_reference.py.')
    rows = []
    for task in io.TASKS:
        for env in _envs(a):
            kind = 'fields' if task == 'field_scale' else 'samples'
            mdir = io.part_dir(root, task, env, kind)
            rdir = ref / ('fields' if task == 'field_scale' else 'volumes') / io.SLUG[env]
            if not (mdir.exists() and rdir.exists()):
                continue
            ctx = {'azimuth': 0.0}
            targets = io.load_targets(ref, env, column=a.target_column)
            for mod in _checks.needing(task, 'samples'):
                try:
                    s, _ = _score_part(mod, mdir, rdir, ctx, targets)
                except (io.SubmissionError, StopIteration) as e:
                    print(f'  skipped {task}/{env}/{mod.NAME}: {e}', file=sys.stderr)
                    continue
                rows.append({'task': task, 'environment': env,
                             'check': mod.NAME, 's': s})
            # Repeats-based checks: many runs of one input. Field scale has
            # none (spec section 4), so this loop is empty there.
            reps = _checks.needing(task, 'repeats')
            if reps:
                m_rep = io.part_dir(root, task, env, 'repeats')
                r_rep = ref / 'repeats' / io.SLUG[env]
                for mod in reps:
                    try:
                        s, _ = _score_repeats(mod, m_rep, r_rep, task, f'{task}/{env}')
                    except io.SubmissionError as e:
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
    envs_help = 'comma-separated subset of environments (default: all eight; a ranked submission needs all eight)'
    p = sub.add_parser('validate'); p.add_argument('submission')
    p.add_argument('--envs', help=envs_help); p.set_defaults(fn=cmd_validate)
    p = sub.add_parser('score'); p.add_argument('submission')
    p.add_argument('--reference'); p.add_argument('--out')
    p.add_argument('--envs', help=envs_help)
    p.add_argument('--target-column', default='ntg',
                   help="manifest column net_to_gross scores against; 'ntg' (the field's "
                        "realized value, the published condition) unless the submission "
                        "predates the manifest and was conditioned on 'ntg_source_cube'")
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser('figures'); p.add_argument('results')
    p.add_argument('--out', default='figures'); p.set_defaults(fn=cmd_figures)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
