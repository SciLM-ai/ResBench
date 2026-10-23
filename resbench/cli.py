"""resbench — validate and score a submission.

    resbench download
    resbench validate ./my_submission
    resbench score    ./my_submission [--reference ./reference]
    resbench figures  results.json

The reference lives on HuggingFace (REFERENCE_REPO), pinned to one revision so
that a later upload can never change anyone's score. `download` fetches it into
the HuggingFace cache; `score` uses that copy, downloading it on first use,
unless --reference PATH or RESBENCH_REFERENCE points at a local one.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

from . import checks as _checks
from . import bands, io, score

REFERENCE_ENV = 'RESBENCH_REFERENCE'
REFERENCE_REPO = 'SciLM/ResBench-reference'
REFERENCE_REVISION = '9087d11c9f1c18697f01be5a98358a0a52329944'


def _fetch_reference(offline=False):
    """Path of the pinned reference in the HuggingFace cache, downloading it
    unless ``offline``."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit('Downloading the reference needs huggingface_hub: '
                         'pip install "resbench[download]" (or pass --reference PATH).')
    return Path(snapshot_download(REFERENCE_REPO, repo_type='dataset',
                                  revision=REFERENCE_REVISION, local_files_only=offline))


def _reference_root(arg):
    root = arg or os.environ.get(REFERENCE_ENV)
    if root:
        return Path(root)
    print(f'reference: {REFERENCE_REPO} @ {REFERENCE_REVISION[:7]}', flush=True)
    return _fetch_reference()


def cmd_download(a):
    path = _fetch_reference()
    print(f'reference {REFERENCE_REPO} @ {REFERENCE_REVISION[:7]} is at {path}')
    return 0


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
    ref = getattr(a, 'reference', None) or os.environ.get(REFERENCE_ENV)    # local only; validate never downloads
    ref = Path(ref) if ref else None
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
                            if ref is not None and len(vols) != io.repeat_run_count(ref, env, cond):
                                raise io.SubmissionError(f'{w}: {len(vols)} runs, need one per ResMill run '
                                                         f'({io.repeat_run_count(ref, env, cond)})')
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


def _per_volume(mod, directory, ctx, targets, where):
    """One summary per volume, in the directory's shard order (per-volume targets sliced along)."""
    out = []
    for ids, vols in io.iter_shards(directory):
        c = _shard_ctx(ctx, ids, targets, where)
        for j in range(len(vols)):
            cj = c if 'target_ntg' not in c else {**c, 'target_ntg': np.asarray(c['target_ntg'])[j:j + 1]}
            out.append(mod.summarize(vols[j:j + 1], cj))
    return out


def _score_part(mod, model_dir, ref_dir, ctx, targets=None):
    if getattr(mod, 'TOLERANCE', None) is None:
        m = _per_volume(mod, model_dir, ctx, targets, str(model_dir))
        r = _per_volume(mod, ref_dir, ctx, targets, str(ref_dir))
        d = bands.matched_distance(mod, m, r)
        b = bands.expected_band(mod, r)
        return score.s_for_check(d, b), {'parts': d}
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

    The band is ResMill's own runs split in half at the same condition, averaged over many splits. The model's
    runs tied to one half of ResMill's runs are compared with the other half, so `s = 1` means the model's spread
    is as far from ResMill's as two halves of ResMill are from each other.
    """
    model = io.load_repeats(model_dir, task, where=f'{where} (submission)')
    ref = io.load_repeats(ref_dir, task, where=f'{where} (reference)')
    groups = {}
    for cond in io.CONDITIONS[task]:
        _, mv, _ = model[cond]
        _, rv, extras = ref[cond]
        if len(mv) < len(rv) // 2:
            raise io.SubmissionError(f'{where}/{cond}: {len(mv)} runs, need one per ResMill run ({len(rv)})')
        if len(mv) != len(rv):
            print(f'  note {where}/{cond}: {len(mv)} model runs for {len(rv)} ResMill runs '
                  f'(one per ResMill run is asked for, see `resbench repeat-inputs`)', file=sys.stderr)
        ctx = io.repeats_ctx(cond, extras)      # the reference defines the well
        groups[cond] = (rv, ctx)
    mg = {cond: model[cond][1] for cond in io.CONDITIONS[task]}
    mids = {cond: model[cond][0] for cond in io.CONDITIONS[task]}
    d = bands.matched_distance_repeats(mod, mg, groups, model_ids=mids)
    b = bands.expected_band_repeats(mod, groups)
    return score.s_for_check(d, b), {'parts': d}


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


def cmd_repeat_inputs(a):
    """Write the repeat inputs: per environment, task and condition, the realised sand fraction of
    each ResMill run. Generate one volume per listed value, with the condition's other parameters."""
    ref = _reference_root(a.reference)
    out = {}
    for env in _envs(a):
        out[env] = {task: {c: [round(float(x), 8) for x in io.repeat_run_ntg(ref, env, c)]
                           for c in io.CONDITIONS[task]} for task in ('unconditional', 'well_conditioned')}
    Path(a.out).write_text(json.dumps(out))
    n = sum(len(v) for e in out.values() for t in e.values() for v in t.values())
    print(f'wrote {a.out}: {n} runs over {len(out)} environment(s)')
    return 0


def cmd_figures(a):
    from . import figures as _f
    for p in _f.make(a.results, a.out):
        print(f'wrote {p}')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='resbench')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('download'); p.set_defaults(fn=cmd_download)
    envs_help = 'comma-separated subset of environments (default: all eight; a ranked submission needs all eight)'
    p = sub.add_parser('validate'); p.add_argument('submission')
    p.add_argument('--reference', help='local reference; checks one repeat run per ResMill run')
    p.add_argument('--envs', help=envs_help); p.set_defaults(fn=cmd_validate)
    p = sub.add_parser('score'); p.add_argument('submission')
    p.add_argument('--reference'); p.add_argument('--out')
    p.add_argument('--envs', help=envs_help)
    p.add_argument('--target-column', default='ntg',
                   help="manifest column net_to_gross scores against; 'ntg' (the field's "
                        "realized value, the published condition) unless the submission "
                        "predates the manifest and was conditioned on 'ntg_source_cube'")
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser('repeat-inputs'); p.add_argument('--reference'); p.add_argument('--envs', help=envs_help)
    p.add_argument('--out', default='repeat_inputs.json'); p.set_defaults(fn=cmd_repeat_inputs)
    p = sub.add_parser('figures'); p.add_argument('results')
    p.add_argument('--out', default='figures'); p.set_defaults(fn=cmd_figures)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
