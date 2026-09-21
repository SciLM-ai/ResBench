"""The scoring path end to end, including the three repeats-based checks.

The invariant that matters: a submission that IS the reference scores zero on
every check the task lists, and a submission with a known defect scores
strictly positive on the check that is supposed to catch it. A check that
cannot fail is worse than no check -- net_to_gross once passed a submission
30 points wrong.
"""
import csv
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from resbench import checks, cli, io, bands, score

RNG = np.random.default_rng(7)
ENV = 'lobe'
SLUG = io.SLUG[ENV]


def smooth_noise(n, p=0.45, seed=0, shape=(64, 64, 32)):
    r = np.random.default_rng(seed)
    v = r.random((n,) + shape)
    return (ndimage.uniform_filter(v, size=(1, 3, 3, 3)) < p).astype(np.int8)


def pin_column(vols, xy, pattern):
    v = vols.copy(); v[:, xy[0], xy[1], :] = pattern; return v


# ---- the three repeats checks, one known failure each ----------------------

def test_variety_catches_a_collapsed_model():
    m = checks.CHECKS['variety']
    ref = smooth_noise(256, seed=1)                     # reference-sized ensemble
    ok = smooth_noise(io.K_REPEATS, seed=2)             # a different but honest draw
    collapsed = np.repeat(smooth_noise(1, seed=3), io.K_REPEATS, axis=0)
    ctx = {'condition_id': 'cond0'}
    b = bands.band_for_repeats(m, {'cond0': (ref, ctx)})
    s_ok = score.s_for_check(m.compare(m.summarize(ok, ctx), m.summarize(ref, ctx))['parts'], b)
    s_bad = score.s_for_check(m.compare(m.summarize(collapsed, ctx), m.summarize(ref, ctx))['parts'], b)
    assert s_ok < score.CLOSE, s_ok                     # honest: inside or close
    assert s_bad > score.CLOSE and s_bad > 3 * s_ok     # collapsed: distinguishable
    assert m.compare(m.summarize(ref, ctx), m.summarize(ref, ctx))['parts']['entropy_mae'] == 0.0


def test_calibration_catches_inverted_probabilities():
    # Entropy is symmetric, so variety cannot see this; calibration must.
    m, v = checks.CHECKS['calibration'], checks.CHECKS['variety']
    ref = smooth_noise(64, seed=4, p=0.3)
    inverted = 1 - ref
    ctx = {'condition_id': 'cond0'}
    d_cal = m.compare(m.summarize(inverted, ctx), m.summarize(ref, ctx))['parts']['ece']
    d_var = v.compare(v.summarize(inverted, ctx), v.summarize(ref, ctx))['parts']['entropy_mae']
    assert d_var == pytest.approx(0.0, abs=1e-12)
    assert d_cal > 0.3


def test_well_blending_catches_a_hard_seam():
    m = checks.CHECKS['well_blending']
    xy, pat = (30, 30), np.array([1] * 12 + [0] * 8 + [1] * 12, np.int8)
    ref = pin_column(smooth_noise(64, seed=5), xy, pat)
    # over-confident: everything within 6 cells of the well pinned to the pattern
    seam = ref.copy()
    d = io.metrics_chebyshev = None
    from resbench import metrics as M
    near = M.chebyshev_distance_map(ref.shape[1:], xy) <= 6
    seam[:, near] = np.broadcast_to(pat, (64, 64, 32))[near]
    ctx = {'condition_id': 'well1', 'well_xy': xy}
    r = m.compare(m.summarize(seam, ctx), m.summarize(ref, ctx))
    assert r['parts']['profile_mae'] > 0.05
    assert r['near_field_offset'] < 0                  # more certain than reality
    assert m.compare(m.summarize(ref, ctx), m.summarize(ref, ctx))['parts']['profile_mae'] == 0.0


# ---- the CLI, all twelve checks, reference against itself ------------------

def _write_stack(path, vols, prefix):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array([f'{prefix}|{i}' for i in range(len(vols))], dtype=object),
                        volumes=vols)


def _build_reference(ref, vols, reps, wells):
    _write_stack(ref / 'volumes' / SLUG / 'volumes.npz', vols, 'vol')
    _write_stack(ref / 'fields' / SLUG / 'fields.npz', vols[:8], 'field')
    for cond, v in reps.items():
        _write_stack(ref / 'repeats' / SLUG / f'{cond}.npz', v, cond)
    for cond, (v, xy, pat) in wells.items():
        p = ref / 'repeats' / SLUG / f'{cond}.npz'
        mask = np.zeros((64, 64, 32), np.uint8); mask[xy[0], xy[1], :] = 1
        np.savez_compressed(p, ids=np.array([f'{cond}|{i}' for i in range(len(v))], dtype=object),
                            volumes=v, pattern=pat, well_mask=mask, well_xy=np.array(xy))
    with open(ref / 'manifest.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['task', 'environment', 'id', 'ntg'])
        w.writeheader()
        for i, v in enumerate(vols):
            w.writerow({'task': 'unconditional', 'environment': ENV, 'id': f'vol|{i}', 'ntg': float(v.mean())})
        for i in range(8):
            w.writerow({'task': 'field_scale', 'environment': ENV, 'id': f'field|{i}', 'ntg': float(vols[i].mean())})


def _build_submission(sub, vols, reps, wells, fields=None):
    for task in ('unconditional', 'well_conditioned'):
        _write_stack(sub / task / SLUG / 'samples' / 'shard0.npz', vols, 'vol')
    for cond, v in reps.items():
        _write_stack(sub / 'unconditional' / SLUG / 'repeats' / f'{cond}.npz', v, cond)
    for cond, (v, xy, pat) in wells.items():
        _write_stack(sub / 'well_conditioned' / SLUG / 'repeats' / f'{cond}.npz', v, cond)
    _write_stack(sub / 'field_scale' / SLUG / 'fields' / 'shard0.npz',
                 vols[:8] if fields is None else fields, 'field')
    (sub / 'submission.yaml').write_text('model: test\nsampler: none\nnfe: 1\n')


@pytest.fixture
def small_fields(monkeypatch):
    # Field references at 512x512 are far too large for a unit test.
    for e in io.ENVIRONMENTS:
        monkeypatch.setitem(io.FIELD_EXTENT, e, (64, 64, 32))


def test_cli_scores_all_twelve_and_reference_against_itself_is_zero(tmp_path, small_fields, capsys):
    vols = smooth_noise(24, seed=10)
    reps = {c: smooth_noise(io.K_REPEATS, seed=20 + i) for i, c in enumerate(io.CONDITIONS['unconditional'])}
    pat = np.array([1] * 16 + [0] * 16, np.int8)
    wells = {c: (pin_column(smooth_noise(io.K_REPEATS, seed=40 + i), (20 + i, 20), pat), (20 + i, 20), pat)
             for i, c in enumerate(io.CONDITIONS['well_conditioned'])}
    ref, sub = tmp_path / 'ref', tmp_path / 'sub'
    _build_reference(ref, vols, reps, wells)
    _build_submission(sub, vols, reps, wells)

    assert cli.main(['validate', str(sub), '--envs', ENV]) == 0, capsys.readouterr().out
    out = tmp_path / 'r.json'
    assert cli.main(['score', str(sub), '--reference', str(ref), '--out', str(out)]) == 0
    import json
    res = json.load(open(out))
    scored = {(r['task'], r['check']): r['s'] for r in res['rows']}
    for task in io.TASKS:
        expected = {m.NAME for m in checks.for_task(task)}
        got = {c for (t, c) in scored if t == task}
        assert got == expected, f'{task}: scored {sorted(got)}, expected {sorted(expected)}'
    assert all(abs(s) < 1e-9 for s in scored.values()), scored
    assert len(scored) == 11 + 11 + 9


def test_cli_a_defective_submission_scores_positive_where_it_should(tmp_path, small_fields):
    vols = smooth_noise(24, seed=10)
    reps = {c: smooth_noise(io.K_REPEATS, seed=20 + i) for i, c in enumerate(io.CONDITIONS['unconditional'])}
    pat = np.array([1] * 16 + [0] * 16, np.int8)
    wells = {c: (pin_column(smooth_noise(io.K_REPEATS, seed=40 + i), (20 + i, 20), pat), (20 + i, 20), pat)
             for i, c in enumerate(io.CONDITIONS['well_conditioned'])}
    ref, sub = tmp_path / 'ref', tmp_path / 'sub'
    _build_reference(ref, vols, reps, wells)
    bad_reps = {c: np.repeat(v[:1], io.K_REPEATS, axis=0) for c, v in reps.items()}   # collapsed
    _build_submission(sub, vols, bad_reps, wells, fields=1 - vols[:8])              # inverted fields
    out = tmp_path / 'r.json'
    assert cli.main(['score', str(sub), '--reference', str(ref), '--out', str(out)]) == 0
    import json
    scored = {(r['task'], r['check']): r['s'] for r in json.load(open(out))['rows']}
    assert scored[('unconditional', 'variety')] > 2
    assert scored[('field_scale', 'net_to_gross')] > 2
    assert scored[('unconditional', 'net_to_gross')] == pytest.approx(0.0, abs=1e-9)


def test_score_refuses_a_reference_without_a_manifest(tmp_path, small_fields):
    vols = smooth_noise(4, seed=1)
    ref, sub = tmp_path / 'ref', tmp_path / 'sub'
    _write_stack(ref / 'volumes' / SLUG / 'volumes.npz', vols, 'vol')
    _write_stack(sub / 'unconditional' / SLUG / 'samples' / 's.npz', vols, 'vol')
    with pytest.raises(SystemExit, match='manifest'):
        cli.main(['score', str(sub), '--reference', str(ref)])


def test_figures_command_writes_pdfs(tmp_path, small_fields):
    vols = smooth_noise(24, seed=10)
    reps = {c: smooth_noise(io.K_REPEATS, seed=20 + i) for i, c in enumerate(io.CONDITIONS['unconditional'])}
    pat = np.array([1] * 16 + [0] * 16, np.int8)
    wells = {c: (pin_column(smooth_noise(io.K_REPEATS, seed=40 + i), (20 + i, 20), pat), (20 + i, 20), pat)
             for i, c in enumerate(io.CONDITIONS['well_conditioned'])}
    ref, sub = tmp_path / 'ref', tmp_path / 'sub'
    _build_reference(ref, vols, reps, wells); _build_submission(sub, vols, reps, wells)
    out = tmp_path / 'r.json'
    assert cli.main(['score', str(sub), '--reference', str(ref), '--out', str(out), '--envs', ENV]) == 0
    assert cli.main(['figures', str(out), '--out', str(tmp_path / 'figs')]) == 0
    pdfs = sorted(p.name for p in (tmp_path / 'figs').glob('*.pdf'))
    assert pdfs == ['scores_field_scale.pdf', 'scores_overview.pdf', 'scores_unconditional.pdf', 'scores_well_conditioned.pdf']
