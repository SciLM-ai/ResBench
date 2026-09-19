"""Tests for the v1 check modules and their primitives.

The important ones are the invariants: Euler against shapes whose answer is
known by hand, histogram W1 against exact W1 on raw samples, and the rule that
merging two summaries equals summarizing the concatenation -- which is what
lets a submission be scored shard by shard without ever holding an ensemble in
memory.
"""
import numpy as np
import pytest

from resbench import checks, bands, score
from resbench import metrics as M
from resbench.checks._base import hist_w1

RNG = np.random.default_rng(0)


def blobs(n=6, shape=(24, 24, 16), p=0.3, seed=0):
    r = np.random.default_rng(seed)
    v = (r.random((n,) + shape) < p).astype(np.int8)
    # smooth so bodies are not pure noise
    from scipy import ndimage
    return (ndimage.uniform_filter(v.astype(float), size=(1, 3, 3, 3)) > p).astype(np.int8)


# ---- primitives ---------------------------------------------------------

@pytest.mark.parametrize('build,expected', [
    (lambda: _cube(), 1),            # one solid piece
    (lambda: _tunnel(), 0),          # one piece with one tunnel through it
    (lambda: _two_cubes(), 2),       # two separate pieces
    (lambda: _shell(), 2),           # one piece enclosing one cavity
    (lambda: np.zeros((5, 5, 5), np.int8), 0),
])
def test_euler_known_shapes(build, expected):
    assert M.euler_characteristic(build()) == expected


def _cube():
    v = np.zeros((8, 8, 8), np.int8); v[2:5, 2:5, 2:5] = 1; return v


def _tunnel():
    v = np.zeros((10, 10, 10), np.int8); v[2:8, 2:8, 2:8] = 1; v[4:6, 4:6, :] = 0; return v


def _two_cubes():
    v = np.zeros((9, 9, 9), np.int8); v[1:3, 1:3, 1:3] = 1; v[5:7, 5:7, 5:7] = 1; return v


def _shell():
    v = np.zeros((9, 9, 9), np.int8); v[1:8, 1:8, 1:8] = 1; v[3:6, 3:6, 3:6] = 0; return v


def test_hist_w1_matches_exact_w1():
    a = RNG.gamma(2, 3, 100000); b = RNG.gamma(2, 3.4, 100000)
    edges = np.linspace(0, 60, 601)
    ha = np.histogram(a, bins=edges)[0]; hb = np.histogram(b, bins=edges)[0]
    assert hist_w1(ha, hb, edges) == pytest.approx(M.w1(a, b), rel=0.05)
    assert hist_w1(ha, ha, edges) == 0.0


def test_jsd_is_symmetric_and_bounded():
    p, q = RNG.random(256) + 1e-6, RNG.random(256) + 1e-6
    assert M.jsd_bits(p, q) == pytest.approx(M.jsd_bits(q, p))
    assert M.jsd_bits(p, p) == pytest.approx(0.0, abs=1e-12)
    assert M.jsd_bits([1, 0], [0, 1]) == pytest.approx(1.0)
    assert 0.0 <= M.jsd_bits(p, q) <= 1.0


def test_run_lengths_counts_beds_and_keeps_truncated_ones():
    col = np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1]).reshape(1, 1, 10)
    # leading run of 2 touches the top, trailing run of 1 touches the bottom;
    # both are kept at their truncated length, identically on both sides.
    assert sorted(M.run_lengths(col, axis=2).tolist()) == [1, 2, 3]


def test_gamma_global_bounds():
    assert M.gamma_global([100]) == pytest.approx(1.0)
    assert M.gamma_global([10] * 10) == pytest.approx(0.1)


def test_entropy_is_symmetric_which_is_why_calibration_exists():
    # H(p) == H(1-p): entropy says HOW uncertain, never WHICH WAY. A model
    # that inverts every probability has zero entropy error, so `variety`
    # cannot see it and `calibration` must.
    p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    assert M.entropy_map_from_p(p) == pytest.approx(M.entropy_map_from_p(1 - p))
    ref = RNG.beta(2, 2, 20000)
    inverted = 1 - ref
    assert np.abs(M.entropy_map_from_p(inverted)
                  - M.entropy_map_from_p(ref)).mean() == pytest.approx(0.0, abs=1e-12)
    assert M.expected_calibration_error(inverted, ref) > 0.3


def test_mps_hist_counts_every_offset():
    v = np.zeros((5, 5, 5), np.int8)
    assert M.mps_hist(v).sum() == 4 * 4 * 4
    assert M.mps_hist(v)[0] == 4 * 4 * 4          # all-shale blocks


# ---- the registry -------------------------------------------------------

def test_task_coverage():
    assert [len(checks.for_task(t)) for t in
            ('unconditional', 'well_conditioned', 'field_scale')] == [11, 11, 9]
    assert 'well_blending' not in [m.NAME for m in checks.for_task('unconditional')]
    assert 'variety' not in [m.NAME for m in checks.for_task('well_conditioned')]


def test_every_check_declares_the_interface():
    for m in checks.MODULES:
        assert isinstance(m.NAME, str) and m.NAME
        assert m.NEEDS in ('samples', 'repeats')
        assert set(m.TASKS) <= set(checks.TASKS)
        for fn in ('summarize', 'merge', 'compare'):
            assert callable(getattr(m, fn)), f'{m.NAME} missing {fn}'


# ---- merge equals summarize-on-the-whole --------------------------------

@pytest.mark.parametrize('name', ['variogram', 'patterns', 'bed_thickness',
                                  'connectivity', 'compartments', 'body_size',
                                  'speckle'])
def test_merge_equals_whole(name):
    m = checks.CHECKS[name]
    v = blobs(8, seed=3)
    ctx = {'azimuth': 0.0}
    whole = m.summarize(v, ctx)
    parts = m.merge([m.summarize(v[:3], ctx), m.summarize(v[3:], ctx)])
    d = m.compare(parts, whole)['parts']
    for k, val in d.items():
        assert val == pytest.approx(0.0, abs=1e-9), f'{name}/{k} = {val}'


def test_identical_ensembles_score_zero():
    v = blobs(8, seed=5)
    ctx = {'azimuth': 0.0}
    for m in checks.needing('unconditional', 'samples'):
        s = m.summarize(v, ctx)
        for k, val in m.compare(s, s)['parts'].items():
            assert val == pytest.approx(0.0, abs=1e-9), f'{m.NAME}/{k}'


def test_net_to_gross_measures_against_its_condition():
    v = blobs(6, seed=7)
    target = v.reshape(len(v), -1).mean(1)
    m = checks.CHECKS['net_to_gross']
    exact = m.compare(m.summarize(v, {'target_ntg': target}), None)['parts']
    assert exact['ntg_error'] == pytest.approx(0.0, abs=1e-12)
    off = m.compare(m.summarize(v, {'target_ntg': target + 0.05}), None)['parts']
    assert off['ntg_error'] == pytest.approx(0.05, abs=1e-9)


def test_score_verdicts():
    assert score.verdict(0.5) == 'matched'
    assert score.verdict(1.5) == 'close'
    assert score.verdict(3.0) == 'distinguishable'


def test_split_halves_is_reproducible_and_disjoint():
    a, b = bands.split_halves(101)
    a2, _ = bands.split_halves(101)
    assert np.array_equal(a, a2)
    assert not set(a.tolist()) & set(b.tolist())
    assert len(a) == len(b) == 50
