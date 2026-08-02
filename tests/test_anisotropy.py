"""Validation of the segmentation-free anisotropy estimators.

The headline check is analytic. For a 2-D ellipse with semi-axes a and b,
the mean chord length along the a-direction is

    (1 / 2b) * INT_{-b}^{b} 2a sqrt(1 - (y/b)^2) dy = a * pi / 2

and likewise b * pi / 2 along the b-direction, so the ratio of mean
chords is exactly a / b -- the aspect ratio -- independent of size. That
gives `chord_anisotropy` a ground truth that owes nothing to connected
components, which is the whole point of the estimator.
"""
import numpy as np
import pytest

from resbench import anisotropy as an


def ellipse_volume(nx=192, ny=192, nz=8, a=30.0, b=15.0, azimuth_deg=0.0,
                   centres=((96, 96),)):
    """Binary volume of one or more identical elliptical cylinders.

    Ellipses are elongated along `azimuth_deg` (clockwise from +x), the
    same convention as ResMill.
    """
    vol = np.zeros((nx, ny, nz), dtype=np.int8)
    xs = np.arange(nx)[:, None]
    ys = np.arange(ny)[None, :]
    t = np.deg2rad(azimuth_deg)
    # Rotate the query point into the ellipse's own frame. The forward
    # map is clockwise, so the inverse is counter-clockwise.
    for (cx, cy) in centres:
        dx, dy = xs - cx, ys - cy
        u = dx * np.cos(t) - dy * np.sin(t)
        v = dx * np.sin(t) + dy * np.cos(t)
        inside = (u / a) ** 2 + (v / b) ** 2 <= 1.0
        vol[inside, :] = 1
    return vol


@pytest.mark.parametrize("asp", [1.0, 1.5, 2.0, 2.5])
def test_chord_anisotropy_recovers_aspect_ratio(asp):
    b = 16.0
    a = b * asp
    vol = ellipse_volume(a=a, b=b, azimuth_deg=0.0)
    got = an.chord_anisotropy([vol], azimuth_deg=0.0)
    assert got == pytest.approx(asp, rel=0.05), f"asp={asp} got={got}"


@pytest.mark.parametrize("azimuth", [0.0, 45.0, 90.0])
def test_chord_anisotropy_on_axis_azimuths(azimuth):
    """At azimuths where the digital line is exactly periodic, no bias."""
    asp, b = 2.0, 16.0
    vol = ellipse_volume(a=b * asp, b=b, azimuth_deg=azimuth)
    got = an.chord_anisotropy([vol], azimuth_deg=azimuth)
    assert got == pytest.approx(asp, rel=0.05), f"az={azimuth} got={got}"


@pytest.mark.parametrize("azimuth", [15.0, 30.0, 60.0, 135.0])
def test_offaxis_bias_is_bounded_and_curable(azimuth):
    """Off-axis chord splitting is real; document its size and its cure.

    The default (gap_close=0) is allowed to be up to ~15% low, because it
    must never close a genuine drape. gap_close=2 brings it to a few
    percent, which is what the across-azimuth calibration path uses.
    """
    asp, b = 2.0, 16.0
    vol = ellipse_volume(a=b * asp, b=b, azimuth_deg=azimuth)
    raw = an.chord_anisotropy([vol], azimuth_deg=azimuth)
    closed = an.chord_anisotropy([vol], azimuth_deg=azimuth, gap_close=2)
    assert 0.85 * asp <= raw <= 1.05 * asp, f"az={azimuth} raw={raw}"
    assert closed == pytest.approx(asp, rel=0.05), f"az={azimuth} closed={closed}"
    assert abs(closed - asp) < abs(raw - asp) + 1e-9


def test_default_does_not_close_thin_drapes():
    """Guard the property the whole metric exists to measure.

    Two blocks separated by a 2-cell mud drape must read as two chords
    with the default settings. If this ever starts passing only with
    gap_close > 0, the estimator has gone blind to drape welding.
    """
    vol = np.zeros((64, 64, 4), dtype=np.int8)
    vol[10:30, 20:40, :] = 1
    vol[32:52, 20:40, :] = 1   # 2-cell gap at x = 30, 31
    ch = an.directional_chords(vol, azimuth_deg=0.0)
    assert ch.size > 0
    assert ch.max() == pytest.approx(20.0), "drape was welded shut"
    welded = an.directional_chords(vol, azimuth_deg=0.0, gap_close=2)
    assert welded.max() == pytest.approx(42.0), "gap_close should weld it"


def test_mean_chord_matches_pi_over_two_times_semi_axis():
    """Absolute calibration, not just the ratio."""
    a, b = 32.0, 16.0
    vol = ellipse_volume(a=a, b=b, azimuth_deg=0.0)
    major = an.ensemble_chords([vol], 0.0)
    minor = an.ensemble_chords([vol], 90.0)
    assert major.mean() == pytest.approx(a * np.pi / 2, rel=0.05)
    assert minor.mean() == pytest.approx(b * np.pi / 2, rel=0.05)


def test_censored_chords_are_dropped():
    """A body running off the domain edge must not contribute a chord."""
    nx = 64
    vol = np.zeros((nx, 64, 4), dtype=np.int8)
    vol[:, 20:30, :] = 1          # spans the full x extent -> censored
    vol[10:20, 40:50, :] = 1      # interior block -> 10-cell chords in x
    ch = an.directional_chords(vol, azimuth_deg=0.0)
    assert ch.size > 0
    assert ch.max() == pytest.approx(10.0), "censored chord leaked in"


def test_isotropic_geology_gives_ratio_one():
    rng = np.random.default_rng(0)
    vol = (rng.random((96, 96, 8)) < 0.3).astype(np.int8)
    got = an.chord_anisotropy([vol], azimuth_deg=37.0)
    assert got == pytest.approx(1.0, rel=0.08)


def test_variogram_anisotropy_tracks_elongation():
    """Range ratio should be > 1 for elongated bodies, ~1 for round ones.

    Centres are random rather than a lattice: regularly spaced bodies
    produce a hole effect whose oscillation can keep the pooled variogram
    from cleanly attaining its sill, which is a property of the test
    geometry and not of the estimator.
    """
    b = 10.0
    rng = np.random.default_rng(7)
    centres = [tuple(c) for c in rng.integers(30, 290, size=(40, 2))]
    round_vol = ellipse_volume(nx=320, ny=320, a=b, b=b, centres=centres)
    long_vol = ellipse_volume(nx=320, ny=320, a=b * 2.5, b=b, centres=centres)
    r_round = an.ensemble_variogram_anisotropy([round_vol], 0.0, max_lag=64)
    r_long = an.ensemble_variogram_anisotropy([long_vol], 0.0, max_lag=64)
    assert r_round == pytest.approx(1.0, rel=0.15), f"round={r_round}"
    assert r_long > r_round * 1.4, f"round={r_round} long={r_long}"


def test_variogram_range_returns_nan_when_threshold_unreached():
    """Structure longer than max_lag must surface as NaN, not as max_lag."""
    gamma = np.linspace(0.0, 0.1, 16)
    assert np.isnan(an.variogram_range(gamma, sill=10.0))


def test_variogram_uses_theoretical_sill_not_empirical_max():
    """A hole effect must not inflate the threshold out of reach."""
    # gamma rises to the sill then overshoots (regularly spaced bodies).
    p = 0.25
    sill = p * (1 - p)
    gamma = np.concatenate([np.linspace(0, sill, 10),
                            np.full(6, sill * 1.4)])
    assert an.variogram_range(gamma, sill=sill) < 10
    # With the empirical max as sill the threshold would sit at 1.4*sill,
    # which the rising limb never reaches until the overshoot.
    assert an.variogram_range(gamma) > 9


def test_chord_w1_zero_for_identical_distributions():
    x = np.array([2.0, 4.0, 8.0, 16.0])
    assert an.chord_w1(x, x) == pytest.approx(0.0)
