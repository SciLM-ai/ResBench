"""Segmentation-free body-scale and anisotropy estimators.

Motivation (EVAL.md Addendum G): the existing `geobody_w1` / `extent_w1`
metrics rest on 6-connected component labelling, which merges bodies
wherever sand amalgamates. In the `lobe` environment amalgamation is
physically real -- the engine reference has merged bodies too -- so any
per-object "width", "length" or "aspect ratio" built on connected
components inherits an arbitrary decision about where one lobe ends and
the next begins.

The estimators here avoid that decision entirely:

* `directional_chords` -- linear-intercept (chord-length) sampling along
  an arbitrary in-plane azimuth. Classic stereology: the run lengths of
  contiguous sand along parallel scan lines. No objects are identified.
* `directional_variogram` / `variogram_range` -- indicator variogram
  along an arbitrary azimuth and its practical range.

Both give an anisotropy ratio as (major-direction statistic) / (minor-
direction statistic), which is directly comparable to the engine at the
same condition and, for lobes, to the requested `asp`.

Conventions match `resbench.metrics`: binary (X, Y, Z) arrays with
sand = 1, axes (0, 1, 2) = (x, y, z). Azimuth is degrees measured
clockwise from +x, matching ResMill's `LobeLayer.create_geology`.

Censoring: a chord touching either end of its scan line is truncated by
the domain, not by geology, so it is discarded. Without this the chord
distribution is biased short, and biased *differently* for volumes of
different sizes -- which would corrupt any native-vs-assembly-tile
comparison.
"""
import numpy as np
from scipy import ndimage
from scipy.stats import wasserstein_distance

# Scan lines are laid perpendicular to the sampling direction at this
# spacing (cells). 1.0 = every cell offset; larger values subsample.
DEFAULT_LINE_SPACING = 1.0
# Practical range = first lag at which gamma reaches this fraction of sill.
RANGE_SILL_FRACTION = 0.95


def _unit(azimuth_deg):
    """In-plane unit vector for an azimuth in degrees clockwise from +x.

    Clockwise, so azimuth 0 -> +x and azimuth 90 -> -y, matching
    ResMill's `LobeLayer.create_geology` docstring. Getting the handedness
    wrong would silently swap the major and minor directions for every
    azimuth that is not a multiple of 90.
    """
    a = np.deg2rad(float(azimuth_deg))
    return np.array([np.cos(a), -np.sin(a)], dtype=np.float64)


def _scan_lines(shape_xy, azimuth_deg, spacing=DEFAULT_LINE_SPACING):
    """Parallel scan lines covering an (nx, ny) plane along `azimuth_deg`.

    Yields (xs, ys) float coordinate arrays at unit step along the
    direction. Lines are seeded on the perpendicular axis across the
    full projected width of the rectangle, and each line is clipped to
    the samples that fall inside the plane.
    """
    nx, ny = shape_xy
    u = _unit(azimuth_deg)
    p = np.array([-u[1], u[0]])  # perpendicular

    corners = np.array([[0, 0], [nx - 1, 0], [0, ny - 1], [nx - 1, ny - 1]],
                       dtype=np.float64)
    t_along = corners @ u
    t_perp = corners @ p
    n_steps = int(np.floor(t_along.max() - t_along.min())) + 1
    if n_steps < 2:
        return

    offsets = np.arange(t_perp.min(), t_perp.max() + spacing, spacing)
    steps = np.arange(n_steps, dtype=np.float64)
    origin = u * t_along.min()

    for off in offsets:
        base = origin + p * off
        xs = base[0] + u[0] * steps
        ys = base[1] + u[1] * steps
        inside = (xs >= 0) & (xs <= nx - 1) & (ys >= 0) & (ys <= ny - 1)
        if inside.sum() < 2:
            continue
        # Keep the contiguous in-bounds run (a straight line enters and
        # leaves a convex rectangle exactly once).
        idx = np.flatnonzero(inside)
        sl = slice(idx[0], idx[-1] + 1)
        yield xs[sl], ys[sl]


def _runs_from_line(vals, drop_censored=True):
    """Run lengths of contiguous 1s in a 1-D binary sample sequence."""
    v = np.asarray(vals, dtype=np.int8)
    if v.size == 0:
        return []
    padded = np.concatenate(([0], v, [0]))
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    lengths = ends - starts
    if drop_censored and lengths.size:
        keep = (starts > 0) & (ends < v.size)
        lengths = lengths[keep]
    return lengths.tolist()


def directional_chords(vol, azimuth_deg, spacing=DEFAULT_LINE_SPACING,
                       z_stride=1, drop_censored=True, gap_close=0):
    """Pooled sand chord lengths along `azimuth_deg`, over all z slices.

    Sampling is linear (order=1) thresholded at 0.5, NOT nearest
    neighbour. On an off-axis direction a scan line grazes the staircase
    boundary of a rasterized body, and nearest-neighbour rounding
    zig-zags in and out of it, shattering one chord into several: on an
    isolated ellipse of aspect ratio 2 that drives the measured ratio
    down to 1.58 at azimuth 45 (and *worse* with a finer step -- 1.03 at
    step 0.25 -- because more samples means more chances to zig-zag).
    Linear interpolation smooths the staircase and recovers 2.09.

    A residual bias remains at off-axis azimuths, up to ~13% on the
    aspect ratio at 30 degrees: a digital line running nearly parallel to
    a body's boundary dips below threshold for a cell or two mid-body and
    splits one chord into several (the chord *count* rises ~27% at 45
    degrees, which is the tell). It is a property of the grid, not the
    geology, so it applies identically to engine and model volumes and
    cancels whenever both sides are measured at the same azimuth -- which
    is why the headline comparisons are all at one fixed azimuth. Do not
    read absolute chord lengths at off-axis azimuths as exact, and never
    compare a measured anisotropy directly against a requested `asp`.

    `gap_close` > 0 closes gaps of that many cells along the scan line
    before run-length encoding, which cuts the off-axis bias to ~3% at
    gap_close=2 (saturating -- 3 and 4 give the same answer, so it is a
    knee rather than a fitted parameter). **It defaults to 0 and should
    stay there for body-scale scoring**: a genuine 1-2 cell mud drape
    between two lobes is exactly what MultiDiffusion blending is
    suspected of welding shut, and closing gaps would make the estimator
    blind to the effect it exists to measure. Use it only for
    across-azimuth calibration, where absolute accuracy matters and no
    drape-welding claim is being made.

    Returns a float64 array of chord lengths (may be empty).
    """
    v = np.asarray(vol)
    nx, ny, nz = v.shape
    lines = list(_scan_lines((nx, ny), azimuth_deg, spacing))
    if not lines:
        return np.empty(0, dtype=np.float64)

    out = []
    for z in range(0, nz, z_stride):
        plane = (v[:, :, z] > 0).astype(np.float32)
        for xs, ys in lines:
            sampled = ndimage.map_coordinates(
                plane, np.vstack([xs, ys]), order=1, mode='nearest') > 0.5
            if gap_close > 0:
                sampled = ndimage.binary_closing(
                    sampled, structure=np.ones(gap_close + 1, dtype=bool))
            out.extend(_runs_from_line(sampled,
                                       drop_censored=drop_censored))
    return np.asarray(out, dtype=np.float64)


def directional_variogram(vol, azimuth_deg, max_lag=32,
                          spacing=DEFAULT_LINE_SPACING, z_stride=1):
    """Indicator variogram gamma(h), h = 1..max_lag, along `azimuth_deg`.

    Pools squared indicator differences over all scan lines and z slices,
    so gamma(h) = 0.5 * n_differing / n_pairs -- the same estimator as
    `metrics.variogram_sums`, just on an arbitrary direction.
    """
    v = np.asarray(vol)
    nx, ny, nz = v.shape
    lines = list(_scan_lines((nx, ny), azimuth_deg, spacing))
    sq = np.zeros(max_lag, dtype=np.int64)
    npair = np.zeros(max_lag, dtype=np.int64)
    if not lines:
        return np.full(max_lag, np.nan)

    for z in range(0, nz, z_stride):
        plane = (v[:, :, z] > 0).astype(np.float32)
        for xs, ys in lines:
            s = ndimage.map_coordinates(
                plane, np.vstack([xs, ys]), order=1, mode='nearest') > 0.5
            for h in range(1, max_lag + 1):
                if s.size <= h:
                    break
                d = s[h:] != s[:-h]
                sq[h - 1] += int(d.sum())
                npair[h - 1] += int(d.size)

    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * sq / np.where(npair > 0, npair, np.nan)


def variogram_range(gamma, sill=None, frac=RANGE_SILL_FRACTION):
    """Practical range: first lag where gamma reaches `frac` * sill.

    Linearly interpolated between bracketing lags. `sill` defaults to the
    maximum of gamma. Returns NaN if the threshold is never reached
    (structure longer than `max_lag`) -- callers should widen max_lag
    rather than silently treat that as "range == max_lag".
    """
    g = np.asarray(gamma, dtype=np.float64)
    if np.all(np.isnan(g)):
        return float('nan')
    s = float(np.nanmax(g)) if sill is None else float(sill)
    if not np.isfinite(s) or s <= 0:
        return float('nan')
    thr = frac * s
    hit = np.flatnonzero(g >= thr)
    if hit.size == 0:
        return float('nan')
    i = int(hit[0])
    if i == 0:
        return 1.0
    g0, g1 = g[i - 1], g[i]
    if not np.isfinite(g0) or g1 == g0:
        return float(i + 1)
    return float(i + (thr - g0) / (g1 - g0))


# -- ensemble-level estimators -------------------------------------------

def ensemble_chords(vols, azimuth_deg, **kw):
    """Chord lengths pooled across an ensemble."""
    parts = [directional_chords(v, azimuth_deg, **kw) for v in vols]
    parts = [p for p in parts if p.size]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)


def chord_anisotropy(vols, azimuth_deg, **kw):
    """Mean chord along the azimuth / mean chord perpendicular to it.

    > 1 means bodies are elongated along `azimuth_deg`. For lobes this is
    the direct analogue of the requested `asp` (semi-major / semi-minor),
    though the two are not numerically identical -- chords average over
    all offsets through a body, not just its central section -- so it is
    only interpretable against the engine measured the same way.
    """
    major = ensemble_chords(vols, azimuth_deg, **kw)
    minor = ensemble_chords(vols, azimuth_deg + 90.0, **kw)
    if major.size == 0 or minor.size == 0:
        return float('nan')
    return float(major.mean() / minor.mean())


def ensemble_variogram_anisotropy(vols, azimuth_deg, max_lag=32, **kw):
    """Practical-range ratio (along azimuth / perpendicular).

    Variograms are pooled across the ensemble before the range is taken,
    so the ratio is an ensemble property rather than a mean of noisy
    per-volume ranges.
    """
    gm = np.nanmean([directional_variogram(v, azimuth_deg, max_lag, **kw)
                     for v in vols], axis=0)
    gp = np.nanmean([directional_variogram(v, azimuth_deg + 90.0, max_lag, **kw)
                     for v in vols], axis=0)
    # Sill: the THEORETICAL indicator sill p(1-p), not the empirical max
    # of gamma. The max is wrong twice over -- a hole effect (regularly
    # spaced bodies) pushes gamma above the sill, inflating the
    # threshold until no lag reaches it; and a direction whose structure
    # is longer than max_lag never attains its own max. p(1-p) is a
    # property of the ensemble's sand fraction and is shared by both
    # directions, which is what makes the two ranges comparable.
    p = float(np.mean([np.mean(np.asarray(v) > 0) for v in vols]))
    sill = p * (1.0 - p)
    rm = variogram_range(gm, sill=sill)
    rp = variogram_range(gp, sill=sill)
    if not (np.isfinite(rm) and np.isfinite(rp)) or rp == 0:
        return float('nan')
    return float(rm / rp)


def chord_w1(chords_a, chords_b):
    """W1 between two chord-length distributions on log10 lengths.

    Matches the `geobody_w1` convention (log10 domain) so the numbers sit
    on a comparable scale in the master table.
    """
    a = np.asarray(chords_a, dtype=np.float64)
    b = np.asarray(chords_b, dtype=np.float64)
    a = a[a > 0]
    b = b[b > 0]
    if a.size == 0 or b.size == 0:
        return float('nan')
    return float(wasserstein_distance(np.log10(a), np.log10(b)))
