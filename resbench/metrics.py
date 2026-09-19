"""Core geostatistical metrics on binary (X, Y, Z) facies volumes.

Per-volume functions return raw *counts/sums* so that ensemble statistics can
be pooled exactly (EVAL.md §4); `ensemble_*` helpers do the pooling. All
functions accept int8/bool arrays with sand = 1 and treat axes (0, 1, 2) =
(x, y, z). Definitions are frozen in EVAL.md; every function is unit-tested
against brute force in tests/test_metrics.py.
"""
from itertools import product

import numpy as np
from scipy import ndimage
from scipy.stats import wasserstein_distance

from . import MAX_LAGS

# 6-connectivity (faces only): the 3D cross-shaped structuring element.
STRUCT_6 = ndimage.generate_binary_structure(3, 1)


# -- per-volume counts ----------------------------------------------------

def ntg(vol):
    """Facies proportion (net-to-gross): mean of the binary volume."""
    return float(np.asarray(vol).mean())


def variogram_sums(vol, max_lags=MAX_LAGS):
    """Axis-aligned indicator variogram numerators/denominators.

    Returns {axis: (sq_sum, n_pairs)} where sq_sum[h-1] is the number of
    differing pairs at lag h along that axis (squared indicator difference
    summed) and n_pairs[h-1] the pair count. gamma(h) = 0.5 * sq_sum / n_pairs.
    """
    v = np.asarray(vol).astype(np.int8)
    out = {}
    for axis, L in enumerate(max_lags):
        sq = np.empty(L, dtype=np.int64)
        n = np.empty(L, dtype=np.int64)
        for h in range(1, L + 1):
            a = np.take(v, np.arange(h, v.shape[axis]), axis=axis)
            b = np.take(v, np.arange(0, v.shape[axis] - h), axis=axis)
            d = a != b
            sq[h - 1] = int(d.sum())
            n[h - 1] = d.size
        out[axis] = (sq, n)
    return out


def label_geobodies(vol):
    """Label sand geobodies with 6-connectivity. Returns (labels, n_bodies)."""
    labels, n = ndimage.label(np.asarray(vol) > 0, structure=STRUCT_6)
    return labels, int(n)


def connectivity_counts(vol, max_lags=MAX_LAGS, labels=None):
    """Connectivity-function counts per Renard & Allard.

    Returns {axis: (n_same, n_sand_pairs)} over voxel pairs at lag h along
    each axis with BOTH voxels sand; tau(h) = n_same / n_sand_pairs.
    """
    v = np.asarray(vol) > 0
    if labels is None:
        labels, _ = label_geobodies(v)
    out = {}
    for axis, L in enumerate(max_lags):
        same = np.empty(L, dtype=np.int64)
        pairs = np.empty(L, dtype=np.int64)
        for h in range(1, L + 1):
            la = np.take(labels, np.arange(h, v.shape[axis]), axis=axis)
            lb = np.take(labels, np.arange(0, v.shape[axis] - h), axis=axis)
            both = (la > 0) & (lb > 0)
            pairs[h - 1] = int(both.sum())
            same[h - 1] = int((both & (la == lb)).sum())
        out[axis] = (same, pairs)
    return out


def geobody_sizes(vol, labels=None):
    """Sizes (voxel counts) of all 6-connected sand components, descending."""
    if labels is None:
        labels, n = label_geobodies(vol)
    else:
        n = int(labels.max())
    if n == 0:
        return np.empty(0, dtype=np.int64)
    sizes = np.bincount(labels.ravel())[1:]  # drop background
    return np.sort(sizes)[::-1].astype(np.int64)


ARTIFACT_MAX = 8  # Addendum F.5: artifacts are 6-connected sand bodies < 8 voxels


def artifact_counts(vols, max_size=ARTIFACT_MAX):
    """Per-volume count of artifact bodies (Addendum F.5).

    One definition shared by the F.5 component (analysis/acceptance_ext.py)
    and the additive assembly diagnostics (analysis/assembly_stats_ext.py),
    so the threshold cannot drift between them.
    """
    return np.array([(geobody_sizes(v) < max_size).sum() for v in vols],
                    dtype=np.int64)


def largest_fraction(vol, labels=None):
    """Largest geobody's fraction of total sand volume (0 if no sand)."""
    sizes = geobody_sizes(vol, labels=labels)
    total = int(sizes.sum())
    return float(sizes[0]) / total if total > 0 else 0.0


def exactitude_counts(gen, ref, mask):
    """Well exactitude: (n_mismatch, n_mask_voxels) at mask == 1."""
    m = np.asarray(mask) > 0
    g = np.asarray(gen)[m] > 0
    r = np.asarray(ref)[m] > 0
    return int((g != r).sum()), int(m.sum())


# -- ensemble pooling -----------------------------------------------------

def ensemble_ntg(vols):
    """Per-volume NTG array for an (N, X, Y, Z) stack."""
    return np.asarray(vols).mean(axis=(1, 2, 3)).astype(np.float64)


def ensemble_variogram(vols, max_lags=MAX_LAGS):
    """Pooled gamma curves: {axis: array of len L}."""
    acc = {a: [np.zeros(L, np.int64), np.zeros(L, np.int64)]
           for a, L in enumerate(max_lags)}
    for v in vols:
        for a, (sq, n) in variogram_sums(v, max_lags).items():
            acc[a][0] += sq
            acc[a][1] += n
    return {a: 0.5 * acc[a][0] / acc[a][1] for a in acc}


def ensemble_connectivity(vols, max_lags=MAX_LAGS):
    """Pooled tau curves {axis: array of len L}. Lags with zero sand pairs
    anywhere in the ensemble yield NaN."""
    acc = {a: [np.zeros(L, np.int64), np.zeros(L, np.int64)]
           for a, L in enumerate(max_lags)}
    for v in vols:
        labels, _ = label_geobodies(v)
        for a, (same, pairs) in connectivity_counts(v, max_lags, labels=labels).items():
            acc[a][0] += same
            acc[a][1] += pairs
    out = {}
    for a in acc:
        same, pairs = acc[a]
        with np.errstate(invalid='ignore'):
            out[a] = np.where(pairs > 0, same / np.maximum(pairs, 1), np.nan)
    return out


def ensemble_geobodies(vols):
    """(pooled sizes over all volumes, per-volume largest fractions)."""
    sizes_all, fracs = [], []
    for v in vols:
        labels, _ = label_geobodies(v)
        s = geobody_sizes(v, labels=labels)
        sizes_all.append(s)
        total = int(s.sum())
        fracs.append(float(s[0]) / total if total > 0 else 0.0)
    pooled = (np.concatenate(sizes_all) if sizes_all else np.empty(0, np.int64))
    return pooled, np.asarray(fracs, dtype=np.float64)


def ensemble_exactitude(gens, refs, masks):
    """Pooled (mismatch_fraction, n_mask_voxels) over aligned stacks."""
    bad = tot = 0
    for g, r, m in zip(gens, refs, masks):
        b, t = exactitude_counts(g, r, m)
        bad += b
        tot += t
    return (bad / tot if tot > 0 else np.nan), tot


# -- scalar comparisons (master-table cells, EVAL.md §5) -------------------

def variogram_mae(gamma_a, gamma_b, sill):
    """Mean |gamma_a - gamma_b| over all (axis, lag), normalized by sill."""
    diffs = [np.abs(gamma_a[a] - gamma_b[a]) for a in gamma_a]
    return float(np.concatenate(diffs).mean() / sill)


def connectivity_mae(tau_a, tau_b):
    """Mean |tau_a - tau_b| over all (axis, lag), NaN lags dropped pairwise."""
    diffs = []
    for a in tau_a:
        d = np.abs(tau_a[a] - tau_b[a])
        diffs.append(d[~np.isnan(d)])
    return float(np.concatenate(diffs).mean())


def geobody_w1(sizes_a, sizes_b):
    """Wasserstein-1 between log10 geobody-size distributions."""
    if len(sizes_a) == 0 or len(sizes_b) == 0:
        return np.nan
    return float(wasserstein_distance(np.log10(sizes_a), np.log10(sizes_b)))


# -- sanity assertions (EVAL.md §4) ---------------------------------------

def sanity_check(ntg_mean, gamma, tau, tau_tol=0.02):
    """Assert frozen sanity conditions on ensemble curves. Returns warnings
    list for soft conditions; raises AssertionError on hard violations."""
    warnings = []
    assert 0.0 <= ntg_mean <= 1.0, f"NTG {ntg_mean} outside [0, 1]"
    sill = ntg_mean * (1.0 - ntg_mean)
    for a, g in gamma.items():
        assert np.all(g >= 0), f"negative variogram values on axis {a}"
        plateau = g[-max(1, len(g) // 4):].mean()
        if sill > 0.01 and not (0.5 * sill <= plateau <= 2.0 * sill):
            warnings.append(
                f"axis {a}: variogram plateau {plateau:.4f} far from sill {sill:.4f}")
    for a, t in tau.items():
        tt = t[~np.isnan(t)]
        if len(tt) > 1 and np.any(np.diff(tt) > tau_tol):
            warnings.append(f"axis {a}: tau increases beyond tolerance {tau_tol}")
    return warnings


# -- v1 primitives ---------------------------------------------------------
# Low-level estimators added for the v1 check modules. Everything above this
# line is unchanged and still unit-tested against brute force.

def max_lags_for(shape):
    """Half the extent on each axis: the longest lag a volume can support."""
    return tuple(max(int(n) // 2, 1) for n in shape[:3])


def w1(a, b):
    """Wasserstein-1 between two 1-D samples (area between the CDFs)."""
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    if a.size == 0 or b.size == 0:
        return float('nan')
    q = np.linspace(0.0, 1.0, 1001)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def mps_hist(vol):
    """Counts of each 2x2x2 sand/shale pattern, 256 bins, every offset.

    The eight cells of a block are read in a fixed order and packed into one
    byte, so a block is a single label in 0..255 rather than a point in 8-D.
    """
    v = np.asarray(vol, np.uint8)
    nx, ny, nz = v.shape[0] - 1, v.shape[1] - 1, v.shape[2] - 1
    acc = np.zeros((nx, ny, nz), np.uint8)
    for i, j, k in product((0, 1), repeat=3):
        acc |= (v[i:nx + i, j:ny + j, k:nz + k] << (4 * i + 2 * j + k))
    return np.bincount(acc.ravel(), minlength=256).astype(np.int64)


def jsd_bits(p, q, eps=1e-300):
    """Jensen-Shannon divergence in bits between two histograms.

    Symmetric and bounded in [0, 1]; KL is neither, and KL is infinite as soon
    as one side holds a pattern the other never produced, which with 256 bins
    always happens.
    """
    p = np.asarray(p, float); p = p / p.sum()
    q = np.asarray(q, float); q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        s = a > 0
        return float((a[s] * np.log2(a[s] / np.maximum(b[s], eps))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def run_lengths(vol, axis=2):
    """Lengths of every unbroken sand run along `axis`.

    axis=2 walks each (x, y) column top to bottom, which is what a vertical
    well logs. Runs that reach the end of the volume are kept at their
    truncated length, identically on both sides of any comparison.
    """
    a = np.moveaxis(np.asarray(vol, np.int8), axis, -1)
    pad = np.zeros(a.shape[:-1] + (a.shape[-1] + 2,), np.int8)
    pad[..., 1:-1] = a
    d = np.diff(pad, axis=-1)
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    return (ends[:, -1] - starts[:, -1]).astype(np.int64)


def gamma_global(sizes):
    """P(two sand cells drawn anywhere land in the same body).

    sum(n_i^2) / (sum n_i)^2 -- the n^2 deliberately weights toward the
    largest bodies, because that is the fraction of the sand one well reaches.
    """
    s = np.asarray(sizes, float)
    tot = s.sum()
    return float((s ** 2).sum() / tot ** 2) if tot > 0 else 0.0


def euler_characteristic(vol):
    """pieces - tunnels + enclosed cavities, for the 6-connected sand phase.

    Computed from the alternating sum of k-cell counts (vertices, edges,
    faces, cubes) of the cubical complex, which is exact and needs no
    labelling. Separates rocks that gamma cannot: a body riddled with shale
    tunnels and a solid one of the same size share a gamma.
    """
    v = np.asarray(vol) > 0
    # k-cell counts of the cubical complex whose 3-cells are the sand voxels
    c3 = int(v.sum())
    c2 = (int((v[:-1] | v[1:]).sum()) + int((v[:, :-1] | v[:, 1:]).sum())
          + int((v[:, :, :-1] | v[:, :, 1:]).sum()))
    c1 = 0
    for ax in ((0, 1), (0, 2), (1, 2)):
        s = v
        for a in ax:
            s = np.moveaxis(s, a, 0)
            s = s[:-1] | s[1:]
            s = np.moveaxis(s, 0, a)
        c1 += int(s.sum())
    s = v
    for a in (0, 1, 2):
        s = np.moveaxis(s, a, 0)
        s = s[:-1] | s[1:]
        s = np.moveaxis(s, 0, a)
    c0 = int(s.sum())
    return int(c0 - c1 + c2 - c3)


def entropy_map(vols):
    """Per-cell Shannon entropy in bits over a stack of repeats.

    p = how often each cell came out sand; H = -p log2 p - (1-p) log2(1-p).
    A cell that is the same rock every time scores 0, a coin-flip cell 1.
    H is symmetric in p, so it says how uncertain, never which way: that is
    what `calibration` is for.
    """
    p = np.asarray(vols, np.float64).mean(axis=0)
    e = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(e * np.log2(e) + (1 - e) * np.log2(1 - e))


def entropy_map_from_p(p):
    """Per-cell entropy in bits from an existing probability map."""
    e = np.clip(np.asarray(p, np.float64), 1e-12, 1.0 - 1e-12)
    return -(e * np.log2(e) + (1 - e) * np.log2(1 - e))


def prob_map(vols):
    """Per-cell P(sand) over a stack of repeats."""
    return np.asarray(vols, np.float64).mean(axis=0)


def expected_calibration_error(p_model, p_ref, mask=None, bins=10):
    """Bin cells by the model's stated probability, compare to the truth rate.

    Uses p itself rather than its entropy, which is why it catches a model
    that inverts every probability: such a model has zero entropy error.
    """
    pm = np.asarray(p_model, float).ravel()
    pr = np.asarray(p_ref, float).ravel()
    if mask is not None:
        m = np.asarray(mask).ravel().astype(bool)
        pm, pr = pm[~m], pr[~m]
    idx = np.clip((pm * bins).astype(int), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(pm[sel].mean() - pr[sel].mean())
    return float(ece)


def chebyshev_distance_map(shape, xy):
    """Per-cell max(|x-wx|, |y-wy|), broadcast down the full depth."""
    nx, ny, nz = shape[:3]
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    d = np.maximum(np.abs(ix - int(xy[0])), np.abs(iy - int(xy[1])))
    return np.repeat(d[:, :, None], nz, axis=2)
