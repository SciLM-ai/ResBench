"""Brute-force verification of every metric on tiny 8x8x8 arrays (EVAL.md §4)."""
import numpy as np
import pytest
from scipy import ndimage

from resbench import metrics

RNG = np.random.default_rng(1234)
SHAPE = (8, 8, 8)
LAGS = (4, 4, 4)


def random_vol(p=0.4, seed=None):
    rng = np.random.default_rng(seed)
    return (rng.random(SHAPE) < p).astype(np.int8)


# -- brute-force references ------------------------------------------------

def brute_variogram(vol, axis, h):
    X, Y, Z = vol.shape
    diffs = []
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                idx = [x, y, z]
                idx[axis] += h
                if idx[axis] < vol.shape[axis]:
                    diffs.append((int(vol[x, y, z]) - int(vol[tuple(idx)])) ** 2)
    return 0.5 * np.mean(diffs), np.sum(diffs), len(diffs)


def brute_connectivity(vol, labels, axis, h):
    X, Y, Z = vol.shape
    same = pairs = 0
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                idx = [x, y, z]
                idx[axis] += h
                if idx[axis] < vol.shape[axis]:
                    a, b = labels[x, y, z], labels[tuple(idx)]
                    if a > 0 and b > 0:
                        pairs += 1
                        if a == b:
                            same += 1
    return same, pairs


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_variogram_vs_brute_force(seed, axis):
    vol = random_vol(seed=seed)
    sums = metrics.variogram_sums(vol, LAGS)
    sq, n = sums[axis]
    for h in range(1, LAGS[axis] + 1):
        gamma_bf, sq_bf, n_bf = brute_variogram(vol, axis, h)
        assert sq[h - 1] == sq_bf
        assert n[h - 1] == n_bf
        assert 0.5 * sq[h - 1] / n[h - 1] == pytest.approx(gamma_bf)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_connectivity_vs_brute_force(seed, axis):
    vol = random_vol(p=0.55, seed=seed)
    labels, _ = metrics.label_geobodies(vol)
    counts = metrics.connectivity_counts(vol, LAGS, labels=labels)
    same, pairs = counts[axis]
    for h in range(1, LAGS[axis] + 1):
        same_bf, pairs_bf = brute_connectivity(vol, labels, axis, h)
        assert same[h - 1] == same_bf
        assert pairs[h - 1] == pairs_bf


def test_labeling_is_6_connective():
    # Two voxels touching only at an edge (diagonal in x-y) must be separate
    # bodies under 6-connectivity, one body under 18/26-connectivity.
    vol = np.zeros(SHAPE, dtype=np.int8)
    vol[3, 3, 3] = 1
    vol[4, 4, 3] = 1
    _, n = metrics.label_geobodies(vol)
    assert n == 2
    # Face-touching voxels merge.
    vol[4, 3, 3] = 1
    _, n = metrics.label_geobodies(vol)
    assert n == 1


def test_geobody_sizes_and_largest_fraction():
    vol = np.zeros(SHAPE, dtype=np.int8)
    vol[0:3, 0, 0] = 1          # size-3 rod
    vol[6, 6, 6] = 1            # size-1
    sizes = metrics.geobody_sizes(vol)
    assert sizes.tolist() == [3, 1]
    assert metrics.largest_fraction(vol) == pytest.approx(3 / 4)
    # brute force on random volumes vs scipy sum-per-label
    for seed in range(3):
        v = random_vol(seed=seed)
        labels, n = metrics.label_geobodies(v)
        sizes = metrics.geobody_sizes(v, labels=labels)
        assert int(sizes.sum()) == int(v.sum())
        assert len(sizes) == n
        bf = sorted((int((labels == i).sum()) for i in range(1, n + 1)),
                    reverse=True)
        assert sizes.tolist() == bf


def test_ntg():
    vol = np.zeros(SHAPE, dtype=np.int8)
    vol[0, 0, :4] = 1
    assert metrics.ntg(vol) == pytest.approx(4 / vol.size)
    assert 0.0 <= metrics.ntg(random_vol(seed=5)) <= 1.0


def test_exactitude():
    ref = random_vol(seed=7)
    mask = np.zeros(SHAPE, dtype=np.int8)
    mask[2, 2, :] = 1
    gen = ref.copy()
    bad, tot = metrics.exactitude_counts(gen, ref, mask)
    assert (bad, tot) == (0, 8)
    gen[2, 2, 0] = 1 - gen[2, 2, 0]
    gen[0, 0, 0] = 1 - gen[0, 0, 0]  # outside mask: must not count
    bad, tot = metrics.exactitude_counts(gen, ref, mask)
    assert (bad, tot) == (1, 8)


def test_ensemble_pooling_matches_per_volume():
    vols = np.stack([random_vol(seed=s) for s in range(4)])
    gamma = metrics.ensemble_variogram(vols, LAGS)
    # pooled == mean of per-volume sums (equal pair counts per volume)
    sq0 = sum(metrics.variogram_sums(v, LAGS)[0][0] for v in vols)
    n0 = sum(metrics.variogram_sums(v, LAGS)[0][1] for v in vols)
    np.testing.assert_allclose(gamma[0], 0.5 * sq0 / n0)

    tau = metrics.ensemble_connectivity(vols, LAGS)
    same = np.zeros(LAGS[2], np.int64)
    pairs = np.zeros(LAGS[2], np.int64)
    for v in vols:
        s, p = metrics.connectivity_counts(v, LAGS)[2]
        same += s
        pairs += p
    np.testing.assert_allclose(tau[2], same / pairs)


def test_variogram_plateau_near_sill_for_iid():
    # iid Bernoulli(p) voxels: gamma(h) -> p(1-p) at all lags.
    vol = (np.random.default_rng(0).random((32, 32, 32)) < 0.3).astype(np.int8)
    gamma = metrics.ensemble_variogram(vol[None], (16, 16, 16))
    sill = 0.3 * 0.7
    for a in gamma:
        assert np.all(np.abs(gamma[a] - sill) < 0.15 * sill)


def test_connectivity_bounds_and_solid_block():
    # Fully connected solid block: tau == 1 at every lag.
    vol = np.ones(SHAPE, dtype=np.int8)
    tau = metrics.ensemble_connectivity(vol[None], LAGS)
    for a in tau:
        np.testing.assert_allclose(tau[a], 1.0)


def test_w1_identical_is_zero():
    s = np.array([1, 5, 20, 100])
    assert metrics.geobody_w1(s, s) == 0.0
    assert metrics.geobody_w1(s, s * 10) == pytest.approx(1.0)  # log10 shift


def test_sanity_check_runs():
    vols = np.stack([random_vol(p=0.35, seed=s) for s in range(6)])
    gamma = metrics.ensemble_variogram(vols, LAGS)
    tau = metrics.ensemble_connectivity(vols, LAGS)
    warnings = metrics.sanity_check(float(vols.mean()), gamma, tau)
    assert isinstance(warnings, list)
