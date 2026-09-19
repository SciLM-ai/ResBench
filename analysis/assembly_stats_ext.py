"""Additive assembly diagnostics (proposed; the frozen protocol is unchanged).

Everything in EVAL.md §4/§5 and Addenda E/F/G keeps its definitions; this
script only *applies* metrics the benchmark already defines to the assembly
task, which currently scores six columns and uses none of Addendum F or G,
and adds two scalars from the connectivity review ResBench already cites.

Motivation, from four days of model comparison (2026-09-10..13):
  * CFG > 3 improved the frozen geobody W1 fourfold (0.231 -> 0.047) while
    lobes visibly merged. F.5's artifact rate would have caught it.
  * A conv decoder head improved connectivity_mae 0.033 -> 0.026 while
    doubling geobody W1, purely by adding isolated cells. Same.
  * Neither the amalgamation that Addendum G was written to chase nor the
    speckle that both exploits used has a column in the assembly table.

What is here and where each piece comes from:

1. `artifacts_per_volume` - Addendum F.5 verbatim: 6-connected sand bodies
   < 8 voxels. Defined and banded in ResBench for the master table; never
   applied to assemblies.
2. `euler_per_1e6` - Euler characteristic of the foreground cubical complex,
   V - E + F - C, the topological member of the Minkowski functionals and
   the standard descriptor of "too many fragments / too many holes". Exact
   for the 6-connectivity ResBench labels with; validated in
   tests/test_assembly_ext.py against solid, hollow and disjoint shapes.
3. `gamma_global` - Renard & Allard (2013), Adv. Water Resour. 51:168-196,
   already cited by EVAL.md §4 for the connectivity function tau(h): the
   probability that two randomly chosen cells of the phase belong to the
   SAME cluster, sum(n_i^2) / (sum n_i)^2. This is the scalar that measures
   amalgamation directly. `percolates_{x,y,z}` is the traversing
   probability from the same review.
4. Directional chords via `resbench.anisotropy` (Addendum G.6) - NOT a
   reimplementation. Those estimators are validated against an analytic
   ellipse and deliberately do not close 1-2 cell gaps, because those gaps
   are the mud drapes under study. Nothing in the repository imported that
   module before this script.
5. `mass_geobody_w1` - the frozen geobody W1 weights every body equally,
   and 62% of bodies in a scored ensemble are < 27 voxels, so it is largely
   a speckle statistic. Weighting each body by its voxel count is the
   standard volume-weighted form used for connected-volume reporting.
   `geobody_w1_min27` is the same restricted to bodies >= 3^3.
6. `largest_fraction` - EVAL.md §5 has this column; `ens_summary()` in
   assembly_stats.py computes it via `ensemble_geobodies()` and discards it.
7. `tau_at_*` - tau at named lags. The frozen connectivity_mae averages all
   (axis, lag) equally, and tau(1) is exactly 1 for every model by
   construction, so roughly half the terms carry no information.

Tiles are cut with `assembly_stats.tile_origin`, i.e. centred at any extent.
Every quantity gets a split-half band from the reference ensemble, computed
the same way EVAL.md §5 does, so "inside / near / outside" means something.

  python analysis/assembly_stats_ext.py --assembly-dir Y --engine-dir Z \
      --env-slug lobe --out-dir W [--native-dir X] [--engine-big-dir B] \
      [--azimuth 95]
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import wasserstein_distance

from resbench import anisotropy, metrics

_spec = importlib.util.spec_from_file_location(
    'assembly_stats', Path(__file__).resolve().parent / 'assembly_stats.py')
_asm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_asm)

ARTIFACT_MAX = metrics.ARTIFACT_MAX    # F.5, defined in resbench.metrics
MIN_BODY = 27             # 3^3
NAMED_LAGS = (2, 4, 8, 16, 32)


# ---------------------------------------------------------------- topology

def euler_characteristic(vol):
    """chi = V - E + F - C of the foreground cubical complex.

    Exact for 6-connectivity (the connectivity ResBench labels bodies with).
    chi = #components - #tunnels + #cavities.
    """
    v = np.asarray(vol) > 0
    V = int(v.sum())
    E = int(sum((v[:-1] & v[1:]).sum() for v in
                (v, np.moveaxis(v, 1, 0), np.moveaxis(v, 2, 0))))
    F = int((v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]).sum()
            + (v[:-1, :, :-1] & v[1:, :, :-1] & v[:-1, :, 1:] & v[1:, :, 1:]).sum()
            + (v[:, :-1, :-1] & v[:, 1:, :-1] & v[:, :-1, 1:] & v[:, 1:, 1:]).sum())
    C = int((v[:-1, :-1, :-1] & v[1:, :-1, :-1] & v[:-1, 1:, :-1] & v[:-1, :-1, 1:]
             & v[1:, 1:, :-1] & v[1:, :-1, 1:] & v[:-1, 1:, 1:] & v[1:, 1:, 1:]).sum())
    return V - E + F - C


def gamma_global(sizes):
    """Renard & Allard global connectivity: P(two random sand cells are in
    the same cluster) = sum(n_i^2) / (sum n_i)^2."""
    s = np.asarray(sizes, dtype=np.float64)
    tot = s.sum()
    return float((s ** 2).sum() / tot ** 2) if tot > 0 else 0.0


def percolates(labels, axis):
    """Does any single cluster touch both faces normal to `axis`?"""
    lo = set(np.unique(np.take(labels, 0, axis=axis))) - {0}
    hi = set(np.unique(np.take(labels, -1, axis=axis))) - {0}
    return bool(lo & hi)


# ---------------------------------------------------------------- per-set

def summarise(vols, azimuth):
    sizes_all, fracs, arts, eulers, gammas, perc = [], [], [], [], [], {0: 0, 1: 0, 2: 0}
    ncells = 0
    for v in vols:
        v = np.asarray(v)
        ncells += v.size
        labels, n = metrics.label_geobodies(v)
        s = metrics.geobody_sizes(v, labels=labels)
        if s.size:
            sizes_all.append(s)
            fracs.append(float(s[0]) / s.sum())
            arts.append(int((s < ARTIFACT_MAX).sum()))
            gammas.append(gamma_global(s))
        eulers.append(euler_characteristic(v))
        for ax in (0, 1, 2):
            perc[ax] += int(percolates(labels, ax))
    pooled = np.concatenate(sizes_all) if sizes_all else np.empty(0)
    out = {
        'n_volumes': len(vols),
        'n_bodies': int(pooled.size),
        'frac_bodies_below_27': float((pooled < MIN_BODY).mean()) if pooled.size else np.nan,
        'largest_fraction': float(np.mean(fracs)) if fracs else np.nan,
        'artifacts_per_volume': float(np.mean(arts)) if arts else np.nan,
        'euler_per_1e6': 1e6 * float(np.sum(eulers)) / max(ncells, 1),
        'gamma_global': float(np.mean(gammas)) if gammas else np.nan,
        'percolates_x': perc[0] / max(len(vols), 1),
        'percolates_y': perc[1] / max(len(vols), 1),
        'percolates_z': perc[2] / max(len(vols), 1),
    }
    tau = metrics.ensemble_connectivity(vols)
    for ax, nm in ((0, 'x'), (1, 'y'), (2, 'z')):
        for h in NAMED_LAGS:
            if h <= len(tau[ax]):
                out[f'tau_{nm}_{h}'] = float(tau[ax][h - 1])
    out['_sizes'] = pooled
    out['_chords_az'] = anisotropy.ensemble_chords(vols, azimuth)
    out['_chords_perp'] = anisotropy.ensemble_chords(vols, azimuth + 90.0)
    out['chord_anisotropy'] = (float(np.mean(out['_chords_az']) / np.mean(out['_chords_perp']))
                               if out['_chords_perp'].size and out['_chords_az'].size else np.nan)
    return out


def compare(a, b):
    sa, sb = a['_sizes'], b['_sizes']
    ka, kb = sa[sa >= MIN_BODY], sb[sb >= MIN_BODY]
    d = {
        'mass_geobody_w1': (float(wasserstein_distance(np.log10(sa), np.log10(sb),
                                                       u_weights=sa, v_weights=sb))
                            if sa.size and sb.size else np.nan),
        'geobody_w1_min27': (float(wasserstein_distance(np.log10(ka), np.log10(kb)))
                             if ka.size and kb.size else np.nan),
        'chord_w1_azimuth': anisotropy.chord_w1(a['_chords_az'], b['_chords_az']),
        'chord_w1_perp': anisotropy.chord_w1(a['_chords_perp'], b['_chords_perp']),
    }
    for k in ('largest_fraction', 'artifacts_per_volume', 'euler_per_1e6',
              'gamma_global', 'chord_anisotropy'):
        d[f'd_{k}'] = abs(a[k] - b[k])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assembly-dir', required=True)
    ap.add_argument('--engine-dir', required=True,
                    help='frozen 64^3 engine reference (ensemble C)')
    ap.add_argument('--engine-big-dir', default=None,
                    help='OPTIONAL field-scale engine reference. Without it '
                         'nothing above 64 cells can be compared with ground '
                         'truth, which is the deepest limitation of the '
                         'assembly task as frozen.')
    ap.add_argument('--native-dir', default=None)
    ap.add_argument('--env-slug', default='lobe')
    ap.add_argument('--azimuth', type=float, default=95.0,
                    help='conditioning azimuth; chords are measured along it '
                         'and perpendicular to it, never against a requested '
                         'asp (Addendum G.6 disclosed rasterisation bias)')
    ap.add_argument('--split-seed', type=int, default=20260815)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    B, fields = _asm.cut_tiles(args.assembly_dir)[0], [
        np.load(f)['binary'] for f in sorted(Path(args.assembly_dir).glob('assembly_*.npz'))]
    C = _asm.load_npz_dir(args.engine_dir, args.env_slug)
    sets = {'B_tiles': B, 'C_engine': C}
    if args.native_dir:
        sets['A_native'] = _asm.load_npz_dir(args.native_dir, args.env_slug)

    res = {k: summarise(v, args.azimuth) for k, v in sets.items()}
    comps = {}
    for a, b in (('B_tiles', 'C_engine'), ('A_native', 'C_engine'), ('B_tiles', 'A_native')):
        if a in res and b in res:
            comps[f'{a}_vs_{b}'] = compare(res[a], res[b])

    # split-half band of the frozen reference, EVAL.md §5 machinery
    perm = np.random.default_rng(args.split_seed).permutation(len(C))
    h1, h2 = C[perm[:len(C) // 2]], C[perm[len(C) // 2:]]
    band = compare(summarise(h1, args.azimuth), summarise(h2, args.azimuth))

    # field-scale comparison: whole assemblies vs whole engine fields
    field = None
    if args.engine_big_dir:
        big = _asm.load_npz_dir(args.engine_big_dir, args.env_slug) \
            if (Path(args.engine_big_dir) / args.env_slug).exists() else \
            np.load(sorted(Path(args.engine_big_dir).glob('volumes_*.npz'))[0],
                    allow_pickle=True)['volumes']
        sF, sE = summarise(fields, args.azimuth), summarise(big, args.azimuth)
        pe = np.random.default_rng(args.split_seed + 1).permutation(len(big))
        fband = compare(summarise(big[pe[:len(big) // 2]], args.azimuth),
                        summarise(big[pe[len(big) // 2:]], args.azimuth))
        field = {'assembly': {k: v for k, v in sF.items() if not k.startswith('_')},
                 'engine_big': {k: v for k, v in sE.items() if not k.startswith('_')},
                 'comparison': compare(sF, sE), 'band': fband,
                 'engine_big_n': int(len(big)), 'extent': list(np.shape(fields[0]))}

    for k in res:
        for kk in ('_sizes', '_chords_az', '_chords_perp'):
            res[k].pop(kk, None)
    payload = {'per_ensemble': res, 'comparisons': comps, 'band_split_half': band,
               'field_scale': field, 'azimuth': args.azimuth,
               'note': 'additive; EVAL.md frozen metrics unchanged'}
    (out / 'assembly_stats_ext.json').write_text(json.dumps(payload, indent=2, default=float))

    def verdict(v, b):
        return 'inside' if v <= b else 'near' if v <= 2 * b else 'outside'
    print(f"\n{'metric':22s} {'B_vs_C':>10s} {'band':>10s}  verdict")
    for k, v in comps.get('B_tiles_vs_C_engine', {}).items():
        print(f'{k:22s} {v:10.4f} {band[k]:10.4f}  {verdict(v, band[k])}')
    if field:
        print(f"\nFIELD SCALE ({field['extent']}, {field['engine_big_n']} engine fields)")
        print(f"{'metric':22s} {'assembly vs engine':>18s} {'band':>10s}  verdict")
        for k, v in field['comparison'].items():
            print(f"{k:22s} {v:18.4f} {field['band'][k]:10.4f}  {verdict(v, field['band'][k])}")
    print(f'\nwrote {out}/assembly_stats_ext.json')


if __name__ == '__main__':
    main()
