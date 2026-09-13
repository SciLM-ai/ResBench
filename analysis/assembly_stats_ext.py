"""Additive diagnostics for the assembly task (proposed, NOT part of the frozen protocol).

Everything in EVAL.md §4/§5 and Addendum E stays exactly as it is; this script
adds columns that four days of model comparison showed to be missing, and
re-expresses two existing ones in forms that cannot be gamed the way the
frozen versions were. Run it alongside assembly_stats.py, never instead.

Why each addition exists, with the measurement that motivated it:

1. mass_geobody_w1 - geobody W1 weighted by VOLUME, not by body count.
   The frozen geobody_w1 is W1 between pooled log10(size) distributions with
   one sample per body, and 27-37% of bodies in a real field are <= 8 voxels,
   so the statistic is dominated by speckle. Measured consequences: raising
   CFG 3 -> 7.5 improved it 0.231 -> 0.047 while lobes visibly merged; adding
   a full-resolution conv head worsened it 0.089 -> 0.191 purely by adding
   isolated cells. Weighting each body by its voxel count makes a 1-voxel
   speck contribute one voxel of mass instead of one sample.
2. geobody_w1_min27 - the same W1 restricted to bodies >= 27 voxels (3^3),
   with the discarded fraction reported, so merging is measured on objects a
   geologist would call bodies.
3. speckle_* - isolated single cells per phase and bodies <= 8 voxels, per
   10^6 cells. Both frozen headline columns can be moved by speckle alone, in
   opposite directions, so it must be visible.
4. regions_per_slice / largest_region_share / long_chord_frac - plan-view
   amalgamation. Merging two 700-voxel lobes barely moves a log10 histogram,
   so the benchmark's most important geological failure currently has no
   column at all. These are what actually separated the models in practice.
5. tau_at_lags - connectivity at named lags instead of a mean over all lags.
   The frozen connectivity_mae averages lag 1 (near unity, no information)
   with lag 32 (most of the information) at equal weight.
6. largest_fraction_delta - restores the column that ens_summary() in
   assembly_stats.py computes via ensemble_geobodies() and then discards,
   even though the master table in EVAL.md §5 has it.
7. Tiles are derived from the array size instead of the hard-coded 5x5 grid at
   origin (52,52), which samples only the central 36% of a 532-cell field and
   4% of a 1572-cell one, discarding exactly the edges where artefacts live.

Usage:
  python analysis/assembly_stats_ext.py --assembly-dir Y --engine-dir Z \
      --env-slug lobe --out-dir W [--native-dir X]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import wasserstein_distance

from resbench import metrics

TILE = 64
S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])          # 4-connectivity, plan view
S6 = ndimage.generate_binary_structure(3, 1)
NAMED_LAGS = (2, 4, 8, 16, 32)
SPECKLE_MAX = 8
MIN_BODY = 27


def load_ref(d, slug):
    f = sorted(Path(d, slug).glob('volumes_*.npz'))[0]
    return np.load(f, allow_pickle=True)['volumes']


def cut_tiles_all(assembly_dir, margin=8):
    """Every non-overlapping 64^3 tile that fits, after trimming `margin`
    cells from each side. Uses the whole field instead of a fixed 5x5 grid."""
    tiles, fields = [], []
    for f in sorted(Path(assembly_dir).glob('assembly_*.npz')):
        b = np.load(f)['binary']
        fields.append(b)
        nx = (b.shape[0] - 2 * margin) // TILE
        ny = (b.shape[1] - 2 * margin) // TILE
        ox = margin + ((b.shape[0] - 2 * margin) - nx * TILE) // 2
        oy = margin + ((b.shape[1] - 2 * margin) - ny * TILE) // 2
        for i in range(nx):
            for j in range(ny):
                tiles.append(b[ox + i * TILE:ox + (i + 1) * TILE,
                               oy + j * TILE:oy + (j + 1) * TILE, :])
    return np.stack(tiles), fields


def body_stats(vols):
    """Pooled sizes, per-body mass weights, extents, largest fractions and
    speckle counts over a stack of volumes."""
    sizes, extents, fracs = [], [], []
    singles = {'sand': 0, 'shale': 0}
    tiny = 0
    ncells = 0
    for v in vols:
        v = np.asarray(v) > 0
        ncells += v.size
        lab, n = ndimage.label(v, structure=S6)
        if n:
            s = np.bincount(lab.ravel())[1:]
            sizes.append(s)
            fracs.append(s.max() / s.sum())
            tiny += int((s <= SPECKLE_MAX).sum())
            for sl in ndimage.find_objects(lab):
                if sl is not None:
                    extents.append(max(sl[0].stop - sl[0].start,
                                       sl[1].stop - sl[1].start))
        for phase, name in ((v, 'sand'), (~v, 'shale')):
            l2, _ = ndimage.label(phase, structure=S6)
            if l2.max():
                singles[name] += int((np.bincount(l2.ravel())[1:] == 1).sum())
    pooled = np.concatenate(sizes) if sizes else np.empty(0)
    return {
        'sizes': pooled,
        'extents': np.asarray(extents, dtype=np.float64),
        'largest_frac': float(np.mean(fracs)) if fracs else 0.0,
        'speckle_sand_singletons_per_1e6': 1e6 * singles['sand'] / max(ncells, 1),
        'speckle_shale_singletons_per_1e6': 1e6 * singles['shale'] / max(ncells, 1),
        'speckle_bodies_le8_per_1e6': 1e6 * tiny / max(ncells, 1),
    }


def plan_view_stats(vols, z_step=3):
    """Amalgamation seen the way a geologist looks at a map: per-slice
    2D sand regions per 10^4 cells, largest-region share, and the fraction of
    sand chords longer than 96 cells (9.6 km)."""
    dens, share, long_frac, n = [], [], 0.0, 0
    for v in vols:
        v = np.asarray(v) > 0
        for z in range(1, v.shape[2] - 1, z_step):
            lab, k = ndimage.label(v[:, :, z], structure=np.ones((3, 3)))
            if not k:
                continue
            s = np.bincount(lab.ravel())[1:]
            dens.append(1e4 * k / (v.shape[0] * v.shape[1]))
            share.append(s.max() / s.sum())
        for axis in (0, 1):
            a = np.moveaxis(v, axis, -1).reshape(-1, v.shape[axis])
            pad = np.zeros((a.shape[0], 1), bool)
            d = np.diff(np.concatenate([pad, a, pad], 1).astype(np.int8), axis=1)
            _, c0 = np.nonzero(d == 1)
            _, c1 = np.nonzero(d == -1)
            L = c1 - c0
            long_frac += float((L > 96).sum())
            n += len(L)
    return {'regions_per_1e4': float(np.mean(dens)) if dens else np.nan,
            'largest_region_share': float(np.mean(share)) if share else np.nan,
            'long_chord_frac': long_frac / max(n, 1)}


def tau_named(vols):
    tau = metrics.ensemble_connectivity(vols)
    out = {}
    for ax, name in ((0, 'x'), (1, 'y'), (2, 'z')):
        for h in NAMED_LAGS:
            if h <= len(tau[ax]):
                out[f'tau_{name}_{h}'] = float(tau[ax][h - 1])
    return out


def mass_w1(sizes_a, sizes_b):
    """W1 between log10-size distributions weighted by body VOLUME."""
    if len(sizes_a) == 0 or len(sizes_b) == 0:
        return float('nan')
    return float(wasserstein_distance(np.log10(sizes_a), np.log10(sizes_b),
                                      u_weights=sizes_a, v_weights=sizes_b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assembly-dir', required=True)
    ap.add_argument('--engine-dir', required=True)
    ap.add_argument('--native-dir', default=None)
    ap.add_argument('--env-slug', default='lobe')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--margin', type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    B, fields = cut_tiles_all(args.assembly_dir, args.margin)
    C = load_ref(args.engine_dir, args.env_slug)
    print(f'B(tiles)={B.shape} from {len(fields)} fields, C(engine)={C.shape}')
    sets = {'B_tiles': B, 'C_engine': C}
    if args.native_dir:
        sets['A_native'] = load_ref(args.native_dir, args.env_slug)

    res = {}
    for k, v in sets.items():
        bs = body_stats(v)
        res[k] = {kk: vv for kk, vv in bs.items() if kk not in ('sizes', 'extents')}
        res[k].update(plan_view_stats(v))
        res[k].update(tau_named(v))
        res[k]['_sizes'] = bs['sizes']
        res[k]['n_bodies'] = int(len(bs['sizes']))
        res[k]['frac_bodies_below_27'] = float((bs['sizes'] < MIN_BODY).mean())
    # the whole field, not re-cut into tiles: what the frozen protocol cannot see
    res['B_fullfield'] = plan_view_stats(fields)

    comps = {}
    for a, b in (('B_tiles', 'C_engine'), ('A_native', 'C_engine'), ('B_tiles', 'A_native')):
        if a not in res or b not in res:
            continue
        sa, sb = res[a]['_sizes'], res[b]['_sizes']
        ka, kb = sa[sa >= MIN_BODY], sb[sb >= MIN_BODY]
        comps[f'{a}_vs_{b}'] = {
            'mass_geobody_w1': mass_w1(sa, sb),
            'geobody_w1_min27': (float(wasserstein_distance(np.log10(ka), np.log10(kb)))
                                 if len(ka) and len(kb) else float('nan')),
            'largest_fraction_delta': abs(res[a]['largest_frac'] - res[b]['largest_frac']),
            'regions_per_1e4_delta': abs(res[a]['regions_per_1e4'] - res[b]['regions_per_1e4']),
            'largest_region_share_delta': abs(res[a]['largest_region_share']
                                              - res[b]['largest_region_share']),
            'long_chord_frac_delta': abs(res[a]['long_chord_frac'] - res[b]['long_chord_frac']),
        }
    for k in res:
        res[k].pop('_sizes', None)
    payload = {'per_ensemble': res, 'comparisons': comps,
               'notes': 'additive diagnostics; frozen EVAL.md metrics unchanged'}
    (out / 'assembly_stats_ext.json').write_text(json.dumps(payload, indent=2, default=float))
    print(json.dumps(payload, indent=2, default=float))


if __name__ == '__main__':
    main()
