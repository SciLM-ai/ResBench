"""The E.2 tile grid must stay bit-identical for the 424-cell overlap-24
assembly it was frozen on, and must stay centred at any other extent."""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    'assembly_stats', Path(__file__).resolve().parents[1] / 'analysis' / 'assembly_stats.py')
asm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(asm)


def test_frozen_424_layout_unchanged():
    """52 is the frozen E.2 origin and is exactly centred for 424."""
    assert asm.tile_origin(424) == 52


def test_grid_is_centred_at_other_extents():
    span = asm.TILE_N * asm.TILE
    for ext in (424, 496, 532, 1024, 1572):
        o = asm.tile_origin(ext)
        assert o >= 0 and o + span <= ext
        # margins differ by at most one cell (integer division)
        assert abs((ext - span - o) - o) <= 1, (ext, o)


def test_tiles_cover_the_centre(tmp_path):
    """A marker written at the centre of a 532-cell field must land in a tile;
    under the old fixed origin it did not."""
    ext = 532
    b = np.zeros((ext, ext, 32), dtype=np.int8)
    c = ext // 2
    b[c - 2:c + 2, c - 2:c + 2, :] = 1
    np.savez_compressed(tmp_path / 'assembly_00.npz', binary=b)
    tiles, _, _ = asm.cut_tiles(tmp_path)
    assert tiles.shape == (asm.TILE_N ** 2, asm.TILE, asm.TILE, 32)
    assert tiles.sum() == b.sum(), 'the centre of the field must be scored'
