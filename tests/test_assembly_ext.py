"""Euler characteristic and global connectivity must be exact on shapes with
known topology, since both are used to judge fragmentation and amalgamation."""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    'ext', Path(__file__).resolve().parents[1] / 'analysis' / 'assembly_stats_ext.py')
ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ext)


def test_euler_solid_block_is_one():
    v = np.zeros((8, 8, 8), np.int8); v[2:6, 2:6, 2:6] = 1
    assert ext.euler_characteristic(v) == 1


def test_euler_two_disjoint_blocks_is_two():
    v = np.zeros((12, 6, 6), np.int8); v[1:4, 1:4, 1:4] = 1; v[7:10, 1:4, 1:4] = 1
    assert ext.euler_characteristic(v) == 2


def test_euler_single_voxel_is_one():
    v = np.zeros((4, 4, 4), np.int8); v[2, 2, 2] = 1
    assert ext.euler_characteristic(v) == 1


def test_euler_hollow_shell_is_two():
    """A shell enclosing one cavity: chi = components + cavities = 1 + 1."""
    v = np.zeros((9, 9, 9), np.int8); v[2:7, 2:7, 2:7] = 1; v[3:6, 3:6, 3:6] = 0
    assert ext.euler_characteristic(v) == 2


def test_gamma_global_bounds():
    # one cluster holding everything -> 1; N equal clusters -> 1/N
    assert ext.gamma_global([100]) == 1.0
    assert abs(ext.gamma_global([10, 10, 10, 10]) - 0.25) < 1e-12


def test_percolation_detects_a_spanning_body():
    from resbench import metrics
    v = np.zeros((10, 6, 6), np.int8); v[:, 2:4, 2:4] = 1
    labels, _ = metrics.label_geobodies(v)
    assert ext.percolates(labels, 0)
    assert not ext.percolates(labels, 1)
