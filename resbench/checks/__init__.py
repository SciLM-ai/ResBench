"""The twelve checks.

Every module here exports the same small interface (see `_base`), so adding a
check means writing one module and one test: there is nowhere else to put it,
which is what stops a benchmark growing appendices.

Which check runs in which task:

    unconditional     11   everything except well_blending
    well_conditioned  11   everything except variety
    field_scale        9   the structural nine; variety, well_blending and
                           calibration all need many runs of one fixed input,
                           which at 512 x 512 x 32 is unaffordable
"""
from . import (net_to_gross, variogram, patterns, bed_thickness, connectivity,
               compartments, body_size, chord_lengths, speckle, variety,
               well_blending, calibration)

MODULES = (net_to_gross, variogram, patterns, bed_thickness, connectivity,
           compartments, body_size, chord_lengths, speckle, variety,
           well_blending, calibration)

CHECKS = {m.NAME: m for m in MODULES}
TASKS = ('unconditional', 'well_conditioned', 'field_scale')


def for_task(task):
    """Checks that run in `task`, in the canonical order above."""
    if task not in TASKS:
        raise KeyError(f'unknown task {task!r}; expected one of {TASKS}')
    return [m for m in MODULES if task in m.TASKS]


def needing(task, needs):
    return [m for m in for_task(task) if m.NEEDS == needs]


__all__ = ['CHECKS', 'MODULES', 'TASKS', 'for_task', 'needing']
