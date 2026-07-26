"""ResBench: geostatistical minimum-acceptance benchmark for binary facies models.

Scores directories of saved volumes against a reference ensemble; imports
nothing from any generative model or data engine. Protocol: EVAL.md.
"""

__version__ = "0.1.0"

LAYER_TYPES = [
    'lobe',
    'channel:PV_SHOESTRING',
    'channel:CB_LABYRINTH',
    'channel:CB_JIGSAW',
    'channel:SH_DISTAL',
    'channel:SH_PROXIMAL',
    'channel:MEANDER_OXBOW',
    'delta',
]

VOLUME_SHAPE = (64, 64, 32)   # (X, Y, Z); z = depth is the LAST axis
MAX_LAGS = (32, 32, 16)       # half the axis extent per axis (EVAL.md §4)

MANIFEST_SEED = 20260726
SPLIT_HALF_SEED = 20260727
BOOTSTRAP_SEED = 20260728


def env_slug(layer_type: str) -> str:
    """Directory-safe environment name: 'channel:CB_JIGSAW' -> 'channel_CB_JIGSAW'."""
    return layer_type.replace(':', '_')
