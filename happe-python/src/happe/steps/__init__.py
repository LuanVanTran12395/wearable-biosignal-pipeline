"""Processing steps, one module per major HAPPE stage.

The step *order* mirrors the MATLAB HAPPE v4.1 pipeline and must not be
reordered, merged, or dropped (see :mod:`happe.pipeline`).
"""

from . import (
    channels,
    line_noise,
    resample,
    filtering,
    bad_channels,
    ecgone,
    wavelet,
    muscil,
    segment,
    baseline,
    interpolate,
    reref,
)

__all__ = [
    "channels",
    "line_noise",
    "resample",
    "filtering",
    "bad_channels",
    "ecgone",
    "wavelet",
    "muscil",
    "segment",
    "baseline",
    "interpolate",
    "reref",
]
