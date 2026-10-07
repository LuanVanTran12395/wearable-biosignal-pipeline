"""Step 4 — Optional resampling (applied before filtering)."""

from __future__ import annotations

import mne

from ..config import Params


def apply(raw: mne.io.BaseRaw, params: Params) -> mne.io.BaseRaw:
    ds = params.downsample
    if ds.enabled and ds.srate and ds.srate != raw.info["sfreq"]:
        raw.resample(ds.srate, verbose="ERROR")
    return raw
