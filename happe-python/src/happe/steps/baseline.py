"""Step 15 — Baseline correction (ERP only, optional).

Subtract the mean amplitude of a user-specified baseline window (ms) from each
epoch/channel via ``epochs.apply_baseline``.
"""

from __future__ import annotations

import mne

from ..config import Params


def apply(epochs: mne.BaseEpochs, params: Params) -> mne.BaseEpochs:
    if not (params.baseCorr.enabled and params.paradigm.erp):
        return epochs
    tmin = params.baseCorr.start_ms / 1000.0
    tmax = params.baseCorr.end_ms / 1000.0
    return epochs.apply_baseline((tmin, tmax), verbose="ERROR")
