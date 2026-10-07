"""Steps 5 & 11 — Band-pass filtering.

Step 5 (non-ERP only): zero-phase FIR band-pass with the configured
highpass/lowpass (matches ``pop_eegfiltnew`` defaults). ERP paradigms defer
filtering to step 11 with ERP-specific cutoffs.

Step 11 (ERP only): either the same FIR filter, or a Butterworth IIR filter
replicating ERPLAB's ``pop_basicfilter`` (``scipy.signal.butter`` + zero-phase
``filtfilt``).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

import mne

from ..config import Params


def bandpass(obj, params: Params):
    """Apply the configured band-pass to a Raw or Epochs object, in place."""
    f = params.filt
    if f.method == "butter":
        _butter_filter(obj, f.highpass, f.lowpass, f.butter_order)
    else:
        obj.filter(l_freq=f.highpass, h_freq=f.lowpass, phase="zero",
                   picks="eeg", verbose="ERROR")
    return obj


def _butter_filter(obj, l_freq, h_freq, order) -> None:
    """ERPLAB-style Butterworth band-pass via zero-phase filtfilt."""
    sfreq = obj.info["sfreq"]
    nyq = sfreq / 2.0
    wl = max(l_freq / nyq, 1e-6)
    wh = min(h_freq / nyq, 0.999999)
    b, a = butter(order, [wl, wh], btype="band")
    picks = mne.pick_types(obj.info, eeg=True)
    data = obj.get_data()
    if data.ndim == 3:  # epochs
        for e in range(data.shape[0]):
            data[e, picks] = filtfilt(b, a, data[e, picks], axis=-1)
        obj._data = data
    else:
        data[picks] = filtfilt(b, a, data[picks], axis=-1)
        obj._data = data
