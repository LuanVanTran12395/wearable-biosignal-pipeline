"""Step 7.5 (DEVIATION, local, not in original MATLAB HAPPE) — optional
Temporal Derivative Distribution Repair (TDDR) pre-clean, inserted right
before wavelet thresholding.

See ``config.PreWaveletTddr`` for the full rationale. In short: on this
2-channel dry-electrode wearable dataset, wavelet's per-subband BayesShrink
threshold (see ``wavelet.py``) can be dominated by large contact-potential
step artifacts that leak broadband spectral energy into the same wavelet
subband as genuine alpha-band (8-13Hz) oscillations, causing wavelet to
erase real signal along with the artifact. TDDR corrects persistent
step-like level shifts in the time domain, per channel, restricted to
content BELOW ``cutoff_hz`` by construction -- content above the cutoff
passes through completely unmodified, so keeping the cutoff below the
alpha band (default 4.0Hz, targeting delta/theta) structurally guarantees
this step cannot itself remove alpha content; it only reduces the
low-frequency artifact energy reaching wavelet's subband threshold
estimation.

The algorithm/implementation is the same one already validated and in
production use for this project's fNIRS motion-correction pipeline
(``lib/fnirs_pipeline_v2.py::tddr()``, cross-checked against an
independent reimplementation and a synthetic step-shift unit test there).
Vendored here (rather than cross-importing) to keep this package
self-contained per its own packaging philosophy.

Citation: Fishburn FA, Ludlum RS, Vaidya CJ, Medvedev AV. "Temporal
Derivative Distribution Repair (TDDR): A motion correction method for
fNIRS." Neuroimage. 2019;184:171-179.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

import mne

from ..config import Params


def tddr(signal: np.ndarray, sample_rate: float, filter_cutoff_hz: float = 4.0,
          filter_order: int = 3) -> Tuple[np.ndarray, np.ndarray, int]:
    """Temporal Derivative Distribution Repair on one channel.

    Returns ``(corrected_signal, sample_weights, n_iterations)``.
    ``sample_weights`` (length n-1) close to 0 = that derivative sample was
    heavily downweighted as a step artifact.
    """
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n < int(sample_rate * 4):
        return x.copy(), np.ones(max(n - 1, 0)), 0

    nyq = sample_rate / 2.0
    fc_norm = filter_cutoff_hz / nyq
    if 0 < fc_norm < 1:
        sos = butter(filter_order, fc_norm, btype="low", output="sos")
        signal_low = sosfiltfilt(sos, x)
    else:
        signal_low = x.copy()
    signal_high = x - signal_low

    D = np.diff(signal_low)
    w = np.ones(len(D))
    mu = np.inf
    tune = 4.685
    iteration = 0
    while abs(mu) > 1e-8 and iteration < 50:
        mu = float(np.sum(D * w) / np.sum(w))
        D = D - mu
        sigma = float(np.sqrt(np.sum(w * D ** 2) / np.sum(w)))
        if sigma < 1e-12:
            break
        r = D / sigma / tune
        w = np.where(np.abs(r) < 1, (1 - r ** 2) ** 2, 0.0)
        iteration += 1

    new_D = w * D
    new_signal_low = np.cumsum(np.concatenate(([signal_low[0]], new_D)))
    corrected = new_signal_low + signal_high
    return corrected, w, iteration


def apply(raw: mne.io.BaseRaw, params: Params) -> mne.io.BaseRaw:
    """Apply TDDR per EEG channel, in place, if enabled in config."""
    cfg = params.preWaveletTddr
    if not cfg.enabled:
        return raw
    picks = mne.pick_types(raw.info, eeg=True)
    sfreq = raw.info["sfreq"]
    data = raw.get_data()
    for ch in picks:
        corrected, _weights, _n_iter = tddr(
            data[ch], sfreq, filter_cutoff_hz=cfg.cutoff_hz,
            filter_order=cfg.filter_order,
        )
        data[ch] = corrected
    raw._data = data
    return raw
