"""Step 7 — ECGone (optional cardiac-artifact removal).

Removes cardiac (ECG/QRS) artifact without necessarily having a dedicated ECG
channel. Best-effort: any failure logs a warning and returns the data
unchanged so the file continues to wavelet thresholding.

Procedure (per the MATLAB ECGone):
    1. Segment continuous data into fixed-length epochs (``epoch_len_s``).
    2. Per epoch obtain an ECG proxy: the designated ECG channel(s), or the
       mean of artifact-laden channels after <100 Hz low-pass + squaring
       (emphasizes QRS peaks), excluding outlier channels.
    3. Detect peaks (``find_peaks`` with min distance = peakWinSize * srate);
       discard peaks at epoch edges.
    4. Channel-proxy path only: check "peakiness" (median relative peak
       deviation > threshold), else skip that epoch.
    5. Build a peak-locked ECG template and subtract a least-squares-scaled
       version from each EEG channel; reconstruct the cleaned continuous signal.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

import mne

from ..config import Params

log = logging.getLogger("happe.ecgone")


def apply(raw: mne.io.BaseRaw, params: Params) -> mne.io.BaseRaw:
    if not params.ecgone.enabled:
        return raw
    try:
        return _run(raw, params)
    except Exception as exc:  # best-effort: never abort the file here
        log.warning("ECGone failed, continuing without it: %s", exc)
        return raw


def _run(raw: mne.io.BaseRaw, params: Params) -> mne.io.BaseRaw:
    eg = params.ecgone
    sfreq = raw.info["sfreq"]
    picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data()
    n = data.shape[1]
    ep_len = int(eg.epoch_len_s * sfreq)
    win = max(1, int(eg.peak_win_s * sfreq))
    edge = win  # discard peaks within one window of the epoch edge

    # Designated ECG proxy channels, if any.
    ecg_idx = [raw.info["ch_names"].index(c) for c in eg.ecg_channels
               if c in raw.info["ch_names"]]
    use_channel_proxy = len(ecg_idx) == 0

    b, a = butter(4, min(eg.lowpass_hz / (sfreq / 2), 0.999), btype="low")

    for start in range(0, n, ep_len):
        stop = min(start + ep_len, n)
        if stop - start < 3 * win:
            continue
        seg = data[:, start:stop]

        if use_channel_proxy:
            filt = filtfilt(b, a, seg[picks], axis=-1)
            proxy = np.mean(filt ** 2, axis=0)  # squared, mean over channels
        else:
            proxy = np.mean(np.abs(seg[ecg_idx]), axis=0)

        peaks, _ = find_peaks(proxy, distance=win)
        peaks = peaks[(peaks >= edge) & (peaks < (stop - start) - edge)]
        if len(peaks) < 2:
            continue

        if use_channel_proxy:
            med = np.median(proxy)
            peakiness = np.median(proxy[peaks]) / (med + 1e-20)
            if peakiness < eg.peakiness_thresh:
                continue  # not peaky enough — skip this epoch

        # Build a peak-locked template per channel and subtract a scaled copy.
        half = win
        for ch in picks:
            x = seg[ch]
            stacks = []
            for pk in peaks:
                lo, hi = pk - half, pk + half
                if lo < 0 or hi >= x.shape[0]:
                    continue
                stacks.append(x[lo:hi])
            if len(stacks) < 2:
                continue
            template = np.mean(np.vstack(stacks), axis=0)
            for pk in peaks:
                lo, hi = pk - half, pk + half
                if lo < 0 or hi >= x.shape[0]:
                    continue
                seg_win = x[lo:hi]
                # Least-squares scale of the template to this window.
                denom = float(template @ template) + 1e-20
                scale = float(seg_win @ template) / denom
                x[lo:hi] = seg_win - scale * template
            seg[ch] = x
        data[:, start:stop] = seg

    raw._data = data
    return raw
