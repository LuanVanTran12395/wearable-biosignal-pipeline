"""Step 3 — Line-noise reduction.

Default method ("CleanLine"-equivalent): a Thomson multi-taper *regression*
that iteratively removes deterministic sinusoidal line components per channel.
EEGLAB's Cleanline plugin (``cleanLineNoise``) fits sine/cosine amplitude at
each target frequency inside a sliding window using DPSS (Slepian) tapers and
subtracts the fitted sinusoid where an F-test is significant.

MNE ships exactly this algorithm as ``mne.filter.notch_filter(...,
method="spectrum_fit")`` — the Thomson F-test multi-taper line-removal of
Mitra & Pesaran (1999), which is the same math Cleanline implements. We use it
as the faithful substitute and expose the same window / bandwidth / p-value
knobs. A hand-rolled sliding-window sine-regression fallback is provided for
environments where the MNE routine is unavailable, and is documented as a
DEVIATION because its stopping rule is simpler than Cleanline's iterative
F-test.

Legacy/alternate method: a plain zero-phase band-stop notch filter around the
line frequency (``revfilt``-style band reject).
"""

from __future__ import annotations

import numpy as np

import mne

from ..config import Params


def apply(raw: mne.io.BaseRaw, params: Params) -> mne.io.BaseRaw:
    """Remove line noise in place-safe fashion, returning the cleaned raw."""
    ln = params.lineNoise
    targets = _filter_targets(ln)

    if ln.method == "notch":
        low = ln.notch_low if ln.notch_low is not None else ln.freq - 1.0
        high = ln.notch_high if ln.notch_high is not None else ln.freq + 1.0
        width = high - low
        # BUGFIX (local): originally only notched a single `center` frequency
        # derived from notch_low/notch_high, silently ignoring `targets`
        # (fundamental + 2nd harmonic + user harmonics) computed above --
        # confirmed via lineNoise_coherence_100Hz staying ~1.0 (unfiltered)
        # while lineNoise_coherence_50Hz dropped, on a run with 2 targets
        # {50, 100}. MNE's raw.notch_filter accepts a list of freqs directly
        # (same as the cleanline branch below), applying a true zero-phase
        # band-stop FIR notch at every target frequency with the same width.
        raw.notch_filter(freqs=targets, picks="eeg", notch_widths=width,
                         verbose="ERROR")
        return raw

    # Default: multitaper spectrum-fit (Cleanline-equivalent).
    try:
        raw.notch_filter(
            freqs=targets,
            method="spectrum_fit",
            mt_bandwidth=ln.bandwidth,
            p_value=ln.p_value,
            filter_length=f"{ln.window_s}s",
            picks="eeg",
            verbose="ERROR",
        )
    except Exception:
        # DEVIATION: fall back to explicit sliding-window sine regression.
        _sine_regression(raw, targets, ln)
    return raw


def _filter_targets(ln) -> list[float]:
    """Frequencies actually removed: fundamental + 2nd harmonic + user extras.

    (Neighbor offsets are for QC only, not filtering — see qc.metrics.)
    """
    return sorted({ln.freq, 2 * ln.freq, *ln.harmonics})


def _sine_regression(raw: mne.io.BaseRaw, freqs, ln) -> None:
    """Sliding-window least-squares sine/cosine removal (fallback path).

    DEVIATION vs Cleanline: this removes the fitted sinusoid unconditionally
    each iteration rather than gating on a per-window F-test. It converges to a
    very similar result for stationary line noise but is not bit-identical.
    """
    sfreq = raw.info["sfreq"]
    picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data()
    win = int(round(ln.window_s * sfreq))
    step = int(round(ln.step_s * sfreq))
    n = data.shape[1]

    for ch in picks:
        x = data[ch]
        start = 0
        while start < n:
            stop = min(start + win, n)
            seg = x[start:stop]
            t = np.arange(stop - start) / sfreq
            for _ in range(ln.max_iter):
                removed_any = False
                for f in freqs:
                    c = np.cos(2 * np.pi * f * t)
                    s = np.sin(2 * np.pi * f * t)
                    A = np.column_stack([c, s])
                    coef, *_ = np.linalg.lstsq(A, seg, rcond=None)
                    fit = A @ coef
                    if np.linalg.norm(fit) > 1e-12:
                        seg = seg - fit
                        removed_any = True
                if not removed_any:
                    break
            x[start:stop] = seg
            start += step
        data[ch] = x
    raw._data = data
