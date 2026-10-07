"""Step 12 — Quality metrics helper (used after steps 3 and 8).

``assess_pipeline_step`` compares data before/after a processing step and
returns the same battery of metrics HAPPE writes to its pipeline-QC table.

Cross-correlation, coherence, RMSE, MAE, SNR and PeakSNR formulas follow the
MATLAB HAPPE ``assessPipelineStep`` implementation exactly (see docstrings on
each helper for the corresponding MATLAB expression).
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from scipy.signal import coherence


def line_noise_freqs_of_interest(freq: float, harmonics: Iterable[float],
                                 neighbors: Iterable[float]) -> List[float]:
    """Build the QC frequency list for line-noise reduction (section 3).

    = fundamental + 2nd harmonic + user harmonics, each with +/- neighbor
    offsets (used only for QC coherence, not for filtering).
    """
    base = [freq, 2 * freq] + list(harmonics)
    out: List[float] = []
    for f in base:
        out.append(f)
        for n in neighbors:
            out.append(f - n)
            out.append(f + n)
    # Deduplicate, keep positive, sorted.
    return sorted({round(f, 6) for f in out if f > 0})


def _flatten(pre: np.ndarray, post: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse epochs into 2-D (channels, timepoints) for metric math."""
    if pre.ndim == 3:
        pre = np.concatenate(list(pre), axis=1)
        post = np.concatenate(list(post), axis=1)
    return pre, post


def assess_pipeline_step(key: str, pre: np.ndarray, post: np.ndarray,
                         srate: float,
                         freqs_of_interest: Iterable[float]) -> Dict[str, float]:
    """Compare ``pre`` and ``post`` arrays and return QC metrics.

    Parameters
    ----------
    key
        Identifier for the step, e.g. ``"line_noise"`` or ``"wavelet"``. For
        keys other than ``"line_noise"`` the amplitude metrics
        (RMSE/MAE/SNR/PeakSNR) are additionally computed.
    pre, post
        Arrays shaped (n_channels, n_times) or (n_epochs, n_channels, n_times).
    srate
        Sampling rate in Hz.
    freqs_of_interest
        Frequencies at which to report magnitude-squared coherence.
    """
    pre, post = _flatten(np.asarray(pre, float), np.asarray(post, float))
    metrics: Dict[str, float] = {}

    # --- Cross-correlation: Pearson r of flattened pre/post (numpy.corrcoef).
    a = pre.ravel()
    b = post.ravel()
    if a.std() == 0 or b.std() == 0:
        metrics["cross_correlation"] = float("nan")
    else:
        metrics["cross_correlation"] = float(np.corrcoef(a, b)[0, 1])

    # --- Per-frequency magnitude-squared coherence, averaged across channels.
    nperseg = int(min(1000, pre.shape[1]))
    foi = list(freqs_of_interest)
    coh_per_freq = {f: [] for f in foi}
    for ch in range(pre.shape[0]):
        f_axis, cxy = coherence(pre[ch], post[ch], fs=srate,
                                nperseg=nperseg, noverlap=0)
        for f in foi:
            idx = int(np.argmin(np.abs(f_axis - f)))
            coh_per_freq[f].append(cxy[idx])
    for f in foi:
        vals = coh_per_freq[f]
        metrics[f"coherence_{f:g}Hz"] = float(np.mean(vals)) if vals else float("nan")

    if key == "line_noise":
        return metrics

    # --- Amplitude metrics (all steps except line-noise reduction). --------
    diff = pre - post
    # RMSE per channel then averaged.
    metrics["rmse"] = float(np.mean(np.sqrt(np.mean(diff ** 2, axis=1))))
    # MAE.
    metrics["mae"] = float(np.mean(np.mean(np.abs(diff), axis=1)))

    # SNR (dB): mean over channels of 20*log10(||post|| / ||post-pre||).
    num = np.sqrt(np.sum(post ** 2, axis=1))
    den = np.sqrt(np.sum(diff ** 2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = 20 * np.log10(num / den)
    metrics["snr_db"] = float(np.mean(snr[np.isfinite(snr)])) if np.isfinite(snr).any() else float("nan")

    # Peak SNR (dB): per-channel MSE, then 20*log10(max(post)/sqrt(MSE)).
    mse = np.mean(diff ** 2, axis=1)
    peak = np.max(post, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        psnr = 20 * np.log10(peak / np.sqrt(mse))
    metrics["peak_snr_db"] = float(np.mean(psnr[np.isfinite(psnr)])) if np.isfinite(psnr).any() else float("nan")

    return metrics
