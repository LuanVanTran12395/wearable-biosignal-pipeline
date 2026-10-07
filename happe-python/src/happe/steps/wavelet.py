"""Step 8 — Wavelet thresholding (core denoising step).

This replaces ICA-based cleaning as HAPPE's primary artifact remover. The
wavelet-domain output is the *artifact estimate*; the cleaned signal is
``original - artifact``.

Family / level selection (per paradigm and sampling rate):
    Resting/non-ERP : bior4.4 ; level 10 (>500 Hz), 9 (250-500], 8 (<=250)
    ERP             : coif4   ; level 11 / 10 / 9 for the same srate bands

Denoising method: Bayesian shrinkage with level-dependent noise estimation,
mirroring MATLAB ``wdenoise(..., 'DenoisingMethod','Bayes',
'NoiseEstimate','LevelDependent')``. Implemented here as:

    1. ``pywt.wavedec`` to the chosen level.
    2. Per detail sub-band, estimate noise sigma via the MAD estimator
       ``sigma = median(|d|) / 0.6745`` (level-dependent, so each sub-band gets
       its own sigma).
    3. Apply a BayesShrink adaptive threshold ``T = sigma^2 / sigma_signal``
       (soft or hard) per sub-band per channel.
    4. ``pywt.waverec`` to reconstruct the artifact estimate.

Threshold rule: Hard by default; Soft only for ERP paradigms.

A legacy wavelet-ICA hybrid path (``params.wavelet.method == "legacy"``) is
provided: extended-Infomax ICA, global soft threshold on each component
(coiflet level 5), reconstruct artifact = mixing matrix x thresholded sources.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt

import mne

from ..config import Params


@dataclass
class WaveletResult:
    raw: mne.io.BaseRaw
    #: var(post)/var(pre) * 100 — "percent variance retained post-wavelet".
    percent_var_retained: float


def _select_wavelet_and_level(params: Params, srate: float) -> tuple[str, int]:
    if params.wavelet.wavelet and params.wavelet.level:
        return params.wavelet.wavelet, params.wavelet.level
    if params.paradigm.erp:
        wav = "coif4"
        level = 11 if srate > 500 else (10 if srate > 250 else 9)
    else:
        wav = "bior4.4"
        level = 10 if srate > 500 else (9 if srate > 250 else 8)
    if params.wavelet.wavelet:
        wav = params.wavelet.wavelet
    if params.wavelet.level:
        level = params.wavelet.level
    return wav, level


def _bayes_threshold_block(detail: np.ndarray, mode: str) -> np.ndarray:
    """BayesShrink adaptive threshold on one detail sub-band (one channel).

    ``sigma`` (noise) from the MAD estimator; ``sigma_signal`` from the
    residual signal variance; ``T = sigma^2 / sigma_signal`` (BayesShrink).
    """
    sigma = np.median(np.abs(detail)) / 0.6745
    if sigma <= 0:
        return detail
    var_obs = np.mean(detail ** 2)
    var_signal = max(var_obs - sigma ** 2, 1e-12)
    thresh = sigma ** 2 / np.sqrt(var_signal)
    return pywt.threshold(detail, thresh, mode=mode)


def _local_threshold_value(window: np.ndarray) -> float:
    """BayesShrink threshold *value* (no thresholding applied) for one window."""
    sigma = np.median(np.abs(window)) / 0.6745
    if sigma <= 0:
        return 0.0
    var_obs = np.mean(window ** 2)
    var_signal = max(var_obs - sigma ** 2, 1e-12)
    return float(sigma ** 2 / np.sqrt(var_signal))


def _bayes_threshold_smooth(detail: np.ndarray, mode: str, block_size: int,
                            hop: int) -> np.ndarray:
    """Threshold with a smoothly-varying (overlapping-window) threshold value.

    Estimates the local threshold at a series of window centers spaced
    ``hop`` apart (each window ``block_size`` wide), then linearly
    interpolates the threshold *value* across every coefficient position and
    applies it pointwise to the ORIGINAL coefficient array. This is not an
    overlap-add of thresholded segments (which would double-count overlapping
    samples) -- only the scalar threshold blends smoothly, avoiding the step
    discontinuity a hard block boundary would otherwise leave in the artifact
    estimate.
    """
    n = len(detail)
    half = block_size // 2
    centers = list(range(0, n, hop))
    if centers[-1] != n - 1:
        centers.append(n - 1)
    centers = np.array(centers)
    thresh_vals = np.array([
        _local_threshold_value(detail[max(0, c - half):min(n, c + half)])
        for c in centers
    ])
    thresh = np.interp(np.arange(n), centers, thresh_vals)
    if mode == "soft":
        return np.sign(detail) * np.maximum(np.abs(detail) - thresh, 0.0)
    return np.where(np.abs(detail) > thresh, detail, 0.0)


def _bayes_threshold(detail: np.ndarray, mode: str,
                     block_size: int | None = None,
                     block_overlap: float = 0.0) -> np.ndarray:
    """BayesShrink threshold, optionally computed independently per block.

    DEVIATION (local): when ``block_size`` is given, sigma/threshold is
    re-estimated from each block's own coefficients rather than the whole
    sub-band at once, so a severe artifact in one block cannot raise the
    threshold applied to a clean block elsewhere (see ``Wavelet.block_seconds``
    in ``config.py``). ``block_size`` is in coefficient-array units (already
    converted from seconds via that sub-band's decimated sample rate). When
    ``block_overlap`` > 0, uses :func:`_bayes_threshold_smooth` instead of
    hard-edged blocks (see ``Wavelet.block_overlap``).
    """
    if not block_size or block_size >= len(detail):
        return _bayes_threshold_block(detail, mode)
    if block_overlap and block_overlap > 0:
        hop = max(1, int(round(block_size * (1 - block_overlap))))
        return _bayes_threshold_smooth(detail, mode, block_size, hop)
    n_blocks = max(1, round(len(detail) / block_size))
    blocks = np.array_split(detail, n_blocks)
    return np.concatenate([_bayes_threshold_block(b, mode) for b in blocks])


def apply(raw: mne.io.BaseRaw, params: Params) -> WaveletResult:
    """Denoise ``raw`` and return cleaned raw + percent-variance-retained QC."""
    srate = raw.info["sfreq"]

    if params.wavelet.method == "legacy":
        return _legacy_wica(raw, params)

    wav, level = _select_wavelet_and_level(params, srate)
    mode = "soft" if params.wavelet.threshold_rule == "soft" else "hard"
    picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data()
    pre_var = np.var(data)

    w = pywt.Wavelet(wav)
    for ch in picks:
        sig = data[ch]
        max_lvl = pywt.dwt_max_level(len(sig), w.dec_len)
        use_lvl = min(level, max_lvl)
        coeffs = pywt.wavedec(sig, w, level=use_lvl, mode="periodization")
        # coeffs[0] = approximation (kept as-is); coeffs[1:] = details.
        artifact_coeffs = [np.zeros_like(coeffs[0])]
        for i, d in enumerate(coeffs[1:]):
            block_size = None
            if params.wavelet.block_seconds:
                # detail index i (0-based) is decomposition level (use_lvl - i);
                # that sub-band's coefficients are decimated by 2**level vs srate.
                sub_level = use_lvl - i
                coeff_rate = srate / (2 ** sub_level)
                block_size = max(1, round(params.wavelet.block_seconds * coeff_rate))
            # Artifact estimate = the *thresholded* detail (what wdenoise keeps
            # as signal is removed from the raw; here we build the complement).
            kept = _bayes_threshold(d, mode, block_size, params.wavelet.block_overlap)
            artifact_coeffs.append(kept)
        artifact = pywt.waverec(artifact_coeffs, w, mode="periodization")
        artifact = artifact[: len(sig)]
        data[ch] = sig - artifact  # cleaned = original - artifact

    raw._data = data
    post_var = np.var(data)
    pct = float(post_var / pre_var * 100) if pre_var > 0 else float("nan")
    return WaveletResult(raw=raw, percent_var_retained=pct)


def _legacy_wica(raw: mne.io.BaseRaw, params: Params) -> WaveletResult:
    """Legacy wavelet-ICA hybrid (optional ``method='legacy'``).

    Extended-Infomax ICA; each source gets a global soft wavelet threshold
    (coiflet level 5); artifact = mixing_matrix @ thresholded_sources; cleaned
    = original - artifact.
    """
    pre_var = np.var(raw.get_data())
    ica = mne.preprocessing.ICA(method="infomax",
                                fit_params=dict(extended=True),
                                max_iter="auto", random_state=97,
                                verbose="ERROR")
    ica.fit(raw, verbose="ERROR")
    sources = ica.get_sources(raw).get_data()  # (n_comp, n_times)

    thr_sources = np.empty_like(sources)
    w = pywt.Wavelet("coif4")
    for i, s in enumerate(sources):
        max_lvl = pywt.dwt_max_level(len(s), w.dec_len)
        lvl = min(5, max_lvl)
        coeffs = pywt.wavedec(s, w, level=lvl, mode="periodization")
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        thr = sigma * np.sqrt(2 * np.log(len(s))) if sigma > 0 else 0.0
        coeffs = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft")
                                for c in coeffs[1:]]
        rec = pywt.waverec(coeffs, w, mode="periodization")
        thr_sources[i] = rec[: len(s)]

    mixing = ica.mixing_matrix_
    # Project the thresholded (artifact) sources back to sensor space.
    pca_comp = ica.pca_components_[: ica.n_components_]
    artifact = pca_comp.T @ (mixing @ thr_sources)
    data = raw.get_data()
    # Undo the ICA mean/whitening offset approximately by working on picks only.
    picks = mne.pick_types(raw.info, eeg=True)
    artifact = artifact[: len(picks)]
    data[picks] = data[picks] - artifact
    raw._data = data
    post_var = np.var(data)
    pct = float(post_var / pre_var * 100) if pre_var > 0 else float("nan")
    return WaveletResult(raw=raw, percent_var_retained=pct)
