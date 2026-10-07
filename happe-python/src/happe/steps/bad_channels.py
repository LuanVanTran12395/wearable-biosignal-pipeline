"""Steps 6 & 9 — Bad-channel detection (pre- and post-wavelet passes).

Two-tier detection depending on montage density.

Low-density montages:
    * flatline detection (near-zero variance), plus
    * spectral-outlier rejection: normalized log power spectrum over 1-100 Hz,
      z-scored across channels, reject |z| > 2.75 (replicates
      ``pop_rejchan(..., 'measure','spec','freqrange',[1 100])``).

High-density montages:
    * the same spectral-outlier rejection with an asymmetric threshold
      (z in [-5, +1.8935] pre-wavelet, [-5, +2.1316] post-wavelet), plus
    * a clean_rawdata-equivalent flag using a channel-correlation criterion
      (~0.485 pre-wavelet) and a line-noise criterion (~7.1 pre-wavelet,
      disabled post-wavelet). Uses ``pyprep.NoisyChannels`` if installed;
      otherwise a custom implementation of the same criteria.

Detection can run before OR after wavelet thresholding per ``badChans.order``.
The result marks channels as ``raw.info['bads']`` (they are interpolated back
in step 18). A summary dict is returned for the per-file QC report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy import signal

import mne

from ..config import Params


@dataclass
class BadChanResult:
    n_selected: int
    bad_ids: List[str] = field(default_factory=list)

    @property
    def n_good(self) -> int:
        return self.n_selected - len(self.bad_ids)

    @property
    def pct_good(self) -> float:
        return 100.0 * self.n_good / self.n_selected if self.n_selected else float("nan")


def _flatline_channels(data: np.ndarray, ch_names, tol: float = 1e-12) -> List[str]:
    var = np.var(data, axis=1)
    return [ch_names[i] for i in np.where(var <= tol)[0]]


def _spectral_outliers(data: np.ndarray, ch_names, srate: float,
                       z_low: float, z_high: float) -> List[str]:
    """Normalized log-power (1-100 Hz) z-scored across channels, thresholded."""
    nperseg = int(min(srate * 2, data.shape[1]))
    if nperseg < 8:
        return []
    freqs, psd = signal.welch(data, fs=srate, nperseg=nperseg, axis=1)
    band = (freqs >= 1) & (freqs <= 100)
    logp = np.log10(psd[:, band] + 1e-20)
    feat = logp.mean(axis=1)  # per-channel mean log power in band
    mu, sd = feat.mean(), feat.std()
    if sd == 0:
        return []
    z = (feat - mu) / sd
    bad = np.where((z < z_low) | (z > z_high))[0]
    return [ch_names[i] for i in bad]


def _correlation_criterion(data: np.ndarray, ch_names,
                           min_corr: float) -> List[str]:
    """clean_rawdata ChannelCriterion: flag channels poorly correlated with peers."""
    # Max absolute correlation of each channel with every other channel.
    c = np.corrcoef(data)
    np.fill_diagonal(c, 0.0)
    max_corr = np.nanmax(np.abs(c), axis=1)
    bad = np.where(max_corr < min_corr)[0]
    return [ch_names[i] for i in bad]


def _line_noise_criterion(data: np.ndarray, ch_names, srate: float,
                          line_freq: float, thresh: float) -> List[str]:
    """clean_rawdata LineNoiseCriterion: ratio of line-band to broadband power."""
    nperseg = int(min(srate * 2, data.shape[1]))
    if nperseg < 8:
        return []
    freqs, psd = signal.welch(data, fs=srate, nperseg=nperseg, axis=1)
    line_band = (freqs >= line_freq - 2) & (freqs <= line_freq + 2)
    broad = (freqs >= 1) & (freqs <= min(100, srate / 2 - 1))
    line_p = psd[:, line_band].sum(axis=1)
    broad_p = psd[:, broad].sum(axis=1) + 1e-20
    ratio = 10 * np.log10(line_p / broad_p + 1e-20)
    # Z-score the dB ratio; flag channels whose line noise stands out.
    z = (ratio - np.median(ratio)) / (np.std(ratio) + 1e-20)
    bad = np.where(z > thresh)[0]
    return [ch_names[i] for i in bad]


def detect(raw: mne.io.BaseRaw, params: Params, *, after_wavelet: bool) -> BadChanResult:
    """Detect bad channels and record them in ``raw.info['bads']``.

    ``after_wavelet`` selects the pass-2 thresholds where they differ.
    """
    bc = params.badChans
    picks = mne.pick_types(raw.info, eeg=True)
    ch_names = [raw.info["ch_names"][i] for i in picks]
    data = raw.get_data(picks=picks)
    srate = raw.info["sfreq"]

    bad: set[str] = set()

    if bc.density == "low":
        bad.update(_flatline_channels(data, ch_names))
        bad.update(_spectral_outliers(data, ch_names, srate,
                                      -bc.low_z, bc.low_z))
    else:
        z_low, z_high = bc.high_z_post if after_wavelet else bc.high_z_pre
        bad.update(_spectral_outliers(data, ch_names, srate, z_low, z_high))
        bad.update(_flatline_channels(data, ch_names))
        # clean_rawdata-equivalent criteria (prefer pyprep if available).
        used_pyprep = False
        try:  # pragma: no cover - optional dependency
            from pyprep import NoisyChannels

            nd = NoisyChannels(raw.copy().pick(picks), do_detrend=False,
                               random_state=97)
            nd.find_bad_by_correlation(correlation_threshold=bc.corr_thresh_pre)
            if not after_wavelet:
                nd.find_bad_by_hfnoise()
            bad.update(nd.get_bads())
            used_pyprep = True
        except Exception:
            used_pyprep = False
        if not used_pyprep:
            bad.update(_correlation_criterion(data, ch_names, bc.corr_thresh_pre))
            if not after_wavelet:  # line-noise criterion disabled post-wavelet
                bad.update(_line_noise_criterion(
                    data, ch_names, srate, params.lineNoise.freq, bc.line_thresh_pre))

    bad_list = sorted(bad)
    raw.info["bads"] = sorted(set(raw.info["bads"]) | set(bad_list))
    return BadChanResult(n_selected=len(ch_names), bad_ids=bad_list)
