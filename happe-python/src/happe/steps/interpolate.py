"""Steps 16 & 18 — bad-channel interpolation.

Step 16 (within-segment, FASTER-style): for each epoch independently compute
four per-channel quality measures (variance, correlation with other channels,
Hurst exponent, max gradient/amplitude — Nolan et al. 2010), z-score each
across channels within that epoch, and flag any channel with |z| > 3 on any
measure as bad *for that epoch only*. Those channel-epoch cells are then
interpolated with spherical-spline interpolation.

Step 18 (full-dataset): after segment rejection, interpolate any channels still
marked bad (steps 6/9) back to the original full montage so every output has a
consistent channel count/order.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

import mne

from ..config import Params


# --------------------------------------------------------------------------- #
# Step 16 — FASTER per-epoch channel properties
# --------------------------------------------------------------------------- #
def _hurst(x: np.ndarray) -> float:
    """Rescaled-range Hurst exponent estimate (FASTER measure 3)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 8:
        return 0.5
    mean = x.mean()
    z = np.cumsum(x - mean)
    r = z.max() - z.min()
    s = x.std()
    if s == 0:
        return 0.5
    return float(np.log(r / s) / np.log(n))


def _single_epoch_channel_properties(epoch: np.ndarray) -> np.ndarray:
    """Return (n_channels, 4) FASTER measures for one epoch (n_ch, n_times)."""
    n_ch = epoch.shape[0]
    var = np.var(epoch, axis=1)
    c = np.corrcoef(epoch)
    np.fill_diagonal(c, np.nan)
    mean_corr = np.nanmean(np.abs(c), axis=1)
    hurst = np.array([_hurst(epoch[i]) for i in range(n_ch)])
    grad = np.max(np.abs(np.diff(epoch, axis=1)), axis=1)
    return np.column_stack([var, mean_corr, hurst, grad])


def interpolate_within_segments(epochs: mne.BaseEpochs,
                                params: Params) -> Dict[int, List[str]]:
    """Flag+interpolate per-epoch bad channels (FASTER). Returns {epoch: [chs]}."""
    if not params.segment.interp_bad_per_seg:
        return {}
    z_thresh = params.segment.faster_z
    ch_names = epochs.info["ch_names"]
    data = epochs.get_data(copy=True)
    per_epoch: Dict[int, List[str]] = {}

    for ei in range(data.shape[0]):
        props = _single_epoch_channel_properties(data[ei])
        bad_mask = np.zeros(data.shape[1], dtype=bool)
        for m in range(props.shape[1]):
            col = props[:, m]
            sd = np.nanstd(col)
            if sd == 0:
                continue
            z = (col - np.nanmean(col)) / sd
            bad_mask |= np.abs(z) > z_thresh
        bad = [ch_names[i] for i in np.where(bad_mask)[0]]
        if bad:
            per_epoch[ei] = bad

    if per_epoch:
        _interp_per_epoch_cells(epochs, per_epoch)
    return per_epoch


def _interp_per_epoch_cells(epochs: mne.BaseEpochs,
                            per_epoch: Dict[int, List[str]]) -> None:
    """Spherical-spline interpolation applied epoch-by-epoch for flagged cells.

    Each epoch is temporarily wrapped as an EvokedArray with its own ``bads``
    so MNE's spherical-spline interpolation runs on just that epoch.
    """
    data = epochs.get_data(copy=True)
    for ei, bads in per_epoch.items():
        ev = mne.EvokedArray(data[ei], epochs.info.copy(), tmin=epochs.tmin,
                             verbose="ERROR")
        ev.info["bads"] = list(bads)
        try:
            ev.interpolate_bads(reset_bads=True, mode="accurate", verbose="ERROR")
            data[ei] = ev.data
        except Exception:
            # DEVIATION: interpolation needs a montage; skip if unavailable.
            pass
    epochs._data = data


# --------------------------------------------------------------------------- #
# Step 18 — full-dataset interpolation
# --------------------------------------------------------------------------- #
def interpolate_full(obj, params: Params):
    """Interpolate any remaining bad channels back to the full montage."""
    if not obj.info["bads"]:
        return obj
    try:
        obj.interpolate_bads(reset_bads=True, mode="accurate", verbose="ERROR")
    except Exception:
        # DEVIATION: no montage positions -> cannot spline-interpolate; leave
        # bads recorded so the QC report still lists them.
        pass
    return obj
