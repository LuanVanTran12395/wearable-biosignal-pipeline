"""Steps 14, 17 & 20 — Segmentation, epoch rejection, and tag/condition split.

Step 14 — Segmentation:
    * Task/ERP: epoch around event tag(s) (``mne.Epochs`` with tmin/tmax from
      segment start/end in ms). For ERP, first shift event onsets by the
      configured offset (ms) to correct trigger delay. If none of the tags are
      present, raise :class:`NoTagsError`.
    * Resting: fixed-length non-overlapping epochs (``mne.make_fixed_length_epochs``),
      no baseline removal here.
    * Never re-segment already-epoched data.

Step 17 — Epoch rejection: amplitude and/or joint-probability ("similarity")
criteria, optionally restricted to a ROI. Raises :class:`AllTrialsRejectedError`
if everything would be rejected, and :class:`SingleTrialRejectionError` if run
on a single-epoch dataset.

Step 20 — Split by tag/condition for multi-tag task datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

import mne

from ..config import Params
from ..exceptions import (
    AllTrialsRejectedError,
    NoTagsError,
    SingleTrialRejectionError,
)


# --------------------------------------------------------------------------- #
# Step 14 — Segmentation
# --------------------------------------------------------------------------- #
def segment(raw: mne.io.BaseRaw, params: Params) -> mne.BaseEpochs:
    if isinstance(raw, mne.BaseEpochs):
        return raw  # never re-segment
    if params.paradigm.task or params.paradigm.erp:
        return _segment_events(raw, params)
    return _segment_regular(raw, params)


def _segment_regular(raw: mne.io.BaseRaw, params: Params) -> mne.BaseEpochs:
    return mne.make_fixed_length_epochs(
        raw, duration=params.segment.reg_len_s, preload=True, verbose="ERROR")


def _segment_events(raw: mne.io.BaseRaw, params: Params) -> mne.BaseEpochs:
    try:
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    except Exception:
        events, event_id = np.empty((0, 3), int), {}

    tags = params.paradigm.onset_tags
    wanted = {k: v for k, v in event_id.items() if k in tags} if tags else event_id
    if tags and not wanted:
        raise NoTagsError(
            f"None of the configured tags {tags} found in file "
            f"(present: {sorted(event_id)})")

    # ERP trigger-delay correction: shift onsets by offset (ms).
    if params.paradigm.erp and params.paradigm.onset_offset_ms and len(events):
        shift = int(round(params.paradigm.onset_offset_ms / 1000.0
                          * raw.info["sfreq"]))
        events = events.copy()
        events[:, 0] += shift

    tmin = params.segment.start_ms / 1000.0
    tmax = params.segment.end_ms / 1000.0
    return mne.Epochs(raw, events, event_id=wanted or None, tmin=tmin, tmax=tmax,
                      baseline=None, preload=True, verbose="ERROR")


# --------------------------------------------------------------------------- #
# Step 17 — Epoch rejection
# --------------------------------------------------------------------------- #
@dataclass
class RejectResult:
    epochs: mne.BaseEpochs
    n_pre: int
    n_post: int
    kept_indices: List[int] = field(default_factory=list)
    per_tag: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def pct_retained(self) -> float:
        return 100.0 * self.n_post / self.n_pre if self.n_pre else float("nan")


def _roi_picks(epochs: mne.BaseEpochs, params: Params) -> List[int]:
    sr = params.segRej
    names = epochs.info["ch_names"]
    if sr.roi_mode == "include":
        return [i for i, n in enumerate(names) if n in sr.roi_labels]
    if sr.roi_mode == "exclude":
        return [i for i, n in enumerate(names) if n not in sr.roi_labels]
    return list(range(len(names)))


def _joint_prob_bad(data: np.ndarray, picks: List[int], sd: float) -> np.ndarray:
    """EEGLAB pop_jointprob-style rejection over epochs.

    For each channel, estimate an amplitude probability density across epochs,
    sum the (negative) log-probability across ROI channels per epoch, z-score
    across epochs, and flag epochs beyond ``sd`` std.
    """
    n_ep = data.shape[0]
    # Per-epoch, per-channel feature: mean absolute amplitude.
    feat = np.mean(np.abs(data[:, picks, :]), axis=2)  # (n_ep, n_roi)
    logp = np.zeros(n_ep)
    for c in range(feat.shape[1]):
        col = feat[:, c]
        mu, sigma = col.mean(), col.std() + 1e-20
        # Gaussian surprisal (negative log-likelihood) as the density proxy.
        nll = 0.5 * ((col - mu) / sigma) ** 2
        logp += nll
    z = (logp - logp.mean()) / (logp.std() + 1e-20)
    return z > sd


def reject(epochs: mne.BaseEpochs, params: Params) -> RejectResult:
    sr = params.segRej
    n_pre = len(epochs)
    if not sr.enabled:
        return RejectResult(epochs=epochs, n_pre=n_pre, n_post=n_pre,
                            kept_indices=list(range(n_pre)))
    if n_pre <= 1:
        raise SingleTrialRejectionError(
            "Epoch rejection requested on a single-epoch dataset")

    picks = _roi_picks(epochs, params)
    data = epochs.get_data(copy=True)
    bad = np.zeros(n_pre, dtype=bool)

    if sr.by_amplitude:
        roi = data[:, picks, :]
        amp_bad = (roi.max(axis=(1, 2)) > sr.amp_max) | \
                  (roi.min(axis=(1, 2)) < sr.amp_min)
        bad |= amp_bad

    if sr.by_similarity:
        bad |= _joint_prob_bad(data, picks, sr.similarity_sd)

    kept = np.where(~bad)[0].tolist()
    if not kept:
        raise AllTrialsRejectedError("Segment rejection removed all epochs")

    out = epochs[kept]
    per_tag = _per_tag_counts(epochs, out)
    return RejectResult(epochs=out, n_pre=n_pre, n_post=len(kept),
                        kept_indices=kept, per_tag=per_tag)


def _per_tag_counts(pre: mne.BaseEpochs, post: mne.BaseEpochs) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    if not pre.event_id or len(pre.event_id) <= 1:
        return out
    for tag in pre.event_id:
        try:
            out[tag] = {"pre": len(pre[tag]), "post": len(post[tag])}
        except Exception:
            out[tag] = {"pre": 0, "post": 0}
    return out


# --------------------------------------------------------------------------- #
# Step 20 — split by tag / condition
# --------------------------------------------------------------------------- #
def split_by_tag(epochs: mne.BaseEpochs, params: Params) -> Dict[str, mne.BaseEpochs]:
    """Return {name: epochs} for each individual tag and each condition group."""
    splits: Dict[str, mne.BaseEpochs] = {}
    tags = params.paradigm.onset_tags
    if not tags or len(tags) <= 1:
        return splits
    for tag in tags:
        try:
            if tag in epochs.event_id:
                splits[tag] = epochs[tag]
        except Exception:
            pass
    for cond, cond_tags in params.paradigm.conditions.items():
        present = [t for t in cond_tags if t in epochs.event_id]
        if present:
            try:
                splits[cond] = epochs[present]
            except Exception:
                pass
    return splits
