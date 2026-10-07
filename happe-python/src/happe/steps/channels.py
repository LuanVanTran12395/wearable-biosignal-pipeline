"""Step 2 — Channel corrections, and step 13 — re-insert the reference channel.

* EGI net layouts: reassign channel positions from a standard montage matching
  the geodesic net (via ``read_custom_montage`` / ``make_standard_montage`` +
  ``set_montage``).
* Reference-channel handling: if a flatline/unused reference channel is present
  and average-style re-referencing is requested later, remove it now (stored in
  ``raw.info['temp']['happe_ref']``) so it is never flagged as bad, and
  re-insert it as an all-zero channel just before re-referencing (step 13).
* Channel-of-interest selection: ``all`` / ``include`` / ``exclude``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

import mne

from ..config import Params

_EGI_MONTAGE = {
    32: "GSN-HydroCel-32",
    64: "GSN-HydroCel-64_1.0",
    128: "GSN-HydroCel-128",
    256: "GSN-HydroCel-256",
}


def apply_montage_for_net(raw: mne.io.BaseRaw, params: Params) -> None:
    """Reassign positions for EGI Hydrocel/GSN nets (step 2, first bullet)."""
    net = params.loadInfo.net_type
    if net is None:
        return
    try:
        if Path(net).exists():
            montage = mne.channels.read_custom_montage(net)
        else:
            name = net
            if net.isdigit():
                name = _EGI_MONTAGE.get(int(net), net)
            montage = mne.channels.make_standard_montage(name)
        raw.set_montage(montage, on_missing="warn", verbose="ERROR")
    except Exception:
        # DEVIATION: montage mismatch is non-fatal; positions left unchanged.
        pass


def pull_reference_channel(raw: mne.io.BaseRaw, params: Params) -> Optional[str]:
    """Remove the flatline reference channel and stash it for step 13.

    Only acts when average-style re-referencing is configured.
    Returns the removed channel name (or ``None``).
    """
    ref = params.chans.ref_chan
    if not ref or params.reref.method != "average":
        return None
    if ref not in raw.info["ch_names"]:
        return None
    pos = None
    montage = raw.get_montage()
    if montage is not None and ref in montage.ch_names:
        idx = montage.ch_names.index(ref)
        pos = montage.get_positions()["ch_pos"].get(ref)
    raw.info["temp"] = {"happe_ref": ref,
                        "happe_ref_pos": None if pos is None else np.asarray(pos)}
    raw.drop_channels([ref])
    return ref


def reinsert_reference_channel(raw: mne.io.BaseRaw, params: Params) -> Optional[str]:
    """Step 13 — re-append the reference as an all-zero channel."""
    temp = raw.info.get("temp") or {}
    ref = temp.get("happe_ref") if isinstance(temp, dict) else None
    if not ref:
        return None
    data = np.zeros((1, raw.n_times))
    info = mne.create_info([ref], raw.info["sfreq"], ["eeg"])
    ref_raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.add_channels([ref_raw], force_update_info=True)
    pos = temp.get("happe_ref_pos")
    if pos is not None:
        montage = raw.get_montage()
        if montage is not None:
            d = montage.get_positions()["ch_pos"]
            d[ref] = np.asarray(pos)
            new_m = mne.channels.make_dig_montage(ch_pos=d, coord_frame="head")
            raw.set_montage(new_m, on_missing="ignore", verbose="ERROR")
    return ref


def select_channels_of_interest(raw: mne.io.BaseRaw, params: Params) -> None:
    """Apply ``all`` / ``include`` / ``exclude`` channel selection."""
    mode = params.chans.mode
    if mode == "all":
        return
    present = set(raw.info["ch_names"])
    labels = [c for c in params.chans.labels if c in present]
    if mode == "include":
        keep = labels
        # Never drop a stashed reference channel here.
        raw.pick(keep)
    elif mode == "exclude":
        raw.drop_channels(labels)
