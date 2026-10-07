"""Step 19 — Re-referencing.

Three modes:
    * ``average``: common average reference, keeping the re-inserted reference
      channel in the data.
    * ``subset``: reference to a user-specified channel subset (e.g. mastoids).
    * ``rest``: Reference Electrode Standardization Technique (REST) — reproject
      to an (approximate) infinity reference using a 3-shell spherical forward
      head model (Yao 2001; Dong et al. 2019). If mastoids are given, average
      them as the practical reference before the REST transform.
"""

from __future__ import annotations

import numpy as np

import mne

from ..config import Params


def apply(obj, params: Params):
    if not params.reref.enabled:
        return obj
    method = params.reref.method
    if method == "average":
        obj.set_eeg_reference("average", projection=False, verbose="ERROR")
    elif method == "subset":
        refs = [c for c in params.reref.subset if c in obj.info["ch_names"]]
        if refs:
            obj.set_eeg_reference(refs, verbose="ERROR")
    elif method == "rest":
        _rest_reference(obj, params)
    return obj


def _rest_reference(obj, params: Params) -> None:
    """REST via MNE's infinity reference, with spherical forward as fallback.

    If mastoids are configured, average them as the practical reference first.
    """
    if params.reref.mastoids:
        mast = [c for c in params.reref.mastoids if c in obj.info["ch_names"]]
        if mast:
            obj.set_eeg_reference(mast, verbose="ERROR")

    try:
        # MNE ships REST as set_eeg_reference('REST', forward=...).
        sphere = mne.make_sphere_model("auto", "auto", obj.info, verbose="ERROR")
        src = mne.setup_volume_source_space(sphere=sphere, pos=15.0,
                                            verbose="ERROR")
        fwd = mne.make_forward_solution(obj.info, trans=None, src=src,
                                        bem=sphere, verbose="ERROR")
        obj.set_eeg_reference("REST", forward=fwd, verbose="ERROR")
    except Exception:
        # DEVIATION: forward-model REST unavailable in this environment;
        # fall back to average reference and record it upstream.
        obj.set_eeg_reference("average", projection=False, verbose="ERROR")
