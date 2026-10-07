"""Step 22 — Save outputs, and intermediate stage saving.

Supports MNE ``.fif``, EEGLAB ``.set`` (via ``mne.export``), raw ``.mat``, and
flat-text per-trial / trial-averaged exports (channels x timepoints,
transposed, 8-decimal precision) — for the full dataset and each
per-tag/per-condition split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

import mne

from ..config import Params

DataObj = Union[mne.io.BaseRaw, mne.BaseEpochs]


def save_intermediate(obj: DataObj, folder: Path, stem: str) -> Path:
    """Persist an intermediate stage as ``.fif`` (mirrors pop_saveset).

    Saving after each major step lets reprocessing resume from any stage.
    """
    folder.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, mne.BaseEpochs):
        out = folder / f"{stem}-epo.fif"
        obj.save(out, overwrite=True, verbose="ERROR")
    else:
        out = folder / f"{stem}-raw.fif"
        obj.save(out, overwrite=True, verbose="ERROR")
    return out


def save_outputs(obj: DataObj, folder: Path, stem: str, params: Params) -> list[Path]:
    """Write the final deliverable(s) in every configured format."""
    folder.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in params.outputFormat.formats:
        if fmt == "fif":
            written.append(save_intermediate(obj, folder, stem))
        elif fmt == "set":
            written.append(_save_set(obj, folder, stem))
        elif fmt == "mat":
            written.append(_save_mat(obj, folder, stem))
        elif fmt == "txt":
            written.extend(_save_txt(obj, folder, stem, params))
    return written


def _save_set(obj: DataObj, folder: Path, stem: str) -> Path:
    out = folder / f"{stem}.set"
    try:
        obj.export(out, fmt="eeglab", overwrite=True, verbose="ERROR")
    except Exception:
        # eeglabio missing -> documented deviation: fall back to fif.
        out = save_intermediate(obj, folder, stem)
    return out


def _save_mat(obj: DataObj, folder: Path, stem: str) -> Path:
    from scipy.io import savemat

    out = folder / f"{stem}.mat"
    data = obj.get_data()
    savemat(out, {
        "data": data,
        "srate": float(obj.info["sfreq"]),
        "chanlabels": np.array(obj.info["ch_names"], dtype=object),
    })
    return out


def _save_txt(obj: DataObj, folder: Path, stem: str, params: Params) -> list[Path]:
    """Flat text export: channels x timepoints, transposed, 8 decimals."""
    written: list[Path] = []
    data = obj.get_data()
    if data.ndim == 3:  # epochs: (n_epochs, n_ch, n_times)
        # Per-trial concatenation and (optionally) trial-averaged export.
        per_trial = np.concatenate(list(data), axis=1)  # (n_ch, epochs*times)
        out = folder / f"{stem}_pertrial.txt"
        np.savetxt(out, per_trial.T, fmt="%.8f")
        written.append(out)
        if params.outputFormat.trial_average_txt:
            avg = data.mean(axis=0)  # (n_ch, n_times)
            outa = folder / f"{stem}_average.txt"
            np.savetxt(outa, avg.T, fmt="%.8f")
            written.append(outa)
    else:  # continuous
        out = folder / f"{stem}.txt"
        np.savetxt(out, data.T, fmt="%.8f")
        written.append(out)
    return written
