"""Step 1 — Data loading.

Supports EEGLAB ``.set``, EDF/BDF, EGI ``.mff``/``.raw``, and generic
``.mat``/matrix inputs (with a user-supplied sampling rate + channel-location
file). The sampling rate is always determined from the loaded object.

For task/ERP paradigms, event annotations are attached (equivalent of
``pop_importevent``). The loaded object is validated (channel count, non-empty
data) before returning; validation failure raises :class:`LoadFailError`,
which the batch loop catches and logs without crashing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

import mne

from ..config import Params
from ..exceptions import LoadFailError

_EXT_FORMAT = {
    ".set": "set",
    ".edf": "edf",
    ".bdf": "bdf",
    ".mff": "mff",
    ".raw": "egi_raw",
    ".mat": "mat",
    ".fif": "fif",
}


def _infer_format(path: Path, params: Params) -> str:
    if params.loadInfo.file_format:
        return params.loadInfo.file_format
    fmt = _EXT_FORMAT.get(path.suffix.lower())
    if fmt is None:
        raise LoadFailError(f"Cannot infer file format for {path.name!r}")
    return fmt


def load_raw(path: str, params: Params) -> mne.io.BaseRaw:
    """Load ``path`` into an :class:`mne.io.Raw`, validated and event-tagged.

    Raises
    ------
    LoadFailError
        If the format is unsupported, the reader fails, or validation fails.
    """
    p = Path(path)
    if not p.exists():
        raise LoadFailError(f"File does not exist: {path}")

    fmt = _infer_format(p, params)
    try:
        raw = _read_by_format(p, fmt, params)
    except LoadFailError:
        raise
    except Exception as exc:  # wrap any backend error into LoadFailError
        raise LoadFailError(f"Failed reading {p.name} as {fmt}: {exc}") from exc

    _attach_events(raw, params)
    _validate(raw, p)
    return raw


def _read_by_format(p: Path, fmt: str, params: Params) -> mne.io.BaseRaw:
    if fmt == "set":
        return mne.io.read_raw_eeglab(p, preload=True, verbose="ERROR")
    if fmt == "edf":
        return mne.io.read_raw_edf(p, preload=True, verbose="ERROR")
    if fmt == "bdf":
        return mne.io.read_raw_bdf(p, preload=True, verbose="ERROR")
    if fmt in ("mff", "egi_raw"):
        return mne.io.read_raw_egi(p, preload=True, verbose="ERROR")
    if fmt == "mat":
        return _read_matrix(p, params)
    if fmt == "fif":
        # DEVIATION (local addition, not upstream MATLAB HAPPE): native MNE
        # .fif support. Added for this project because Layer 2 exports
        # labeled EEG as eeg_labeled_raw.fif -- .fif is MNE's own format, so
        # this is a direct read (no lossy round-trip through EDF/16-bit
        # integer quantization needed to reuse the pipeline).
        return mne.io.read_raw_fif(p, preload=True, verbose="ERROR")
    raise LoadFailError(f"Unsupported format: {fmt!r}")


def _read_matrix(p: Path, params: Params) -> mne.io.BaseRaw:
    """Generic matrix loader for ``.mat``/``.npy`` (channels x timepoints).

    Requires ``params.loadInfo.srate`` and, ideally, a channel-location file.
    """
    srate = params.loadInfo.srate
    if srate is None:
        raise LoadFailError("Matrix input requires loadInfo.srate to be set")

    if p.suffix.lower() == ".npy":
        data = np.load(p)
    else:
        from scipy.io import loadmat

        mat = loadmat(p)
        arrays = [v for k, v in mat.items() if not k.startswith("__")
                  and isinstance(v, np.ndarray) and v.ndim == 2]
        if not arrays:
            raise LoadFailError(f"No 2-D data matrix found in {p.name}")
        data = max(arrays, key=lambda a: a.size).astype(float)

    if data.shape[0] > data.shape[1]:
        # Heuristic: EEG has more timepoints than channels; transpose if needed.
        data = data.T
    n_ch = data.shape[0]
    ch_names = [f"E{i + 1}" for i in range(n_ch)]
    info = mne.create_info(ch_names, sfreq=float(srate), ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    if params.loadInfo.chanlocs:
        _apply_montage(raw, params.loadInfo.chanlocs)
    return raw


def _apply_montage(raw: mne.io.BaseRaw, chanlocs: str) -> None:
    """Set channel positions from a .sfp file or standard montage name."""
    try:
        if Path(chanlocs).exists():
            montage = mne.channels.read_custom_montage(chanlocs)
        else:
            montage = mne.channels.make_standard_montage(chanlocs)
        raw.set_montage(montage, on_missing="warn", verbose="ERROR")
    except Exception as exc:
        raise LoadFailError(f"Failed to apply montage {chanlocs!r}: {exc}") from exc


def _attach_events(raw: mne.io.BaseRaw, params: Params) -> None:
    """Attach event annotations for task/ERP paradigms (pop_importevent)."""
    if not (params.paradigm.task or params.paradigm.erp):
        return
    ev = params.loadInfo.event_file
    if not ev:
        return  # events may already be embedded (e.g. .set / .mff)
    p = Path(ev)
    if not p.exists():
        raise LoadFailError(f"Event file not found: {ev}")
    try:
        if p.suffix.lower() == ".csv":
            import pandas as pd

            df = pd.read_csv(p)
            onset = df["onset"].to_numpy(dtype=float)
            desc = df.get("type", df.get("description")).astype(str).tolist()
            ann = mne.Annotations(onset=onset, duration=np.zeros_like(onset),
                                  description=desc)
            raw.set_annotations(ann)
        else:  # let MNE try
            ann = mne.read_annotations(p)
            raw.set_annotations(ann)
    except Exception as exc:
        raise LoadFailError(f"Failed importing events from {ev}: {exc}") from exc


def _validate(raw: mne.io.BaseRaw, p: Path) -> None:
    if raw.info["nchan"] < 1:
        raise LoadFailError(f"{p.name}: no channels present")
    if raw.n_times < 1:
        raise LoadFailError(f"{p.name}: empty data (0 timepoints)")
    data = raw.get_data()
    if not np.isfinite(data).any():
        raise LoadFailError(f"{p.name}: data contains no finite samples")
    if raw.info["sfreq"] <= 0:
        raise LoadFailError(f"{p.name}: invalid sampling rate {raw.info['sfreq']}")
