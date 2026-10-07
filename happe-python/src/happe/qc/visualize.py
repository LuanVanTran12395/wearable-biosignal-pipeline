"""Step 21 — Visualization (optional).

One figure per file: ERP butterfly + topography at a user time window
(``pop_timtopo`` equivalent) for ERP paradigms; a power-spectrum plot with
topography over a user frequency range (``pop_spectopo`` equivalent) otherwise.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne

from ..config import Params


def make_figure(obj, params: Params, folder: Path, stem: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{stem}_vis.png"
    try:
        if params.paradigm.erp and isinstance(obj, mne.BaseEpochs):
            fig = _erp_figure(obj, params)
        else:
            fig = _spectrum_figure(obj, params)
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        # DEVIATION: plotting may fail without a montage; emit a placeholder.
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "visualization unavailable", ha="center")
        fig.savefig(out, dpi=90)
        plt.close(fig)
    return out


def _erp_figure(epochs: mne.BaseEpochs, params: Params):
    evoked = epochs.average()
    fig = evoked.plot_joint(show=False)
    return fig if not isinstance(fig, list) else fig[0]


def _spectrum_figure(obj, params: Params):
    fmin, fmax = params.vis.freq_range_hz
    spec = obj.compute_psd(fmin=fmin, fmax=fmax, verbose="ERROR")
    fig = spec.plot(show=False)
    return fig
