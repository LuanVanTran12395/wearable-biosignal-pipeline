"""Step 10 — MuscIL: ICA-based muscle-artifact rejection (optional).

* Fit extended-Infomax ICA on a copy high-pass filtered at 1 Hz (ICA fits
  better on high-pass data; the *unfiltered* wavelet-cleaned data is what gets
  cleaned).
* Classify components with an ICLabel-equivalent classifier
  (``mne_icalabel.label_components``, method="iclabel").
* Flag any component with muscle probability >= 0.25.
* Transfer the decomposition onto the original (unfiltered) copy and remove
  only the flagged components.
* If every component is flagged, raise :class:`AllICsRejectedError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

import mne

from ..config import Params
from ..exceptions import AllICsRejectedError


@dataclass
class MuscILResult:
    raw: mne.io.BaseRaw
    n_components: int
    n_rejected: int

    @property
    def pct_rejected(self) -> float:
        return 100.0 * self.n_rejected / self.n_components if self.n_components else 0.0


def apply(raw: mne.io.BaseRaw, params: Params) -> MuscILResult:
    if not params.muscIL.enabled:
        return MuscILResult(raw=raw, n_components=0, n_rejected=0)

    ica = mne.preprocessing.ICA(
        method="infomax", fit_params=dict(extended=True),
        max_iter="auto", random_state=97, verbose="ERROR")

    fit_copy = raw.copy().filter(l_freq=params.muscIL.ica_highpass, h_freq=None,
                                 picks="eeg", verbose="ERROR")
    ica.fit(fit_copy, verbose="ERROR")

    muscle_idx = _classify_muscle(ica, fit_copy, params.muscIL.muscle_thresh)

    n_comp = ica.n_components_
    if len(muscle_idx) >= n_comp and n_comp > 0:
        raise AllICsRejectedError(
            f"All {n_comp} ICs flagged as muscle — file unprocessable")

    ica.exclude = muscle_idx
    ica.apply(raw, verbose="ERROR")  # applied to the unfiltered data
    return MuscILResult(raw=raw, n_components=n_comp, n_rejected=len(muscle_idx))


def _classify_muscle(ica, inst, thresh: float) -> List[int]:
    try:
        from mne_icalabel import label_components

        labels = label_components(inst, ica, method="iclabel")
        probs = labels["y_pred_proba"]
        classes = labels["labels"]
        idx = [i for i, (c, p) in enumerate(zip(classes, probs))
               if c == "muscle artifact" and p >= thresh]
        return idx
    except Exception:
        # DEVIATION: mne-icalabel unavailable -> spectral-slope heuristic.
        # Muscle ICs have rising high-frequency power; flag positive-slope comps.
        sources = ica.get_sources(inst).get_data()
        sfreq = inst.info["sfreq"]
        from scipy.signal import welch

        flagged = []
        for i, s in enumerate(sources):
            f, p = welch(s, fs=sfreq, nperseg=int(min(sfreq * 2, len(s))))
            band = (f >= 20) & (f <= min(90, sfreq / 2 - 1))
            if band.sum() < 4:
                continue
            slope = np.polyfit(np.log(f[band]), np.log(p[band] + 1e-20), 1)[0]
            # A flat/positive log-log slope in 20-90 Hz suggests muscle.
            if slope > -0.5:
                flagged.append(i)
        return flagged
