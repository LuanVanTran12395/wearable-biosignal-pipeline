"""Synthetic fixtures: simulated sine + line noise + eyeblink + white noise."""

from __future__ import annotations

import numpy as np
import pytest

import mne


def _standard_montage_names(n):
    m = mne.channels.make_standard_montage("standard_1020")
    return m.ch_names[:n]


@pytest.fixture
def srate():
    return 250.0


@pytest.fixture
def synthetic_raw(srate):
    """A 10-channel, 20 s Raw with alpha + 60 Hz line + eyeblink + noise."""
    rng = np.random.default_rng(0)
    n_ch = 10
    dur = 20.0
    n = int(srate * dur)
    t = np.arange(n) / srate
    names = _standard_montage_names(n_ch)

    data = np.zeros((n_ch, n))
    for ch in range(n_ch):
        alpha = np.sin(2 * np.pi * 10 * t) * 20e-6
        line = np.sin(2 * np.pi * 60 * t) * 8e-6
        noise = rng.standard_normal(n) * 5e-6
        data[ch] = alpha + line + noise

    # Eyeblink bursts on frontal channels.
    for onset in (2.0, 7.5, 13.0):
        i0 = int(onset * srate)
        blink = np.hanning(int(0.3 * srate)) * 80e-6
        data[0, i0:i0 + len(blink)] += blink
        data[1, i0:i0 + len(blink)] += blink

    info = mne.create_info(names, srate, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_montage("standard_1020", on_missing="ignore", verbose="ERROR")
    return raw


@pytest.fixture
def synthetic_raw_with_events(synthetic_raw, srate):
    """Add periodic 'stim' annotations for task/ERP tests."""
    onsets = np.arange(1.0, 18.0, 2.0)
    ann = mne.Annotations(onset=onsets, duration=np.zeros_like(onsets),
                          description=["stim"] * len(onsets))
    synthetic_raw.set_annotations(ann)
    return synthetic_raw


@pytest.fixture
def synthetic_epochs(synthetic_raw):
    return mne.make_fixed_length_epochs(synthetic_raw, duration=2.0,
                                        preload=True, verbose="ERROR")
