#!/usr/bin/env python3
"""
Multimodal QC for wearable EEG/PPG/fNIRS headband CSV recordings (v2.1).

Outputs
-------
- qc_summary.csv                 One row per signal with QC metrics and label
- qc_summary.json                Full machine-readable report
- session_label.csv              Session-level decision and reasons
- window_qc.csv                  Window-by-window metrics and artifact labels
- artifact_intervals.csv         Merged artifact intervals
- signal_overview.png            Representative clean/dirty signal overview
- psd_overview.png               PSD overview
- eeg_raw_block_XX.edf           EEG AF3/AF4 raw ADC-count EDF
- ppg_raw_block_XX.edf           PPG raw-count EDF
- fnirs_raw_block_XX.csv         fNIRS raw optical-intensity CSV
- signal_export_manifest.csv     Export details

Labels
------
PASS
CONDITIONAL_PASS
FAIL
NOT_ASSESSABLE

Important interpretation
------------------------
- EEG amplitude constants (ADC_MIDPOINT, UV_PER_COUNT) live in
  lib/eeg_units.py -- single source of truth, imported here.
- Window/artifact QC (EEG_AMPLITUDE_LIMIT_UV=150, filtered_rms_uv,
  filtered_peak_to_peak_uv) already runs on values converted to microvolts
  via eeg_units.scale_to_uv() -- scale-only, no ADC_MIDPOINT subtraction,
  because those inputs are already filtered/detrended (AC-coupled, DC
  already removed by the filter).
- eeg_raw*.edf export (raw, unfiltered) additionally subtracts ADC_MIDPOINT
  via eeg_units.adc_to_uv(), since the DC offset is still present there.
- fNIRS channels are treated as raw optical intensity. HbO/HbR readiness is
  reported as NOT_ASSESSABLE unless wavelength/calibration metadata are
  available from the study protocol.
- A session can be complete in duration but still fail biological signal QC.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, filtfilt, find_peaks, iirnotch, sosfiltfilt, welch
from scipy.stats import kurtosis

import eeg_units

try:
    from pyedflib import FILETYPE_EDF, highlevel
except ImportError:
    FILETYPE_EDF = None
    highlevel = None


EEG_FS = 244
OPTICAL_FS = 100
PACKET_SIZE = 244
OPTICAL_SAMPLES_PER_PACKET = 100

# ADC -> microvolt conversion constants live in eeg_units.py (single source
# of truth, shared with the raw eeg_raw*.edf/fif export path).
ADC_MIDPOINT = eeg_units.ADC_MIDPOINT
EEG_UV_PER_COUNT = eeg_units.UV_PER_COUNT
EEG_AMPLITUDE_LIMIT_UV = 150.0  # per-sample amplitude implausible for cortical EEG beyond this
EEG_AMPLITUDE_BAD_FRACTION = 0.15  # window is amplitude-bad once >15% of samples exceed the limit

# Fnir_1/Fnir_2 are two wavelengths co-located at a single optode on this
# device (confirmed against hardware photos -- only one optical module per
# side of the band), so a proper Scalp Coupling Index applies: both
# wavelengths at a well-coupled site should share the same cardiac
# pulsation. Threshold follows the SCI literature: ~0.75 is the conventional
# good/marginal cut, ~0.5 marginal/poor. Citation:
#   Pollonini L, Olds C, Abaya H, Bortfeld H, Beauchamp MS, Oghalai JS.
#   "Auditory cortex activation to natural speech and simulated cochlear
#   implant speech measured with functional near-infrared spectroscopy."
#   Hear Res. 2014;309:84-93.
#   Hernandez SM, Pollonini L. "NIRSplot: a tool for quality assessment of
#   fNIRS scans." Optica Biophotonics Congress: Biomedical Optics 2020
#   (Translational, Microscopy, OCT, OTS, BRAIN), paper BM2C.5.
FNIRS_CARDIAC_LOW_HZ = 0.8
FNIRS_CARDIAC_HIGH_HZ = 2.5
FNIRS_SCI_FAIL_THRESHOLD = 0.5
FNIRS_SCI_CONDITIONAL_THRESHOLD = 0.75

COLUMN_ALIASES = {
    "timestamp": ["timestamp", "ingestTimestamp"],
    "AF3": ["AF_3", "af3", "AF3"],
    "AF4": ["AF_4", "af4", "AF4"],
    "PPG": ["PPG", "ppg"],
    "FNIRS1": ["Fnir_1", "fnirs_1", "fNIRS_1", "FNIRS_1"],
    "FNIRS2": ["Fnir_2", "fnirs_2", "fNIRS_2", "FNIRS_2"],
}

LABEL_ORDER = {
    "PASS": 0,
    "CONDITIONAL_PASS": 1,
    "FAIL": 2,
    "NOT_ASSESSABLE": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Improved QC and labeling for multimodal meditation CSV files."
    )
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gap-ms", type=int, default=2500)
    parser.add_argument("--minimum-duration", type=float, default=840.0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--setup-trim-seconds", type=float, default=10.0)
    parser.add_argument("--overview-seconds", type=int, default=10)
    parser.add_argument("--skip-edf", action="store_true")
    parser.add_argument("--edf-only-longest", action="store_true")
    return parser.parse_args()


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for logical_name, aliases in COLUMN_ALIASES.items():
        match = next((name for name in aliases if name in df.columns), None)
        if match is None:
            raise ValueError(
                f"Missing column for {logical_name}. Accepted names: {aliases}"
            )
        resolved[logical_name] = match
    return resolved


def numeric_with_nan(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def fill_for_filtering(x: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(x, dtype=float))
    if series.isna().all():
        return np.zeros(len(series), dtype=float)
    return (
        series.interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(dtype=float)
    )


def safe_percent(condition: np.ndarray) -> float:
    condition = np.asarray(condition)
    return float(np.mean(condition) * 100.0) if condition.size else float("nan")


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    if not np.isfinite(mad) or mad == 0:
        scale = np.nanstd(values)
        if not np.isfinite(scale) or scale == 0:
            return np.zeros_like(values)
        return (values - med) / scale
    return 0.67448975 * (values - med) / mad


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.nanmean(x * x))) if x.size else float("nan")


def eeg_counts_to_uv(x: np.ndarray) -> np.ndarray:
    """Scale AC-coupled (already detrended/filtered) EEG samples from raw ADC
    counts to microvolts. No offset subtraction: the DC midpoint is already
    removed by filtering, so only the count->volt scale factor applies.
    Thin wrapper around eeg_units.scale_to_uv() (single source of truth)."""
    return eeg_units.scale_to_uv(x)


def longest_flatline_seconds(x: np.ndarray, fs: float, tolerance: float = 0.0) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    d = np.abs(np.diff(x))
    same = np.isfinite(d) & (d <= tolerance)
    longest = current = 0
    for value in same:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float((longest + 1) / fs) if longest else 0.0


def clipping_percent_heuristic(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan")
    mn, mx = np.min(finite), np.max(finite)
    return safe_percent((finite == mn) | (finite == mx))


def detect_blocks(packet_timestamps: np.ndarray, gap_ms: int) -> List[np.ndarray]:
    if len(packet_timestamps) == 0:
        return []
    split_points = np.where(np.diff(packet_timestamps) > gap_ms)[0] + 1
    return list(np.split(np.arange(len(packet_timestamps)), split_points))


def get_packet_timestamps(df: pd.DataFrame, timestamp_col: str) -> np.ndarray:
    ts = numeric_with_nan(df[timestamp_col])
    ts = fill_for_filtering(ts)
    return ts[::PACKET_SIZE]


def get_block_dataframe(df: pd.DataFrame, block: np.ndarray) -> pd.DataFrame:
    row_start = int(block[0] * PACKET_SIZE)
    row_end = int((block[-1] + 1) * PACKET_SIZE)
    return df.iloc[row_start:row_end].reset_index(drop=True)


def unpack_optical(values: np.ndarray) -> np.ndarray:
    n_packets = len(values) // PACKET_SIZE
    if n_packets == 0:
        return np.array([], dtype=float)
    trimmed = values[: n_packets * PACKET_SIZE].reshape(n_packets, PACKET_SIZE)
    return trimmed[:, :OPTICAL_SAMPLES_PER_PACKET].reshape(-1).astype(float)


def notch_filter(x: np.ndarray, fs: float, frequency: float = 50.0, q: float = 30.0) -> np.ndarray:
    if len(x) < 20 or frequency >= fs / 2:
        return np.asarray(x, dtype=float)
    b, a = iirnotch(frequency / (fs / 2), q)
    return filtfilt(b, a, np.asarray(x, dtype=float))


def bandpass_zero_phase(
    x: np.ndarray, fs: float, low: float, high: float, order: int = 4
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < max(20, order * 6):
        return x
    nyquist = fs / 2
    high = min(high, nyquist * 0.95)
    sos = butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def filter_eeg(x: np.ndarray) -> np.ndarray:
    x = detrend(fill_for_filtering(x), type="linear")
    x = notch_filter(x, EEG_FS, 50.0, 30.0)
    return bandpass_zero_phase(x, EEG_FS, 0.5, 45.0, 4)


def filter_ppg(x: np.ndarray) -> np.ndarray:
    x = detrend(fill_for_filtering(x), type="linear")
    return bandpass_zero_phase(x, OPTICAL_FS, 0.5, 8.0, 3)


def filter_fnirs(x: np.ndarray) -> np.ndarray:
    x = detrend(fill_for_filtering(x), type="linear")
    return bandpass_zero_phase(x, OPTICAL_FS, 0.01, 0.5, 3)


def fnirs_pulsatility_index(raw: np.ndarray, fs: float) -> float:
    """
    Fraction of spectral power sitting in the cardiac band vs. a broader
    band -- evidence that real physiological pulsation is reaching the
    detector. Usable per channel even without a paired wavelength.

    Formula: PI = bandpower(FNIRS_CARDIAC_LOW_HZ..FNIRS_CARDIAC_HIGH_HZ,
    default 0.8-2.5Hz) / bandpower(0.5-5.0Hz), both via Welch PSD (see
    band_power()). Not a single named metric in the literature under this
    exact form -- a lightweight heuristic in the same spirit as
    cardiac-pulsation-based fNIRS signal-quality indices, see e.g.:
      Sappia MS, Hakimi N, Colier WNJM, Horschig JM. "Signal quality index:
      an algorithm for quantitative assessment of functional near infrared
      spectroscopy signal quality." Biomed Opt Express. 2020;11(11):6732-6754.
    """
    x = fill_for_filtering(raw)
    cardiac = band_power(x, fs, FNIRS_CARDIAC_LOW_HZ, FNIRS_CARDIAC_HIGH_HZ)
    broad = band_power(x, fs, 0.5, 5.0)
    return float(cardiac / broad) if broad and np.isfinite(broad) and broad > 0 else float("nan")


def fnirs_scalp_coupling_index(fn1_raw: np.ndarray, fn2_raw: np.ndarray, fs: float) -> float:
    """
    Scalp Coupling Index (SCI): zero-lag Pearson correlation between the two
    co-located wavelength channels after bandpassing to the cardiac band
    (FNIRS_CARDIAC_LOW_HZ..FNIRS_CARDIAC_HIGH_HZ, default 0.8-2.5Hz). A
    well-coupled optode sees the same cardiac pulsation on both wavelengths
    (SCI near 1); a poorly-coupled one (hair, air gap, motion) does not.

    Citation:
      Pollonini L, Olds C, Abaya H, Bortfeld H, Beauchamp MS, Oghalai JS.
      "Auditory cortex activation to natural speech and simulated cochlear
      implant speech measured with functional near-infrared spectroscopy."
      Hear Res. 2014;309:84-93.
      Hernandez SM, Pollonini L. "NIRSplot: a tool for quality assessment of
      fNIRS scans." Optica Biophotonics Congress: Biomedical Optics 2020,
      paper BM2C.5.
    """
    n = min(len(fn1_raw), len(fn2_raw))
    if n < fs * 4:
        return float("nan")
    a = bandpass_zero_phase(
        fill_for_filtering(fn1_raw[:n]), fs, FNIRS_CARDIAC_LOW_HZ, FNIRS_CARDIAC_HIGH_HZ, order=3
    )
    b = bandpass_zero_phase(
        fill_for_filtering(fn2_raw[:n]), fs, FNIRS_CARDIAC_LOW_HZ, FNIRS_CARDIAC_HIGH_HZ, order=3
    )
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def band_power(x: np.ndarray, fs: float, low: float, high: float) -> float:
    if len(x) < fs * 2:
        return float("nan")
    f, p = welch(x, fs=fs, nperseg=min(int(fs * 4), len(x)))
    mask = (f >= low) & (f < high)
    return float(np.trapz(p[mask], f[mask])) if np.any(mask) else float("nan")


def line_noise_ratio(x: np.ndarray, fs: float) -> float:
    if fs <= 100:
        return float("nan")
    line = band_power(x, fs, 49.0, 51.0)
    nearby = band_power(x, fs, 45.0, 49.0) + band_power(x, fs, 51.0, 55.0)
    return float(line / nearby) if nearby and np.isfinite(nearby) else float("nan")


def dominant_frequencies(
    x: np.ndarray, fs: float, max_frequency: float, top_n: int = 5
) -> List[float]:
    if len(x) < fs * 2:
        return []
    f, p = welch(x, fs=fs, nperseg=min(int(fs * 4), len(x)))
    mask = (f >= 0.3) & (f <= max_frequency)
    f, p = f[mask], p[mask]
    peaks, _ = find_peaks(p)
    if len(peaks) == 0:
        return []
    strongest = peaks[np.argsort(p[peaks])[-top_n:]][::-1]
    return [round(float(f[i]), 2) for i in strongest]


def build_windows(x: np.ndarray, fs: float, seconds: float) -> List[Tuple[int, int, np.ndarray]]:
    n = max(1, int(round(fs * seconds)))
    windows = []
    for start in range(0, len(x) - n + 1, n):
        end = start + n
        windows.append((start, end, x[start:end]))
    return windows


def eeg_window_table(
    name: str, filtered: np.ndarray, seconds: float, setup_trim_seconds: float
) -> pd.DataFrame:
    rows = []
    for idx, (start, end, segment) in enumerate(build_windows(filtered, EEG_FS, seconds)):
        segment_uv = eeg_counts_to_uv(segment)
        rows.append(
            {
                "signal": name,
                "window_index": idx,
                "start_seconds": start / EEG_FS,
                "end_seconds": end / EEG_FS,
                "rms": rms(segment),
                "peak_to_peak": float(np.ptp(segment)),
                "kurtosis": float(kurtosis(segment, fisher=False, bias=False)),
                "max_abs": float(np.max(np.abs(segment))),
                "peak_to_peak_uv": float(np.ptp(segment_uv)),
                "high_amplitude_fraction": safe_percent(np.abs(segment_uv) > EEG_AMPLITUDE_LIMIT_UV) / 100.0,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["z_rms"] = robust_z(out["rms"].to_numpy())
    out["z_p2p"] = robust_z(out["peak_to_peak"].to_numpy())
    out["z_kurtosis"] = robust_z(out["kurtosis"].to_numpy())

    med_rms = max(float(np.nanmedian(out["rms"])), 1e-12)
    med_p2p = max(float(np.nanmedian(out["peak_to_peak"])), 1e-12)
    out["rms_ratio_to_median"] = out["rms"] / med_rms
    out["p2p_ratio_to_median"] = out["peak_to_peak"] / med_p2p

    # Absolute physiological amplitude check (dry-electrode motion/EMG artifact),
    # independent of this session's own median so it cannot be fooled by a
    # uniformly noisy recording the way the relative z-score checks can.
    out["high_amplitude_artifact"] = out["high_amplitude_fraction"] > EEG_AMPLITUDE_BAD_FRACTION

    out["artifact"] = (
        (out["z_rms"] > 6)
        | (out["z_p2p"] > 6)
        | (out["z_kurtosis"] > 8)
        | (out["rms_ratio_to_median"] > 12)
        | (out["p2p_ratio_to_median"] > 20)
        | out["high_amplitude_artifact"]
        | (out["start_seconds"] < setup_trim_seconds)
    )
    out["artifact_reason"] = ""
    reason_map = [
        ("setup", out["start_seconds"] < setup_trim_seconds),
        ("high_rms", (out["z_rms"] > 6) | (out["rms_ratio_to_median"] > 12)),
        ("high_peak_to_peak", (out["z_p2p"] > 6) | (out["p2p_ratio_to_median"] > 20)),
        ("high_kurtosis", out["z_kurtosis"] > 8),
        ("high_amplitude_uv", out["high_amplitude_artifact"]),
    ]
    for reason, mask in reason_map:
        out.loc[mask, "artifact_reason"] = out.loc[mask, "artifact_reason"].apply(
            lambda old: reason if old == "" else f"{old};{reason}"
        )
    return out


def ppg_peak_metrics(filtered: np.ndarray) -> Dict[str, float]:
    # Peak count remains valid for short QC windows. HRV metrics require longer
    # segments, but returning zero peaks for all windows shorter than 10 s
    # falsely labels clean PPG as 100% artifact.
    if len(filtered) < int(OPTICAL_FS * 1.5):
        return {
            "n_peaks": 0,
            "valid_ibi_percent": float("nan"),
            "median_hr_bpm": float("nan"),
            "ibi_cv_percent": float("nan"),
            "rmssd_ms": float("nan"),
        }

    prominence = max(np.nanmedian(np.abs(filtered - np.nanmedian(filtered))) * 0.8, 1e-9)
    peaks, _ = find_peaks(
        filtered,
        distance=int(0.35 * OPTICAL_FS),
        prominence=prominence,
    )
    if len(peaks) < 3:
        return {
            "n_peaks": int(len(peaks)),
            "valid_ibi_percent": 0.0,
            "median_hr_bpm": float("nan"),
            "ibi_cv_percent": float("nan"),
            "rmssd_ms": float("nan"),
        }

    ibi = np.diff(peaks) / OPTICAL_FS
    valid = (ibi >= 0.4) & (ibi <= 1.5)
    valid_ibi = ibi[valid]
    if valid_ibi.size == 0:
        return {
            "n_peaks": int(len(peaks)),
            "valid_ibi_percent": 0.0,
            "median_hr_bpm": float("nan"),
            "ibi_cv_percent": float("nan"),
            "rmssd_ms": float("nan"),
        }

    hr = 60.0 / valid_ibi
    rmssd = np.sqrt(np.mean(np.diff(valid_ibi) ** 2)) * 1000 if len(valid_ibi) > 1 else np.nan
    return {
        "n_peaks": int(len(peaks)),
        "valid_ibi_percent": float(np.mean(valid) * 100.0),
        "median_hr_bpm": float(np.median(hr)),
        "ibi_cv_percent": float(np.std(valid_ibi) / np.mean(valid_ibi) * 100.0),
        "rmssd_ms": float(rmssd),
    }


def ppg_window_table(
    filtered: np.ndarray, seconds: float, setup_trim_seconds: float
) -> pd.DataFrame:
    rows = []
    for idx, (start, end, segment) in enumerate(build_windows(filtered, OPTICAL_FS, seconds)):
        metrics = ppg_peak_metrics(segment)
        rows.append(
            {
                "signal": "PPG",
                "window_index": idx,
                "start_seconds": start / OPTICAL_FS,
                "end_seconds": end / OPTICAL_FS,
                "rms": rms(segment),
                "peak_to_peak": float(np.ptp(segment)),
                "kurtosis": float(kurtosis(segment, fisher=False, bias=False)),
                "n_peaks": metrics["n_peaks"],
                "valid_ibi_percent": metrics["valid_ibi_percent"],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["z_rms"] = robust_z(out["rms"].to_numpy())
    out["z_p2p"] = robust_z(out["peak_to_peak"].to_numpy())
    med_rms = max(float(np.nanmedian(out["rms"])), 1e-12)
    med_p2p = max(float(np.nanmedian(out["peak_to_peak"])), 1e-12)
    out["rms_ratio_to_median"] = out["rms"] / med_rms
    out["p2p_ratio_to_median"] = out["peak_to_peak"] / med_p2p

    # For a window of W seconds, plausible pulse count for 35-200 bpm is
    # approximately W*35/60 to W*200/60, with one-count tolerance because a
    # beat may fall just outside a window boundary.
    window_duration = out["end_seconds"] - out["start_seconds"]
    min_peaks = np.maximum(1, np.floor(window_duration * 35.0 / 60.0 - 1)).astype(int)
    max_peaks = np.ceil(window_duration * 200.0 / 60.0 + 1).astype(int)
    peak_bad = (out["n_peaks"] < min_peaks) | (out["n_peaks"] > max_peaks)
    out["artifact"] = (
        (out["z_rms"] > 6)
        | (out["z_p2p"] > 6)
        | (out["rms_ratio_to_median"] > 12)
        | (out["p2p_ratio_to_median"] > 20)
        | peak_bad
        | (out["start_seconds"] < setup_trim_seconds)
    )
    out["artifact_reason"] = ""
    for reason, mask in [
        ("setup", out["start_seconds"] < setup_trim_seconds),
        ("high_rms", (out["z_rms"] > 6) | (out["rms_ratio_to_median"] > 12)),
        ("high_peak_to_peak", (out["z_p2p"] > 6) | (out["p2p_ratio_to_median"] > 20)),
        ("implausible_peak_count", peak_bad),
    ]:
        out.loc[mask, "artifact_reason"] = out.loc[mask, "artifact_reason"].apply(
            lambda old: reason if old == "" else f"{old};{reason}"
        )
    return out


def fnirs_window_table(
    name: str, raw: np.ndarray, filtered: np.ndarray, seconds: float, setup_trim_seconds: float
) -> pd.DataFrame:
    raw_windows = build_windows(raw, OPTICAL_FS, seconds)
    filt_windows = build_windows(filtered, OPTICAL_FS, seconds)
    rows = []
    for idx, ((start, end, raw_seg), (_, _, filt_seg)) in enumerate(zip(raw_windows, filt_windows)):
        finite = raw_seg[np.isfinite(raw_seg)]
        mean_raw = float(np.mean(finite)) if finite.size else np.nan
        rows.append(
            {
                "signal": name,
                "window_index": idx,
                "start_seconds": start / OPTICAL_FS,
                "end_seconds": end / OPTICAL_FS,
                "raw_mean": mean_raw,
                "raw_cv_percent": float(np.std(finite) / abs(mean_raw) * 100.0)
                if finite.size and mean_raw != 0
                else np.nan,
                "rms": rms(filt_seg),
                "peak_to_peak": float(np.ptp(filt_seg)),
                "max_first_difference": float(np.max(np.abs(np.diff(fill_for_filtering(raw_seg)))))
                if len(raw_seg) > 1
                else 0.0,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["z_p2p"] = robust_z(out["peak_to_peak"].to_numpy())
    out["z_diff"] = robust_z(out["max_first_difference"].to_numpy())
    out["artifact"] = (
        (out["z_p2p"] > 6)
        | (out["z_diff"] > 6)
        | (out["raw_cv_percent"] > 10)
        | (out["start_seconds"] < setup_trim_seconds)
    )
    out["artifact_reason"] = ""
    for reason, mask in [
        ("setup", out["start_seconds"] < setup_trim_seconds),
        ("high_peak_to_peak", out["z_p2p"] > 6),
        ("motion_step", out["z_diff"] > 6),
        ("high_local_cv", out["raw_cv_percent"] > 10),
    ]:
        out.loc[mask, "artifact_reason"] = out.loc[mask, "artifact_reason"].apply(
            lambda old: reason if old == "" else f"{old};{reason}"
        )
    return out


def artifact_percent(window_df: pd.DataFrame) -> float:
    if window_df.empty:
        return 100.0
    return float(window_df["artifact"].mean() * 100.0)


def dynamic_ratio(
    window_df: pd.DataFrame,
    metric: str = "rms",
    percentile: float = 100.0,
    floor: float = 1e-9,
) -> float:
    """Ratio of a high percentile to the median of `metric` across windows.
    `percentile=100` (default) reproduces the original max/median ratio for
    callers that haven't opted into the more robust behavior. `floor` bounds
    the denominator so a session with a near-zero median doesn't blow the
    ratio up to an arbitrarily large (or numerically infinite) value."""
    if window_df.empty or metric not in window_df:
        return float("nan")
    values = window_df[metric].to_numpy(dtype=float)
    med = float(np.nanmedian(values))
    peak = float(np.nanpercentile(values, percentile))
    denom = max(med, floor)
    return float(peak / denom) if denom > 0 else float("inf")


def eeg_interchannel_correlation(af3_filtered: np.ndarray, af4_filtered: np.ndarray) -> float:
    """Pearson correlation between the two filtered frontal channels over the
    longest continuous block. Bilateral frontal EEG is expected to correlate
    moderately via volume conduction; a session-wide collapse toward zero is
    the signature of a poorly-contacted dry electrode on one channel, and is
    used to distinguish that from genuine bilateral low-frequency (meditative)
    power increases."""
    n = min(len(af3_filtered), len(af4_filtered))
    if n < 2:
        return float("nan")
    a = np.asarray(af3_filtered[:n], dtype=float)
    b = np.asarray(af4_filtered[:n], dtype=float)
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def eeg_summary(
    name: str,
    raw: np.ndarray,
    filtered: np.ndarray,
    windows: pd.DataFrame,
    interchannel_corr: float = float("nan"),
) -> Dict[str, object]:
    missing = safe_percent(np.isnan(raw))
    clipping = clipping_percent_heuristic(raw)
    flatline = longest_flatline_seconds(raw, EEG_FS)
    artifact = artifact_percent(windows)
    # p95/median instead of max/median, floored at a 2uV-equivalent RMS: a
    # single spiky window (or a session with a near-silent median) can no
    # longer send this ratio to an arbitrarily large value on its own.
    dyn = dynamic_ratio(
        windows, "rms", percentile=95.0, floor=2.0 / EEG_UV_PER_COUNT
    )
    amplitude_bad_percent = (
        float(windows["high_amplitude_artifact"].mean() * 100.0)
        if not windows.empty and "high_amplitude_artifact" in windows
        else 0.0
    )
    total_1_45 = band_power(filtered, EEG_FS, 1.0, 45.0)
    low = band_power(filtered, EEG_FS, 1.0, 4.0)
    alpha = band_power(filtered, EEG_FS, 8.0, 13.0)
    low_ratio = float(low / total_1_45) if total_1_45 and np.isfinite(total_1_45) else np.nan
    alpha_ratio = float(alpha / total_1_45) if total_1_45 and np.isfinite(total_1_45) else np.nan

    reasons: List[str] = []
    label = "PASS"

    if missing > 2 or clipping > 1 or flatline > 1:
        label = "FAIL"
        reasons.append("structural_signal_failure")
    if artifact > 30 or amplitude_bad_percent > 30:
        label = "FAIL"
        reasons.append("severe_recurrent_artifact")
    elif label != "FAIL" and (artifact > 10 or amplitude_bad_percent > 10 or dyn > 20):
        label = "CONDITIONAL_PASS"
        reasons.append("moderate_artifact_burden")
    # Deep meditation naturally raises low-frequency (theta/delta) power, so
    # only treat low-frequency dominance as suspect when it also coincides
    # with low inter-channel correlation -- the actual signature of a
    # poorly-contacted dry electrode rather than a genuine meditative state.
    uncorrelated = np.isfinite(interchannel_corr) and interchannel_corr < 0.1
    if label != "FAIL" and np.isfinite(low_ratio) and low_ratio > 0.80 and uncorrelated:
        label = "CONDITIONAL_PASS"
        reasons.append("low_frequency_dominance_uncorrelated")
    if not reasons:
        reasons.append("acceptable_qc_metrics")

    return {
        "signal": name,
        "modality": "EEG",
        "label": label,
        "reasons": ";".join(reasons),
        "n_samples": int(len(raw)),
        "sampling_rate_hz": EEG_FS,
        "duration_seconds": round(len(raw) / EEG_FS, 3),
        "missing_percent": round(missing, 4),
        "zero_percent": round(safe_percent(raw == 0), 4),
        "clipping_percent_heuristic": round(clipping, 4),
        "longest_flatline_seconds": round(flatline, 4),
        "raw_min": float(np.nanmin(raw)),
        "raw_max": float(np.nanmax(raw)),
        "filtered_rms": rms(filtered),
        "filtered_rms_uv": round(rms(filtered) * EEG_UV_PER_COUNT, 4),
        "filtered_peak_to_peak": float(np.ptp(filtered)),
        "filtered_peak_to_peak_uv": round(float(np.ptp(filtered)) * EEG_UV_PER_COUNT, 4),
        "artifact_window_percent": round(artifact, 2),
        "high_amplitude_window_percent": round(amplitude_bad_percent, 2),
        "max_to_median_rms_ratio": round(dyn, 2),
        "interchannel_correlation": (
            round(interchannel_corr, 4) if np.isfinite(interchannel_corr) else None
        ),
        "low_frequency_power_ratio_1_4_over_1_45": round(low_ratio, 4),
        "alpha_power_ratio_8_13_over_1_45": round(alpha_ratio, 4),
        "line_noise_ratio_50hz": round(line_noise_ratio(filtered, EEG_FS), 4),
        "dominant_frequencies_hz": dominant_frequencies(filtered, EEG_FS, 45.0),
        "unit_interpretation": "raw_adc_count",
        "uv_per_count": EEG_UV_PER_COUNT,
    }


def ppg_summary(raw: np.ndarray, filtered: np.ndarray, windows: pd.DataFrame) -> Dict[str, object]:
    missing = safe_percent(np.isnan(raw))
    clipping = clipping_percent_heuristic(raw)
    flatline = longest_flatline_seconds(raw, OPTICAL_FS)
    artifact = artifact_percent(windows)
    # p95/median instead of max/median: a single spiky 2s window (motion,
    # cuff/strap shift) can no longer send this ratio past the FAIL
    # threshold on its own when the far stronger valid_ibi_percent /
    # good_window_percent peak-detection metrics say the pulse is fine.
    dyn = dynamic_ratio(windows, "rms", percentile=95.0, floor=1.0)
    peaks = ppg_peak_metrics(filtered)
    good_windows = 100.0 - artifact

    reasons: List[str] = []
    label = "PASS"

    if missing > 2 or clipping > 1 or flatline > 1:
        label = "FAIL"
        reasons.append("structural_signal_failure")
    if peaks["valid_ibi_percent"] < 80 or good_windows < 60 or dyn > 100:
        label = "FAIL"
        reasons.append("poor_pulse_detectability")
    elif label != "FAIL" and (
        peaks["valid_ibi_percent"] < 90 or good_windows < 80 or dyn > 20
    ):
        label = "CONDITIONAL_PASS"
        reasons.append("moderate_ppg_artifact")
    if np.isfinite(peaks["median_hr_bpm"]) and not (40 <= peaks["median_hr_bpm"] <= 180):
        label = "FAIL"
        reasons.append("implausible_median_heart_rate")
    if not reasons:
        reasons.append("stable_pulse_waveform")

    return {
        "signal": "PPG",
        "modality": "PPG",
        "label": label,
        "reasons": ";".join(reasons),
        "n_samples": int(len(raw)),
        "sampling_rate_hz": OPTICAL_FS,
        "duration_seconds": round(len(raw) / OPTICAL_FS, 3),
        "missing_percent": round(missing, 4),
        "zero_percent": round(safe_percent(raw == 0), 4),
        "clipping_percent_heuristic": round(clipping, 4),
        "longest_flatline_seconds": round(flatline, 4),
        "raw_min": float(np.nanmin(raw)),
        "raw_max": float(np.nanmax(raw)),
        "filtered_rms": rms(filtered),
        "filtered_peak_to_peak": float(np.ptp(filtered)),
        "artifact_window_percent": round(artifact, 2),
        "good_window_percent": round(good_windows, 2),
        "max_to_median_rms_ratio": round(dyn, 2),
        "n_detected_peaks": peaks["n_peaks"],
        "valid_ibi_percent": round(peaks["valid_ibi_percent"], 2),
        "median_hr_bpm": round(peaks["median_hr_bpm"], 2)
        if np.isfinite(peaks["median_hr_bpm"])
        else None,
        "ibi_cv_percent": round(peaks["ibi_cv_percent"], 2)
        if np.isfinite(peaks["ibi_cv_percent"])
        else None,
        "rmssd_ms_unedited": round(peaks["rmssd_ms"], 2)
        if np.isfinite(peaks["rmssd_ms"])
        else None,
        "dominant_frequencies_hz": dominant_frequencies(filtered, OPTICAL_FS, 8.0),
        "unit_interpretation": "raw_count",
    }


def fnirs_summary(
    name: str,
    raw: np.ndarray,
    filtered: np.ndarray,
    windows: pd.DataFrame,
    sci: float = float("nan"),
) -> Dict[str, object]:
    finite = raw[np.isfinite(raw)]
    missing = safe_percent(np.isnan(raw))
    clipping = clipping_percent_heuristic(raw)
    flatline = longest_flatline_seconds(raw, OPTICAL_FS)
    artifact = artifact_percent(windows)
    mean_raw = float(np.mean(finite)) if finite.size else np.nan
    cv = float(np.std(finite) / abs(mean_raw) * 100.0) if finite.size and mean_raw != 0 else np.nan
    pulsatility = fnirs_pulsatility_index(raw, OPTICAL_FS)

    reasons: List[str] = []
    label = "PASS"

    if missing > 2 or clipping > 1 or flatline > 1:
        label = "FAIL"
        reasons.append("structural_signal_failure")

    poor_coupling = np.isfinite(sci) and sci < FNIRS_SCI_FAIL_THRESHOLD
    marginal_coupling = (
        np.isfinite(sci) and FNIRS_SCI_FAIL_THRESHOLD <= sci < FNIRS_SCI_CONDITIONAL_THRESHOLD
    )
    severe_cv = np.isfinite(cv) and cv > 20
    moderate_cv = np.isfinite(cv) and cv > 5

    if label != "FAIL" and (artifact > 30 or severe_cv or poor_coupling):
        label = "FAIL"
        reasons.append("poor_scalp_coupling" if poor_coupling else "severe_optical_instability")
    elif label != "FAIL" and (artifact > 10 or moderate_cv or marginal_coupling):
        label = "CONDITIONAL_PASS"
        reasons.append("marginal_scalp_coupling" if marginal_coupling else "moderate_optical_instability")
    if not reasons:
        reasons.append("stable_raw_optical_intensity")

    return {
        "signal": name,
        "modality": "fNIRS_raw",
        "label": label,
        "reasons": ";".join(reasons),
        "n_samples": int(len(raw)),
        "sampling_rate_hz": OPTICAL_FS,
        "duration_seconds": round(len(raw) / OPTICAL_FS, 3),
        "missing_percent": round(missing, 4),
        "zero_percent": round(safe_percent(raw == 0), 4),
        "clipping_percent_heuristic": round(clipping, 4),
        "longest_flatline_seconds": round(flatline, 4),
        "raw_min": float(np.nanmin(raw)),
        "raw_max": float(np.nanmax(raw)),
        "raw_mean": mean_raw,
        "raw_cv_percent": round(cv, 4),
        "filtered_rms": rms(filtered),
        "filtered_peak_to_peak": float(np.ptp(filtered)),
        "artifact_window_percent": round(artifact, 2),
        "scalp_coupling_index": round(sci, 4) if np.isfinite(sci) else None,
        "pulsatility_index_cardiac_band": round(pulsatility, 4) if np.isfinite(pulsatility) else None,
        "unit_interpretation": "raw_optical_count",
        "hbo_hbr_readiness": "NOT_ASSESSABLE",
    }


def merge_intervals(window_df: pd.DataFrame) -> pd.DataFrame:
    if window_df.empty:
        return pd.DataFrame(
            columns=["signal", "start_seconds", "end_seconds", "duration_seconds", "reason"]
        )
    intervals = []
    for signal, group in window_df[window_df["artifact"]].groupby("signal"):
        group = group.sort_values("start_seconds")
        if group.empty:
            continue
        start = float(group.iloc[0]["start_seconds"])
        end = float(group.iloc[0]["end_seconds"])
        reasons = set(str(group.iloc[0]["artifact_reason"]).split(";"))
        for _, row in group.iloc[1:].iterrows():
            row_start = float(row["start_seconds"])
            row_end = float(row["end_seconds"])
            if row_start <= end + 1e-9:
                end = max(end, row_end)
                reasons.update(str(row["artifact_reason"]).split(";"))
            else:
                intervals.append(
                    {
                        "signal": signal,
                        "start_seconds": start,
                        "end_seconds": end,
                        "duration_seconds": end - start,
                        "reason": ";".join(sorted(r for r in reasons if r)),
                    }
                )
                start, end = row_start, row_end
                reasons = set(str(row["artifact_reason"]).split(";"))
        intervals.append(
            {
                "signal": signal,
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": end - start,
                "reason": ";".join(sorted(r for r in reasons if r)),
            }
        )
    return pd.DataFrame(intervals)


def add_global_artifact_windows(window_df: pd.DataFrame) -> pd.DataFrame:
    if window_df.empty:
        return window_df
    key_cols = ["start_seconds", "end_seconds"]
    counts = (
        window_df[window_df["artifact"]]
        .groupby(key_cols)["signal"]
        .nunique()
        .rename("n_signals_artifact")
        .reset_index()
    )
    out = window_df.merge(counts, on=key_cols, how="left")
    out["n_signals_artifact"] = out["n_signals_artifact"].fillna(0).astype(int)
    out["global_multimodal_artifact"] = out["n_signals_artifact"] >= 3
    return out


def packet_qc(packet_timestamps: np.ndarray) -> Dict[str, object]:
    intervals = np.diff(packet_timestamps)
    if len(intervals) == 0:
        return {
            "estimated_packets": int(len(packet_timestamps)),
            "median_packet_interval_ms": None,
            "min_packet_interval_ms": None,
            "max_packet_interval_ms": None,
            "duplicate_timestamp_count": 0,
            "intervals_over_1500_ms": 0,
        }
    return {
        "estimated_packets": int(len(packet_timestamps)),
        "median_packet_interval_ms": float(np.median(intervals)),
        "min_packet_interval_ms": float(np.min(intervals)),
        "max_packet_interval_ms": float(np.max(intervals)),
        "duplicate_timestamp_count": int(pd.Series(packet_timestamps).duplicated().sum()),
        "intervals_over_1500_ms": int(np.sum(intervals > 1500)),
    }


def choose_overview_start(window_qc: pd.DataFrame, duration: float, span: float) -> float:
    if window_qc.empty:
        return max(0.0, duration / 2 - span / 2)
    grouped = (
        window_qc.groupby(["start_seconds", "end_seconds"])["artifact"]
        .mean()
        .reset_index()
        .sort_values(["artifact", "start_seconds"])
    )
    for _, row in grouped.iterrows():
        start = float(row["start_seconds"])
        if start + span <= duration:
            return start
    return max(0.0, duration / 2 - span / 2)


def save_overview_plot(
    af3: np.ndarray,
    af4: np.ndarray,
    ppg: np.ndarray,
    fnirs1: np.ndarray,
    fnirs2: np.ndarray,
    output_path: Path,
    window_qc: pd.DataFrame,
    seconds: int,
) -> None:
    duration = min(len(af3) / EEG_FS, len(ppg) / OPTICAL_FS)
    start_seconds = choose_overview_start(window_qc, duration, seconds)
    eeg_start = int(start_seconds * EEG_FS)
    eeg_n = min(int(seconds * EEG_FS), len(af3) - eeg_start)
    opt_start = int(start_seconds * OPTICAL_FS)
    opt_n = min(int(seconds * OPTICAL_FS), len(ppg) - opt_start)

    eeg_t = np.arange(eeg_n) / EEG_FS + start_seconds
    opt_t = np.arange(opt_n) / OPTICAL_FS + start_seconds

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    axes[0].plot(eeg_t, af3[eeg_start:eeg_start + eeg_n], lw=0.8, label="AF3")
    axes[0].plot(eeg_t, af4[eeg_start:eeg_start + eeg_n], lw=0.8, label="AF4")
    axes[0].set_title("EEG filtered")
    axes[0].set_ylabel("ADC count")
    axes[0].legend()

    axes[1].plot(opt_t, ppg[opt_start:opt_start + opt_n], lw=0.8)
    axes[1].set_title("PPG filtered")
    axes[1].set_ylabel("Raw count")

    axes[2].plot(opt_t, fnirs1[opt_start:opt_start + opt_n], lw=0.8)
    axes[2].set_title("fNIRS 1 filtered")
    axes[2].set_ylabel("Raw count")

    axes[3].plot(opt_t, fnirs2[opt_start:opt_start + opt_n], lw=0.8)
    axes[3].set_title("fNIRS 2 filtered")
    axes[3].set_ylabel("Raw count")
    axes[3].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(alpha=0.2)

    fig.suptitle(f"Representative segment: {start_seconds:.1f}-{start_seconds + seconds:.1f} s")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_psd_plot(af3: np.ndarray, af4: np.ndarray, ppg: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for signal, label in [(af3, "AF3"), (af4, "AF4")]:
        f, p = welch(signal, fs=EEG_FS, nperseg=min(EEG_FS * 4, len(signal)))
        mask = (f >= 0.5) & (f <= 45)
        axes[0].semilogy(f[mask], p[mask], lw=0.9, label=label)
    axes[0].set_title("EEG PSD")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("PSD")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    f, p = welch(ppg, fs=OPTICAL_FS, nperseg=min(OPTICAL_FS * 8, len(ppg)))
    mask = (f >= 0.3) & (f <= 8)
    axes[1].semilogy(f[mask], p[mask], lw=0.9)
    axes[1].set_title("PPG PSD")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def timestamp_to_datetime(value: float) -> datetime:
    try:
        value = float(value)
        if value > 10_000_000_000:
            return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).replace(tzinfo=None)
        if value > 1_000_000_000:
            return datetime.fromtimestamp(value, tz=timezone.utc).replace(tzinfo=None)
    except Exception:
        pass
    return datetime.now(timezone.utc).replace(tzinfo=None)


def safe_physical_limits(signal: np.ndarray) -> Tuple[float, float]:
    finite = np.asarray(signal, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    mn, mx = float(np.min(finite)), float(np.max(finite))
    if mn == mx:
        margin = max(abs(mn) * 0.01, 1.0)
        return mn - margin, mx + margin
    margin = (mx - mn) * 0.01
    return mn - margin, mx + margin


def make_edf_signal_header(
    label: str, signal: np.ndarray, sampling_rate: int, dimension: str
) -> Dict[str, object]:
    mn, mx = safe_physical_limits(signal)
    return highlevel.make_signal_header(
        label=label,
        dimension=dimension,
        sample_frequency=sampling_rate,
        physical_min=mn,
        physical_max=mx,
        digital_min=-32768,
        digital_max=32767,
        transducer="Meditation wearable",
        prefiler="RAW; no software filtering",
    )



def _block_start_datetime(
    block_df: pd.DataFrame,
    columns: Dict[str, str],
) -> datetime:
    first_timestamp = fill_for_filtering(
        numeric_with_nan(block_df[columns["timestamp"]])
    )[0]
    return timestamp_to_datetime(first_timestamp)


def export_eeg_block_to_edf(
    block_df: pd.DataFrame,
    columns: Dict[str, str],
    output_path: Path,
) -> Dict[str, object]:
    """Export AF3 and AF4, converted to microvolts (see lib/eeg_units.py)."""
    if highlevel is None:
        raise RuntimeError("pyedflib is not installed. Run: pip install pyedflib")

    af3_raw = fill_for_filtering(numeric_with_nan(block_df[columns["AF3"]]))
    af4_raw = fill_for_filtering(numeric_with_nan(block_df[columns["AF4"]]))

    n_packets = len(block_df) // PACKET_SIZE
    expected_samples = n_packets * EEG_FS
    af3_raw = af3_raw[:expected_samples]
    af4_raw = af4_raw[:expected_samples]

    af3_uv = eeg_units.adc_to_uv(af3_raw)
    af4_uv = eeg_units.adc_to_uv(af4_raw)

    headers = [
        make_edf_signal_header("AF3_RAW", af3_uv, EEG_FS, "uV"),
        make_edf_signal_header("AF4_RAW", af4_uv, EEG_FS, "uV"),
    ]
    header = highlevel.make_header(
        recording_additional="EEG microvolts (converted from raw ADC count); 244 Hz; classic EDF",
        equipment="Meditation wearable",
        startdate=_block_start_datetime(block_df, columns),
    )

    success = highlevel.write_edf(
        str(output_path),
        [af3_uv.astype(np.float64), af4_uv.astype(np.float64)],
        headers,
        header=header,
        digital=False,
        file_type=FILETYPE_EDF,
    )
    if not success:
        raise RuntimeError(f"Failed to write EEG EDF: {output_path}")

    return {
        "file_type": "EEG",
        "file_path": str(output_path),
        "channels": "AF3_RAW,AF4_RAW",
        "sampling_rate_hz": EEG_FS,
        "unit": "uV",
        "duration_seconds": expected_samples / EEG_FS,
        "n_samples_per_channel": expected_samples,
        "precision_note": "Converted ADC->uV (see lib/eeg_units.py); keep source CSV for bit-exact raw ADC values.",
    }


def export_ppg_block_to_edf(
    block_df: pd.DataFrame,
    columns: Dict[str, str],
    output_path: Path,
) -> Dict[str, object]:
    """Export PPG only at its native effective sampling rate."""
    if highlevel is None:
        raise RuntimeError("pyedflib is not installed. Run: pip install pyedflib")

    ppg = unpack_optical(numeric_with_nan(block_df[columns["PPG"]]))
    ppg = fill_for_filtering(ppg)

    n_packets = len(block_df) // PACKET_SIZE
    expected_samples = n_packets * OPTICAL_FS
    ppg = ppg[:expected_samples]

    headers = [
        make_edf_signal_header("PPG_RAW", ppg, OPTICAL_FS, "raw_count"),
    ]
    header = highlevel.make_header(
        recording_additional="PPG raw counts; 100 Hz; classic EDF",
        equipment="Meditation wearable",
        startdate=_block_start_datetime(block_df, columns),
    )

    success = highlevel.write_edf(
        str(output_path),
        [ppg.astype(np.float64)],
        headers,
        header=header,
        digital=False,
        file_type=FILETYPE_EDF,
    )
    if not success:
        raise RuntimeError(f"Failed to write PPG EDF: {output_path}")

    return {
        "file_type": "PPG",
        "file_path": str(output_path),
        "channels": "PPG_RAW",
        "sampling_rate_hz": OPTICAL_FS,
        "unit": "raw_count",
        "duration_seconds": expected_samples / OPTICAL_FS,
        "n_samples": expected_samples,
    }


def export_fnirs_block_to_csv(
    block_df: pd.DataFrame,
    columns: Dict[str, str],
    output_path: Path,
) -> Dict[str, object]:
    """Export fNIRS raw optical intensities to a query-friendly CSV."""
    fnirs1 = unpack_optical(numeric_with_nan(block_df[columns["FNIRS1"]]))
    fnirs2 = unpack_optical(numeric_with_nan(block_df[columns["FNIRS2"]]))

    n_packets = len(block_df) // PACKET_SIZE
    expected_samples = n_packets * OPTICAL_FS
    fnirs1 = fnirs1[:expected_samples]
    fnirs2 = fnirs2[:expected_samples]

    packet_timestamps = fill_for_filtering(
        numeric_with_nan(block_df[columns["timestamp"]])
    )[::PACKET_SIZE][:n_packets]

    sample_index = np.arange(expected_samples, dtype=np.int64)
    repeated_packet_timestamps = np.repeat(
        packet_timestamps,
        OPTICAL_SAMPLES_PER_PACKET,
    )[:expected_samples]

    out = pd.DataFrame(
        {
            "sample_index": sample_index,
            "time_seconds": sample_index / OPTICAL_FS,
            "packet_timestamp": repeated_packet_timestamps,
            "fnirs_1_raw": fnirs1,
            "fnirs_2_raw": fnirs2,
        }
    )
    out.to_csv(output_path, index=False)

    return {
        "file_type": "fNIRS",
        "file_path": str(output_path),
        "channels": "fnirs_1_raw,fnirs_2_raw",
        "sampling_rate_hz": OPTICAL_FS,
        "unit": "raw_optical_count",
        "duration_seconds": expected_samples / OPTICAL_FS,
        "n_samples_per_channel": expected_samples,
    }

def session_decision(
    duration: float,
    minimum_duration: float,
    packet_report: Dict[str, object],
    summaries: Sequence[Dict[str, object]],
    window_qc: pd.DataFrame,
) -> Dict[str, object]:
    by_signal = {row["signal"]: row for row in summaries}
    reasons: List[str] = []

    recording_label = "PASS"
    if duration < minimum_duration:
        recording_label = "FAIL"
        reasons.append("longest_continuous_block_below_minimum")
    if packet_report["intervals_over_1500_ms"] > 0:
        recording_label = "CONDITIONAL_PASS" if recording_label != "FAIL" else "FAIL"
        reasons.append("packet_gaps_detected")
    if packet_report["duplicate_timestamp_count"] > 0:
        recording_label = "CONDITIONAL_PASS" if recording_label != "FAIL" else "FAIL"
        reasons.append("duplicate_packet_timestamps")

    eeg_labels = [by_signal["AF3"]["label"], by_signal["AF4"]["label"]]
    ppg_label = by_signal["PPG"]["label"]
    fnirs_labels = [by_signal["fNIRS_1"]["label"], by_signal["fNIRS_2"]["label"]]

    # With only AF3 and AF4, failure of either channel invalidates bilateral
    # EEG and frontal-asymmetry analyses.
    eeg_modality = (
        "FAIL"
        if "FAIL" in eeg_labels
        else "CONDITIONAL_PASS"
        if "CONDITIONAL_PASS" in eeg_labels
        else "PASS"
    )
    fnirs_modality = (
        "FAIL"
        if fnirs_labels.count("FAIL") == 2
        else "CONDITIONAL_PASS"
        if "FAIL" in fnirs_labels or "CONDITIONAL_PASS" in fnirs_labels
        else "PASS"
    )

    global_artifact_percent = (
        float(
            window_qc.drop_duplicates(["start_seconds", "end_seconds"])[
                "global_multimodal_artifact"
            ].mean()
            * 100.0
        )
        if not window_qc.empty
        else 100.0
    )

    if recording_label == "FAIL":
        overall = "FAIL"
    elif eeg_modality == "FAIL" and ppg_label == "FAIL":
        overall = "FAIL"
        reasons.append("eeg_and_ppg_failed")
    elif global_artifact_percent > 30:
        overall = "FAIL"
        reasons.append("severe_multimodal_artifact")
    elif (
        recording_label == "CONDITIONAL_PASS"
        or eeg_modality != "PASS"
        or ppg_label != "PASS"
        or fnirs_modality != "PASS"
        or global_artifact_percent > 10
    ):
        overall = "CONDITIONAL_PASS"
        reasons.append("usable_after_cleaning_or_partial_modality_limit")
    else:
        overall = "PASS"
        reasons.append("all_primary_qc_layers_passed")

    return {
        "recording_label": recording_label,
        "eeg_label": eeg_modality,
        "ppg_label": ppg_label,
        "fnirs_raw_label": fnirs_modality,
        "fnirs_hbo_hbr_readiness": "NOT_ASSESSABLE",
        "overall_session_label": overall,
        "global_multimodal_artifact_percent": round(global_artifact_percent, 2),
        "analysis_status": (
            "included"
            if overall == "PASS"
            else "included_after_cleaning"
            if overall == "CONDITIONAL_PASS"
            else "excluded_from_primary_analysis"
        ),
        "reasons": ";".join(dict.fromkeys(reasons)),
    }


def main() -> int:
    args = parse_args()

    if not args.csv_file.exists():
        print(f"File not found: {args.csv_file}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or args.csv_file.with_name(args.csv_file.stem + "_qc_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_file)
    try:
        columns = resolve_columns(df)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    packet_timestamps = get_packet_timestamps(df, columns["timestamp"])
    blocks = detect_blocks(packet_timestamps, args.gap_ms)
    if not blocks:
        print("No continuous recording block found.", file=sys.stderr)
        return 3

    block_lengths_seconds = [int(len(block)) for block in blocks]
    longest_index = int(np.argmax(block_lengths_seconds))
    longest_df = get_block_dataframe(df, blocks[longest_index])

    af3_raw = numeric_with_nan(longest_df[columns["AF3"]])
    af4_raw = numeric_with_nan(longest_df[columns["AF4"]])
    ppg_raw = unpack_optical(numeric_with_nan(longest_df[columns["PPG"]]))
    fnirs1_raw = unpack_optical(numeric_with_nan(longest_df[columns["FNIRS1"]]))
    fnirs2_raw = unpack_optical(numeric_with_nan(longest_df[columns["FNIRS2"]]))

    af3_f = filter_eeg(af3_raw)
    af4_f = filter_eeg(af4_raw)
    ppg_f = filter_ppg(ppg_raw)
    fnirs1_f = filter_fnirs(fnirs1_raw)
    fnirs2_f = filter_fnirs(fnirs2_raw)

    eeg_interchannel_corr = eeg_interchannel_correlation(af3_f, af4_f)
    fnirs_sci = fnirs_scalp_coupling_index(fnirs1_raw, fnirs2_raw, OPTICAL_FS)

    af3_w = eeg_window_table("AF3", af3_f, args.window_seconds, args.setup_trim_seconds)
    af4_w = eeg_window_table("AF4", af4_f, args.window_seconds, args.setup_trim_seconds)
    ppg_w = ppg_window_table(ppg_f, args.window_seconds, args.setup_trim_seconds)
    fnirs1_w = fnirs_window_table(
        "fNIRS_1", fnirs1_raw, fnirs1_f, args.window_seconds, args.setup_trim_seconds
    )
    fnirs2_w = fnirs_window_table(
        "fNIRS_2", fnirs2_raw, fnirs2_f, args.window_seconds, args.setup_trim_seconds
    )

    window_qc = pd.concat([af3_w, af4_w, ppg_w, fnirs1_w, fnirs2_w], ignore_index=True, sort=False)
    window_qc = add_global_artifact_windows(window_qc)
    window_qc.to_csv(output_dir / "window_qc.csv", index=False)

    artifact_intervals = merge_intervals(window_qc)
    artifact_intervals.to_csv(output_dir / "artifact_intervals.csv", index=False)

    summaries = [
        eeg_summary("AF3", af3_raw, af3_f, af3_w, interchannel_corr=eeg_interchannel_corr),
        eeg_summary("AF4", af4_raw, af4_f, af4_w, interchannel_corr=eeg_interchannel_corr),
        ppg_summary(ppg_raw, ppg_f, ppg_w),
        fnirs_summary("fNIRS_1", fnirs1_raw, fnirs1_f, fnirs1_w, sci=fnirs_sci),
        fnirs_summary("fNIRS_2", fnirs2_raw, fnirs2_f, fnirs2_w, sci=fnirs_sci),
    ]
    qc_df = pd.DataFrame(summaries)
    qc_df.to_csv(output_dir / "qc_summary.csv", index=False)

    packet_report = packet_qc(packet_timestamps)
    duration = float(block_lengths_seconds[longest_index])
    decision = session_decision(
        duration,
        args.minimum_duration,
        packet_report,
        summaries,
        window_qc,
    )
    pd.DataFrame([decision]).to_csv(output_dir / "session_label.csv", index=False)

    save_overview_plot(
        af3_f,
        af4_f,
        ppg_f,
        fnirs1_f,
        fnirs2_f,
        output_dir / "signal_overview.png",
        window_qc,
        args.overview_seconds,
    )
    save_psd_plot(af3_f, af4_f, ppg_f, output_dir / "psd_overview.png")

    signal_export_manifest: List[Dict[str, object]] = []
    signal_export_error = None
    if not args.skip_edf:
        selected = (
            [longest_index]
            if args.edf_only_longest
            else list(range(len(blocks)))
        )
        multiple_blocks = len(selected) > 1

        for index in selected:
            block_df = get_block_dataframe(df, blocks[index])
            suffix = f"_block_{index + 1:02d}" if multiple_blocks else ""

            eeg_path = output_dir / f"eeg_raw{suffix}.edf"
            ppg_path = output_dir / f"ppg_raw{suffix}.edf"
            fnirs_path = output_dir / f"fnirs_raw{suffix}.csv"

            exporters = [
                ("EEG", export_eeg_block_to_edf, eeg_path),
                ("PPG", export_ppg_block_to_edf, ppg_path),
                ("fNIRS", export_fnirs_block_to_csv, fnirs_path),
            ]

            for modality, exporter, file_path in exporters:
                try:
                    result = exporter(block_df, columns, file_path)
                    result["block_index"] = index
                    signal_export_manifest.append(result)
                except Exception as error:
                    signal_export_manifest.append(
                        {
                            "block_index": index,
                            "file_type": modality,
                            "file_path": str(file_path),
                            "error": str(error),
                        }
                    )

        pd.DataFrame(signal_export_manifest).to_csv(
            output_dir / "signal_export_manifest.csv",
            index=False,
        )

    fnirs_corr_raw = float(
        np.corrcoef(fill_for_filtering(fnirs1_raw), fill_for_filtering(fnirs2_raw))[0, 1]
    )
    eeg_rms_ratio = max(rms(af3_f), rms(af4_f)) / max(min(rms(af3_f), rms(af4_f)), 1e-12)

    # Channel-level imbalance can reveal one severely unstable EEG electrode.
    if eeg_rms_ratio > 10 and decision["overall_session_label"] != "FAIL":
        decision["overall_session_label"] = "CONDITIONAL_PASS"
        decision["analysis_status"] = "included_after_cleaning"
        decision["reasons"] += ";large_eeg_channel_imbalance"
        pd.DataFrame([decision]).to_csv(output_dir / "session_label.csv", index=False)

    summary = {
        "input_file": str(args.csv_file),
        "resolved_columns": columns,
        "continuous_blocks_seconds": block_lengths_seconds,
        "selected_longest_block_index": longest_index,
        "selected_longest_block_seconds": duration,
        "minimum_duration_seconds": args.minimum_duration,
        "packet_qc": packet_report,
        "session_decision": decision,
        "cross_channel_metrics": {
            "eeg_filtered_rms_ratio_max_over_min": round(float(eeg_rms_ratio), 4),
            "fnirs_raw_correlation": round(fnirs_corr_raw, 4),
        },
        "signals": summaries,
        "artifact_intervals": artifact_intervals.to_dict(orient="records"),
        "signal_exports": signal_export_manifest,
        "signal_export_error": signal_export_error,
        "notes": [
            "QC is calculated on the longest continuous block.",
            "A complete recording can still fail biological signal QC.",
            "EEG is exported separately as raw ADC counts, not microvolts.",
            "PPG is exported separately as raw counts in EDF.",
            "fNIRS is exported separately as raw optical-intensity CSV.",
            "fNIRS is labeled as raw optical intensity; HbO/HbR is not assessed.",
            "Initial setup windows are flagged automatically.",
            "Global multimodal artifact means at least three signals were flagged in the same window.",
            "Thresholds are heuristic and should be validated against manually reviewed project data.",
        ],
    }

    (output_dir / "qc_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== IMPROVED QC COMPLETED ===")
    print(qc_df[["signal", "label", "reasons", "artifact_window_percent"]].to_string(index=False))
    print("\nSession decision:")
    print(pd.DataFrame([decision]).to_string(index=False))
    print(f"\nOutput directory: {output_dir}")
    if signal_export_error:
        print(f"\nSignal export error: {signal_export_error}")
    elif not args.skip_edf:
        print("\nSignal export manifest:")
        print(pd.DataFrame(signal_export_manifest).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
