"""
fNIRS hemodynamic pipeline v2 -- xử lý CONTINUOUS session trước, cắt phase
sau. Thay thế cách làm cũ (lib/fnirs_metrics.py + notebook v1) vốn lọc/tính
chỉ số riêng lẻ trên từng phase đã cắt sẵn.

Nguyên tắc thiết kế (bắt buộc, xem yêu cầu gốc):
- Không filter riêng từng phase -- lọc trên toàn bộ tín hiệu liên tục của
  session (1 block dài nhất do Layer 2 xác định) rồi mới cắt phase.
- Không detrend riêng từng phase -- detrend trên toàn session.
- Không reset baseline tại mỗi phase -- baseline lấy 1 lần từ REST_BEFORE,
  áp dụng cho toàn bộ các phase khác trong CÙNG session.
- Không âm thầm loại bỏ dữ liệu -- mọi trường hợp loại/đánh dấu đều có
  QC flag + lý do + ngưỡng + session/participant/phase ghi trong output.

Nguồn dữ liệu (xem `load_session_data()`):
- Ưu tiên `fnirs_raw.csv` (raw intensity 740nm/850nm, 100Hz, do Layer 2
  xuất qua export_fnirs_block_to_csv() -- xem
  lib/signal_qc.py) -- pipeline đầy đủ
  "Trường hợp A": raw intensity -> QC -> OD -> TDDR -> MBLL -> HbO/HbR.
- Fallback "Trường hợp B" (chỉ có HbO/HbR có sẵn, vd fnirs_hbo_hbr.csv của
  Layer 2/v1) nếu không tìm thấy fnirs_raw.csv -- pipeline rút gọn, GHI RÕ
  input_signal_stage="concentration_fallback" trong output, KHÔNG coi tương
  đương pipeline đầy đủ.

Event alignment: latency trong event_labels_block.csv của Layer 2 tính theo
sample EEG (244Hz). fNIRS xuất riêng ở 100Hz nhưng
CÙNG block gốc (cùng block_df, cùng n_packets -- xem
export_eeg_block_to_edf()/export_fnirs_block_to_csv() trong
signal_qc.py), nên về lý thuyết cùng
mốc 0 tuyệt đối. `validate_and_align_events()` KIỂM CHỨNG giả định này bằng
cách so n_packets suy từ độ dài EEG vs độ dài fNIRS thay vì mặc định đúng.

CHƯA THỂ XÁC NHẬN (do thiếu metadata thiết bị): source-detector geometry
thực tế ngoài L=3cm mặc định, có short-channel để tách
tín hiệu ngoài da/hệ thống hay không, độ trễ đồng bộ phần cứng giữa 2 kênh
quang trong cùng optode (giả định = 0 vì cùng 1 module quang theo xác nhận
ảnh phần cứng trong signal_qc.py).
"""
from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import theilslopes

import fnirs_hbo_hbr as hbo_mod

PIPELINE_VERSION = "fnirs_hemo_v2.0"

# ════════════════════════════════════════════════════════════════════════
# QC FLAG CONSTANTS (xem docstring gốc -- không dùng string tự do rải rác)
# ════════════════════════════════════════════════════════════════════════

INPUT_QC_FLAGS = {
    "MISSING_FILE", "EMPTY_FILE", "INVALID_COLUMNS",
    "INVALID_SAMPLING_RATE", "DUPLICATED_TIMESTAMP", "NON_MONOTONIC_TIME",
}
SIGNAL_QC_FLAGS = {
    "HIGH_NAN_RATIO", "FLAT_SIGNAL", "SATURATION", "EXTREME_PEAK_TO_PEAK",
    "EXTREME_SD", "EXTREME_SLOPE", "BASELINE_SHIFT", "EXCESSIVE_MOTION",
    "HBT_INCONSISTENT",
}
EVENT_QC_FLAGS = {
    "MISSING_PHASE", "INCOMPLETE_PHASES", "EVENT_OUT_OF_RANGE",
    "EVENT_CLIPPED", "PHASE_OVERLAP", "INVALID_EVENT_ORDER",
    "DURATION_MISMATCH", "ALIGNMENT_UNCERTAIN",
}
BASELINE_QC_FLAGS = {"MISSING_BASELINE", "INVALID_BASELINE", "BASELINE_HIGH_VARIABILITY"}
STAT_QC_FLAGS = {"ROBUST_OUTLIER"}

ALL_QC_FLAGS = INPUT_QC_FLAGS | SIGNAL_QC_FLAGS | EVENT_QC_FLAGS | BASELINE_QC_FLAGS | STAT_QC_FLAGS

EXPECTED_PHASE_ORDER = ["REST_BEFORE", "MEDITATION", "REST_AFTER"]


# ════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ════════════════════════════════════════════════════════════════════════

def load_session_data(
    fnirs_raw_path: Optional[Path],
    hbo_hbr_fallback_path: Optional[Path],
    event_csv_path: Optional[Path],
    optical_fs_expected: float,
) -> Dict[str, Any]:
    """
    Đọc dữ liệu 1 session -- ưu tiên fnirs_raw.csv (raw intensity, "Trường
    hợp A"), fallback fnirs_hbo_hbr.csv nếu không có raw ("Trường hợp B").

    Trả về dict:
        input_signal_stage: "raw_intensity" | "concentration_fallback"
        fnirs_1_raw, fnirs_2_raw: np.ndarray | None (chỉ có khi raw_intensity)
        hbo_uM, hbr_uM, hbt_uM: np.ndarray | None (chỉ có khi concentration_fallback)
        time_seconds, packet_timestamp: np.ndarray | None
        n_samples: int
        sampling_rate_hz: float (đo THỰC TẾ từ time_seconds, không giả định)
        events: pd.DataFrame | None (đọc thẳng event_csv_path, chưa xử lý)
        input_qc_flags: List[str]
        source_files: dict đường dẫn đã dùng

    Không raise exception cho lỗi dữ liệu thông thường (file thiếu/rỗng) --
    trả về input_qc_flags để caller quyết định, raise chỉ cho lỗi hệ thống
    thực sự (permission, v.v.) qua exception tự nhiên của pandas/pathlib.
    """
    flags: List[str] = []
    result: Dict[str, Any] = {
        "input_signal_stage": None,
        "fnirs_1_raw": None, "fnirs_2_raw": None,
        "hbo_uM": None, "hbr_uM": None, "hbt_uM": None,
        "time_seconds": None, "packet_timestamp": None,
        "n_samples": 0, "sampling_rate_hz": np.nan,
        "events": None,
        "input_qc_flags": flags,
        "source_files": {
            "fnirs_raw_path": str(fnirs_raw_path) if fnirs_raw_path else None,
            "hbo_hbr_fallback_path": str(hbo_hbr_fallback_path) if hbo_hbr_fallback_path else None,
            "event_csv_path": str(event_csv_path) if event_csv_path else None,
        },
    }

    df = None
    if fnirs_raw_path is not None and Path(fnirs_raw_path).exists():
        df = pd.read_csv(fnirs_raw_path)
        if df.empty:
            flags.append("EMPTY_FILE")
            df = None
        elif not {"fnirs_1_raw", "fnirs_2_raw", "time_seconds"}.issubset(df.columns):
            flags.append("INVALID_COLUMNS")
            df = None
        else:
            result["input_signal_stage"] = "raw_intensity"
            result["fnirs_1_raw"] = df["fnirs_1_raw"].to_numpy(dtype=np.float64)
            result["fnirs_2_raw"] = df["fnirs_2_raw"].to_numpy(dtype=np.float64)
            if "packet_timestamp" in df.columns:
                result["packet_timestamp"] = df["packet_timestamp"].to_numpy()
    elif fnirs_raw_path is not None:
        flags.append("MISSING_FILE")

    if df is None and hbo_hbr_fallback_path is not None and Path(hbo_hbr_fallback_path).exists():
        df = pd.read_csv(hbo_hbr_fallback_path)
        if df.empty:
            flags.append("EMPTY_FILE")
            df = None
        elif not {"HbO_uM", "HbR_uM", "time_seconds"}.issubset(df.columns):
            flags.append("INVALID_COLUMNS")
            df = None
        else:
            result["input_signal_stage"] = "concentration_fallback"
            result["hbo_uM"] = df["HbO_uM"].to_numpy(dtype=np.float64)
            result["hbr_uM"] = df["HbR_uM"].to_numpy(dtype=np.float64)
            result["hbt_uM"] = df["HbT_uM"].to_numpy(dtype=np.float64) if "HbT_uM" in df.columns else None
    elif df is None and hbo_hbr_fallback_path is not None:
        flags.append("MISSING_FILE")

    if df is None:
        return result

    result["time_seconds"] = df["time_seconds"].to_numpy(dtype=np.float64)
    result["n_samples"] = len(df)

    # Sampling rate ĐO THỰC TẾ từ time_seconds (median dt), không giả định cứng.
    if len(df) >= 2:
        dt = np.diff(result["time_seconds"])
        if np.any(dt <= 0):
            flags.append("NON_MONOTONIC_TIME")
        if np.any(dt == 0):
            flags.append("DUPLICATED_TIMESTAMP")
        median_dt = float(np.median(dt[dt > 0])) if np.any(dt > 0) else np.nan
        result["sampling_rate_hz"] = 1.0 / median_dt if median_dt and median_dt > 0 else np.nan
        if not np.isfinite(result["sampling_rate_hz"]) or abs(result["sampling_rate_hz"] - optical_fs_expected) > 1.0:
            flags.append("INVALID_SAMPLING_RATE")
    else:
        flags.append("INVALID_SAMPLING_RATE")

    if event_csv_path is not None and Path(event_csv_path).exists():
        result["events"] = pd.read_csv(event_csv_path)

    return result


# ════════════════════════════════════════════════════════════════════════
# 2. VALIDATE SIGNAL (input QC + signal QC trên tín hiệu THÔ, trước xử lý)
# ════════════════════════════════════════════════════════════════════════

def validate_signal(
    values: np.ndarray,
    sampling_rate_hz: float,
    min_finite_fraction: float,
    label: str = "signal",
) -> Tuple[List[str], Dict[str, float]]:
    """
    QC trên 1 kênh tín hiệu thô (raw intensity HOẶC HbO/HbR fallback) --
    KHÔNG sửa dữ liệu, chỉ phát hiện + trả flag. Ngưỡng dùng thống kê ROBUST
    của chính đoạn tín hiệu (median/MAD) thay vì hằng số thiết bị cố định,
    vì KHÔNG có tài liệu xác nhận dải bão hoà (saturation ceiling) thực tế
    của cảm biến fNIRS trên thiết bị này (khác EEG có tài liệu ADC 24-bit
    rõ ràng) -- xem mục "CHƯA THỂ XÁC NHẬN" ở đầu file.

    Trả về (flags, stats) -- stats để ghi vào QC report, không chỉ pass/fail.
    """
    flags: List[str] = []
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    stats: Dict[str, float] = {f"{label}_n_samples": n}

    finite = np.isfinite(x)
    finite_fraction = float(finite.sum()) / n if n else 0.0
    stats[f"{label}_finite_fraction"] = finite_fraction
    if finite_fraction < min_finite_fraction:
        flags.append("HIGH_NAN_RATIO")

    xf = x[finite]
    if len(xf) < 2:
        flags.append("FLAT_SIGNAL")
        return flags, stats

    # Flatline: chuỗi giá trị giống hệt liên tiếp dài bất thường (>=2s liên tục).
    same_run = 0
    max_run = 0
    for i in range(1, len(xf)):
        if xf[i] == xf[i - 1]:
            same_run += 1
            max_run = max(max_run, same_run)
        else:
            same_run = 0
    max_flat_s = max_run / sampling_rate_hz if sampling_rate_hz else np.nan
    stats[f"{label}_max_flatline_s"] = max_flat_s
    if max_flat_s >= 2.0:
        flags.append("FLAT_SIGNAL")

    median = float(np.median(xf))
    mad = float(np.median(np.abs(xf - median))) * 1.4826
    stats[f"{label}_median"] = median
    stats[f"{label}_mad"] = mad

    # "Saturation" thao tác hoá bằng robust z cực đoan (>=15 MAD) -- không
    # phải ngưỡng ADC thiết bị (chưa có tài liệu xác nhận), chỉ là phát hiện
    # giá trị áp trần/đáy lặp lại bất thường trong CHÍNH đoạn tín hiệu này.
    if mad > 0:
        robust_z = 0.6745 * (xf - median) / mad
        extreme_frac = float(np.mean(np.abs(robust_z) > 15))
        stats[f"{label}_extreme_value_fraction"] = extreme_frac
        if extreme_frac > 0.01:
            flags.append("SATURATION")

    return flags, stats


# ════════════════════════════════════════════════════════════════════════
# 3. TDDR (Temporal Derivative Distribution Repair)
# ════════════════════════════════════════════════════════════════════════

def tddr(signal: np.ndarray, sample_rate: float, filter_cutoff_hz: float = 0.5, filter_order: int = 3) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Temporal Derivative Distribution Repair -- sửa artefact vận động (đặc
    biệt baseline shift/step) trên tín hiệu Optical Density, KHÔNG cần biết
    trước thời điểm artefact xảy ra.

    Thuật toán (theo đúng tài liệu tham khảo gốc):
    1. Tách tín hiệu thành thành phần tần số thấp (<filter_cutoff_hz, mặc
       định 0.5Hz) và phần dư tần số cao.
    2. Lấy đạo hàm bậc 1 của thành phần tần số thấp.
    3. Ước lượng lặp (iteratively reweighted) trọng số Tukey's biweight cho
       từng mẫu đạo hàm để hạ trọng số các bước nhảy bất thường (artefact),
       hội tụ khi mean đạo hàm đã trừ trọng số ~ 0.
    4. Tái tạo thành phần tần số thấp đã sửa bằng cumsum(đạo hàm*trọng số).
    5. Cộng lại phần tần số cao KHÔNG đổi (TDDR chỉ sửa artefact ở dải thấp,
       không đụng vào dao động sinh lý tần số cao hơn cutoff).

    Citation:
      Fishburn FA, Ludlum RS, Vaidya CJ, Medvedev AV. "Temporal Derivative
      Distribution Repair (TDDR): A motion correction method for fNIRS."
      Neuroimage. 2019;184:171-179.

    Trả về (corrected_signal, sample_weights, n_iterations). sample_weights
    (độ dài n-1, gắn với từng cặp mẫu liên tiếp) gần 0 = artefact bị hạ trọng
    số mạnh -- dùng để tính `motion_fraction` cho QC report.
    """
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n < int(sample_rate * 4):
        # Quá ngắn để tách tần số/đạo hàm đáng tin cậy -- trả nguyên vẹn.
        return x.copy(), np.ones(max(n - 1, 0)), 0

    nyq = sample_rate / 2.0
    fc_norm = filter_cutoff_hz / nyq
    if 0 < fc_norm < 1:
        sos = butter(filter_order, fc_norm, btype="low", output="sos")
        signal_low = sosfiltfilt(sos, x)
    else:
        signal_low = x.copy()
    signal_high = x - signal_low

    D = np.diff(signal_low)
    w = np.ones(len(D))
    mu = np.inf
    tune = 4.685
    iteration = 0
    while abs(mu) > 1e-8 and iteration < 50:
        mu = float(np.sum(D * w) / np.sum(w))
        D = D - mu
        sigma = float(np.sqrt(np.sum(w * D ** 2) / np.sum(w)))
        if sigma < 1e-12:
            break
        r = D / sigma / tune
        w = np.where(np.abs(r) < 1, (1 - r ** 2) ** 2, 0.0)
        iteration += 1

    new_D = w * D
    new_signal_low = np.cumsum(np.concatenate(([signal_low[0]], new_D)))
    corrected = new_signal_low + signal_high
    return corrected, w, iteration


# ════════════════════════════════════════════════════════════════════════
# 4. PREPROCESS CONTINUOUS SESSION
# ════════════════════════════════════════════════════════════════════════

def _lowpass(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 3) -> np.ndarray:
    nyq = fs / 2.0
    fc = cutoff_hz / nyq
    if not (0 < fc < 1):
        return x.copy()
    sos = butter(order, fc, btype="low", output="sos")
    return sosfiltfilt(sos, x)


def _highpass(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 3) -> np.ndarray:
    nyq = fs / 2.0
    fc = cutoff_hz / nyq
    if not (0 < fc < 1):
        return x.copy()
    sos = butter(order, fc, btype="high", output="sos")
    return sosfiltfilt(sos, x)


def _theilslopes_capped(t: np.ndarray, x: np.ndarray, max_points: int = 2000) -> Tuple[float, float]:
    """
    Slope/intercept Theil-Sen -- robust nhưng `scipy.stats.theilslopes` tính
    TẤT CẢ slope từng cặp điểm nội bộ, tức O(n^2) cả thời gian lẫn bộ nhớ.
    Với tín hiệu fNIRS nguyên vẹn (100Hz, hàng chục nghìn mẫu/session hay
    /phase MEDITATION dài), gọi trực tiếp có thể tốn hàng GB RAM và hàng
    chục giây -- đã xác nhận qua smoke test (48000 mẫu -> bị OS kill vì hết
    bộ nhớ). Downsample về tối đa `max_points` bin (median mỗi bin) trước
    khi gọi theilslopes nếu chuỗi dài hơn ngưỡng này; slope theo ĐƠN VỊ THỜI
    GIAN của `t` không đổi vì Theil-Sen slope ước lượng cùng 1 xu hướng dù
    tính trên toàn bộ điểm hay trên trung vị các bin đều nhau.
    """
    n = len(x)
    if n <= max_points:
        slope, intercept, _, _ = theilslopes(x, t)
        return float(slope), float(intercept)
    bin_edges = np.linspace(0, n, max_points + 1).astype(int)
    t_bin, x_bin = [], []
    for i in range(max_points):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi <= lo:
            continue
        t_bin.append(float(np.median(t[lo:hi])))
        x_bin.append(float(np.median(x[lo:hi])))
    slope, intercept, _, _ = theilslopes(np.array(x_bin), np.array(t_bin))
    return float(slope), float(intercept)


def _robust_linear_detrend(x: np.ndarray, fs: float) -> np.ndarray:
    """Trừ đường hồi quy Theil-Sen (robust, không bị outlier kéo lệch như
    detrend tuyến tính OLS thông thường) trên TOÀN BỘ tín hiệu liên tục."""
    n = len(x)
    finite = np.isfinite(x)
    if finite.sum() < max(10, n // 20):
        return x - np.nanmedian(x)
    t = np.arange(n) / fs
    slope, intercept = _theilslopes_capped(t[finite], x[finite])
    trend = slope * t + intercept
    return x - trend


def _robust_rolling_median_detrend(x: np.ndarray, fs: float, window_s: float = 90.0) -> np.ndarray:
    """
    Detrend bằng baseline rolling-median (cửa sổ trượt, center=True) thay vì
    MỘT độ dốc tuyến tính duy nhất cho toàn phiên (`_robust_linear_detrend`).

    Lý do tồn tại riêng hàm này: `_robust_linear_detrend` chỉ khử được drift
    TUYẾN TÍNH đều suốt session (~15 phút). Drift cảm biến/optode thực tế
    thường KHÔNG tuyến tính (vd trôi dạng bậc thang do nhiệt độ, tiếp xúc da
    thay đổi khi participant cử động nhẹ) -- 1 độ dốc duy nhất không nắm bắt
    được, để lại low-frequency residual lẫn vào cả HbO và HbR CÙNG CHIỀU,
    thổi phồng tương quan dương giả tạo giữa 2 kênh.

    Kỳ vọng lý thuyết: sau khi khử trôi nền đúng cách, tương quan
    r(HbO,HbR) TRONG-PHIÊN nên ÂM -- đúng chiều lý thuyết của neurovascular
    coupling (Kirilina et al. 2012; Scholkmann et al. 2014 -- tương quan
    dương mạnh là dấu hiệu kinh điển của nhiễu hệ thống/ngoài sọ lấn át tín
    hiệu thần kinh thật).

    Đây là OPTION MỚI (giá trị `detrend_method="robust_median_rolling_90s"`
    trong `PROCESSING_CONFIG`), KHÔNG thay thế `_robust_linear_detrend` --
    hàm/giá trị config cũ vẫn giữ nguyên hành vi, không phá code path hiện
    có. `preprocess_continuous_session()` chỉ gọi hàm này khi được chọn rõ.
    """
    n = len(x)
    window_samples = max(3, int(round(window_s * fs)))
    if window_samples % 2 == 0:
        window_samples += 1  # center=True cần cửa sổ lẻ để đối xứng quanh mẫu hiện tại
    if n < window_samples:
        return x - np.nanmedian(x)

    s = pd.Series(x)
    min_periods = max(3, window_samples // 4)
    baseline = s.rolling(window=window_samples, center=True, min_periods=min_periods).median()
    # Rìa đầu/cuối không đủ điểm trong cửa sổ -- lấp bằng giá trị baseline
    # gần nhất thay vì để NaN lan vào tín hiệu đã detrend.
    baseline = baseline.bfill().ffill()
    return (s - baseline).to_numpy()


def preprocess_continuous_session(
    loaded: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Xử lý TOÀN BỘ tín hiệu liên tục của session (KHÔNG cắt phase trước khi
    xử lý) -- thứ tự đúng theo yêu cầu:

    (nếu input_signal_stage == "raw_intensity", "Trường hợp A"):
      raw intensity -> Optical Density (I0 = mean toàn session, KHÔNG phải
      baseline phase cụ thể -- baseline theo phase áp dụng RIÊNG ở bước
      compute_baseline(), không trộn với bước OD ở đây)
      -> motion correction (TDDR) trên OD
      -> Modified Beer-Lambert -> HbO/HbR/HbT (motion-corrected)
      -> lowpass toàn session
      -> (tuỳ chọn) highpass toàn session (mặc định TẮT -- xem config)
      -> robust linear detrend toàn session

    (nếu input_signal_stage == "concentration_fallback", "Trường hợp B"):
      HbO/HbR có sẵn -> (tuỳ chọn) motion correction trên concentration nếu
      config bật -- GHI RÕ đây là "concentration-level fallback", không
      tương đương xử lý từ optical density
      -> lowpass -> (tuỳ chọn) highpass -> robust linear detrend

    Không filter/detrend riêng bất kỳ đoạn con nào -- hàm này chạy đúng 1
    lần cho toàn bộ mảng đầu vào.
    """
    stage = loaded["input_signal_stage"]
    fs = loaded["sampling_rate_hz"]
    if not np.isfinite(fs) or fs <= 0:
        fs = config["optical_sampling_rate_hz"]

    n_before = loaded["n_samples"]
    motion_info: Dict[str, Any] = {
        "motion_correction_method": None,
        "motion_correction_stage": None,
        "motion_fraction_ch1": np.nan,
        "motion_fraction_ch2": np.nan,
    }

    if stage == "raw_intensity":
        ch1 = loaded["fnirs_1_raw"]
        ch2 = loaded["fnirs_2_raw"]
        od1, ref1 = hbo_mod.raw_intensity_to_od(ch1, reference=None)
        od2, ref2 = hbo_mod.raw_intensity_to_od(ch2, reference=None)

        if config["motion_correction_method"] == "TDDR":
            od1_c, w1, it1 = tddr(od1, fs)
            od2_c, w2, it2 = tddr(od2, fs)
            motion_info["motion_correction_method"] = "TDDR"
            motion_info["motion_correction_stage"] = "optical_density"
            motion_info["motion_fraction_ch1"] = float(np.mean(w1 < 0.5))
            motion_info["motion_fraction_ch2"] = float(np.mean(w2 < 0.5))
            od1, od2 = od1_c, od2_c
        else:
            motion_info["motion_correction_method"] = "none"
            motion_info["motion_correction_stage"] = "none"

        conc = hbo_mod.concentration_from_od(
            od1, od2,
            l_cm=config.get("source_detector_distance_cm", hbo_mod.L_CM_DEFAULT),
            age=config.get("age_years", hbo_mod.AGE_DEFAULT),
        )
        hbo = conc["HbO_uM"].to_numpy()
        hbr = conc["HbR_uM"].to_numpy()

    elif stage == "concentration_fallback":
        hbo = loaded["hbo_uM"].copy()
        hbr = loaded["hbr_uM"].copy()
        if config.get("motion_correction_on_concentration_fallback", False) and config["motion_correction_method"] == "TDDR":
            hbo, w1, _ = tddr(hbo, fs)
            hbr, w2, _ = tddr(hbr, fs)
            motion_info["motion_correction_method"] = "TDDR"
            motion_info["motion_correction_stage"] = "concentration_fallback"
            motion_info["motion_fraction_ch1"] = float(np.mean(w1 < 0.5))
            motion_info["motion_fraction_ch2"] = float(np.mean(w2 < 0.5))
        else:
            motion_info["motion_correction_method"] = "none"
            motion_info["motion_correction_stage"] = "none"
    else:
        raise ValueError(f"Không xác định được input_signal_stage: {stage!r}")

    # --- filter + detrend TOÀN SESSION (KHÔNG cắt phase trước bước này) ---
    filter_log = {"lowpass_hz": config["lowpass_hz"], "highpass_hz": config["highpass_hz"], "detrend_method": config["detrend_method"]}

    def _filter_channel(x: np.ndarray) -> np.ndarray:
        y = x.copy()
        finite = np.isfinite(y)
        if finite.sum() < len(y):
            fill = np.nanmedian(y) if np.isfinite(np.nanmedian(y)) else 0.0
            y = np.where(finite, y, fill)
        if config["highpass_hz"]:
            y = _highpass(y, fs, config["highpass_hz"])
        if config["lowpass_hz"]:
            y = _lowpass(y, fs, config["lowpass_hz"])
        if config["detrend_method"] == "robust_linear_whole_session":
            y = _robust_linear_detrend(y, fs)
        elif config["detrend_method"] == "robust_median_rolling_90s":
            y = _robust_rolling_median_detrend(y, fs, window_s=config.get("detrend_rolling_window_s", 90.0))
        return y

    hbo_processed = _filter_channel(hbo)
    hbr_processed = _filter_channel(hbr)
    hbt_processed = hbo_processed + hbr_processed

    n_after = len(hbo_processed)

    return {
        "hbo_uM": hbo_processed,
        "hbr_uM": hbr_processed,
        "hbt_uM": hbt_processed,
        "sampling_rate_hz": fs,
        "input_signal_stage": stage,
        "n_samples_before": n_before,
        "n_samples_after": n_after,
        "length_unchanged": bool(n_before == n_after),
        **motion_info,
        **filter_log,
    }


# ════════════════════════════════════════════════════════════════════════
# 5. EVENT ALIGNMENT VALIDATION
# ════════════════════════════════════════════════════════════════════════

def validate_and_align_events(
    events_raw: Optional[pd.DataFrame],
    n_signal_samples: int,
    signal_fs: float,
    event_fs: float,
    session_duration_from_eeg_s: Optional[float],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Kiểm chứng (không mặc định đúng) và quy đổi event từ sample EEG sang
    sample fNIRS. KHÔNG dùng max()/min() để clip âm thầm -- ghi rõ từng
    event có bị clip hay không, offset căn chỉnh, và trạng thái.

    Trả về (events_aligned, event_qc_flags). events_aligned có cột:
      description, event_code,
      event_start_requested_s, event_end_requested_s,
      event_start_used_s, event_end_used_s,
      event_start_index, event_end_index,
      signal_duration_s, event_clipped, alignment_offset_s, alignment_status
    """
    flags: List[str] = []
    signal_duration_s = n_signal_samples / signal_fs if signal_fs else np.nan

    if events_raw is None or events_raw.empty:
        flags.append("MISSING_PHASE")
        return pd.DataFrame(), flags

    # Kiểm chứng time origin chung EEG/fNIRS bằng cách so thời lượng session
    # suy ra từ EEG (nếu có) với thời lượng suy ra từ chính tín hiệu fNIRS --
    # xem docstring đầu file: cả 2 phải xuất phát từ CÙNG block_df ở Layer 2.
    alignment_offset_s = 0.0
    if session_duration_from_eeg_s is not None and np.isfinite(session_duration_from_eeg_s):
        alignment_offset_s = float(signal_duration_s - session_duration_from_eeg_s)
        if abs(alignment_offset_s) > 1.0:
            flags.append("ALIGNMENT_UNCERTAIN")

    rows = []
    prev_end = -np.inf
    order_ok = True
    seen_phases = set()

    for _, ev in events_raw.iterrows():
        start_req_s = float(ev["latency"]) / event_fs
        end_req_s = float(ev["latency"] + ev["duration_samples"]) / event_fs
        desc = ev["description"]
        seen_phases.add(desc)

        clipped = False
        status_parts = []

        start_used_s = start_req_s
        end_used_s = end_req_s

        if end_req_s < 0 or start_req_s > signal_duration_s:
            status_parts.append("EVENT_OUT_OF_RANGE")
            flags.append("EVENT_OUT_OF_RANGE")
            start_used_s, end_used_s = np.nan, np.nan
        else:
            if start_req_s < 0:
                start_used_s = 0.0
                clipped = True
            if end_req_s > signal_duration_s:
                end_used_s = signal_duration_s
                clipped = True
            if clipped:
                status_parts.append("EVENT_CLIPPED")
                flags.append("EVENT_CLIPPED")

        if desc in EXPECTED_PHASE_ORDER and start_req_s < prev_end - 1e-6:
            status_parts.append("PHASE_OVERLAP")
            flags.append("PHASE_OVERLAP")
        if desc in EXPECTED_PHASE_ORDER:
            prev_end = end_req_s

        start_idx = int(round(start_used_s * signal_fs)) if np.isfinite(start_used_s) else None
        end_idx = int(round(end_used_s * signal_fs)) if np.isfinite(end_used_s) else None
        if start_idx is not None:
            start_idx = max(0, min(start_idx, n_signal_samples))
        if end_idx is not None:
            end_idx = max(0, min(end_idx, n_signal_samples))

        rows.append({
            "description": desc,
            "event_code": ev.get("event_code"),
            "event_start_requested_s": start_req_s,
            "event_end_requested_s": end_req_s,
            "event_start_used_s": start_used_s,
            "event_end_used_s": end_used_s,
            "event_start_index": start_idx,
            "event_end_index": end_idx,
            "signal_duration_s": signal_duration_s,
            "event_clipped": clipped,
            "alignment_offset_s": alignment_offset_s,
            "alignment_status": ";".join(status_parts) if status_parts else "OK",
        })

    # Thứ tự chuẩn REST_BEFORE -> MEDITATION -> REST_AFTER
    present_order = [r["description"] for r in rows if r["description"] in EXPECTED_PHASE_ORDER]
    expected_present = [p for p in EXPECTED_PHASE_ORDER if p in present_order]
    if present_order != expected_present:
        flags.append("INVALID_EVENT_ORDER")

    missing = set(EXPECTED_PHASE_ORDER) - seen_phases
    if missing:
        flags.append("INCOMPLETE_PHASES" if len(missing) < len(EXPECTED_PHASE_ORDER) else "MISSING_PHASE")

    return pd.DataFrame(rows), sorted(set(flags))


# ════════════════════════════════════════════════════════════════════════
# 6. BASELINE
# ════════════════════════════════════════════════════════════════════════

def compute_baseline(
    hbo: np.ndarray,
    hbr: np.ndarray,
    hbt: np.ndarray,
    fs: float,
    events_aligned: pd.DataFrame,
    baseline_source: str,
    baseline_window_s: float,
    baseline_statistic: str,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Baseline lấy 1 LẦN DUY NHẤT từ `baseline_source` (mặc định REST_BEFORE)
    của session -- KHÔNG reset lại ở mỗi phase. Dùng `baseline_window_s`
    giây CUỐI của phase baseline (sau khi phase đã đủ dài) để tránh onset
    transient ngay đầu rest.

    Trả về (baseline_dict, qc_flags). baseline_dict rỗng nếu không hợp lệ,
    kèm flag MISSING_BASELINE/INVALID_BASELINE/BASELINE_HIGH_VARIABILITY.
    """
    flags: List[str] = []
    if events_aligned.empty or "description" not in events_aligned.columns:
        # Session không có event_labels_block.csv (hoặc rỗng) -- không có
        # cách nào xác định REST_BEFORE, KHÔNG raise lỗi cứng (sẽ làm mất
        # nguyên session khỏi output, vi phạm yêu cầu không âm thầm loại bỏ
        # dữ liệu) -- trả về "không có baseline", caller vẫn xử lý phần
        # OVERALL/absolute-only cho session này.
        flags.append("MISSING_BASELINE")
        return {}, flags

    row = events_aligned[events_aligned["description"] == baseline_source]
    if row.empty or pd.isna(row.iloc[0]["event_start_index"]):
        flags.append("MISSING_BASELINE")
        return {}, flags

    r = row.iloc[0]
    s_idx, e_idx = int(r["event_start_index"]), int(r["event_end_index"])
    phase_len_s = (e_idx - s_idx) / fs
    win_samples = int(round(baseline_window_s * fs))

    if phase_len_s < 10:
        flags.append("INVALID_BASELINE")
        return {}, flags

    # baseline_window_s CUỐI của phase (gần lúc chuyển sang phase kế tiếp
    # nhất -- ổn định hơn vài giây đầu rest, vẫn còn trong REST_BEFORE).
    win_start = max(s_idx, e_idx - win_samples)
    seg_hbo = hbo[win_start:e_idx]
    seg_hbr = hbr[win_start:e_idx]
    seg_hbt = hbt[win_start:e_idx]

    stat_fn = np.nanmedian if baseline_statistic == "median" else np.nanmean
    baseline = {
        "baseline_hbo_uM": float(stat_fn(seg_hbo)),
        "baseline_hbr_uM": float(stat_fn(seg_hbr)),
        "baseline_hbt_uM": float(stat_fn(seg_hbt)),
        "baseline_n_samples": int(len(seg_hbo)),
        "baseline_window_s_used": float(len(seg_hbo) / fs),
        "baseline_source": baseline_source,
        "baseline_statistic": baseline_statistic,
    }

    # KHÔNG dùng CV = SD/|mean| -- HbO/HbR đã là tín hiệu bien-thien-tuong-doi
    # (relative to whole-session OD reference), trung bình dao động quanh 0
    # theo đúng thiết kế MBLL ở đây, nên |mean| gần 0 làm CV vô nghĩa/bùng nổ
    # số học (phát hiện qua smoke test: CV=3.09 dù baseline hoàn toàn "sạch").
    # Thay bằng so sánh độ phân tán CỦA CHÍNH cửa sổ baseline với độ phân tán
    # robust (MAD) của TOÀN BỘ session -- baseline "bất thường ồn" nếu ồn hơn
    # hẳn phần còn lại của session, không phụ thuộc vào giá trị trung bình.
    session_mad_hbo = float(np.median(np.abs(hbo[np.isfinite(hbo)] - np.median(hbo[np.isfinite(hbo)])))) * 1.4826
    baseline_sd_hbo = float(np.nanstd(seg_hbo))
    baseline["baseline_hbo_sd_uM"] = baseline_sd_hbo
    baseline["session_hbo_mad_uM"] = session_mad_hbo
    ratio = baseline_sd_hbo / session_mad_hbo if session_mad_hbo > 1e-9 else np.nan
    baseline["baseline_to_session_dispersion_ratio"] = ratio
    if not np.isfinite(ratio) or ratio > 3.0:
        flags.append("BASELINE_HIGH_VARIABILITY")

    return baseline, flags


# ════════════════════════════════════════════════════════════════════════
# 7. PHASE SEGMENTATION (transition exclusion, full vs stable)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class PhaseSegment:
    description: str
    full_slice: Tuple[int, int]
    stable_slice: Tuple[int, int]
    full_duration_s: float
    stable_duration_s: float
    transition_excluded_s: float
    qc_flags: List[str] = field(default_factory=list)


def segment_phases(
    events_aligned: pd.DataFrame,
    n_signal_samples: int,
    fs: float,
    transition_exclusion_s: Dict[str, float],
    min_phase_duration_s: float,
) -> List[PhaseSegment]:
    """
    Cắt phase từ tín hiệu ĐÃ XỬ LÝ LIÊN TỤC (preprocess_continuous_session)
    -- hàm này chỉ cắt index, không filter/detrend gì thêm.

    Mỗi phase có 2 slice:
    - full_slice: toàn bộ phase như alignment đã tính.
    - stable_slice: bỏ `transition_exclusion_s[phase]` giây ĐẦU phase (loại
      transient chuyển pha) -- nếu phase ngắn hơn min_phase_duration_s SAU
      khi loại transition, stable_slice rỗng và gắn cờ tương ứng qua caller.
    """
    segments: List[PhaseSegment] = []
    if events_aligned.empty or "description" not in events_aligned.columns:
        # Không có event_labels_block.csv hợp lệ -- không cắt được phase nào,
        # trả về [] (caller tạo 1 phase "OVERALL" phủ toàn tín hiệu thay thế,
        # xem notebook xử lý) thay vì raise lỗi làm mất cả session.
        return segments
    for _, r in events_aligned.iterrows():
        desc = r["description"]
        if pd.isna(r["event_start_index"]) or pd.isna(r["event_end_index"]):
            continue
        s_idx, e_idx = int(r["event_start_index"]), int(r["event_end_index"])
        s_idx = max(0, min(s_idx, n_signal_samples))
        e_idx = max(0, min(e_idx, n_signal_samples))
        if e_idx <= s_idx:
            continue

        full_duration_s = (e_idx - s_idx) / fs
        exclude_s = transition_exclusion_s.get(desc, 0.0)
        exclude_n = int(round(exclude_s * fs))
        stable_s_idx = min(s_idx + exclude_n, e_idx)
        stable_duration_s = (e_idx - stable_s_idx) / fs

        flags = []
        if full_duration_s < min_phase_duration_s:
            flags.append("DURATION_MISMATCH")

        segments.append(PhaseSegment(
            description=desc,
            full_slice=(s_idx, e_idx),
            stable_slice=(stable_s_idx, e_idx),
            full_duration_s=full_duration_s,
            stable_duration_s=stable_duration_s,
            transition_excluded_s=exclude_s,
            qc_flags=flags,
        ))
    return segments


# ════════════════════════════════════════════════════════════════════════
# 8. ROBUST FEATURE HELPERS
# ════════════════════════════════════════════════════════════════════════

def _trimmed_mean(x: np.ndarray, proportion: float = 0.1) -> float:
    x = np.sort(x[np.isfinite(x)])
    n = len(x)
    if n == 0:
        return np.nan
    k = int(np.floor(n * proportion))
    trimmed = x[k: n - k] if n - 2 * k > 0 else x
    return float(np.mean(trimmed))


def _mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.median(np.abs(x - np.median(x))) * 1.4826)


def _iqr(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


def _theil_sen_slope_per_min(x: np.ndarray, fs: float) -> float:
    """Dùng `_theilslopes_capped()` -- KHÔNG gọi theilslopes trực tiếp, vì
    phase MEDITATION (hàng chục nghìn mẫu ở 100Hz) đủ dài để làm
    scipy.stats.theilslopes (O(n^2) bộ nhớ) hết RAM (xem ghi chú tại
    `_theilslopes_capped`)."""
    finite = np.isfinite(x)
    if finite.sum() < 4:
        return np.nan
    t_min = np.arange(len(x))[finite] / fs / 60.0
    slope, _ = _theilslopes_capped(t_min, x[finite])
    return float(slope)


def _ordinary_slope_per_min(x: np.ndarray, fs: float) -> float:
    finite = np.isfinite(x)
    if finite.sum() < 2:
        return np.nan
    t_min = np.arange(len(x))[finite] / fs / 60.0
    slope, _ = np.polyfit(t_min, x[finite], 1)
    return float(slope)


def _auc_relative_to_baseline(x_delta: np.ndarray, fs: float) -> float:
    """Diện tích dưới đường cong ΔX theo thời gian (µM*giây) -- x_delta đã
    là bien thien so với baseline (0 = baseline)."""
    finite = np.isfinite(x_delta)
    if finite.sum() < 2:
        return np.nan
    return float(np.trapz(np.nan_to_num(x_delta[finite]), dx=1.0 / fs))


def robust_channel_features(x_absolute: np.ndarray, x_delta: Optional[np.ndarray], fs: float, prefix: str) -> Dict[str, float]:
    """Bộ feature robust đầy đủ cho 1 kênh (HbO/HbR/HbT), cả absolute và
    (nếu truyền x_delta) baseline-corrected."""
    x = np.asarray(x_absolute, dtype=np.float64)
    finite = np.isfinite(x)
    out: Dict[str, float] = {
        f"{prefix}_finite_fraction": float(finite.mean()) if len(x) else np.nan,
        f"{prefix}_valid_duration_s": float(finite.sum() / fs) if fs else np.nan,
    }
    xf = x[finite]
    if len(xf) < 2:
        for k in ["mean", "median", "trimmed_mean10", "sd", "mad", "iqr", "peak_to_peak", "max", "min",
                  "slope_ordinary_uM_per_min", "slope_theilsen_uM_per_min"]:
            out[f"{prefix}_absolute_{k}_uM"] = np.nan
        return out

    out.update({
        f"{prefix}_absolute_mean_uM": float(np.mean(xf)),
        f"{prefix}_absolute_median_uM": float(np.median(xf)),
        f"{prefix}_absolute_trimmed_mean10_uM": _trimmed_mean(xf, 0.1),
        f"{prefix}_absolute_sd_uM": float(np.std(xf, ddof=1)) if len(xf) > 1 else np.nan,
        f"{prefix}_absolute_mad_uM": _mad(xf),
        f"{prefix}_absolute_iqr_uM": _iqr(xf),
        f"{prefix}_absolute_peak_to_peak_uM": float(np.ptp(xf)),
        f"{prefix}_absolute_max_uM": float(np.max(xf)),
        f"{prefix}_absolute_min_uM": float(np.min(xf)),
        f"{prefix}_slope_ordinary_uM_per_min": _ordinary_slope_per_min(x, fs),
        f"{prefix}_slope_theilsen_uM_per_min": _theil_sen_slope_per_min(x, fs),
    })

    if x_delta is not None:
        d = np.asarray(x_delta, dtype=np.float64)
        df_ = d[np.isfinite(d)]
        if len(df_) >= 2:
            out.update({
                f"{prefix}_delta_mean_uM": float(np.mean(df_)),
                f"{prefix}_delta_median_uM": float(np.median(df_)),
                f"{prefix}_delta_trimmed_mean10_uM": _trimmed_mean(df_, 0.1),
                f"{prefix}_delta_sd_uM": float(np.std(df_, ddof=1)),
                f"{prefix}_delta_mad_uM": _mad(df_),
                f"{prefix}_delta_iqr_uM": _iqr(df_),
                f"{prefix}_delta_auc_uM_s": _auc_relative_to_baseline(d, fs),
            })
        else:
            for k in ["mean", "median", "trimmed_mean10", "sd", "mad", "iqr"]:
                out[f"{prefix}_delta_{k}_uM"] = np.nan
            out[f"{prefix}_delta_auc_uM_s"] = np.nan

    return out


# ════════════════════════════════════════════════════════════════════════
# 9. SPECTRAL (windowed, tránh nhầm lẫn low-pass cutoff vs band tổng)
# ════════════════════════════════════════════════════════════════════════

def spectral_low_freq_ratio_windowed(
    x: np.ndarray,
    fs: float,
    window_s: float,
    step_s: float,
    low_band: Tuple[float, float],
    total_band: Tuple[float, float],
) -> Dict[str, float]:
    """
    Tính tỉ lệ công suất dải thấp bằng CỬA SỔ CỐ ĐỊNH (window_s/step_s),
    KHÔNG dùng toàn bộ phase 1 lần (rest 2 phút vs thiền 10 phút không thể
    so PSD trực tiếp bằng nhau) -- lấy MEDIAN qua các cửa sổ hợp lệ.

    QUAN TRỌNG: hàm này phải được gọi trên nhánh tín hiệu PHÙ HỢP với
    total_band -- nếu tín hiệu đầu vào đã lowpass ở cutoff_hz thấp hơn
    total_band[1], phổ phía trên cutoff sẽ ~0 và tỉ lệ mất ý nghĩa. Notebook
    gọi hàm này phải tự chọn nhánh tín hiệu đúng (xem PROCESSING_CONFIG
    "spectral_source_branch" trong notebook).
    """
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    n = len(x)
    win_n = int(round(window_s * fs))
    step_n = int(round(step_s * fs))

    if win_n < 8 or n < win_n:
        return {"low_freq_power_ratio": np.nan, "n_spectral_windows": 0, "spectral_frequency_resolution_hz": np.nan, "spectral_valid": False}

    ratios = []
    for start in range(0, n - win_n + 1, step_n):
        seg = x[start:start + win_n]
        if np.mean(np.isfinite(seg)) < 0.95:
            continue
        seg = np.nan_to_num(seg, nan=np.nanmedian(seg))
        freqs, psd = welch(seg - np.mean(seg), fs=fs, window="hann", nperseg=win_n, noverlap=win_n // 2)

        def band_power(lo, hi):
            idx = (freqs >= lo) & (freqs <= hi)
            return float(np.trapz(psd[idx], freqs[idx])) if idx.any() else np.nan

        low = band_power(*low_band)
        total = band_power(*total_band)
        if total and total > 0 and np.isfinite(low):
            ratios.append(low / total)

    if not ratios:
        return {"low_freq_power_ratio": np.nan, "n_spectral_windows": 0, "spectral_frequency_resolution_hz": fs / win_n, "spectral_valid": False}

    return {
        "low_freq_power_ratio": float(np.median(ratios)),
        "n_spectral_windows": len(ratios),
        "spectral_frequency_resolution_hz": fs / win_n,
        "spectral_valid": True,
    }


# ════════════════════════════════════════════════════════════════════════
# 10. QC FLAG ASSIGNMENT (statistical screen, robust z theo phase/metric)
# ════════════════════════════════════════════════════════════════════════

def robust_outlier_flags(df: pd.DataFrame, metric_cols: List[str], group_cols: List[str], z_threshold: float) -> pd.DataFrame:
    """
    Gắn robust_outlier + max_abs_robust_z theo NHÓM (thường group theo
    `phase`) dùng median/MAD -- KHÔNG xoá dòng nào, chỉ đánh dấu để chạy
    sensitivity analysis riêng (xem EDA notebook).

        robust_z = 0.6745 * (x - median) / MAD
    """
    out = df.copy()
    out["max_abs_robust_z"] = 0.0
    out["robust_outlier"] = False

    for _, idx in out.groupby(group_cols, observed=True).groups.items():
        sub = out.loc[idx]
        z_max = pd.Series(0.0, index=sub.index)
        for col in metric_cols:
            if col not in sub.columns:
                continue
            vals = sub[col].to_numpy(dtype=float)
            med = np.nanmedian(vals)
            mad = np.nanmedian(np.abs(vals - med)) * 1.4826
            if mad <= 0 or not np.isfinite(mad):
                continue
            z = np.abs(0.6745 * (vals - med) / mad)
            z_max = np.maximum(z_max, pd.Series(np.nan_to_num(z, nan=0.0), index=sub.index))
        out.loc[idx, "max_abs_robust_z"] = z_max
        out.loc[idx, "robust_outlier"] = z_max > z_threshold

    return out


# ════════════════════════════════════════════════════════════════════════
# 11. REPRODUCIBILITY METADATA
# ════════════════════════════════════════════════════════════════════════

def reproducibility_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    import sys
    import scipy
    meta = {
        "pipeline_version": config.get("pipeline_version", PIPELINE_VERSION),
        "processing_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
    }
    return meta


# ════════════════════════════════════════════════════════════════════════
# 12. MEDITATION SUB-PERIOD SPLIT (EARLY / MIDDLE / LATE)
# ════════════════════════════════════════════════════════════════════════

def split_meditation_subperiods(
    segment: "PhaseSegment",
    fs: float,
    meditation_phase_name: str = "MEDITATION",
    min_duration_s: float = 30.0,
) -> List["PhaseSegment"]:
    """
    Chia phase MEDITATION thành 3 phần bằng nhau EARLY/MIDDLE/LATE, dùng
    `segment.stable_slice` (đã loại transition) làm miền chia -- không chia
    lại transition riêng cho từng phần con vì transition chỉ xảy ra ở đầu
    MEDITATION, không lặp lại giữa các phần con.

    Trả về [] nếu không phải phase MEDITATION hoặc quá ngắn (<min_duration_s)
    để chia có ý nghĩa thống kê.
    """
    if segment.description != meditation_phase_name:
        return []
    s_idx, e_idx = segment.stable_slice
    n = e_idx - s_idx
    if n / fs < min_duration_s:
        return []

    cuts = [s_idx + int(round(n * f)) for f in (0.0, 1 / 3, 2 / 3, 1.0)]
    labels = ["MEDITATION_EARLY", "MEDITATION_MIDDLE", "MEDITATION_LATE"]
    out = []
    for label, a, b in zip(labels, cuts[:-1], cuts[1:]):
        out.append(PhaseSegment(
            description=label,
            full_slice=(a, b),
            stable_slice=(a, b),
            full_duration_s=(b - a) / fs,
            stable_duration_s=(b - a) / fs,
            transition_excluded_s=0.0,
            qc_flags=[],
        ))
    return out


# ════════════════════════════════════════════════════════════════════════
# 13. COMPUTE PHASE METRICS (combines robust features + HbT QC + spectral)
# ════════════════════════════════════════════════════════════════════════

def compute_phase_metrics(
    hbo: np.ndarray,
    hbr: np.ndarray,
    hbt: np.ndarray,
    fs: float,
    baseline: Dict[str, float],
    segment: "PhaseSegment",
    spectral_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Tính đầy đủ feature (absolute + delta-so-baseline, HbO/HbR/HbT, thống kê
    robust, slope, AUC, spectral ratio, HbT-consistency QC) cho 1 phase.

    Trả về DANH SÁCH 2 dict -- 1 cho `feature_variant="full_phase"`
    (`segment.full_slice`) và 1 cho `feature_variant="stable_phase"`
    (`segment.stable_slice`, sau khi loại transition) -- GIỮ CẢ HAI biến
    thể trong output, không ghi đè/loại bỏ cái nào (yêu cầu bắt buộc).
    """
    rows: List[Dict[str, Any]] = []
    variants = [("full_phase", segment.full_slice), ("stable_phase", segment.stable_slice)]
    has_baseline = bool(baseline)

    for variant_name, (s_idx, e_idx) in variants:
        row: Dict[str, Any] = {
            "phase": segment.description,
            "feature_variant": variant_name,
            "phase_start_index": s_idx,
            "phase_end_index": e_idx,
            "phase_duration_s": (e_idx - s_idx) / fs if e_idx > s_idx else 0.0,
            "transition_excluded_s": segment.transition_excluded_s if variant_name == "stable_phase" else 0.0,
        }
        flags = list(segment.qc_flags)

        if e_idx <= s_idx:
            row["qc_flags"] = ";".join(sorted(set(flags) | {"DURATION_MISMATCH"}))
            rows.append(row)
            continue

        seg_hbo = hbo[s_idx:e_idx]
        seg_hbr = hbr[s_idx:e_idx]
        seg_hbt = hbt[s_idx:e_idx]

        if has_baseline:
            delta_hbo = seg_hbo - baseline["baseline_hbo_uM"]
            delta_hbr = seg_hbr - baseline["baseline_hbr_uM"]
            delta_hbt = seg_hbt - baseline["baseline_hbt_uM"]
        else:
            delta_hbo = delta_hbr = delta_hbt = None
            flags.append("MISSING_BASELINE")

        row.update(robust_channel_features(seg_hbo, delta_hbo, fs, "hbo"))
        row.update(robust_channel_features(seg_hbr, delta_hbr, fs, "hbr"))
        row.update(robust_channel_features(seg_hbt, delta_hbt, fs, "hbt"))

        # HbT phải xấp xỉ HbO + HbR THEO ĐỊNH NGHĨA (không phải 2 phép đo độc
        # lập) -- lệch lớn báo lỗi số học ở bước trước, không phải hiện tượng
        # sinh lý mới.
        finite = np.isfinite(seg_hbo) & np.isfinite(seg_hbr) & np.isfinite(seg_hbt)
        if finite.sum() > 0:
            resid = seg_hbt[finite] - (seg_hbo[finite] + seg_hbr[finite])
            max_resid = float(np.max(np.abs(resid)))
            row["hbt_consistency_max_abs_residual_uM"] = max_resid
            if max_resid > 1e-6:
                flags.append("HBT_INCONSISTENT")
        else:
            row["hbt_consistency_max_abs_residual_uM"] = np.nan

        spec = spectral_low_freq_ratio_windowed(
            seg_hbo, fs,
            spectral_config["window_s"], spectral_config["step_s"],
            spectral_config["low_band"], spectral_config["total_band"],
        )
        row.update({f"hbo_spectral_{k}": v for k, v in spec.items()})

        row["qc_flags"] = ";".join(sorted(set(flags))) if flags else ""
        rows.append(row)

    return rows


# ════════════════════════════════════════════════════════════════════════
# 14. SESSION CHANGES (paired phase-to-phase deltas, cho bước thống kê)
# ════════════════════════════════════════════════════════════════════════

def compute_session_changes(
    phase_df: pd.DataFrame,
    phase_pairs: List[Tuple[str, str]],
    metric_cols: List[str],
    group_cols: List[str],
    feature_variant: str = "stable_phase",
) -> pd.DataFrame:
    """
    Thay đổi (change = giá trị phase_to - giá trị phase_from) theo CẶP phase
    trong CÙNG session, cho từng metric -- input cho bước thống kê (Wilcoxon
    /bootstrap chạy trên các "change" này, không phải giá trị tuyệt đối của
    từng phase riêng lẻ). Long-format: group_cols + phase_from/phase_to/
    metric/value_from/value_to/change/feature_variant.
    """
    sub = phase_df[phase_df["feature_variant"] == feature_variant]
    rows = []
    for keys, g in sub.groupby(group_cols, observed=True):
        keys_t = keys if isinstance(keys, tuple) else (keys,)
        key_dict = dict(zip(group_cols, keys_t))
        g_by_phase = g.drop_duplicates(subset="phase").set_index("phase")
        for phase_from, phase_to in phase_pairs:
            if phase_from not in g_by_phase.index or phase_to not in g_by_phase.index:
                continue
            for metric in metric_cols:
                if metric not in g_by_phase.columns:
                    continue
                v_from = g_by_phase.loc[phase_from, metric]
                v_to = g_by_phase.loc[phase_to, metric]
                change = (v_to - v_from) if pd.notna(v_from) and pd.notna(v_to) else np.nan
                rows.append({
                    **key_dict,
                    "phase_from": phase_from, "phase_to": phase_to,
                    "metric": metric,
                    "value_from": v_from, "value_to": v_to,
                    "change": change,
                    "feature_variant": feature_variant,
                })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════
# 15. QC FLAG AGGREGATION (per tầng -- không tạo flag mới, chỉ phân loại)
# ════════════════════════════════════════════════════════════════════════

def assign_qc_flags(df: pd.DataFrame, qc_flags_col: str = "qc_flags") -> pd.DataFrame:
    """
    Tổng hợp cờ QC theo tầng (input/signal/event/baseline) từ cột
    `qc_flags_col` (chuỗi flag phân tách ';') đã gắn ở các bước trước --
    KHÔNG tạo flag mới, chỉ phân loại + gắn `qc_pass`.

    `qc_pass=False` chỉ có nghĩa "cần xem lại" -- KHÔNG bao giờ tự động xoá
    dòng khỏi output (yêu cầu bắt buộc: không âm thầm loại bỏ dữ liệu).
    Flag thống kê (ROBUST_OUTLIER, gắn riêng bởi `robust_outlier_flags()` ở
    bước sau) CỐ Ý không tính vào qc_pass vì đó là cờ nhạy cảm-phân-tích
    (sensitivity), không phải lỗi chất lượng dữ liệu.
    """
    out = df.copy()

    def _flags_of(s):
        return set(s.split(";")) if isinstance(s, str) and s else set()

    flag_sets = out[qc_flags_col].apply(_flags_of)
    out["qc_flag_count"] = flag_sets.apply(len)
    out["has_input_flag"] = flag_sets.apply(lambda s: bool(s & INPUT_QC_FLAGS))
    out["has_signal_flag"] = flag_sets.apply(lambda s: bool(s & SIGNAL_QC_FLAGS))
    out["has_event_flag"] = flag_sets.apply(lambda s: bool(s & EVENT_QC_FLAGS))
    out["has_baseline_flag"] = flag_sets.apply(lambda s: bool(s & BASELINE_QC_FLAGS))
    out["has_stat_flag"] = flag_sets.apply(lambda s: bool(s & STAT_QC_FLAGS))
    out["qc_pass"] = ~(out["has_input_flag"] | out["has_signal_flag"] | out["has_event_flag"] | out["has_baseline_flag"])
    return out


# ════════════════════════════════════════════════════════════════════════
# 16. EXPORT RESULTS
# ════════════════════════════════════════════════════════════════════════

def export_results(
    out_dir: Path,
    phase_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    changes_df: pd.DataFrame,
    events_df: pd.DataFrame,
    run_log_df: pd.DataFrame,
    qc_summary_df: pd.DataFrame,
    config: Dict[str, Any],
    input_file_hashes: Dict[str, str],
) -> Dict[str, Path]:
    """Ghi 7 file CSV/JSON (hậu tố _v2) vào out_dir -- file thứ 8 (boxplot
    tổng quan) được vẽ + lưu riêng ở notebook (cần matplotlib figure, không
    phù hợp đặt trong hàm thư viện thuần data)."""
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "phase_by_phase": out_dir / "fnirs_hemo_by_phase_v2.csv",
        "wide_by_session": out_dir / "fnirs_hemo_wide_by_session_v2.csv",
        "session_changes": out_dir / "fnirs_hemo_session_changes_v2.csv",
        "event_alignment": out_dir / "fnirs_hemo_event_alignment_v2.csv",
        "run_log": out_dir / "fnirs_hemo_run_log_v2.csv",
        "qc_summary": out_dir / "fnirs_hemo_qc_summary_v2.csv",
        "reproducibility": out_dir / "fnirs_hemo_reproducibility_v2.json",
    }

    phase_df.to_csv(paths["phase_by_phase"], index=False)
    wide_df.to_csv(paths["wide_by_session"], index=False)
    changes_df.to_csv(paths["session_changes"], index=False)
    events_df.to_csv(paths["event_alignment"], index=False)
    run_log_df.to_csv(paths["run_log"], index=False)
    qc_summary_df.to_csv(paths["qc_summary"], index=False)

    meta = reproducibility_metadata(config)
    meta["config"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()}
    meta["input_file_hashes"] = input_file_hashes
    meta["n_sessions_processed"] = int(run_log_df["session_id"].nunique()) if "session_id" in run_log_df.columns else None
    meta["n_phase_rows"] = int(len(phase_df))
    with open(paths["reproducibility"], "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return paths
