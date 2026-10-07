"""
Xử lý tín hiệu PPG raw: lọc, xác định đỉnh (systolic peak), tính chỉ số HRV
miền thời gian (HR, RMSSD, SDRR, pRR50, pRR20) và miền tần số (LF/HF) --
dùng chung cho notebook phân tích PPG hàng loạt (batch).

Input mong đợi: PPG raw count (chưa hiệu chỉnh đơn vị -- không cần thiết vì
HRV chỉ phụ thuộc THỜI ĐIỂM đỉnh, không phụ thuộc biên độ tuyệt đối), lấy
mẫu ở OPTICAL_FS = 100Hz (đúng quy ước ppg_raw.edf do Layer 2 xuất, xem
lib/signal_qc.py::export_ppg_block_to_edf).

Băng lọc 0.5-8Hz giống hệt filter_ppg() trong module QC (giữ nhất quán).

================================================================================
CÔNG THỨC & CITATION
================================================================================

1) Xác định đỉnh -- `detect_peaks()`
------------------------------------
`scipy.signal.find_peaks` với khoảng cách tối thiểu 0.35s (giới hạn ~171bpm)
và prominence = 0.8 * median absolute deviation quanh median tín hiệu đã lọc
-- heuristic thích nghi theo biên độ từng đoạn, không dùng ngưỡng cố định.

2) IBI (inter-beat interval) -- `compute_ibi()`
------------------------------------------------
IBI = khoảng cách thời gian giữa 2 đỉnh liên tiếp (giây). Lọc theo khoảng
sinh lý hợp lệ [MIN_IBI_S, MAX_IBI_S] = [0.30, 1.80]s (~33-200 bpm) để loại
nhịp lỡ/nhịp giả do artefact trước khi tính HRV.

3) HRV miền thời gian -- `time_domain_hrv()`
-----------------------------------------------
    HR (bpm)    = 60 / mean(IBI_giây)
    RMSSD (ms)  = sqrt( mean( diff(IBI_ms)^2 ) )
    SDRR (ms)   = std(IBI_ms, ddof=1)                 (= SDNN)
    pRR50 (%)   = % số cặp IBI liên tiếp có |diff| > 50ms
    pRR20 (%)   = % số cặp IBI liên tiếp có |diff| > 20ms

Citation (định nghĩa chuẩn + băng tần):
- Task Force of the European Society of Cardiology and the North American
  Society of Pacing and Electrophysiology. "Heart rate variability:
  standards of measurement, physiological interpretation, and clinical
  use." Circulation. 1996;93(5):1043-1065.
- Shaffer F, Ginsberg JP. "An overview of heart rate variability metrics
  and norms." Front Public Health. 2017;5:258.

4) HRV miền tần số (LF/HF) -- `freq_domain_hrv()`
----------------------------------------------------
Nội suy tuyến tính chuỗi IBI (không đều mẫu, 1 giá trị/nhịp) thành tachogram
đều mẫu ở resample_hz (mặc định 4Hz, theo khuyến nghị Task Force 1996), trừ
mean rồi tính PSD bằng Welch (Hann, cửa sổ tới 120s, chồng lấn 50%):

    LF power = tích phân PSD trong băng 0.04-0.15 Hz
    HF power = tích phân PSD trong băng 0.15-0.40 Hz
    LF/HF    = LF power / HF power

Cần tối thiểu MIN_DURATION_FOR_FREQ_DOMAIN_S = 60 giây IBI liên tục (băng LF
thấp nhất 0.04Hz = chu kỳ 25s, cần nhiều chu kỳ mới ước lượng ổn định) --
ngắn hơn trả NaN thay vì 1 số không đáng tin.

Citation: cùng Task Force 1996 ở mục 3 (băng tần LF/HF chuẩn hoá bởi tài
liệu này).

Lưu ý diễn giải: HRV tính từ PPG (photoplethysmography) không hoàn toàn
tương đương HRV tính từ ECG -- có thêm jitter do pulse transit time, RMSSD
từ PPG thường cao hơn ECG cùng đối tượng. Xem:
- Schäfer A, Vagedes J. "How accurate is pulse rate variability as an
  estimate of heart rate variability? A review on studies comparing
  photoplethysmographic technology with an electrocardiogram."
  Int J Cardiol. 2013;166(1):15-29.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

PPG_FS_DEFAULT = 100.0

# Băng tần HRV chuẩn (Task Force of ESC/NASPE 1996).
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)

# Khoảng IBI hợp lệ về mặt sinh lý (33-200 bpm) -- loại nhịp lỡ/nhịp giả do artefact.
MIN_IBI_S = 0.30
MAX_IBI_S = 1.80

# Cần tối thiểu ngần này giây IBI liên tục mới tính LF/HF (ngắn hơn thì phổ
# tần số thấp không đủ chu kỳ để ước lượng đáng tin cậy).
MIN_DURATION_FOR_FREQ_DOMAIN_S = 60.0


def bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def filter_ppg(raw_ppg: np.ndarray, fs: float = PPG_FS_DEFAULT) -> np.ndarray:
    """Bandpass 0.5-8Hz -- cùng dải với signal_qc.filter_ppg()."""
    x = np.asarray(raw_ppg, dtype=float)
    if np.isnan(x).any():
        median = np.nanmedian(x)
        x = np.where(np.isnan(x), median if np.isfinite(median) else 0.0, x)
    return bandpass_filter(x, fs, 0.5, 8.0, order=3)


def detect_peaks(filtered_ppg: np.ndarray, fs: float = PPG_FS_DEFAULT) -> np.ndarray:
    """Đỉnh tâm thu (systolic peak) -- cùng heuristic prominence/distance với module QC."""
    prominence = max(np.nanmedian(np.abs(filtered_ppg - np.nanmedian(filtered_ppg))) * 0.8, 1e-9)
    peaks, _ = find_peaks(filtered_ppg, distance=int(0.35 * fs), prominence=prominence)
    return peaks


def compute_ibi(peaks: np.ndarray, fs: float = PPG_FS_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    """IBI (giây) giữa các đỉnh liên tiếp + thời điểm (giây, gắn với đỉnh sau)
    của từng IBI -- đã lọc theo khoảng sinh lý hợp lệ [MIN_IBI_S, MAX_IBI_S]."""
    if len(peaks) < 2:
        return np.array([]), np.array([])
    ibi = np.diff(peaks) / fs
    t = peaks[1:] / fs
    valid = (ibi >= MIN_IBI_S) & (ibi <= MAX_IBI_S)
    return ibi[valid], t[valid]


def time_domain_hrv(ibi_s: np.ndarray) -> Dict[str, float]:
    """HR trung bình, RMSSD, SDRR (=SDNN), pRR50, pRR20 -- từ chuỗi IBI (giây) đã lọc hợp lệ."""
    ibi_s = np.asarray(ibi_s, dtype=float)
    if len(ibi_s) < 2:
        return {
            "n_valid_ibi": int(len(ibi_s)),
            "hr_mean_bpm": np.nan,
            "rmssd_ms": np.nan,
            "sdrr_ms": np.nan,
            "prr50_pct": np.nan,
            "prr20_pct": np.nan,
        }
    ibi_ms = ibi_s * 1000.0
    diffs = np.diff(ibi_ms)
    return {
        "n_valid_ibi": int(len(ibi_s)),
        "hr_mean_bpm": float(60.0 / np.mean(ibi_s)),
        "rmssd_ms": float(np.sqrt(np.mean(diffs ** 2))),
        "sdrr_ms": float(np.std(ibi_ms, ddof=1)),
        "prr50_pct": float(np.mean(np.abs(diffs) > 50.0) * 100.0),
        "prr20_pct": float(np.mean(np.abs(diffs) > 20.0) * 100.0),
    }


def freq_domain_hrv(ibi_s: np.ndarray, ibi_times_s: np.ndarray, resample_hz: float = 4.0) -> Dict[str, float]:
    """
    Nội suy chuỗi IBI (không đều mẫu, 1 giá trị/nhịp tim) thành tachogram đều
    mẫu (resample_hz, mặc định 4Hz theo chuẩn HRV), rồi tính PSD bằng Welch
    và lấy công suất trong băng LF (0.04-0.15Hz) / HF (0.15-0.4Hz) -> LF/HF.

    Cần tối thiểu MIN_DURATION_FOR_FREQ_DOMAIN_S giây IBI liên tục, nếu không
    trả về NaN thay vì 1 con số không đáng tin (băng LF cần nhiều chu kỳ mới
    ước lượng ổn định).
    """
    duration = float(ibi_times_s[-1] - ibi_times_s[0]) if len(ibi_times_s) else 0.0
    if len(ibi_s) < 4 or duration < MIN_DURATION_FOR_FREQ_DOMAIN_S:
        return {"lf_power": np.nan, "hf_power": np.nan, "lf_hf_ratio": np.nan, "freq_domain_duration_s": duration}

    t0, t1 = ibi_times_s[0], ibi_times_s[-1]
    t_uniform = np.arange(t0, t1, 1.0 / resample_hz)
    ibi_ms = ibi_s * 1000.0
    rr_interp = np.interp(t_uniform, ibi_times_s, ibi_ms)
    rr_interp = rr_interp - np.mean(rr_interp)

    nperseg = min(len(rr_interp), int(resample_hz * 120))  # cửa sổ Welch tới 120s nếu đủ dài
    if nperseg < 8:
        return {"lf_power": np.nan, "hf_power": np.nan, "lf_hf_ratio": np.nan, "freq_domain_duration_s": duration}

    freqs, psd = welch(rr_interp, fs=resample_hz, window="hann", nperseg=nperseg, noverlap=nperseg // 2)

    def band_power(lo: float, hi: float) -> float:
        idx = (freqs >= lo) & (freqs <= hi)
        return float(np.trapz(psd[idx], freqs[idx])) if idx.any() else np.nan

    lf = band_power(*LF_BAND)
    hf = band_power(*HF_BAND)
    ratio = float(lf / hf) if hf and hf > 0 and np.isfinite(lf) else np.nan
    return {"lf_power": lf, "hf_power": hf, "lf_hf_ratio": ratio, "freq_domain_duration_s": duration}


def analyze_segment(raw_ppg: np.ndarray, fs: float = PPG_FS_DEFAULT) -> Dict[str, float]:
    """Chạy full pipeline (lọc -> đỉnh -> IBI -> HRV miền thời gian + tần số) cho 1 đoạn PPG raw."""
    n = len(raw_ppg)
    result: Dict[str, float] = {
        "segment_duration_s": float(n / fs) if fs else np.nan,
        "n_peaks_detected": 0,
    }
    if n < int(fs * 1.5):
        result.update({
            "n_valid_ibi": 0, "hr_mean_bpm": np.nan, "rmssd_ms": np.nan, "sdrr_ms": np.nan,
            "prr50_pct": np.nan, "prr20_pct": np.nan,
            "lf_power": np.nan, "hf_power": np.nan, "lf_hf_ratio": np.nan, "freq_domain_duration_s": 0.0,
        })
        return result

    filtered = filter_ppg(raw_ppg, fs)
    peaks = detect_peaks(filtered, fs)
    result["n_peaks_detected"] = int(len(peaks))

    ibi_s, ibi_t = compute_ibi(peaks, fs)
    result.update(time_domain_hrv(ibi_s))
    result.update(freq_domain_hrv(ibi_s, ibi_t))
    return result


def split_windows(start_s: float, end_s: float, window_s: float = 120.0) -> list[Tuple[float, float]]:
    """
    Chia [start_s, end_s) thành các cửa sổ liên tiếp dài `window_s` giây
    (mặc định 120s = 2 phút), dùng để theo dõi HRV biến đổi NHƯ THẾ NÀO
    trong lúc thiền thay vì chỉ 1 con số trung bình cho cả MEDITATION.

    Cửa sổ CUỐI có thể ngắn hơn `window_s` (phần dư của phép chia) -- vẫn
    giữ lại (không bỏ), độ dài thực tế phản ánh qua chính (end - start) của
    tuple trả về; `analyze_segment()` gọi sau đó tự đánh giá đủ dài hay
    không qua `hrv_quality_flag` (SHORT_SEGMENT nếu <30s) -- không cần lọc
    trước ở đây.

    Trả về [] nếu đoạn ngắn hơn `window_s` (không đủ cắt dù chỉ 1 cửa sổ
    trọn vẹn) -- caller nên fallback về phân tích nguyên khối MEDITATION.
    """
    duration = end_s - start_s
    if duration < window_s:
        return []
    windows = []
    t = start_s
    while t < end_s:
        w_end = min(t + window_s, end_s)
        windows.append((t, w_end))
        t += window_s
    return windows
