"""
Tính chỉ số tổng hợp từ ΔHbO/ΔHbR/ΔHbT (đã tính bằng lib/fnirs_hbo_hbr.py) --
tương tự lib/ppg_hrv.py về vai trò (chỉ số/segment cho batch notebook), nhưng
khác cách tiếp cận: fNIRS không có nhịp rời rạc như PPG (không "đếm đỉnh tim"
được), tín hiệu là xu hướng huyết động biến đổi chậm. Vì vậy chỉ số ở đây là:

- Miền thời gian: mean / SD / slope (xu hướng tăng-giảm theo thời gian) /
  peak-to-peak cho từng kênh HbO, HbR, HbT.
- Miền tần số: 1 chỉ số đơn giản -- tỉ lệ công suất dải rất thấp (vasomotion,
  0.02-0.1Hz) trên tổng công suất dải thấp (0.02-0.5Hz) của HbO, cần đoạn đủ
  dài (mặc định >=120s) mới ước lượng được, ngắn hơn trả NaN.

Input mong đợi: mảng ΔHbO/ΔHbR/ΔHbT (µM, đã trừ baseline) lấy mẫu ở
OPTICAL_FS = 100Hz -- đúng quy ước fnirs_hbo_hbr.csv do Layer 2 xuất.

================================================================================
CÔNG THỨC & CITATION
================================================================================

1) Lọc lowpass trước khi tính chỉ số -- `smooth_hemodynamic()`
------------------------------------------------------------------
Butterworth lowpass bậc 3, cutoff mặc định 0.2Hz, áp dụng zero-phase
(`scipy.signal.sosfiltfilt`, lọc xuôi+ngược nên không lệch pha/không trễ).
Mục đích: đáp ứng huyết động (hemodynamic response) thực sự nằm ở dải
<0.1-0.2Hz; thành phần tim mạch (~0.8-2Hz) và hô hấp (~0.15-0.4Hz) là nhiễu
"leak" vào tín hiệu MBLL cần loại trước khi tính mean/SD/slope, tương tự
bước tiền xử lý chuẩn trong các pipeline fNIRS (bandpass/lowpass trước phân
tích task/rest), xem:
- Pinti P, Scholkmann F, Hamilton A, Burgess P, Tachtsidis I. "Current
  status and issues regarding pre-processing of fNIRS neuroimaging data:
  an investigation of diverse signal filtering methods within a general
  linear model framework." Front Hum Neurosci. 2019;12:505.

2) Miền thời gian -- `time_domain_metrics()`, `linear_slope_per_min()`
------------------------------------------------------------------------
Không phải chỉ số fNIRS "chuẩn hoá" riêng biệt nào, mà là các mô tả thống kê
mô tả cơ bản thường dùng để tóm tắt đáp ứng huyết động theo block/phase
trong nghiên cứu fNIRS (mean amplitude, biến thiên, xu hướng tuyến tính,
peak response) -- xem cách tiếp cận tương tự (mean amplitude + slope theo
block) trong:
- Herold F, Wiegel P, Scholkmann F, Müller NG. "Applications of Functional
  Near-Infrared Spectroscopy (fNIRS) Neuroimaging in Exercise-Cognition
  Science: A Systematic, Methodology-Focused Review." J Clin Med.
  2018;7(12):466.

    slope (µM/phút) = hệ số góc hồi quy tuyến tính bậc 1 (least-squares,
    `numpy.polyfit`) của giá trị theo thời gian (phút) trong đoạn đang xét.

3) Miền tần số -- `low_freq_power_ratio()`
------------------------------------------------------------------------
fNIRS/NIRS ghi nhận nhiều dải dao động tự phát sinh lý chồng lên đáp ứng
huyết động, phổ biến chia thành:
- VLF/vasomotion:  ~0.02-0.1  Hz (hoạt động cơ trơn mạch máu)
- LF/Mayer waves:  ~0.1-0.15  Hz (điều hoà giao cảm)
- Hô hấp:          ~0.15-0.4  Hz
- Tim mạch:        ~0.8-2     Hz

`low_freq_power_ratio()` tính PSD bằng Welch (Hann, cửa sổ tới 120s, chồng
lấn 50%) trên HbO đã lọc, rồi lấy tỉ lệ công suất trong LOW_FREQ_BAND
(0.02-0.1Hz) trên TOTAL_LOW_BAND (0.02-0.5Hz, gồm cả vasomotion+Mayer+hô
hấp) làm chỉ số đơn giản mô tả tỉ trọng dao động rất chậm so với dao động
chậm nói chung -- lấy cảm hứng từ phân tích dao động tự phát tần số thấp
trong:
- Obrig H, Neufang M, Wenzel R, Kohl M, Steinbrink J, Einhäupl K,
  Villringer A. "Spontaneous low frequency oscillations of cerebral
  hemodynamics and metabolism in human adults." Neuroimage.
  2000;12(6):623-639.
- Tachtsidis I, Scholkmann F. "False positives and false negatives in
  functional near-infrared spectroscopy: issues, challenges, and the way
  forward." Neurophotonics. 2016;3(3):031405.

  Đây KHÔNG phải 1 chỉ số chuẩn hoá có tên riêng trong y văn (khác LF/HF
  của HRV, vốn có băng tần chuẩn hoá bởi Task Force 1996) -- coi là chỉ số
  thăm dò (exploratory), cần diễn giải thận trọng.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

OPTICAL_FS_DEFAULT = 100.0

LOW_FREQ_BAND = (0.02, 0.10)     # vasomotion
TOTAL_LOW_BAND = (0.02, 0.50)    # dải tham chiếu để tính tỉ lệ
MIN_DURATION_FOR_FREQ_DOMAIN_S = 120.0


def smooth_hemodynamic(x: np.ndarray, fs: float = OPTICAL_FS_DEFAULT, cutoff: float = 0.2, order: int = 3) -> np.ndarray:
    """Lowpass <cutoff Hz -- lọc bớt nhiễu tim mạch/hô hấp/vận động còn sót
    lại trên tín hiệu MBLL, giữ lại xu hướng huyết động chậm (đúng dải quan
    tâm khi phân tích theo phase dài chục giây tới vài phút)."""
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        median = np.nanmedian(x)
        x = np.where(np.isnan(x), median if np.isfinite(median) else 0.0, x)
    nyq = fs / 2.0
    sos = butter(order, cutoff / nyq, btype="lowpass", output="sos")
    return sosfiltfilt(sos, x)


def linear_slope_per_min(values: np.ndarray, fs: float = OPTICAL_FS_DEFAULT) -> float:
    """Hệ số góc hồi quy tuyến tính theo thời gian (µM/phút) -- dương = đang
    tăng theo thời gian trong đoạn này, âm = đang giảm."""
    n = len(values)
    if n < 2:
        return np.nan
    t_min = np.arange(n) / fs / 60.0
    slope, _ = np.polyfit(t_min, values, 1)
    return float(slope)


def time_domain_metrics(values: np.ndarray, fs: float, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return {
            f"{prefix}_mean_uM": np.nan, f"{prefix}_sd_uM": np.nan,
            f"{prefix}_slope_uM_per_min": np.nan, f"{prefix}_peak_to_peak_uM": np.nan,
            f"{prefix}_max_uM": np.nan, f"{prefix}_min_uM": np.nan,
        }
    return {
        f"{prefix}_mean_uM": float(np.mean(values)),
        f"{prefix}_sd_uM": float(np.std(values, ddof=1)),
        f"{prefix}_slope_uM_per_min": linear_slope_per_min(values, fs),
        f"{prefix}_peak_to_peak_uM": float(np.ptp(values)),
        f"{prefix}_max_uM": float(np.max(values)),
        f"{prefix}_min_uM": float(np.min(values)),
    }


def low_freq_power_ratio(values: np.ndarray, fs: float = OPTICAL_FS_DEFAULT) -> Dict[str, float]:
    """Welch PSD trên HbO đã lọc -> tỉ lệ công suất vasomotion (0.02-0.1Hz)
    trên tổng công suất dải thấp (0.02-0.5Hz). Cần đủ dài mới đáng tin."""
    n = len(values)
    duration = n / fs if fs else 0.0
    if duration < MIN_DURATION_FOR_FREQ_DOMAIN_S:
        return {"low_freq_power_ratio": np.nan, "freq_domain_duration_s": duration}

    nperseg = min(n, int(fs * 120))
    if nperseg < 8:
        return {"low_freq_power_ratio": np.nan, "freq_domain_duration_s": duration}

    freqs, psd = welch(values - np.mean(values), fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2)

    def band_power(lo: float, hi: float) -> float:
        idx = (freqs >= lo) & (freqs <= hi)
        return float(np.trapz(psd[idx], freqs[idx])) if idx.any() else np.nan

    low = band_power(*LOW_FREQ_BAND)
    total = band_power(*TOTAL_LOW_BAND)
    ratio = float(low / total) if total and total > 0 and np.isfinite(low) else np.nan
    return {"low_freq_power_ratio": ratio, "freq_domain_duration_s": duration}


def analyze_segment(
    hbo: np.ndarray,
    hbr: np.ndarray,
    hbt: Optional[np.ndarray] = None,
    fs: float = OPTICAL_FS_DEFAULT,
) -> Dict[str, float]:
    """Chạy full pipeline (lọc -> mean/SD/slope/peak HbO+HbR(+HbT) -> tỉ lệ
    công suất thấp của HbO) cho 1 đoạn ΔHbO/ΔHbR(/ΔHbT, µM)."""
    n = len(hbo)
    result: Dict[str, float] = {"segment_duration_s": float(n / fs) if fs else np.nan, "n_samples": int(n)}
    if n < int(fs * 2):
        result.update(time_domain_metrics(np.array([]), fs, "hbo"))
        result.update(time_domain_metrics(np.array([]), fs, "hbr"))
        if hbt is not None:
            result.update(time_domain_metrics(np.array([]), fs, "hbt"))
        result.update({"low_freq_power_ratio": np.nan, "freq_domain_duration_s": 0.0})
        return result

    hbo_smooth = smooth_hemodynamic(hbo, fs)
    hbr_smooth = smooth_hemodynamic(hbr, fs)
    result.update(time_domain_metrics(hbo_smooth, fs, "hbo"))
    result.update(time_domain_metrics(hbr_smooth, fs, "hbr"))
    if hbt is not None:
        hbt_smooth = smooth_hemodynamic(hbt, fs)
        result.update(time_domain_metrics(hbt_smooth, fs, "hbt"))
    result.update(low_freq_power_ratio(hbo_smooth, fs))
    return result
