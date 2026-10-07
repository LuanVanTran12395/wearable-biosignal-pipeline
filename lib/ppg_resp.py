"""
Trích xuất nhịp thở từ PPG raw (PPG-derived respiration, PDR) -- dùng phương
pháp "baseline wander" (BW): hô hấp điều biến biên độ/baseline của sóng PPG
qua áp lực lồng ngực thay đổi tuần hoàn tĩnh mạch, tạo dao động tần số thấp
(~0.1-0.5Hz) chồng lên nhịp mạch tâm thu (~0.8-3Hz). Cùng input raw PPG count
100Hz (ppg_raw.edf) và cùng phong cách/quy ước với lib/ppg_hrv.py.

================================================================================
CÔNG THỨC & CITATION
================================================================================

1) Băng lọc hô hấp -- `filter_resp_band()`
---------------------------------------------
Bandpass Butterworth bậc 4, 0.1-0.5Hz (zero-phase, sosfiltfilt) -- khớp dải
nhịp thở người lớn khi nghỉ/thiền 6-30 lần/phút (0.1-0.5 Hz). Cách ly thành
phần baseline wander do hô hấp khỏi nhịp mạch tâm thu (0.5-8Hz, xem
`ppg_hrv.filter_ppg`) và trôi baseline chậm hơn (motion/DC drift, <0.1Hz).

2) Xác định đỉnh hô hấp (breath) -- `detect_breaths()`
---------------------------------------------------------
`scipy.signal.find_peaks` trên tín hiệu đã lọc, khoảng cách tối thiểu 2.0s
(giới hạn <=30 nhịp thở/phút) -- không cần prominence thích nghi phức tạp
như PPG vì biên độ baseline wander tương đối đồng đều sau bandpass hẹp.

3) BBI (breath-to-breath interval) -- `compute_bbi()`
---------------------------------------------------------
BBI = khoảng cách thời gian giữa 2 đỉnh hô hấp liên tiếp (giây), lọc theo
khoảng sinh lý hợp lệ [MIN_BBI_S, MAX_BBI_S] = [2.0, 10.0]s (~6-30 nhịp/phút).

4) Chỉ số hô hấp -- `resp_time_domain()`
---------------------------------------------
    resp_rate_bpm       = 60 / mean(BBI_giây)                 -- nhịp thở/phút
    resp_bbi_cv          = std(BBI) / mean(BBI)                -- biến thiên
                            breath-to-breath (thấp = đều đặn hơn)

QUAN TRỌNG -- đây là module MỚI, CHƯA được validate lâm sàng trên bộ dữ liệu
này (không có cảm biến hô hấp tham chiếu để đối chiếu độ chính xác). Baseline
wander PPG là phương pháp phổ biến trong y văn nhưng độ tin cậy phụ thuộc
nhiều vào chất lượng tiếp xúc cảm biến -- dùng cảm biến tiêu dùng (không
phải oximeter lâm sàng) nên nhiễu chuyển động có thể trội hơn tín hiệu hô
hấp thật. Đọc mọi kết quả từ module này như ƯỚC LƯỢNG THĂM DÒ, không phải
phép đo đã kiểm định.

Citation:
- Charlton PH, Bonnici T, Tarassenko L, Clifton DA, Beale R, Watkinson PJ.
  "An assessment of algorithms to estimate respiratory rate from the
  electrocardiogram and photoplethysmogram." Physiol Meas. 2016;37(4):610-626.
  (Đánh giá tổng quan các phương pháp PDR, gồm baseline-wander/amplitude-
  modulation/frequency-modulation; băng tần hô hấp người lớn dùng ở đây nằm
  trong dải khuyến nghị của bài báo.)
- Meredith DJ, Clifton D, Charlton P, Brooks J, Pugh CW, Tarassenko L.
  "Photoplethysmographic derivation of respiratory rate: a review of
  relevant physiology." J Med Eng Technol. 2012;36(1):1-7.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

PPG_FS_DEFAULT = 100.0

RESP_BAND = (0.1, 0.5)  # Hz, ~6-30 breaths/min

MIN_BBI_S = 2.0   # 30 breaths/min
MAX_BBI_S = 10.0  # 6 breaths/min

MIN_DURATION_FOR_RESP_S = 30.0  # cần >=1 chu kỳ đầy đủ ở tần số thấp nhất (0.1Hz = 10s), lấy dư ra cho ổn định


def filter_resp_band(raw_ppg: np.ndarray, fs: float = PPG_FS_DEFAULT, order: int = 4) -> np.ndarray:
    x = np.asarray(raw_ppg, dtype=float)
    if np.isnan(x).any():
        median = np.nanmedian(x)
        x = np.where(np.isnan(x), median if np.isfinite(median) else 0.0, x)
    nyq = fs / 2.0
    sos = butter(order, [RESP_BAND[0] / nyq, RESP_BAND[1] / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def detect_breaths(filtered: np.ndarray, fs: float = PPG_FS_DEFAULT) -> np.ndarray:
    peaks, _ = find_peaks(filtered, distance=int(MIN_BBI_S * fs))
    return peaks


def compute_bbi(breaths: np.ndarray, fs: float = PPG_FS_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    if len(breaths) < 2:
        return np.array([]), np.array([])
    bbi = np.diff(breaths) / fs
    t = breaths[1:] / fs
    valid = (bbi >= MIN_BBI_S) & (bbi <= MAX_BBI_S)
    return bbi[valid], t[valid]


def resp_time_domain(bbi_s: np.ndarray) -> Dict[str, float]:
    bbi_s = np.asarray(bbi_s, dtype=float)
    if len(bbi_s) < 2:
        return {"n_valid_breaths": int(len(bbi_s)), "resp_rate_bpm": np.nan, "resp_bbi_cv": np.nan}
    return {
        "n_valid_breaths": int(len(bbi_s)),
        "resp_rate_bpm": float(60.0 / np.mean(bbi_s)),
        "resp_bbi_cv": float(np.std(bbi_s, ddof=1) / np.mean(bbi_s)) if np.mean(bbi_s) > 0 else np.nan,
    }


def analyze_segment_resp(raw_ppg: np.ndarray, fs: float = PPG_FS_DEFAULT) -> Dict[str, float]:
    """Chạy full pipeline (lọc band hô hấp -> đỉnh -> BBI -> resp rate/CV) cho 1 đoạn PPG raw."""
    n = len(raw_ppg)
    result: Dict[str, float] = {"segment_duration_s": float(n / fs) if fs else np.nan, "n_breaths_detected": 0}
    if n < int(fs * MIN_DURATION_FOR_RESP_S):
        result.update({"n_valid_breaths": 0, "resp_rate_bpm": np.nan, "resp_bbi_cv": np.nan})
        return result

    filtered = filter_resp_band(raw_ppg, fs)
    breaths = detect_breaths(filtered, fs)
    result["n_breaths_detected"] = int(len(breaths))

    bbi_s, _ = compute_bbi(breaths, fs)
    result.update(resp_time_domain(bbi_s))
    return result
