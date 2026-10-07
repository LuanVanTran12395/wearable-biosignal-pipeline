"""
Chuyển đổi EEG ADC count (thiết bị wearable, kênh AF3/AF4) sang đơn vị điện
thế chuẩn -- dùng chung cho mọi notebook/script cần đọc EEG raw ra µV.

Công thức (ADC N-bit differential, reference +-VREF):
  ADC_MIDPOINT = 2**(N-1)
  EEG_uV = (ADC - ADC_MIDPOINT) * UV_PER_COUNT
  UV_PER_COUNT = 1e6 * VREF / ADC_MIDPOINT / 2

Giá trị mặc định dưới đây là placeholder -- đặt đúng theo datasheet thiết bị
của bạn qua biến môi trường EEG_ADC_BITS / EEG_VREF_V hoặc `configure_adc()`.
"""
from __future__ import annotations

import os

import numpy as np

ADC_BITS = int(os.environ.get("EEG_ADC_BITS", "24"))
VREF_V = float(os.environ.get("EEG_VREF_V", "2.4"))
ADC_MIDPOINT = float(2 ** (ADC_BITS - 1))
UV_PER_COUNT = 1_000_000.0 * VREF_V / ADC_MIDPOINT / 2.0


def configure_adc(adc_bits: int, vref_v: float) -> None:
    """Đặt lại thông số ADC cho thiết bị cụ thể (gọi trước khi import module khác
    đọc ADC_MIDPOINT/UV_PER_COUNT ở cấp module, vd signal_qc)."""
    global ADC_BITS, VREF_V, ADC_MIDPOINT, UV_PER_COUNT
    ADC_BITS, VREF_V = int(adc_bits), float(vref_v)
    ADC_MIDPOINT = float(2 ** (ADC_BITS - 1))
    UV_PER_COUNT = 1_000_000.0 * VREF_V / ADC_MIDPOINT / 2.0


def adc_to_uv(adc: np.ndarray) -> np.ndarray:
    """ADC count (raw) -> microvolt (µV)."""
    adc = np.asarray(adc, dtype=np.float64)
    return (adc - ADC_MIDPOINT) * UV_PER_COUNT


def adc_to_volts(adc: np.ndarray) -> np.ndarray:
    """ADC count (raw) -> Volt (V) -- đơn vị SI mà MNE dùng nội bộ cho kênh ch_type='eeg'."""
    return adc_to_uv(adc) * 1e-6


def scale_to_uv(x: np.ndarray) -> np.ndarray:
    """
    Quy đổi biên độ sang µV CHỈ bằng hệ số scale, KHÔNG trừ ADC_MIDPOINT.

    Dùng cho tín hiệu đã lọc/detrend (AC-coupled): DC offset đã bị filter
    loại bỏ nên trừ thêm ADC_MIDPOINT sẽ sai (dịch cả tín hiệu ra khỏi 0).
    Với dữ liệu ADC thô (còn nguyên DC offset), dùng `adc_to_uv()` ở trên.
    """
    return np.asarray(x, dtype=np.float64) * UV_PER_COUNT
