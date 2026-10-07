"""Feature động học trong-phase (within-session dynamics) -- khác các feature khác trong
project (vốn là trung bình cả 1 phase, 1 số/session/phase), các hàm ở đây trích feature theo
DÒNG THỜI GIAN trong lúc thiền, trực quan hơn cho người dùng cuối:

- `detect_relaxation_onset()`: bao lâu (giây) kể từ lúc bắt đầu MEDITATION thì alpha power
  vượt và giữ trên baseline REST_BEFORE cùng buổi.
- `compute_distraction_gaps()`: đếm khoảng trống giữa các epoch còn lại sau segment
  rejection (mỗi gap = 1+ epoch bị loại vì artifact biên độ, thường do cử động/chớp mắt
  mạnh) -- dùng lại chính log rejection đã có sẵn từ `eeg_qc.ipynb`, không cần QC lại.
- `detect_drowsy_episodes()`: chuỗi epoch liên tục dài nhất mà theta/alpha ratio vượt 1 bội số
  của baseline REST_BEFORE cùng buổi -- bắt được cả episode ngủ gật NGẮN giữa buổi (không cần
  kéo dài liên tục tới hết như `detect_relaxation_onset`). Vẫn ở mức THỬ
  NGHIỆM (chưa có ground-truth đầy đủ để tính độ nhạy/đặc hiệu chính xác).

Các hàm không phụ thuộc MNE trực tiếp (nhận numpy array/list đã trích sẵn) để dễ test và dùng
lại ở nơi khác; `epoch_phase_labels()` là phần duy nhất cần đối tượng `mne.Epochs`.
"""

from __future__ import annotations

import numpy as np


def epoch_phase_labels(epochs) -> list[str]:
    """1 nhãn phase / epoch -- lấy annotation có duration lớn nhất đè lên epoch đó
    (loại các point-event duration=0 như SESSION_START/PRECHECK_SUBMITTED...).

    Giống hệt hàm cùng tên định nghĩa inline trong
    `notebooks/eeg_feature_extraction.ipynb` -- tách ra đây để dùng lại ở
    notebook phân tích động học trong-phiên mà không phải copy-paste.
    """
    per_epoch = epochs.get_annotations_per_epoch()
    labels = []
    for anns in per_epoch:
        candidates = [(dur, desc) for (_onset, dur, desc) in anns if dur > 0]
        if not candidates:
            labels.append("UNLABELED")
            continue
        candidates.sort(key=lambda t: t[0], reverse=True)
        labels.append(candidates[0][1])
    return labels


def detect_relaxation_onset(
    values: np.ndarray,
    baseline: float,
    epoch_sec: float = 2.0,
    smooth_win: int = 3,
    sustain_n: int = 3,
) -> dict:
    """Tìm epoch đầu tiên (theo thứ tự thời gian trong phase) mà `values` (đã rolling-median
    làm mượt) vượt `baseline` VÀ giữ được ở `sustain_n` epoch liên tiếp -- tránh bắt nhầm dao
    động ngẫu nhiên ngắn.

    values:    mảng 1 giá trị/epoch (vd alpha_power_rel trung bình 2 kênh), ĐÃ SẮP XẾP theo
               thời gian trong phase (epoch_idx tăng dần).
    baseline:  ngưỡng so sánh -- thường là median cùng feature ở phase REST_BEFORE cùng buổi.
    epoch_sec: độ dài 1 epoch (giây) -- dùng quy đổi epoch -> giây.
    smooth_win: cửa sổ rolling median (số epoch) để giảm nhiễu trước khi so ngưỡng.
    sustain_n: số epoch liên tiếp phải giữ trên ngưỡng mới tính là "đã đạt".

    Trả về dict: onset_epoch_idx (None nếu không đạt), onset_latency_sec, onset_reached,
    onset_pct_of_session (vị trí đạt ngưỡng / tổng độ dài phase, None nếu không đạt),
    n_epoch (tổng số epoch trong phase), duration_sec.
    """
    values = np.asarray(values, dtype=float)
    n_epoch = len(values)
    duration_sec = n_epoch * epoch_sec

    if n_epoch < smooth_win + sustain_n:
        return {
            "onset_epoch_idx": None, "onset_latency_sec": np.nan, "onset_reached": False,
            "onset_pct_of_session": np.nan, "n_epoch": n_epoch, "duration_sec": duration_sec,
        }

    smoothed = (
        np.convolve(values, np.ones(smooth_win) / smooth_win, mode="same")
        if smooth_win > 1 else values
    )

    onset_idx = None
    for i in range(len(smoothed) - sustain_n + 1):
        if np.all(smoothed[i:i + sustain_n] >= baseline):
            onset_idx = i
            break

    return {
        "onset_epoch_idx": onset_idx,
        "onset_latency_sec": onset_idx * epoch_sec if onset_idx is not None else np.nan,
        "onset_reached": onset_idx is not None,
        "onset_pct_of_session": (onset_idx * epoch_sec / duration_sec) if onset_idx is not None and duration_sec > 0 else np.nan,
        "n_epoch": n_epoch, "duration_sec": duration_sec,
    }


def detect_drowsy_episodes(
    values: np.ndarray,
    baseline: float,
    epoch_sec: float = 2.0,
    smooth_win: int = 3,
    rel_multiplier: float = 2.0,
) -> dict:
    """Chuỗi epoch liên tục dài nhất mà `values` (rolling-mean làm mượt) vượt
    `baseline * rel_multiplier` -- khác `detect_relaxation_onset` ở chỗ không yêu cầu giữ
    ngưỡng tới hết phase, chỉ cần 1 đoạn liên tục bất kỳ (bắt episode ngắn giữa buổi).

    values:         mảng 1 giá trị/epoch (vd theta_alpha_ratio trung bình 2 kênh), ĐÃ SẮP XẾP
                    theo thời gian trong phase.
    baseline:       median cùng feature ở phase REST_BEFORE cùng buổi -- ngưỡng tính theo
                    TỈ LỆ với baseline riêng từng người (không dùng ngưỡng tuyệt đối cố định --
                    đã thử và gây âm tính giả với người có baseline thấp sẵn, xem notebook
                    trial dẫn ở docstring module).
    rel_multiplier: hệ số nhân baseline để thành ngưỡng (mặc định 2.0 -- cố tình chọn LỎNG,
                    ưu tiên độ nhạy hơn độ đặc hiệu ở giai đoạn thăm dò).

    Trả về dict: threshold, drowsy_mask (mảng bool cùng độ dài `values`), pct_epochs_drowsy,
    longest_run_epoch_idx (vị trí bắt đầu chuỗi dài nhất, None nếu không có), longest_run_sec,
    n_epoch, duration_sec.
    """
    values = np.asarray(values, dtype=float)
    n_epoch = len(values)
    duration_sec = n_epoch * epoch_sec
    threshold = baseline * rel_multiplier

    if n_epoch < smooth_win:
        return {
            "threshold": threshold, "drowsy_mask": np.zeros(n_epoch, dtype=bool),
            "pct_epochs_drowsy": np.nan, "longest_run_epoch_idx": None, "longest_run_sec": 0.0,
            "n_epoch": n_epoch, "duration_sec": duration_sec,
        }

    smoothed = (
        np.convolve(values, np.ones(smooth_win) / smooth_win, mode="same")
        if smooth_win > 1 else values
    )
    drowsy_mask = smoothed > threshold

    best_len, best_start, cur_len, cur_start = 0, None, 0, None
    for i, v in enumerate(drowsy_mask):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0

    return {
        "threshold": float(threshold), "drowsy_mask": drowsy_mask,
        "pct_epochs_drowsy": float(drowsy_mask.mean() * 100),
        "longest_run_epoch_idx": best_start, "longest_run_sec": best_len * epoch_sec,
        "n_epoch": n_epoch, "duration_sec": duration_sec,
    }


def compute_distraction_gaps(
    sample_numbers: np.ndarray,
    sfreq: float,
    epoch_sec: float = 2.0,
    gap_tolerance: float = 1.5,
) -> dict:
    """Đếm khoảng trống (gap) giữa các epoch CÒN LẠI (sau segment rejection ở
    `eeg_qc.ipynb`) trong 1 phase -- epoch cách nhau xa hơn `gap_tolerance` lần khoảng cách
    chuẩn (không chồng lấn) nghĩa là có epoch bị loại ở giữa vì artifact biên độ.

    sample_numbers: mảng sample number gốc của từng epoch còn lại trong phase (từ
                    `epochs.events[:, 0]`, lọc theo phase trước khi gọi hàm này), KHÔNG cần
                    sắp xếp trước (hàm tự sort).
    sfreq:          tần số lấy mẫu (Hz) của file gốc (`epochs.info["sfreq"]`).
    epoch_sec:      độ dài 1 epoch (giây).
    gap_tolerance:  hệ số nhân với khoảng cách chuẩn để coi là "có gap" (1.5 = chấp nhận
                    jitter nhỏ do làm tròn sample, chỉ tính gap khi rõ ràng thiếu epoch).

    Trả về dict: n_epoch_kept, duration_sec, n_gaps, n_epoch_missing_est (tổng số epoch ước
    tính đã bị loại giữa các epoch còn lại), gaps_per_min.
    """
    sample_numbers = np.sort(np.asarray(sample_numbers))
    n_epoch_kept = len(sample_numbers)
    duration_sec = n_epoch_kept * epoch_sec

    if n_epoch_kept < 2:
        return {
            "n_epoch_kept": n_epoch_kept, "duration_sec": duration_sec,
            "n_gaps": 0, "n_epoch_missing_est": 0, "gaps_per_min": np.nan,
        }

    expected_step = round(epoch_sec * sfreq)
    diffs = np.diff(sample_numbers)
    gap_mask = diffs > expected_step * gap_tolerance
    n_gaps = int(gap_mask.sum())
    n_epoch_missing_est = (
        int(np.round(diffs[gap_mask] / expected_step).sum() - n_gaps) if n_gaps else 0
    )

    return {
        "n_epoch_kept": n_epoch_kept, "duration_sec": duration_sec,
        "n_gaps": n_gaps, "n_epoch_missing_est": max(n_epoch_missing_est, 0),
        "gaps_per_min": n_gaps / (duration_sec / 60) if duration_sec > 0 else np.nan,
    }
