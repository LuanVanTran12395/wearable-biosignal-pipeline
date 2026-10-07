"""
Đọc báo cáo QC của HAPPE (`notebooks/eeg_happe_qc.ipynb`, xem
`happe-python/`) và gắn cờ chất lượng CẤP SESSION -- dùng để join vào
các bảng phân tích fNIRS/PPG (join theo `session_id`) làm biến hiệp phương
sai/lọc độ nhạy, KHÔNG dùng để tự động xoá session khỏi output nào.

Ngưỡng dùng ở đây (700s/1100s thời lượng, 10% variance retained) là ngưỡng
DỮ LIỆU-DẪN-XUẤT (data-driven, từ phân vị của các session đã chạy HAPPE),
KHÔNG phải ngưỡng chính thức HAPPE công bố -- cần tính lại cho dataset khác.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TYPICAL_LENGTH_MIN_S = 700.0
TYPICAL_LENGTH_MAX_S = 1100.0
MIN_PCT_VAR_RETAINED = 10.0


def compute_happe_quality_flags(row: pd.Series) -> str:
    """Cờ chất lượng session dựa trên báo cáo QC của HAPPE (không xoá dòng
    nào -- chỉ đánh dấu để lọc/kiểm định độ nhạy ở bước phân tích sau)."""
    flags = []
    length_s = row.get("File_Length_s")
    pct_var = row.get("Pct_Var_Retained_PostWavelet")

    if pd.notna(length_s):
        if length_s < TYPICAL_LENGTH_MIN_S:
            flags.append("HAPPE_TRUNCATED_RECORDING")
        elif length_s > TYPICAL_LENGTH_MAX_S:
            flags.append("HAPPE_ANOMALOUS_LONG_RECORDING")

    if pd.notna(pct_var) and pct_var < MIN_PCT_VAR_RETAINED:
        flags.append("HAPPE_LOW_VARIANCE_RETAINED")

    return ";".join(flags)


def load_happe_quality_table(happe_qc_dir: Path) -> pd.DataFrame:
    """
    Đọc `happe_dataQC_with_identity.csv` (xuất bởi `eeg_happe_qc.ipynb`) và
    trả về bảng CẤP SESSION gồm:
      session_id, happe_file_length_s, happe_pct_var_retained_postwavelet,
      happe_pct_good_channels, happe_qc_flags, happe_qc_pass

    `happe_qc_pass = True` khi KHÔNG có cờ nào ở trên (thời lượng bình
    thường + variance retained đạt ngưỡng tối thiểu). Trả về DataFrame rỗng
    (đúng schema cột) nếu chưa chạy `eeg_happe_qc.ipynb` -- caller tự quyết
    định left-join rồi coi thiếu dữ liệu HAPPE là "chưa đánh giá", không
    phải "không đạt".
    """
    path = Path(happe_qc_dir) / "happe_dataQC_with_identity.csv"
    cols = ["session_id", "happe_file_length_s", "happe_pct_var_retained_postwavelet",
            "happe_pct_good_channels", "happe_qc_flags", "happe_qc_pass"]
    if not path.exists():
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(path)
    df["happe_qc_flags"] = df.apply(compute_happe_quality_flags, axis=1)
    df["happe_qc_pass"] = df["happe_qc_flags"] == ""

    out = df.rename(columns={
        "File_Length_s": "happe_file_length_s",
        "Pct_Var_Retained_PostWavelet": "happe_pct_var_retained_postwavelet",
        "Pct_Good_Channels": "happe_pct_good_channels",
    })[cols]
    return out
