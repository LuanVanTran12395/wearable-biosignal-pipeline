#!/usr/bin/env python3
"""
Xuất file .fif (MNE Raw đã gắn annotation phase) sang
1 file JSON "sidecar" để một signal viewer HTML đọc trực tiếp bằng
JavaScript trong trình duyệt.

Vì sao cần bước này: .fif là định dạng nhị phân riêng của MNE, trình duyệt
không đọc được. Exporter này + viewer HTML là MỘT cặp bắt buộc đi cùng nhau:
chạy exporter 1 lần cho mỗi session (hoặc để dashboard tự gọi), sau đó mở
file JSON sinh ra bằng viewer.

Sidecar JSON gồm:
- meta: channel names, sampling rate, số sample, duration, đơn vị (uV)
- channel_data: mỗi kênh 1 chuỗi base64 của mảng Float32 nhị phân (gọn hơn
  nhiều so với ghi số thập phân dạng text trong JSON, ~5.3 byte/sample thay
  vì ~10-15 byte/sample)
- annotations: onset/duration/description (giây, tương đối đầu file) --
  REST_BEFORE/MEDITATION/REST_AFTER...

Dùng:
    python lib/fif_to_json_sidecar.py <fif_path> [-o out.json]
    python lib/fif_to_json_sidecar.py <fif_path> --uv-per-count ...  (không cần,
        .fif build_labeled_raw() đã lưu sẵn đơn vị Volt chuẩn MNE)
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import mne
import numpy as np


def export_sidecar(fif_path: Path, out_path: Path) -> Path:
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)

    data_v = raw.get_data()  # (n_channels, n_samples), Volt (quy ước MNE)
    data_uv = (data_v * 1e6).astype(np.float32)  # -> microvolt, float32 cho gọn

    channel_data = {
        name: base64.b64encode(data_uv[i].tobytes()).decode("ascii")
        for i, name in enumerate(ch_names)
    }

    annotations = [
        {"onset": float(onset), "duration": float(duration), "description": str(description)}
        for onset, duration, description in zip(
            raw.annotations.onset, raw.annotations.duration, raw.annotations.description
        )
    ]

    sidecar = {
        "meta": {
            "source_file": fif_path.name,
            "channels": ch_names,
            "sfreq": sfreq,
            "n_samples": int(data_uv.shape[1]),
            "duration_seconds": float(data_uv.shape[1] / sfreq) if sfreq else 0.0,
            "unit": "uV",
            "dtype": "float32",
            "encoding": "base64",
        },
        "channel_data": channel_data,
        "annotations": annotations,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sidecar, f)

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fif_path", type=Path, help="Đường dẫn file .fif (vd eeg_labeled_raw.fif)")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Đường dẫn JSON output (mặc định: cùng tên, đuôi .sidecar.json)")
    args = parser.parse_args()

    if not args.fif_path.exists():
        print(f"Không tìm thấy file: {args.fif_path}", file=sys.stderr)
        return 1

    out_path = args.out or args.fif_path.with_name(args.fif_path.stem + ".sidecar.json")
    result = export_sidecar(args.fif_path, out_path)
    size_mb = result.stat().st_size / 1e6
    print(f"Đã lưu sidecar: {result} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
