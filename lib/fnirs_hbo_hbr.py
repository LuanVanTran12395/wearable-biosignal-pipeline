"""
Tinh HbO / HbR tu 2 kenh fNIRS raw bang Modified Beer-Lambert Law (MBLL).

THONG SO THIET BI (gia tri mac dinh cho headband fNIRS 2 buoc song, chinh
lai theo thiet bi thuc te):
- Kenh 1 (fnirs_1_raw) = buoc song 740 nm
- Kenh 2 (fnirs_2_raw) = buoc song 850 nm
- Khoang cach nguon-dau do L = 3.0 cm (mac dinh, chinh qua tham so l_cm)
- Vi tri: prefrontal cortex, fs = 100Hz (OPTICAL_FS cua Layer 2)
- Baseline I0 = trung binh cuong do trong `baseline_seconds` giay dau tien
  (gia dinh la giai doan nghi/on dinh dau session)

================================================================================
CONG THUC & CITATION
================================================================================

1) Modified Beer-Lambert Law (MBLL)
------------------------------------
Voi moi buoc song lambda, do hap thu quang (Optical Density) bien thien so
voi baseline la:

    OD_lambda(t) = -log10( I_lambda(t) / I0_lambda )

MBLL lien he OD voi bien thien nong do HbO/HbR:

    OD_lambda(t) = ( eps_HbO(lambda)*dHbO(t) + eps_HbR(lambda)*dHbR(t) ) * L * DPF(lambda)

Voi 2 buoc song (740nm, 850nm) ta co he 2 phuong trinh, 2 an (dHbO, dHbR):

    [OD_740]   [eps_HbO(740) eps_HbR(740)]     [dHbO]
    [       ] = [                        ] * L*[    ]
    [OD_850]   [eps_HbO(850) eps_HbR(850)]     [dHbR]

    => [dHbO; dHbR] = inv(EPS) @ [OD_740/(L*DPF_740); OD_850/(L*DPF_850)]

(code: `EPS_INV @ b`, voi `b` da chia san cho L*DPF tung buoc song)

Citation:
- Delpy DT, Cope M, van der Zee P, Arridge S, Wray S, Wyatt J. "Estimation of
  optical pathlength through tissue from direct time of flight measurement."
  Phys Med Biol. 1988;33(12):1433-1442.
- Cope M, Delpy DT. "System for long-term measurement of cerebral blood and
  tissue oxygenation on newborn infants by near infra-red transillumination."
  Med Biol Eng Comput. 1988;26(3):289-294.

2) DPF (Differential Pathlength Factor)
-----------------------------------------
DPF hieu chinh quang trinh thuc te (photon di theo duong zic-zac trong mo,
dai hon L hinh hoc) theo buoc song va tuoi:

    DPF(lambda, age) = alpha + beta*age^gamma + delta*lambda^3 + eps*lambda^2 + zeta*lambda

    alpha=223.3, beta=0.05624, gamma=0.8493,
    delta=-5.723e-7, eps=0.001245, zeta=-0.9025

    -> DPF(740nm, 28y) = 6.255,  DPF(850nm, 28y) = 5.177

age=28 dung lam mac dinh ("nguoi lon") vi khong co tuoi chinh xac tung
participant trong metadata hien tai -- xem `scholkmann_wolf_dpf()`.

Citation:
- Scholkmann F, Wolf M. "General equation for the differential pathlength
  factor of the frontal human head depending on wavelength and age."
  J Biomed Opt. 2013;18(10):105004.

3) He so hap thu mol (extinction coefficients)
-------------------------------------------------
Don vi mM^-1 cm^-1, tra theo buoc song tu bang tong hop cua Scott Prahl
(Oregon Medical Laser Center) -- xem hang so `_EPS`:

    740 nm: eps_HbO = 0.446,   eps_HbR = 1.11588
    850 nm: eps_HbO = 1.058,   eps_HbR = 0.69132

Citation:
- Prahl S. "Optical Absorption of Hemoglobin." Oregon Medical Laser Center,
  1999. https://omlc.org/spectra/hemoglobin/ -- du lieu goc tong hop tu:
  Wray S, Cope M, Delpy DT, Wyatt JS, Reynolds EO. "Characterization of the
  near infrared absorption spectra of cytochrome aa3 and haemoglobin for the
  non-invasive monitoring of cerebral oxygenation." Biochim Biophys Acta.
  1988;933(1):184-192.

4) Baseline / don vi ket qua
--------------------------------
I0 = trung binh cuong do trong `baseline_seconds` giay dau (mac dinh 10s,
gia dinh giai doan nghi/on dinh). HbO, HbR (va HbT = HbO+HbR) la BIEN THIEN
nong do SO VOI baseline (Delta), don vi micromol/L (uM) -- KHONG PHAI gia
tri tuyet doi. MBLL voi 1 khoang cach nguon-dau co dinh (continuous-wave,
don kenh) khong cho phep tinh nong do tuyet doi dang tin cay -- day la gioi
han da biet cua ky thuat nay, xem thao luan trong:

- Scholkmann F, Kleiser S, Metz AJ, Zimmermann R, Mata Pavia J, Wolf U,
  Wolf M. "A review on continuous wave functional near-infrared spectroscopy
  and imaging instrumentation and methodology." Neuroimage. 2014;85:6-27.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

L_CM_DEFAULT = 3.0
AGE_DEFAULT = 28
OPTICAL_FS_DEFAULT = 100
BASELINE_SECONDS_DEFAULT = 10.0

# [740nm, 850nm] x [HbO, HbR], mM^-1 cm^-1 (Prahl, omlc.org -- xem muc 3 o dau file).
_EPS = np.array([
    [0.446, 1.11588],
    [1.058, 0.69132],
])
_EPS_INV = np.linalg.inv(_EPS)


def scholkmann_wolf_dpf(lam_nm: float, age: float) -> float:
    """DPF theo Scholkmann & Wolf (2013) -- xem CONG THUC & CITATION muc 2 o dau file."""
    alpha, beta, gamma = 223.3, 0.05624, 0.8493
    delta, eps, zeta = -5.723e-7, 0.001245, -0.9025
    return alpha + beta * age**gamma + delta * lam_nm**3 + eps * lam_nm**2 + zeta * lam_nm


def raw_intensity_to_od(
    intensity: np.ndarray,
    reference: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """
    Quy đổi cường độ quang raw sang Optical Density (OD) -- bước 1 của MBLL
    (xem mục 1 ở đầu file): `OD(t) = -log10(I(t) / I_ref)`.

    `reference` là giá trị I_ref dùng làm mốc 0 cho OD (thường là baseline
    trung bình -- xem `compute_hbo_hbr()` dùng baseline_seconds đầu session,
    hoặc pipeline v2 trong lib/fnirs_pipeline_v2.py dùng mean/median toàn
    đoạn liên tục trước khi motion-correct). Nếu không truyền, mặc định
    dùng trung bình của chính `intensity` (mốc "OD tương đối trong đoạn").

    Trả về (OD, reference_value_used) để hàm gọi log lại I_ref đã dùng.
    """
    x = np.asarray(intensity, dtype=np.float64)
    ref = float(reference) if reference is not None else float(np.nanmean(x))
    ref = max(ref, 1e-6)
    x = np.clip(x, 1e-6, None)
    od = -np.log10(x / ref)
    return od, ref


def concentration_from_od(
    od_740: np.ndarray,
    od_850: np.ndarray,
    l_cm: float = L_CM_DEFAULT,
    age: float = AGE_DEFAULT,
) -> pd.DataFrame:
    """
    Bước 2 của MBLL (xem mục 1 ở đầu file): giải hệ 2 phương trình (2 bước
    sóng) để ra ΔHbO/ΔHbR (mM) từ OD đã tính sẵn (`raw_intensity_to_od()`
    hoặc OD đã qua motion-correction/TDDR ở nơi khác, vd
    lib/fnirs_pipeline_v2.py). Tách riêng khỏi bước tính OD để pipeline v2
    có thể chèn bước motion-correction ở GIỮA (trên OD) trước khi gọi hàm
    này -- `compute_hbo_hbr()` bên dưới gọi lại đúng hàm này cho pipeline
    đơn giản (không motion-correction), giữ nguyên hành vi cũ cho Layer 2.

    Trả về DataFrame HbO_uM/HbR_uM/HbT_uM (cùng chiều dài input), không có
    cột thời gian (hàm gọi tự thêm nếu cần).
    """
    n = min(len(od_740), len(od_850))
    dpf_740 = scholkmann_wolf_dpf(740, age)
    dpf_850 = scholkmann_wolf_dpf(850, age)

    b = np.vstack([
        np.asarray(od_740[:n], dtype=np.float64) / (l_cm * dpf_740),
        np.asarray(od_850[:n], dtype=np.float64) / (l_cm * dpf_850),
    ])  # (2, N)

    conc_mM = _EPS_INV @ b  # (2, N) -> [HbO; HbR] in mM
    HbO_uM = conc_mM[0] * 1000.0
    HbR_uM = conc_mM[1] * 1000.0
    HbT_uM = HbO_uM + HbR_uM

    result = pd.DataFrame({"HbO_uM": HbO_uM, "HbR_uM": HbR_uM, "HbT_uM": HbT_uM})
    result.attrs.update({"dpf_740": dpf_740, "dpf_850": dpf_850, "l_cm": l_cm, "age": age})
    return result


def compute_hbo_hbr(
    fnirs_740: np.ndarray,
    fnirs_850: np.ndarray,
    fs: float = OPTICAL_FS_DEFAULT,
    l_cm: float = L_CM_DEFAULT,
    age: float = AGE_DEFAULT,
    baseline_seconds: float = BASELINE_SECONDS_DEFAULT,
) -> pd.DataFrame:
    """
    Tinh HbO/HbR (bien thien so voi baseline, don vi uM) tu 2 kenh fNIRS raw
    bang Modified Beer-Lambert Law -- cong thuc day du + citation xem
    docstring dau file (muc 1-4).

    fnirs_740 = kenh 740nm (fnirs_1_raw), fnirs_850 = kenh 850nm (fnirs_2_raw).
    Cac thong so DPF/I0/l_cm/age dung de tinh ra ket qua duoc gan vao
    `result.attrs` de tien log/ve plot, khong anh huong cac cot du lieu.

    Pipeline don gian (KHONG motion-correction) -- dung cho Layer 2 (block
    dai nhat, khong can xu ly lien tuc toan session). Pipeline day du hon
    (motion-correction TDDR tren OD, xu ly lien tuc ca session truoc khi cat
    phase) nam o lib/fnirs_pipeline_v2.py, dung raw_intensity_to_od() +
    concentration_from_od() o tren de chen buoc TDDR vao giua.
    """
    n = min(len(fnirs_740), len(fnirs_850))
    ch1 = np.asarray(fnirs_740[:n], dtype=np.float64)
    ch2 = np.asarray(fnirs_850[:n], dtype=np.float64)
    time_seconds = np.arange(n) / fs

    baseline_mask = time_seconds <= baseline_seconds
    if not baseline_mask.any():
        # Session ngắn hơn baseline_seconds -- dùng toàn bộ tín hiệu có sẵn
        # thay vì crash, để vẫn ra kết quả (dù kém tin cậy hơn).
        baseline_mask = np.ones(n, dtype=bool)

    I0_1 = float(np.nanmean(ch1[baseline_mask]))
    I0_2 = float(np.nanmean(ch2[baseline_mask]))

    OD1, _ = raw_intensity_to_od(ch1, reference=I0_1)
    OD2, _ = raw_intensity_to_od(ch2, reference=I0_2)

    conc = concentration_from_od(OD1, OD2, l_cm=l_cm, age=age)

    result = pd.DataFrame({
        "sample_index": np.arange(n, dtype=np.int64),
        "time_seconds": time_seconds,
        "HbO_uM": conc["HbO_uM"].to_numpy(),
        "HbR_uM": conc["HbR_uM"].to_numpy(),
        "HbT_uM": conc["HbT_uM"].to_numpy(),
    })
    result.attrs.update({
        "dpf_740": conc.attrs["dpf_740"],
        "dpf_850": conc.attrs["dpf_850"],
        "I0_740": I0_1,
        "I0_850": I0_2,
        "l_cm": l_cm,
        "age": age,
        "fs": fs,
    })
    return result


def compute_hbo_hbr_from_csv(
    csv_path: Union[str, Path],
    fs: float = OPTICAL_FS_DEFAULT,
    l_cm: float = L_CM_DEFAULT,
    age: float = AGE_DEFAULT,
    baseline_seconds: float = BASELINE_SECONDS_DEFAULT,
) -> pd.DataFrame:
    """Tiện ích đọc trực tiếp từ fnirs_raw.csv (cột fnirs_1_raw/fnirs_2_raw) do Layer 2 xuất."""
    df = pd.read_csv(csv_path)
    return compute_hbo_hbr(
        df["fnirs_1_raw"].to_numpy(),
        df["fnirs_2_raw"].to_numpy(),
        fs=fs,
        l_cm=l_cm,
        age=age,
        baseline_seconds=baseline_seconds,
    )


def plot_hbo_hbr(result: pd.DataFrame, output_path: Union[str, Path]) -> None:
    """Vẽ ΔHbO/ΔHbR theo thời gian và lưu PNG (không mở cửa sổ, dùng backend Agg)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpf_740 = result.attrs.get("dpf_740")
    dpf_850 = result.attrs.get("dpf_850")
    age = result.attrs.get("age")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(result["time_seconds"], result["HbO_uM"], label="HbO (oxy-Hb)", color="crimson", linewidth=0.8)
    ax.plot(result["time_seconds"], result["HbR_uM"], label="HbR (deoxy-Hb)", color="royalblue", linewidth=0.8)
    ax.set_xlabel("Thời gian (s)")
    ax.set_ylabel("Δ Nồng độ (µM)")

    title = "ΔHbO / ΔHbR tính từ fNIRS raw (740/850nm"
    if dpf_740 is not None and dpf_850 is not None:
        title += f", DPF={dpf_740:.2f}/{dpf_850:.2f}"
    if age is not None:
        title += f", age={age}"
    title += ")"
    ax.set_title(title)

    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
