"""
Linear mixed-effects model (random intercept theo participant) -- dùng để
kiểm định lại các finding tương quan chính, xử lý vấn đề **pseudo-replication**
thường gặp trong nghiên cứu lặp lại nhiều session: hầu hết phép tương quan trước giờ
(`scipy.stats.spearmanr` trên toàn bộ session) coi mỗi session là 1 quan sát
độc lập, dù N participant thật thường nhỏ, mỗi người lặp lại nhiều session
-- vi phạm giả định độc lập, dễ thổi phồng ý nghĩa thống kê (p bị đánh giá
thấp hơn thực tế).

================================================================================
PHƯƠNG PHÁP
================================================================================

Model:  y_ij = beta0 + beta1 * x_ij + u_i + e_ij

    i = participant, j = session của participant đó
    u_i ~ N(0, tau^2)   -- random intercept riêng cho từng participant
                            (cho phép mỗi người có baseline khác nhau)
    e_ij ~ N(0, sigma^2) -- nhiễu residual trong-người

beta1 (fixed effect slope) là hệ số quan tâm: có phản ánh mối quan hệ x->y
sau khi đã "trừ đi" baseline riêng của từng người hay không. p-value của
beta1 đáng tin hơn hẳn Spearman gộp vì đã tính đúng effective sample size
(gần với N participant hơn là N session).

Dùng REML=False (ML) khi so sánh model qua LRT, REML=True (mặc định của
statsmodels) cho ước lượng variance component ít lệch hơn khi chỉ cần
p-value 1 fixed effect -- ở đây dùng mặc định REML=True vì chỉ cần suy diễn
trên beta1, không so sánh model.

x được chuẩn hoá z-score (mặc định) trước khi fit để hệ số dễ so sánh giữa
các cặp biến có đơn vị khác nhau, và để tối ưu hoá số trị ổn định hơn.

Citation:
- Pinheiro JC, Bates DM. "Mixed-Effects Models in S and S-PLUS." Springer, 2000.
- Aarts E, Verhage M, Veenvliet JV, Dolan CV, van der Sluis S. "A solution
  to dependency: using multilevel analysis to accommodate nested data."
  Nat Neurosci. 2014;17(4):491-6. -- lý do cụ thể cho neuroscience/nhiều
  quan sát lặp lại trên cùng 1 cá thể, đúng tình huống của project này.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

try:
    import statsmodels.formula.api as smf
except ImportError as e:  # pragma: no cover
    raise ImportError("Can 'pip install statsmodels' de dung module nay.") from e


def fit_random_intercept_lmm(
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str = "participant_key",
    zscore_x: bool = True,
) -> dict:
    """Fit LMM 1 fixed effect (x) + random intercept theo group.

    Trả về dict:
        n            -- so quan sat (session) dua vao fit
        n_groups     -- so participant (group) khac nhau
        coef         -- he so fixed effect cua x (slope)
        se           -- standard error cua coef
        t            -- t-stat (coef / se)
        p            -- p-value 2 phia cho coef
        icc          -- intraclass correlation = tau^2 / (tau^2 + sigma^2),
                        ty le variance do khac biet GIUA participant giai
                        thich duoc -- ICC cao (>0.3-0.4) nghia la pseudo-
                        replication anh huong manh neu dung Spearman gop
        converged    -- model co hoi tu khong (False -> ket qua khong dang tin)

    Neu sau khi dropna con < 2 group hoac < 3 quan sat, tra ve None (khong
    du du lieu de fit).
    """
    sub = df[[x, y, group]].dropna()
    if sub[group].nunique() < 2 or len(sub) < 3:
        return None

    sub = sub.rename(columns={x: "_x", y: "_y", group: "_group"})
    if zscore_x:
        std = sub["_x"].std(ddof=0)
        if std > 0:
            sub["_x"] = (sub["_x"] - sub["_x"].mean()) / std

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm("_y ~ _x", data=sub, groups=sub["_group"])
        try:
            result = model.fit(method=["lbfgs"])
        except Exception:
            return None

    tau2 = float(result.cov_re.iloc[0, 0]) if result.cov_re is not None else np.nan
    sigma2 = float(result.scale)
    icc = tau2 / (tau2 + sigma2) if (tau2 + sigma2) > 0 else np.nan

    return {
        "n": int(len(sub)),
        "n_groups": int(sub["_group"].nunique()),
        "coef": float(result.fe_params.get("_x", np.nan)),
        "se": float(result.bse.get("_x", np.nan)),
        "t": float(result.tvalues.get("_x", np.nan)),
        "p": float(result.pvalues.get("_x", np.nan)),
        "icc": icc,
        "converged": bool(result.converged),
    }


def batch_mixed_effects(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    group: str = "participant_key",
    zscore_x: bool = True,
    fdr_method: str = "fdr_bh",
) -> pd.DataFrame:
    """Chay fit_random_intercept_lmm() cho nhieu cap (x, y), gop FDR correction
    tren toan bo tap test cung luc -- cung quy uoc voi cac notebook
    correlation_*.csv da co trong project (multipletests fdr_bh).

    pairs: list [(x1, y1), (x2, y2), ...]
    Tra ve DataFrame 1 dong / cap, cot 'x', 'y' + toan bo key cua
    fit_random_intercept_lmm(), them 'p_fdr', 'significant_fdr05'.
    Cap khong du du lieu (tra ve None) bi bo qua, khong tinh vao FDR.
    """
    rows = []
    for x, y in pairs:
        res = fit_random_intercept_lmm(df, x, y, group=group, zscore_x=zscore_x)
        if res is None:
            continue
        res = {"x": x, "y": y, **res}
        rows.append(res)

    result_df = pd.DataFrame(rows)
    if len(result_df) == 0:
        return result_df

    valid = result_df["p"].notna()
    rej, p_fdr, _, _ = multipletests(result_df.loc[valid, "p"], method=fdr_method)
    result_df.loc[valid, "p_fdr"] = p_fdr
    result_df.loc[valid, "significant_fdr05"] = rej
    return result_df
