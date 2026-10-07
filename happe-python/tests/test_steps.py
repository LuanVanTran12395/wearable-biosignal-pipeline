import numpy as np
import pytest

import mne

from happe.config import default_params
from happe.steps import (
    bad_channels, filtering, line_noise, pre_wavelet_tddr, resample, segment, wavelet, reref,
)
from happe.qc.metrics import assess_pipeline_step, line_noise_freqs_of_interest
from happe.exceptions import NoTagsError, AllTrialsRejectedError


def _line_power(data, srate, f0):
    from scipy.signal import welch
    freqs, psd = welch(data, fs=srate, nperseg=int(min(srate * 2, data.shape[-1])))
    idx = np.argmin(np.abs(freqs - f0))
    return psd[..., idx].mean()


def test_line_noise_reduces_60hz(synthetic_raw, srate):
    p = default_params()
    p.lineNoise.freq = 60
    before = _line_power(synthetic_raw.get_data(), srate, 60)
    line_noise.apply(synthetic_raw, p)
    after = _line_power(synthetic_raw.get_data(), srate, 60)
    assert after < before


def test_line_noise_notch_method(synthetic_raw, srate):
    p = default_params()
    p.lineNoise.method = "notch"
    p.lineNoise.freq = 60
    before = _line_power(synthetic_raw.get_data(), srate, 60)
    line_noise.apply(synthetic_raw, p)
    after = _line_power(synthetic_raw.get_data(), srate, 60)
    assert after < before


def test_wavelet_returns_variance_metric(synthetic_raw):
    p = default_params()
    res = wavelet.apply(synthetic_raw, p)
    assert np.isfinite(res.percent_var_retained)
    assert 0 < res.percent_var_retained <= 100 + 1e-6


def test_wavelet_block_seconds_preserves_alpha_away_from_artifact(srate):
    """Regression test for the whole-session BayesShrink failure mode found in
    the eeg_happe_qc.ipynb investigation: a severe artifact anywhere in the
    session inflates the subband-wide noise sigma, over-suppressing genuine
    alpha content in *unrelated, clean* stretches of the same session. With
    ``wavelet.block_seconds`` set, thresholding is local per time-block, so
    the clean first half's alpha should survive much better than with the
    default whole-session (block_seconds=None) behavior."""
    dur = 40.0
    n = int(srate * dur)
    t = np.arange(n) / srate
    rng = np.random.default_rng(2)
    alpha = np.sin(2 * np.pi * 10.0 * t) * 15e-6  # steady alpha, whole session
    noise = rng.standard_normal(n) * 2e-6
    sig = alpha + noise
    # Severe, large-amplitude artifact confined to the second half only.
    burst = np.zeros(n)
    burst_win = slice(int(srate * 25.0), int(srate * 35.0))
    burst[burst_win] = rng.standard_normal(burst[burst_win].shape) * 200e-6
    data = np.tile((sig + burst), (2, 1))
    info = mne.create_info(["AF3_RAW", "AF4_RAW"], srate, ch_types="eeg")

    def alpha_power_first_half(cleaned):
        from scipy.signal import welch
        half = cleaned[0, : int(srate * 20.0)]
        freqs, psd = welch(half, fs=srate, nperseg=int(srate * 4))
        idx = np.argmin(np.abs(freqs - 10.0))
        return psd[idx]

    from scipy.signal import welch
    freqs0, psd0 = welch(sig[: int(srate * 20.0)], fs=srate, nperseg=int(srate * 4))
    idx0 = np.argmin(np.abs(freqs0 - 10.0))
    orig_alpha_power = psd0[idx0]

    p_global = default_params(erp=False)
    raw_global = mne.io.RawArray(data.copy(), info, verbose="ERROR")
    wavelet.apply(raw_global, p_global)
    global_alpha = alpha_power_first_half(raw_global.get_data())

    p_block = default_params(erp=False)
    p_block.wavelet.block_seconds = 4.0
    raw_block = mne.io.RawArray(data.copy(), info, verbose="ERROR")
    wavelet.apply(raw_block, p_block)
    block_alpha = alpha_power_first_half(raw_block.get_data())

    global_retention = global_alpha / orig_alpha_power
    block_retention = block_alpha / orig_alpha_power
    assert block_retention > global_retention, (
        f"blockwise should retain more first-half alpha than whole-session: "
        f"global={global_retention:.3f} block={block_retention:.3f}"
    )


def test_wavelet_level_selection_by_srate():
    p = default_params()
    from happe.steps.wavelet import _select_wavelet_and_level
    assert _select_wavelet_and_level(p, 1000)[1] == 10
    assert _select_wavelet_and_level(p, 300)[1] == 9
    assert _select_wavelet_and_level(p, 200)[1] == 8
    p.paradigm.erp = True
    assert _select_wavelet_and_level(p, 1000) == ("coif4", 11)


def test_bad_channels_flag_flatline(synthetic_raw):
    p = default_params()
    data = synthetic_raw.get_data()
    data[3] = 0.0  # flatline channel
    synthetic_raw._data = data
    res = bad_channels.detect(synthetic_raw, p, after_wavelet=False)
    assert synthetic_raw.info["ch_names"][3] in res.bad_ids


def test_resample_changes_srate(synthetic_raw):
    p = default_params()
    p.downsample.enabled = True
    p.downsample.srate = 125
    resample.apply(synthetic_raw, p)
    assert synthetic_raw.info["sfreq"] == 125


def test_filtering_fir_and_butter(synthetic_raw):
    p = default_params()
    r1 = synthetic_raw.copy()
    filtering.bandpass(r1, p)
    assert np.isfinite(r1.get_data()).all()
    p.filt.method = "butter"
    r2 = synthetic_raw.copy()
    filtering.bandpass(r2, p)
    assert np.isfinite(r2.get_data()).all()


def test_segment_regular(synthetic_raw):
    p = default_params()
    p.segment.reg_len_s = 2.0
    ep = segment.segment(synthetic_raw, p)
    assert isinstance(ep, mne.BaseEpochs)
    assert len(ep) == 10  # 20 s / 2 s


def test_segment_missing_tag_raises(synthetic_raw_with_events):
    p = default_params()
    p.paradigm.task = True
    p.paradigm.onset_tags = ["does_not_exist"]
    with pytest.raises(NoTagsError):
        segment.segment(synthetic_raw_with_events, p)


def test_segment_events(synthetic_raw_with_events):
    p = default_params()
    p.paradigm.task = True
    p.paradigm.onset_tags = ["stim"]
    p.segment.start_ms = -100
    p.segment.end_ms = 500
    ep = segment.segment(synthetic_raw_with_events, p)
    assert len(ep) >= 5


def test_reject_all_raises(synthetic_epochs):
    p = default_params()
    p.segRej.enabled = True
    p.segRej.by_amplitude = True
    p.segRej.amp_min = -1e-12
    p.segRej.amp_max = 1e-12  # impossible to satisfy -> all rejected
    with pytest.raises(AllTrialsRejectedError):
        segment.reject(synthetic_epochs, p)


def test_reject_keeps_indices(synthetic_epochs):
    p = default_params()
    p.segRej.enabled = True
    p.segRej.amp_min = -1.0
    p.segRej.amp_max = 1.0  # generous -> keep all
    res = segment.reject(synthetic_epochs, p)
    assert res.n_post == res.n_pre
    assert res.kept_indices == list(range(res.n_pre))


def test_qc_line_noise_keys(synthetic_raw, srate):
    pre = synthetic_raw.get_data()
    post = pre * 0.9
    foi = line_noise_freqs_of_interest(60, [], [10, 5, 2, 1])
    m = assess_pipeline_step("line_noise", pre, post, srate, foi)
    assert "cross_correlation" in m
    assert any(k.startswith("coherence_") for k in m)
    assert "rmse" not in m  # amplitude metrics excluded for line-noise


def test_qc_wavelet_keys(synthetic_raw, srate):
    pre = synthetic_raw.get_data()
    post = pre * 0.9
    m = assess_pipeline_step("wavelet", pre, post, srate, [10.0])
    for key in ("rmse", "mae", "snr_db", "peak_snr_db", "cross_correlation"):
        assert key in m


def test_reref_average(synthetic_epochs):
    p = default_params()
    p.reref.method = "average"
    reref.apply(synthetic_epochs, p)
    # After average reference, per-sample mean across channels ~ 0.
    d = synthetic_epochs.get_data()
    assert np.abs(d.mean(axis=1)).max() < 1e-6


def test_pre_wavelet_tddr_disabled_by_default_is_noop(synthetic_raw):
    p = default_params(erp=False)
    assert p.preWaveletTddr.enabled is False
    before = synthetic_raw.get_data().copy()
    pre_wavelet_tddr.apply(synthetic_raw, p)
    np.testing.assert_array_equal(synthetic_raw.get_data(), before)


def test_pre_wavelet_tddr_removes_step_shift(srate):
    """Synthetic step-shift (matches the fNIRS TDDR validation methodology
    in lib/fnirs_pipeline_v2.py::test_tddr_removes_step_shift): inject a
    large baseline step at t=10s into an otherwise flat channel, verify TDDR
    (cutoff below the step's own near-DC nature) removes the large majority
    of it, isolating the step via pre/post median-difference vs a step-free
    reference (avoids drift/noise contaminating the measurement)."""
    rng = np.random.default_rng(1)
    dur = 20.0
    n = int(srate * dur)
    t = np.arange(n) / srate
    noise = rng.normal(0, 2.0, n)  # uV-scale background noise, no drift
    step = np.zeros(n)
    step[t >= 10.0] = 500.0  # uV step -- large relative to background noise
    sig_with_step = noise + step
    sig_no_step = noise.copy()

    corrected, weights, n_iter = pre_wavelet_tddr.tddr(sig_with_step, srate, filter_cutoff_hz=4.0)

    win = int(srate * 2.0)  # 2s windows either side of the step, away from the edge
    idx_pre = slice(int(srate * 6.0), int(srate * 6.0) + win)
    idx_post = slice(int(srate * 14.0), int(srate * 14.0) + win)

    def step_estimate(sig):
        return np.median(sig[idx_post]) - np.median(sig[idx_pre])

    step_before = step_estimate(sig_with_step) - step_estimate(sig_no_step)
    step_after = step_estimate(corrected) - step_estimate(sig_no_step)

    assert abs(step_before - 500.0) < 5.0  # sanity: injected step correctly isolated
    reduction = 1.0 - abs(step_after) / abs(step_before)
    assert reduction > 0.9, f"expected >90% step reduction, got {reduction:.2%}"
    assert n_iter > 0
    assert weights.min() < 0.5  # some derivative samples flagged as artifact


def test_pre_wavelet_tddr_spares_content_above_cutoff(srate):
    """Content structurally above filter_cutoff_hz must pass through
    unmodified (by construction -- signal_high is added back untouched),
    so an oscillation well above cutoff should be ~fully preserved."""
    dur = 20.0
    n = int(srate * dur)
    t = np.arange(n) / srate
    alpha = 10.0 * np.sin(2 * np.pi * 10.0 * t)  # 10Hz "alpha", well above cutoff=4Hz
    step = np.zeros(n)
    step[t >= 10.0] = 300.0

    corrected, _w, _n = pre_wavelet_tddr.tddr(alpha + step, srate, filter_cutoff_hz=4.0)
    # Isolate the residual alpha-band content post-TDDR and compare power to original.
    from scipy.signal import welch
    nperseg = int(srate * 4)
    freqs, psd_orig = welch(alpha, fs=srate, nperseg=nperseg)
    _, psd_corr = welch(corrected - step.mean(), fs=srate, nperseg=nperseg)  # remove residual DC only
    idx = np.argmin(np.abs(freqs - 10.0))
    ratio = psd_corr[idx] / psd_orig[idx]
    assert 0.8 < ratio < 1.25, f"10Hz power should be ~preserved, ratio={ratio:.3f}"


def test_pre_wavelet_tddr_apply_enabled_modifies_raw(synthetic_raw):
    p = default_params(erp=False)
    p.preWaveletTddr.enabled = True
    p.preWaveletTddr.cutoff_hz = 4.0
    before = synthetic_raw.get_data().copy()
    pre_wavelet_tddr.apply(synthetic_raw, p)
    assert not np.array_equal(synthetic_raw.get_data(), before)
