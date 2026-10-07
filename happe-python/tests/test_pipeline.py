"""End-to-end batch test, including per-file error isolation."""

from pathlib import Path

import numpy as np

import mne

from happe.config import default_params
from happe.pipeline import run_pipeline


def _write_raw(path, srate=250.0, n_ch=10, dur=12.0, bad=False):
    rng = np.random.default_rng(1)
    n = int(srate * dur)
    t = np.arange(n) / srate
    m = mne.channels.make_standard_montage("standard_1020")
    names = m.ch_names[:n_ch]
    data = (np.sin(2 * np.pi * 10 * t) * 20e-6
            + np.sin(2 * np.pi * 60 * t) * 8e-6
            + rng.standard_normal((n_ch, n)) * 5e-6)
    info = mne.create_info(names, srate, "eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_montage("standard_1020", on_missing="ignore", verbose="ERROR")
    raw.export(path, fmt="eeglab", overwrite=True, verbose="ERROR") \
        if not bad else Path(path).write_text("not a real EEG file")


def test_batch_runs_and_isolates_errors(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "out"

    try:
        _write_raw(raw_dir / "good1.set")
        _write_raw(raw_dir / "good2.set")
    except Exception:
        import pytest
        pytest.skip("eeglabio not installed; cannot write .set fixtures")
    # A corrupt file to exercise the error log.
    (raw_dir / "broken.set").write_text("garbage")

    p = default_params()
    p.badChans.enabled = True
    p.segment.reg_len_s = 2.0
    report = run_pipeline(p, str(raw_dir), str(out_dir))

    # Staged folders exist.
    assert (out_dir / "1 - intermediate_processing").is_dir()
    assert (out_dir / "6 - processed").is_dir()
    assert (out_dir / "7 - quality_assessment_outputs").is_dir()

    # Error log always written; broken file recorded, good files processed.
    err_csv = out_dir / "7 - quality_assessment_outputs" / "HAPPE_errorLog.csv"
    assert err_csv.exists()
    assert len(report.error_rows) >= 1
    assert len(report.data_rows) >= 1  # batch did not abort


def test_run_config_saved(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "out"
    p = default_params()
    run_pipeline(p, str(raw_dir), str(out_dir))
    assert (out_dir / "run_config.yaml").exists() or (out_dir / "run_config.json").exists()
