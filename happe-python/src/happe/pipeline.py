"""Pipeline orchestration — the per-file loop.

The step order mirrors MATLAB HAPPE v4.1 exactly and is NOT reordered, merged,
or dropped. Each file's processing is wrapped in try/except so one failing file
never aborts the batch; errors are collected into the error-log CSV.

A staged output-folder structure is created per run:

    1 - intermediate_processing
    2 - wavelet_cleaned_continuous
    3 - muscIL                      (if enabled)
    4 - ERP_filtered                (if ERP)
    5 - segmenting
    6 - processed
    7 - quality_assessment_outputs

An intermediate file is saved after each major step so reprocessing can resume
from any stage.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np

import mne

from .config import Params
from .exceptions import HappeError
from .io import load_raw, save_intermediate, save_outputs
from .qc.metrics import assess_pipeline_step, line_noise_freqs_of_interest
from .qc.reports import ReportAccumulator
from .qc import visualize
from .steps import (
    baseline,
    bad_channels,
    channels,
    ecgone,
    filtering,
    interpolate,
    line_noise,
    muscil,
    pre_wavelet_tddr,
    reref,
    resample,
    segment,
    wavelet,
)

log = logging.getLogger("happe")

_STAGE_DIRS = {
    "intermediate": "1 - intermediate_processing",
    "wavelet": "2 - wavelet_cleaned_continuous",
    "muscil": "3 - muscIL",
    "erp": "4 - ERP_filtered",
    "segment": "5 - segmenting",
    "processed": "6 - processed",
    "qc": "7 - quality_assessment_outputs",
}

_RAW_EXTS = {".set", ".edf", ".bdf", ".mff", ".raw", ".mat", ".npy", ".fif"}


def make_stage_dirs(output_dir: Path, params: Params) -> dict:
    dirs = {}
    for key, name in _STAGE_DIRS.items():
        if key == "muscil" and not params.muscIL.enabled:
            continue
        if key == "erp" and not params.paradigm.erp:
            continue
        d = output_dir / name
        d.mkdir(parents=True, exist_ok=True)
        dirs[key] = d
    return dirs


def process_file(path: str, params: Params, dirs: dict,
                 report: ReportAccumulator) -> Optional[List[Path]]:
    """Run the full pipeline on one file. Returns saved output paths.

    Raises are propagated to the caller (:func:`run_pipeline`) which logs them;
    kept as a separate function so a single file can be processed/tested alone.
    """
    stem = Path(path).stem
    srate_freqs = params.lineNoise

    # ---- Step 1: load ------------------------------------------------- #
    raw = load_raw(path, params)
    srate = raw.info["sfreq"]

    # ---- Step 2: channel corrections --------------------------------- #
    channels.apply_montage_for_net(raw, params)
    channels.pull_reference_channel(raw, params)
    channels.select_channels_of_interest(raw, params)

    # ---- Step 3: line-noise reduction (with QC) ---------------------- #
    pre_ln = raw.get_data(picks="eeg").copy()
    line_noise.apply(raw, params)
    foi = line_noise_freqs_of_interest(srate_freqs.freq, srate_freqs.harmonics,
                                       srate_freqs.neighbors)
    ln_qc = assess_pipeline_step("line_noise", pre_ln,
                                 raw.get_data(picks="eeg"), srate, foi)

    # ---- Step 4: resample -------------------------------------------- #
    resample.apply(raw, params)
    srate = raw.info["sfreq"]

    # ---- Step 5: band-pass (non-ERP only) ---------------------------- #
    if not params.paradigm.erp:
        filtering.bandpass(raw, params)
    save_intermediate(raw, dirs["intermediate"], stem)

    # ---- Step 6: bad-channel detection (pass 1, if order == before) -- #
    bc_result = None
    if params.badChans.enabled and params.badChans.order == "before":
        bc_result = bad_channels.detect(raw, params, after_wavelet=False)

    # ---- Step 7: ECGone (optional, best-effort) ---------------------- #
    ecgone.apply(raw, params)

    # ---- Step 7.5: TDDR pre-clean (DEVIATION, local, optional) -------- #
    pre_wavelet_tddr.apply(raw, params)

    # ---- Step 8: wavelet thresholding (core, with QC) ---------------- #
    pre_wav = raw.get_data(picks="eeg").copy()
    if params.paradigm.erp:
        # ERP QC basis: filter both pre & post to the ERP band, then diff.
        pre_ref = raw.copy()
        filtering.bandpass(pre_ref, params)
        pre_wav = pre_ref.get_data(picks="eeg").copy()
    wav_res = wavelet.apply(raw, params)
    post_wav = raw.get_data(picks="eeg")
    if params.paradigm.erp:
        post_ref = raw.copy()
        filtering.bandpass(post_ref, params)
        post_wav = post_ref.get_data(picks="eeg")
    wav_qc = assess_pipeline_step("wavelet", pre_wav, post_wav, srate, foi)
    save_intermediate(raw, dirs["wavelet"], stem)

    # ---- Step 9: bad-channel detection (pass 2, if order == after) --- #
    if params.badChans.enabled and params.badChans.order == "after" and bc_result is None:
        bc_result = bad_channels.detect(raw, params, after_wavelet=True)

    # ---- Step 10: MuscIL (optional) ---------------------------------- #
    musc_res = muscil.apply(raw, params)
    if params.muscIL.enabled and "muscil" in dirs:
        save_intermediate(raw, dirs["muscil"], stem)

    # ---- Step 11: ERP-band filtering (ERP only) ---------------------- #
    if params.paradigm.erp:
        filtering.bandpass(raw, params)
        save_intermediate(raw, dirs["erp"], stem)

    # ---- Step 13: re-insert reference channel ------------------------ #
    channels.reinsert_reference_channel(raw, params)

    # ---- Step 14: segmentation --------------------------------------- #
    epochs = segment.segment(raw, params)

    # ---- Step 15: baseline correction (ERP only) -------------------- #
    epochs = baseline.apply(epochs, params)

    # ---- Step 16: within-segment interpolation (optional) ----------- #
    interp_map = interpolate.interpolate_within_segments(epochs, params)

    # ---- Step 17: segment rejection (optional) ---------------------- #
    rej = segment.reject(epochs, params)
    epochs = rej.epochs
    save_intermediate(epochs, dirs["segment"], stem)

    # ---- Step 18: full-dataset interpolation ------------------------- #
    interpolate.interpolate_full(epochs, params)

    # ---- Step 19: re-referencing ------------------------------------- #
    reref.apply(epochs, params)

    # ---- Step 20: split by tag / condition --------------------------- #
    splits = segment.split_by_tag(epochs, params)

    # ---- Step 21: visualization (optional) --------------------------- #
    if params.vis.enabled:
        visualize.make_figure(epochs, params, dirs["qc"], stem)

    # ---- Step 22: save outputs --------------------------------------- #
    written = save_outputs(epochs, dirs["processed"], stem, params)
    for name, sub in splits.items():
        written.extend(save_outputs(sub, dirs["processed"], f"{stem}_{name}", params))

    # ---- Step 23: accumulate QC rows --------------------------------- #
    report.add_pipeline(Path(path).name, ln_qc, wav_qc)
    report.add_data(_data_qc_row(path, raw, epochs, bc_result, wav_res,
                                 musc_res, interp_map, rej))
    return written


def _data_qc_row(path, raw, epochs, bc_result, wav_res, musc_res,
                 interp_map, rej) -> dict:
    n_sel = bc_result.n_selected if bc_result else raw.info["nchan"]
    row = {
        "File": Path(path).name,
        "File_Length_s": round(raw.n_times / raw.info["sfreq"], 3),
        "N_Channels_Selected": n_sel,
        "N_Good_Channels": bc_result.n_good if bc_result else n_sel,
        "Pct_Good_Channels": round(bc_result.pct_good, 2) if bc_result else 100.0,
        "Rejected_Channel_IDs": ";".join(bc_result.bad_ids) if bc_result else "",
        "Pct_Var_Retained_PostWavelet": round(wav_res.percent_var_retained, 3),
        "N_ICs_Rejected": musc_res.n_rejected,
        "Pct_ICs_Rejected": round(musc_res.pct_rejected, 2),
        "Channels_Interpolated_Per_Segment": _fmt_interp(interp_map),
        "N_Epochs_Pre_Rejection": rej.n_pre,
        "N_Epochs_Post_Rejection": rej.n_post,
        "Pct_Epochs_Retained": round(rej.pct_retained, 2),
        "Retained_Epoch_Indices": ";".join(map(str, rej.kept_indices)),
    }
    for tag, counts in rej.per_tag.items():
        row[f"Tag_{tag}_Pre"] = counts.get("pre")
        row[f"Tag_{tag}_Post"] = counts.get("post")
    return row


def _fmt_interp(interp_map: dict) -> str:
    if not interp_map:
        return ""
    return " | ".join(f"ep{ei}:{','.join(chs)}" for ei, chs in interp_map.items())


def _discover_files(input_dir: Path) -> List[Path]:
    files = [p for p in sorted(input_dir.iterdir())
             if p.is_file() and p.suffix.lower() in _RAW_EXTS]
    return files


def run_pipeline(params: Params, input_dir: str, output_dir: str,
                 save_config: bool = True) -> ReportAccumulator:
    """Batch-process every raw file in ``input_dir`` into ``output_dir``.

    Per-file errors are caught and logged to the error CSV; the batch always
    runs to completion. Returns the populated :class:`ReportAccumulator`.
    """
    params.validate()
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = make_stage_dirs(out_dir, params)
    report = ReportAccumulator()

    if save_config:
        try:
            params.to_yaml(str(out_dir / "run_config.yaml"))
        except Exception:
            params.to_json(str(out_dir / "run_config.json"))

    files = _discover_files(in_dir)
    if not files:
        log.warning("No raw files found in %s", in_dir)

    for f in files:
        try:
            process_file(str(f), params, dirs, report)
            log.info("Processed %s", f.name)
        except (HappeError, Exception) as exc:  # non-fatal per-file isolation
            tb = traceback.format_exc()
            report.add_error(f.name, str(exc), tb.strip().splitlines()[-1]
                             if tb else "")
            log.error("Failed %s: %s", f.name, exc)

    report.write(dirs["qc"])
    return report
