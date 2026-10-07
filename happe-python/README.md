# HAPPE (Python)

A Python port of **HAPPE — Harvard Automated Processing Pipeline for EEG**
(v4.1, originally MATLAB + EEGLAB). The processing order and parameter
semantics of the MATLAB pipeline are reproduced step-for-step, using
MNE-Python as the core EEG data structure / IO backend plus PyWavelets, NumPy,
SciPy, scikit-learn, and (optionally) mne-icalabel and pyprep.

> **Status of this build:** full package scaffold with the core numeric steps
> — line-noise reduction, wavelet thresholding, bad-channel detection, and the
> QC-metric battery — fully implemented, and every other step implemented as a
> real (not placeholder) module. Where an exact MATLAB/EEGLAB routine has no
> drop-in Python equivalent, the substitution is documented below and marked
> `DEVIATION` in the source, never silently approximated.

## Install

```bash
pip install -e .            # core
pip install -e ".[all]"     # + mne-icalabel, pyprep, eeglabio, pytest
```

## Usage

Programmatic:

```python
from happe import run_pipeline, default_params

params = default_params(erp=False)
params.lineNoise.freq = 60
params.badChans.density = "low"
report = run_pipeline(params, input_dir="raw/", output_dir="out/")
```

CLI:

```bash
happe init-config --out config.yaml           # write a default config
happe run --config config.yaml \
          --input-dir raw/ --output-dir out/
```

## Architecture

```
src/happe/
  config.py        Params dataclass tree (mirrors MATLAB `params` struct) + YAML/JSON IO
  exceptions.py    LoadFailError, NoTagsError, AllICsRejectedError, AllTrialsRejectedError, ...
  pipeline.py      Per-file loop, step order, error isolation, staged folders
  cli.py           `happe run` / `happe init-config`
  io/              loaders.py (multi-format load + validation), savers.py (fif/set/mat/txt)
  steps/           channels, line_noise, resample, filtering, bad_channels, ecgone,
                   wavelet, muscil, segment, baseline, interpolate, reref
  qc/              metrics.py (assess_pipeline_step), reports.py (CSVs), visualize.py
```

`Params` serializes to JSON/YAML, replacing the MATLAB `.mat` param file, so a
run is fully reproducible from a saved config (also written to the output
directory as `run_config.yaml`).

Each file is processed inside `try/except`; a failing file is logged to
`HAPPE_errorLog.csv` (File, Error Message, Traceback) and the batch continues.
The per-run staged output folders are created exactly as in MATLAB HAPPE:
`1 - intermediate_processing`, `2 - wavelet_cleaned_continuous`,
`3 - muscIL` (if enabled), `4 - ERP_filtered` (if ERP), `5 - segmenting`,
`6 - processed`, `7 - quality_assessment_outputs`, with an intermediate `.fif`
saved after each major step.

## Step → module map

| # | Step | Module |
|---|------|--------|
| 1 | Data loading | `io/loaders.py` |
| 2 | Channel corrections | `steps/channels.py` |
| 3 | Line-noise reduction | `steps/line_noise.py` |
| 4 | Resampling | `steps/resample.py` |
| 5 | Band-pass (non-ERP) | `steps/filtering.py` |
| 6 / 9 | Bad-channel detection (pre/post wavelet) | `steps/bad_channels.py` |
| 7 | ECGone | `steps/ecgone.py` |
| 8 | Wavelet thresholding | `steps/wavelet.py` |
| 10 | MuscIL | `steps/muscil.py` |
| 11 | ERP-band filtering | `steps/filtering.py` |
| 12 | QC metrics | `qc/metrics.py` |
| 13 | Re-insert reference channel | `steps/channels.py` |
| 14 | Segmentation | `steps/segment.py` |
| 15 | Baseline correction | `steps/baseline.py` |
| 16 | Within-segment interpolation (FASTER) | `steps/interpolate.py` |
| 17 | Segment rejection | `steps/segment.py` |
| 18 | Full-dataset interpolation | `steps/interpolate.py` |
| 19 | Re-referencing | `steps/reref.py` |
| 20 | Split by tag/condition | `steps/segment.py` |
| 21 | Visualization | `qc/visualize.py` |
| 22 | Save outputs | `io/savers.py` |
| 23 | QC reports | `qc/reports.py` |

## MATLAB → Python substitutions (documented deviations)

Every item below is where an exact port was impractical; the source marks each
with a `DEVIATION` comment.

1. **Cleanline line-noise removal** → `mne.filter.notch_filter(method=
   "spectrum_fit")`, the Mitra–Pesaran Thomson F-test multi-taper regression
   that EEGLAB's Cleanline implements. Same math; not bit-identical because MNE
   fits over the whole signal length with its own DPSS taper set rather than
   Cleanline's 4 s sliding window. A pure-NumPy sliding-window sine-regression
   fallback (`_sine_regression`) is used only if the MNE call fails, and drops
   the per-window F-test gating (documented in-source).

2. **`wdenoise(Bayes, LevelDependent)`** → PyWavelets `wavedec`/`waverec` with a
   per-sub-band MAD noise estimate (`sigma = median(|d|)/0.6745`) and a
   BayesShrink threshold `T = sigma^2 / sqrt(max(var_obs - sigma^2, eps))`.
   MATLAB's exact empirical-Bayes posterior shrinkage differs slightly from
   BayesShrink; results correlate very highly but are not identical.

3. **`clean_rawdata` channel/line-noise criteria** → `pyprep.NoisyChannels`
   when installed; otherwise a custom correlation criterion (max off-diagonal
   |r| < threshold) and a line-band/broadband dB-ratio z-score. RANSAC is not
   used (matches HAPPE, which disables it).

4. **ICLabel** → `mne_icalabel.label_components(method="iclabel")` when
   installed; otherwise a spectral-slope heuristic (rising 20–90 Hz power ⇒
   muscle). The heuristic is coarser than ICLabel and is flagged in-source.

5. **REST reference** → `set_eeg_reference("REST", forward=...)` with an
   auto spherical 3-shell forward model; falls back to average reference if a
   forward solution cannot be built in the environment.

6. **FASTER `single_epoch_channel_properties`** → variance, mean |correlation|,
   rescaled-range Hurst exponent, and max gradient per channel-epoch, z-scored
   within-epoch at |z| > 3. Per-epoch spherical-spline interpolation is done by
   wrapping each epoch as an `EvokedArray`.

7. **`pop_jointprob`** → per-channel Gaussian surprisal summed across ROI
   channels, z-scored across epochs, thresholded at N SD (3 standard / 2 low
   density). This is a Gaussian-density stand-in for EEGLAB's empirical
   histogram density.

8. **EEGLAB `.set` export** requires `eeglabio`; without it the saver falls back
   to `.fif` and records the substitution.

## Testing

```bash
pytest
```

Tests build small synthetic `mne.io.Raw`/`Epochs` fixtures (sine + line noise
+ simulated eyeblink + white noise) and exercise each step independently. See
`tests/`.

## Validating against MATLAB HAPPE

The QC metric helper (`assess_pipeline_step`) emits the same
cross-correlation / coherence / RMSE / SNR / PeakSNR values HAPPE writes, so a
Python run and a MATLAB run on the same file can be compared directly within a
tolerance. Provide a MATLAB reference run to close the loop on numerical
equivalence for the steps above.
```

## Attribution & license

This package is an independent Python port of HAPPE. It is not affiliated with
or endorsed by the original authors. The original MATLAB implementation is
maintained at <https://github.com/PINE-Lab/HAPPE> and is distributed under the
GNU General Public License v3.0. As a derivative work, this port is likewise
licensed under **GPL-3.0-or-later** — see [`LICENSE`](LICENSE).

If you use it, please cite the original HAPPE paper:

> Gabard-Durnam, L. J., Mendez Leal, A. S., Wilkinson, C. L., & Levin, A. R. (2018).
> The Harvard Automated Processing Pipeline for Electroencephalography (HAPPE):
> Standardized processing software for developmental and high-artifact data.
> *Frontiers in Neuroscience, 12*, 97. https://doi.org/10.3389/fnins.2018.00097
