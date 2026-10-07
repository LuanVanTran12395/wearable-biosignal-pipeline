# Wearable Biosignal Pipeline

Processing and quality-control code for **multimodal wearable recordings**:
2-channel frontal EEG (AF3/AF4), PPG and dual-wavelength fNIRS from a
forehead headband. It was built for a multi-session meditation study
(rest → meditation → rest protocol).

> **No data is included.** This repository contains code and method notebooks
> only. Participant recordings, survey responses, study results and
> device-vendor material are excluded. Notebook outputs have been cleared.

## What's inside

| Path | What it does |
|---|---|
| `lib/signal_qc.py` | Multimodal QC for one recording. Splits it into blocks at packet gaps, then runs window-level EEG/PPG/fNIRS metrics: flatline, clipping, line-noise ratio, amplitude, scalp-coupling index and pulsatility. It merges artifact intervals and labels each session `PASS / CONDITIONAL_PASS / FAIL / NOT_ASSESSABLE`. It can export EDF and CSV. |
| `lib/eeg_units.py` | ADC count → µV / V conversion. ADC resolution and reference voltage are configurable (`EEG_ADC_BITS`, `EEG_VREF_V`, `configure_adc()`). |
| `lib/features.py` | EEG features: band power, individual alpha-peak features, DWT sub-band features (entropy, Higuchi FD), and cross-channel coherence and phase-locking value (PLV). Hjorth parameters and frontal alpha asymmetry are computed in `notebooks/eeg_feature_extraction.ipynb`. |
| `lib/within_session_dynamics.py` | Time-resolved EEG markers: relaxation onset vs. baseline, drowsy episodes (θ/α ratio) and distraction gaps from epoch rejection. |
| `lib/fnirs_hbo_hbr.py` | Modified Beer–Lambert law: raw intensity → optical density → ΔHbO / ΔHbR / ΔHbT. |
| `lib/fnirs_pipeline_v2.py` | Full fNIRS pipeline. It processes the **continuous session first and segments phases afterwards**: TDDR motion correction, low-pass, robust detrend, a single REST_BEFORE baseline, transition exclusion, robust phase features (median, MAD, IQR, Theil–Sen slope, AUC) and multi-level QC flags that never silently drop data. |
| `lib/fnirs_metrics.py` | Simpler per-segment hemodynamic summary metrics. |
| `lib/ppg_hrv.py` | PPG peak detection → IBI → time-domain HRV (HR, RMSSD, SDRR, pRR50/20) and frequency-domain HRV (LF/HF). |
| `lib/ppg_resp.py` | PPG-derived respiration rate from baseline wander. |
| `lib/happe_qc.py` | Session-level quality flags derived from HAPPE QC reports. |
| `lib/mixed_effects.py` | Random-intercept linear mixed models (participant as grouping factor), so that repeated sessions aren't treated as independent observations (pseudo-replication). |
| `lib/fif_to_json_sidecar.py` | Converts MNE `.fif` files into a compact JSON sidecar (base64 Float32) that a browser-based signal viewer can read. |
| `happe-python/` | Python port of **HAPPE** (Harvard Automated Processing Pipeline for EEG, v4.1) on MNE-Python. Every deviation from the MATLAB original is documented. Licensed **GPL-3.0**; see [its README](happe-python/README.md). |
| `notebooks/` | Batch-processing method notebooks: EEG QC, EEG + HAPPE, EEG feature extraction, fNIRS v2, PPG HRV and PPG respiration. Outputs are cleared. |
| `notebooks/overall/` | **Study-level export.** `layer3_master_tables_export.ipynb` joins session metadata, protocol phases and signal-QC labels into a master table (one row per session). It also exports per-participant yield and coverage, per-study-day summaries, the analysis roster (sessions eligible for the main analysis), a review-flag list, and a study overview figure. |
| `notebooks/personalized/` | **Per-participant export** for a personal results dashboard. It has three tiers: (1) the immediate pre/post change for each session; (2) the person's own trend, shown only with ≥5 valid sessions; (3) where the person sits against a population trend, labelled as reference only. It also exports session time series (EEG band power, HR/RMSSD in 30 s windows, PPG-derived respiration, motion-corrected HbO/HbR, relaxation onset, drowsy episodes) and extra dashboard fields (LF/HF by phase, a pooled-phase composite EEG score). The output is JSON/CSV for a front-end. |

## Design principles

- **Process continuous, then segment.** Filtering, detrending and motion
  correction run once on the whole session. Phases are cut afterwards, so
  phase boundaries don't introduce filter edge effects or baseline resets.
- **Flag, don't delete.** Every exclusion becomes a QC flag that records the
  reason and threshold. Downstream analyses then choose a sensitivity filter.
- **Document every deviation.** Wherever a step departs from a published
  method (HAPPE, TDDR, MBLL), the code says so in a comment and gives the
  reason.
- **Respect the sample size.** Repeated-measures data go through mixed models
  instead of pooled correlations.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e happe-python          # optional: the HAPPE port
```

## Tests

```bash
python lib/test_fnirs_pipeline_v2.py           # TDDR + detrend behaviour on synthetic signals
cd happe-python && pytest -q                   # HAPPE port unit tests
```

## Using the notebooks with your own data

The notebooks expect per-session folders under `data/analytics/` (git-ignored),
in a Hive-style layout:

```
data/analytics/layer2/participant=<id>/study_day=<n>/session_id=<id>/
    eeg_labeled_raw.fif      # EEG with REST_BEFORE / MEDITATION / REST_AFTER annotations
    ppg_raw.edf              # PPG, 100 Hz
    fnirs_raw.csv            # fNIRS raw intensity, 2 wavelengths, 100 Hz
    event_labels_block.csv   # phase onsets (EEG-sample latency)
```

The overall and personalized notebooks also read tables produced upstream
(e.g. `data/analytics/layer1/sessions.csv` and
`data/analytics/metadata_multimodal/session_metadata_survey_features.csv`, a
per-session table joining survey responses with signal features). Those
upstream survey steps are not part of this repository. Participant keys are
shown with a placeholder prefix (`P_A…`).

Set the EEG ADC parameters for your hardware before converting raw counts:

```bash
export EEG_ADC_BITS=24 EEG_VREF_V=<your reference voltage>
```

## Notes

- Code comments and notebook text are mostly in Vietnamese. Function names,
  identifiers and this README are in English.
- Signal-processing methods are cited inline in module docstrings, e.g. MBLL,
  TDDR (Fishburn et al., 2019) and HAPPE (Gabard-Durnam et al., 2018).

## License

- Everything outside `happe-python/`: MIT, see [`LICENSE`](LICENSE).
- `happe-python/`: GPL-3.0-or-later (derivative of HAPPE), see
  [`happe-python/LICENSE`](happe-python/LICENSE).
