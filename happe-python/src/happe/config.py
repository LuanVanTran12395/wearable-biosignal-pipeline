"""Central configuration mirroring the MATLAB HAPPE ``params`` struct.

The MATLAB pipeline stored its configuration in a ``.mat`` param file. Here we
use nested :func:`dataclasses.dataclass` objects that serialize to / from JSON
and YAML so a run is fully reproducible from a saved config file.

Every field name follows the MATLAB ``params`` sub-struct names as closely as
possible: ``loadInfo``, ``chans``, ``lineNoise``, ``downsample``, ``filt``,
``badChans``, ``ecgone``, ``wavelet``, ``muscIL``, ``paradigm``, ``segment``,
``segRej``, ``reref``, ``baseCorr``, ``vis``, ``outputFormat``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, get_type_hints

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is a hard dependency
    yaml = None

from .exceptions import ConfigError


# --------------------------------------------------------------------------- #
# Sub-configuration structs
# --------------------------------------------------------------------------- #
@dataclass
class LoadInfo:
    """File import settings (MATLAB ``params.loadInfo``)."""

    #: One of {"set", "edf", "bdf", "mff", "egi_raw", "mat"}. If ``None`` the
    #: loader infers the format from the file extension.
    file_format: Optional[str] = None
    #: Required for ``mat``/matrix inputs where the sampling rate is not stored.
    srate: Optional[float] = None
    #: Montage / channel-location file (.sfp or a standard MNE montage name).
    chanlocs: Optional[str] = None
    #: For task/ERP runs: path to an event file (equivalent of pop_importevent).
    event_file: Optional[str] = None
    #: Layout string for EGI nets, e.g. "GSN-HydroCel-128".
    net_type: Optional[str] = None


@dataclass
class Chans:
    """Channel-of-interest selection (MATLAB ``params.chans``)."""

    #: One of {"all", "include", "exclude"}.
    mode: str = "all"
    #: Channel labels to include/exclude when mode != "all".
    labels: List[str] = field(default_factory=list)
    #: A flatline/unused reference channel to pull out early and re-insert
    #: before re-referencing (steps 2 & 13). ``None`` = none present.
    ref_chan: Optional[str] = None


@dataclass
class LineNoise:
    """Line-noise reduction settings (MATLAB ``params.lineNoise``)."""

    #: Fundamental line frequency, 50 or 60 Hz.
    freq: float = 60.0
    #: Extra user harmonics (beyond fundamental + 2nd harmonic default).
    harmonics: List[float] = field(default_factory=list)
    #: "cleanline" (multitaper regression) or "notch" (band-stop).
    method: str = "cleanline"
    #: Notch band edges (Hz) when method == "notch".
    notch_low: Optional[float] = None
    notch_high: Optional[float] = None
    #: Cleanline sliding-window parameters.
    window_s: float = 4.0
    step_s: float = 4.0
    bandwidth: float = 2.0
    p_value: float = 0.01
    max_iter: int = 10
    #: Neighbor offsets (Hz) used only for QC coherence, not filtering.
    neighbors: List[float] = field(default_factory=lambda: [10, 5, 2, 1])


@dataclass
class Downsample:
    """Resampling settings (MATLAB ``params.downsample``)."""

    enabled: bool = False
    srate: Optional[float] = None


@dataclass
class Filt:
    """Band-pass filter settings (MATLAB ``params.filt``)."""

    highpass: float = 1.0
    lowpass: float = 100.0
    #: "fir" (pop_eegfiltnew / mne raw.filter) or "butter" (ERPLAB-style IIR).
    method: str = "fir"
    #: Butterworth order when method == "butter".
    butter_order: int = 2


@dataclass
class BadChans:
    """Bad-channel detection settings (MATLAB ``params.badChans``)."""

    enabled: bool = True
    #: "before" or "after" wavelet thresholding (pass 1 vs pass 2).
    order: str = "before"
    #: "low" or "high" density; selects the detection tier.
    density: str = "low"
    #: Low-density spectral-outlier symmetric z threshold.
    low_z: float = 2.75
    #: High-density asymmetric spectral-outlier thresholds (pre-/post-wavelet).
    high_z_pre: tuple = (-5.0, 1.8935)
    high_z_post: tuple = (-5.0, 2.1316)
    #: clean_rawdata-equivalent criteria (high density).
    corr_thresh_pre: float = 0.485
    line_thresh_pre: float = 7.1


@dataclass
class Ecgone:
    """ECGone cardiac-artifact removal (MATLAB ``params.ecgone``)."""

    enabled: bool = False
    epoch_len_s: float = 30.0
    ecg_channels: List[str] = field(default_factory=list)
    peak_win_s: float = 0.5
    peakiness_thresh: float = 2.0
    lowpass_hz: float = 100.0


@dataclass
class Wavelet:
    """Wavelet-thresholding settings (MATLAB ``params.wavelet``)."""

    #: "bayes" (wdenoise Bayes / LevelDependent) or "legacy" (wavelet-ICA).
    method: str = "bayes"
    #: Threshold rule: "hard" (default) or "soft" (ERP only).
    threshold_rule: str = "hard"
    #: Overrides for wavelet family / level; None => paradigm+srate defaults.
    wavelet: Optional[str] = None
    level: Optional[int] = None
    #: DEVIATION (local, not in original MATLAB HAPPE): if set, BayesShrink's
    #: sigma/threshold is estimated per-subband PER TIME BLOCK of roughly this
    #: many seconds, instead of once over the whole continuous session. None
    #: (default) preserves original whole-session behavior. Rationale: a
    #: whole-session threshold lets one severe artifact dominate a subband's
    #: noise estimate, suppressing genuine alpha-band content elsewhere in the
    #: same session (see eeg_happe_qc.ipynb investigation notes). Blocking is
    #: done on the wavelet coefficient array itself (not the raw time series),
    #: so it does not require re-segmenting/reconstructing the signal and
    #: introduces no boundary artifacts in the output.
    block_seconds: Optional[float] = None
    #: DEVIATION (local): fraction (0-1) of overlap between adjacent time
    #: blocks used by ``block_seconds``. 0.0 (default) = hard-edged,
    #: non-overlapping blocks (original block implementation). >0 blends the
    #: per-block threshold smoothly across block centers (linear interpolation
    #: of the threshold *value*, applied pointwise to the original
    #: coefficients -- not an overlap-add of thresholded outputs, which would
    #: double-count), to avoid a step discontinuity in the artifact estimate
    #: at block boundaries. Ignored if ``block_seconds`` is None.
    block_overlap: float = 0.0


@dataclass
class MuscIL:
    """ICA-based muscle-artifact rejection (MATLAB ``params.muscIL``)."""

    enabled: bool = False
    #: Muscle-probability threshold for flagging a component.
    muscle_thresh: float = 0.25
    #: High-pass (Hz) applied to the ICA-fitting copy only.
    ica_highpass: float = 1.0


@dataclass
class Paradigm:
    """Task/ERP vs resting settings (MATLAB ``params.paradigm``)."""

    #: True => ERP/task paradigm; False => resting/baseline.
    erp: bool = False
    task: bool = False
    #: Onset event tags (task/ERP). Grouped conditions map name -> [tags].
    onset_tags: List[str] = field(default_factory=list)
    conditions: Dict[str, List[str]] = field(default_factory=dict)
    #: Trigger-delay correction offset (ms) applied to event latencies (ERP).
    onset_offset_ms: float = 0.0


@dataclass
class Segment:
    """Segmentation settings (MATLAB ``params.segment``)."""

    enabled: bool = True
    #: Task/ERP epoch window relative to event (ms).
    start_ms: float = -100.0
    end_ms: float = 1000.0
    #: Resting fixed-length epoch duration (s).
    reg_len_s: float = 2.0
    #: Within-segment FASTER-style interpolation (step 16).
    interp_bad_per_seg: bool = False
    faster_z: float = 3.0


@dataclass
class SegRej:
    """Segment/epoch rejection settings (MATLAB ``params.segRej``)."""

    enabled: bool = False
    #: Use amplitude criterion.
    by_amplitude: bool = True
    # BUGFIX (local): original MATLAB HAPPE operates internally in microvolts,
    # so its default +-150 IS +-150uV. This port keeps MNE's native Volt scale
    # throughout (see steps/segment.py's reject(), which compares directly
    # against amp_min/amp_max with no unit conversion) -- +-150.0 here would
    # mean +-150 VOLTS, a threshold no real EEG signal ever reaches, silently
    # disabling amplitude-based epoch rejection entirely. Fixed to +-150e-6 V
    # (still +-150uV) to match the intended HAPPE default. Confirmed via a
    # full 54-session run before this fix: Pct_Epochs_Retained was 100.0 for
    # every single session (i.e. by_amplitude rejection never fired at all).
    amp_min: float = -150e-6
    amp_max: float = 150e-6
    #: Use joint-probability ("similarity") criterion.
    by_similarity: bool = False
    #: N std threshold (3 standard density, 2 low density).
    similarity_sd: float = 3.0
    #: Restrict criteria to a ROI subset of channels.
    roi_mode: str = "all"  # "all" | "include" | "exclude"
    roi_labels: List[str] = field(default_factory=list)
    #: NetStation "select good trials only" status filtering.
    select_good_only: bool = False


@dataclass
class Reref:
    """Re-referencing settings (MATLAB ``params.reref``)."""

    enabled: bool = True
    #: "average", "subset", or "rest".
    method: str = "average"
    subset: List[str] = field(default_factory=list)
    #: REST leadfield / electrode file; if None a spherical model is built.
    rest_leadfield: Optional[str] = None
    #: Mastoid channels to average as practical ref before REST.
    mastoids: List[str] = field(default_factory=list)


@dataclass
class BaseCorr:
    """Baseline correction settings (MATLAB ``params.baseCorr``)."""

    enabled: bool = False
    start_ms: float = -100.0
    end_ms: float = 0.0


@dataclass
class Vis:
    """Visualization settings (MATLAB ``params.vis``)."""

    enabled: bool = False
    #: ERP time window (ms) for timtopo, or spectra frequency range (Hz).
    time_window_ms: tuple = (0.0, 500.0)
    freq_range_hz: tuple = (1.0, 50.0)


@dataclass
class PreWaveletTddr:
    """DEVIATION (local, not in original MATLAB HAPPE): optional time-domain
    artifact correction inserted right before wavelet thresholding (after
    line-noise + band-pass, step 5), using Temporal Derivative Distribution
    Repair (Fishburn et al. 2019 -- originally an fNIRS motion-correction
    method, already validated and in production use for this project's
    fNIRS pipeline, see lib/fnirs_pipeline_v2.py::tddr()).

    Rationale: on this 2-channel dry-electrode wearable dataset, wavelet
    thresholding was found to sometimes destroy genuine alpha-band (8-13Hz)
    oscillatory structure -- traced to BayesShrink computing a single
    per-subband noise threshold from the WHOLE session, which large
    contact-potential step artifacts (broadband spectral leakage) can
    dominate even in subbands that also carry real alpha content (bior4.4
    detail level 4 spans ~7.6-15.25Hz at 244Hz sample rate, overlapping
    alpha almost entirely). TDDR corrects persistent step-like level shifts
    in the time domain, per channel, BELOW ``cutoff_hz`` only -- content
    above the cutoff (including alpha, if cutoff is kept below ~8Hz) passes
    through completely unmodified by construction, so this is a
    structurally conservative pre-clean, not a replacement for wavelet.

    Disabled by default -- must be explicitly enabled in config to activate.
    """

    enabled: bool = False
    #: Low-pass cutoff (Hz) separating the "trend" TDDR corrects from the
    #: untouched high-frequency residual. Keep < 8.0 to structurally spare
    #: the alpha band; default 4.0 targets delta/theta, the bands already
    #: found to be most heavily artifact-dominated on this dataset.
    cutoff_hz: float = 4.0
    filter_order: int = 3


@dataclass
class OutputFormat:
    """Output export settings (MATLAB ``params.outputFormat``)."""

    #: Any of {"fif", "set", "mat", "txt"}.
    formats: List[str] = field(default_factory=lambda: ["fif"])
    #: Also write trial-averaged text export.
    trial_average_txt: bool = False


# --------------------------------------------------------------------------- #
# Top-level Params
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    """Full pipeline configuration (MATLAB ``params`` struct)."""

    loadInfo: LoadInfo = field(default_factory=LoadInfo)
    chans: Chans = field(default_factory=Chans)
    lineNoise: LineNoise = field(default_factory=LineNoise)
    downsample: Downsample = field(default_factory=Downsample)
    filt: Filt = field(default_factory=Filt)
    badChans: BadChans = field(default_factory=BadChans)
    ecgone: Ecgone = field(default_factory=Ecgone)
    preWaveletTddr: PreWaveletTddr = field(default_factory=PreWaveletTddr)
    wavelet: Wavelet = field(default_factory=Wavelet)
    muscIL: MuscIL = field(default_factory=MuscIL)
    paradigm: Paradigm = field(default_factory=Paradigm)
    segment: Segment = field(default_factory=Segment)
    segRej: SegRej = field(default_factory=SegRej)
    reref: Reref = field(default_factory=Reref)
    baseCorr: BaseCorr = field(default_factory=BaseCorr)
    vis: Vis = field(default_factory=Vis)
    outputFormat: OutputFormat = field(default_factory=OutputFormat)

    # ---- validation ---------------------------------------------------- #
    def validate(self) -> "Params":
        if self.chans.mode not in ("all", "include", "exclude"):
            raise ConfigError(f"chans.mode invalid: {self.chans.mode!r}")
        if self.lineNoise.method not in ("cleanline", "notch"):
            raise ConfigError(f"lineNoise.method invalid: {self.lineNoise.method!r}")
        if self.badChans.order not in ("before", "after"):
            raise ConfigError(f"badChans.order invalid: {self.badChans.order!r}")
        if self.wavelet.method not in ("bayes", "legacy"):
            raise ConfigError(f"wavelet.method invalid: {self.wavelet.method!r}")
        if self.wavelet.threshold_rule not in ("hard", "soft"):
            raise ConfigError("wavelet.threshold_rule must be 'hard' or 'soft'")
        if self.reref.method not in ("average", "subset", "rest"):
            raise ConfigError(f"reref.method invalid: {self.reref.method!r}")
        for fmt in self.outputFormat.formats:
            if fmt not in ("fif", "set", "mat", "txt"):
                raise ConfigError(f"unknown output format: {fmt!r}")
        # DEVIATION (local): original HAPPE restricts soft-threshold wavelet
        # denoising to ERP paradigms only (better temporal precision for
        # amplitude/latency measures; hard threshold preferred for
        # resting-state spectral analysis). Relaxed here to allow an
        # explicit, requested experiment on resting-state data -- comparing
        # whether soft thresholding reduces destruction of genuine alpha-band
        # oscillatory structure in this 2-channel dry-electrode dataset (see
        # notebooks/eeg_happe_qc.ipynb wavelet-aggressiveness section).
        # NOT a recommendation to make soft the default for resting data.
        return self

    # ---- (de)serialization -------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Params":
        return _from_dict(cls, data)

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, default=_json_default)
        if path:
            Path(path).write_text(text)
        return text

    def to_yaml(self, path: Optional[str] = None) -> str:
        if yaml is None:  # pragma: no cover
            raise ConfigError("PyYAML is required for YAML serialization")
        text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path:
            Path(path).write_text(text)
        return text

    @classmethod
    def load(cls, path: str) -> "Params":
        """Load a config from a .json/.yaml/.yml file (format inferred)."""
        p = Path(path)
        raw = p.read_text()
        if p.suffix.lower() in (".yaml", ".yml"):
            if yaml is None:  # pragma: no cover
                raise ConfigError("PyYAML is required to load YAML configs")
            data = yaml.safe_load(raw)
        else:
            data = json.loads(raw)
        return cls.from_dict(data).validate()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not serializable: {type(obj)}")


def _from_dict(dc_type, data):
    """Recursively rebuild nested dataclasses from a plain dict.

    ``from __future__ import annotations`` turns ``field.type`` into a string,
    so we resolve concrete types with :func:`typing.get_type_hints` before
    testing for nested dataclasses.
    """
    if data is None:
        return dc_type()
    try:
        type_hints = get_type_hints(dc_type)
    except Exception:  # pragma: no cover
        type_hints = {f.name: f.type for f in fields(dc_type)}
    kwargs = {}
    for f in fields(dc_type):
        if f.name not in data:
            continue
        val = data[f.name]
        ftype = type_hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[f.name] = _from_dict(ftype, val)
        elif isinstance(val, list) and any(
            k in f.name for k in ("high_z", "time_window", "freq_range")
        ):
            kwargs[f.name] = tuple(val)
        else:
            kwargs[f.name] = val
    return dc_type(**kwargs)


def default_params(erp: bool = False) -> Params:
    """Return a sensible default config; ``erp=True`` flips paradigm/wavelet."""
    p = Params()
    if erp:
        p.paradigm.erp = True
        p.wavelet.threshold_rule = "soft"
        p.filt.highpass = 0.1
        p.filt.lowpass = 30.0
    return p.validate()
