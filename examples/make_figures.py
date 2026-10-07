"""
Generate the README figures from SYNTHETIC signals.

Every figure runs simulated data through the real functions in `lib/`
(TDDR, PPG peak detection + HRV, relaxation-onset and drowsy-episode
detection). No study data is used -- the numbers on these plots describe the
simulation, not any participant or study result.

Run from the repo root:
    python examples/make_figures.py
Outputs: docs/figures/*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "lib"))
import fnirs_pipeline_v2 as fp  # noqa: E402
import ppg_hrv  # noqa: E402
from within_session_dynamics import detect_drowsy_episodes, detect_relaxation_onset  # noqa: E402

OUT_DIR = REPO_ROOT / "docs" / "figures"
RNG = np.random.default_rng(7)

# ---- palette (validated reference palette, light mode) ----------------------
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
PHASE_BAND = "#f0efec"
BLUE = "#2a78d6"      # categorical slot 1 / diverging cool pole
ORANGE = "#eb6834"    # categorical slot 2
RED = "#e34948"       # diverging warm pole
BEFORE_GRAY = "#a3a29c"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 2.0, "legend.frameon": False,
})
SYNTH_NOTE = "Synthetic data — illustrates the method, not study results"


def _finish(fig, name: str, title: str) -> None:
    fig.suptitle(title, x=0.01, y=1.03, ha="left", fontsize=13, fontweight="bold", color=TEXT)
    fig.text(0.01, -0.04, SYNTH_NOTE, ha="left", va="bottom", fontsize=8.5, color=TEXT_2, style="italic")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT_DIR / name).relative_to(REPO_ROOT))


def _shade_phases(ax, bounds, label_y=None):
    """bounds: list of (start, end, label). MEDITATION gets a neutral band."""
    for start, end, label in bounds:
        if label == "MEDITATION":
            ax.axvspan(start, end, color=PHASE_BAND, zorder=0, lw=0)
        if label_y is not None:
            ax.text((start + end) / 2, label_y, label.replace("_", " "), ha="center", va="bottom",
                    fontsize=8, color=TEXT_2, transform=ax.get_xaxis_transform())


# ---------------------------------------------------------------------------
# 1. fNIRS: TDDR motion correction
# ---------------------------------------------------------------------------
def fig_fnirs_tddr() -> None:
    fs = 10.0
    t = np.arange(0, 300, 1 / fs)
    physiology = 0.010 * np.sin(2 * np.pi * 0.05 * t) + 0.004 * np.sin(2 * np.pi * 0.1 * t + 1)
    drift = 0.00004 * t
    noise = 0.0015 * RNG.standard_normal(t.size)
    clean = physiology + drift
    od = clean + noise
    # motion artefacts: two baseline shifts and one spike
    od = od + 0.03 * (t >= 95) - 0.022 * (t >= 205)
    od = od + 0.025 * np.exp(-0.5 * ((t - 150) / 0.6) ** 2)

    corrected, _, _ = fp.tddr(od, fs)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(t, od, color=BEFORE_GRAY, lw=1.5, label="Raw optical density")
    ax.plot(t, corrected, color=BLUE, lw=2.0, label="After TDDR")
    for x, txt in [(95, "step"), (150, "spike"), (205, "step")]:
        ax.annotate(txt, xy=(x, od[int(x * fs) + 5]), xytext=(x + 6, od[int(x * fs) + 5] + 0.008),
                    fontsize=8.5, color=TEXT_2, arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.8))
    ax.text(t[-1] + 3, od[-1], "Raw", color=TEXT_2, va="center", fontsize=9)
    ax.text(t[-1] + 3, corrected[-1], "TDDR", color=TEXT, va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Optical density (a.u.)")
    ax.set_xlim(0, 318)
    ax.legend(loc="upper right", ncols=2)
    ax.set_title("Baseline shifts and spikes are repaired without knowing when they happen", fontweight="normal",
                 color=TEXT_2, fontsize=10)
    _finish(fig, "fnirs_tddr.png", "fNIRS motion correction (TDDR)")


# ---------------------------------------------------------------------------
# 2. PPG: peak detection -> IBI -> RMSSD per phase
# ---------------------------------------------------------------------------
def _synthetic_ppg(fs: float, phases):
    """Pulse train whose beat-to-beat variability follows breathing (RSA);
    slower, deeper breathing during MEDITATION -> larger RSA."""
    beat_times, t_now = [], 0.0
    for start, end, label in phases:
        resp_hz, rsa = (0.1, 0.09) if label == "MEDITATION" else (0.25, 0.012)
        while t_now < end:
            ibi = 0.86 + rsa * np.sin(2 * np.pi * resp_hz * t_now) + 0.005 * RNG.standard_normal()
            t_now += ibi
            beat_times.append(t_now)
    total = phases[-1][1]
    t = np.arange(0, total, 1 / fs)
    sig = np.zeros_like(t)
    for bt in beat_times:
        sig += np.exp(-0.5 * ((t - bt) / 0.07) ** 2) + 0.35 * np.exp(-0.5 * ((t - bt - 0.28) / 0.1) ** 2)
    sig += 0.25 * np.sin(2 * np.pi * 0.03 * t) + 0.05 * RNG.standard_normal(t.size)
    return t, 2000 + 400 * sig


def fig_ppg_hrv() -> None:
    fs = 100.0
    phases = [(0, 120, "REST_BEFORE"), (120, 420, "MEDITATION"), (420, 540, "REST_AFTER")]
    t, raw = _synthetic_ppg(fs, phases)
    filtered = ppg_hrv.filter_ppg(raw, fs)
    peaks = ppg_hrv.detect_peaks(filtered, fs)
    ibi_s, ibi_t = ppg_hrv.compute_ibi(peaks, fs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [1, 1.3], "hspace": 0.55})

    win = (t >= 130) & (t < 145)
    pk = peaks[(t[peaks] >= 130) & (t[peaks] < 145)]
    ax1.plot(t[win], filtered[win], color=BLUE, lw=1.6)
    ax1.plot(t[pk], filtered[pk], "o", ms=6, color=ORANGE, mec=SURFACE, mew=1.5, label="Detected systolic peak")
    ax1.set_title("1  Band-pass 0.5–8 Hz, then adaptive-prominence peak detection (orange = detected peak)", fontsize=10)
    ax1.set_xlabel("Time (s)")
    ax1.set_yticks([])
    ax1.grid(False)

    _shade_phases(ax2, phases, label_y=1.01)
    ax2.plot(ibi_t, ibi_s * 1000, color=BLUE, lw=1.4)
    for start, end, label in phases:
        m = (ibi_t >= start) & (ibi_t < end)
        hrv = ppg_hrv.time_domain_hrv(ibi_s[m])
        ax2.text((start + end) / 2, 0.03, f"RMSSD {hrv['rmssd_ms']:.0f} ms · HR {hrv['hr_mean_bpm']:.0f} bpm",
                 ha="center", fontsize=8.5, color=TEXT, transform=ax2.get_xaxis_transform())
    ax2.set_title("2  Inter-beat intervals → time-domain HRV per protocol phase", fontsize=10, pad=16)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Inter-beat interval (ms)")
    ax2.set_xlim(0, 540)
    lo, hi = ax2.get_ylim()
    ax2.set_ylim(lo - 0.18 * (hi - lo), hi)
    _finish(fig, "ppg_hrv.png", "PPG → heart-rate variability")


# ---------------------------------------------------------------------------
# 3. EEG within-session dynamics: relaxation onset + drowsy episode
# ---------------------------------------------------------------------------
def fig_eeg_dynamics() -> None:
    epoch = 2.0
    n_rest, n_med = 60, 300  # 2 min rest, 10 min meditation, 2 min rest
    rest_b = 0.22 + 0.02 * RNG.standard_normal(n_rest)
    ramp = np.clip((np.arange(n_med) - 45) / 60, 0, 1)
    med = 0.19 + 0.12 * ramp + 0.02 * RNG.standard_normal(n_med)
    rest_a = 0.27 + 0.025 * RNG.standard_normal(n_rest)
    alpha = np.concatenate([rest_b, med, rest_a])

    theta_alpha_b = 0.9 + 0.08 * RNG.standard_normal(n_rest)
    theta_alpha_m = 0.85 + 0.08 * RNG.standard_normal(n_med)
    theta_alpha_m[190:225] += 1.3  # a short drowsy episode mid-meditation
    theta_alpha = np.concatenate([theta_alpha_b, theta_alpha_m, 0.88 + 0.08 * RNG.standard_normal(n_rest)])

    baseline_alpha = float(np.median(rest_b))
    onset = detect_relaxation_onset(med, baseline_alpha, epoch_sec=epoch)
    drowsy = detect_drowsy_episodes(theta_alpha_m, float(np.median(theta_alpha_b)), epoch_sec=epoch)

    t = np.arange(alpha.size) * epoch / 60  # minutes
    t_med0 = n_rest * epoch / 60
    phases = [(0, t_med0, "REST_BEFORE"), (t_med0, (n_rest + n_med) * epoch / 60, "MEDITATION"),
              ((n_rest + n_med) * epoch / 60, t[-1] + epoch / 60, "REST_AFTER")]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"hspace": 0.35})
    _shade_phases(ax1, phases, label_y=1.01)
    ax1.plot(t, alpha, color=BLUE, lw=1.2, alpha=0.9)
    ax1.axhline(baseline_alpha, color=TEXT_2, lw=1, ls="--")
    ax1.text(13.9, baseline_alpha - 0.012, "REST_BEFORE baseline", fontsize=8.5, color=TEXT_2, ha="right", va="top")
    if onset["onset_reached"]:
        x_on = t_med0 + onset["onset_latency_sec"] / 60
        ax1.axvline(x_on, color=ORANGE, lw=2)
        ax1.text(x_on - 0.15, 0.36, f"Relaxation onset\n{onset['onset_latency_sec']:.0f} s into meditation",
                 fontsize=8.5, color=TEXT, va="top", ha="right")
    ax1.set_ylabel("Relative alpha power")
    ax1.set_title("Alpha rises above the person's own baseline and stays there", fontsize=10, pad=16,
                  fontweight="normal", color=TEXT_2)

    _shade_phases(ax2, phases)
    ax2.plot(t, theta_alpha, color=BLUE, lw=1.2)
    ax2.axhline(drowsy["threshold"], color=TEXT_2, lw=1, ls="--")
    ax2.text(0.3, drowsy["threshold"] + 0.05, "2 × baseline threshold", fontsize=8.5, color=TEXT_2)
    if drowsy["longest_run_epoch_idx"] is not None:
        s = t_med0 + drowsy["longest_run_epoch_idx"] * epoch / 60
        e = s + drowsy["longest_run_sec"] / 60
        ax2.axvspan(s, e, color=ORANGE, alpha=0.18, lw=0)
        ax2.text(e + 0.15, 0.97, f"Drowsy episode\n{drowsy['longest_run_sec']:.0f} s", ha="left",
                 va="top", fontsize=8.5, color=TEXT, transform=ax2.get_xaxis_transform())
    ax2.set_ylabel("Theta / alpha ratio")
    ax2.set_xlabel("Time (min)")
    ax2.set_xlim(0, t[-1] + epoch / 60)
    _finish(fig, "eeg_session_dynamics.png", "EEG dynamics within one session")


# ---------------------------------------------------------------------------
# 4. Personalized report: tier 1 (this session) + tier 2 (own trend)
# ---------------------------------------------------------------------------
def fig_personal_report() -> None:
    items = ["Stress (↓)", "Body tension (↓)", "Sleepiness (↓)", "Calmness", "Focus", "Mood"]
    # positive = improvement (stress/tension/sleepiness already sign-flipped)
    delta = np.array([1.6, 1.1, -0.4, 1.8, 0.7, 1.0])

    days = np.arange(1, 13)
    rmssd = 38 + 0.9 * days + 4 * RNG.standard_normal(days.size)
    valid = np.ones_like(days, dtype=bool)
    valid[[4, 9]] = False  # sessions failing QC are shown but not used for the trend

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.45})

    colors = [BLUE if d >= 0 else RED for d in delta]
    y = np.arange(len(items))[::-1]
    ax1.barh(y, delta, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
    ax1.axvline(0, color=TEXT_2, lw=1)
    for yi, d in zip(y, delta):
        ax1.text(d + (0.08 if d >= 0 else -0.08), yi, f"{d:+.1f}", va="center",
                 ha="left" if d >= 0 else "right", fontsize=9, color=TEXT)
    ax1.set_yticks(y, items)
    ax1.set_xlim(-1.2, 2.4)
    ax1.grid(axis="x", color=GRID)
    ax1.grid(axis="y", visible=False)
    ax1.set_xlabel("Change after session (points, + = better)")
    ax1.set_title("Tier 1 · This session: pre → post", fontsize=10)

    ax2.plot(days[valid], rmssd[valid], "o", ms=8, color=BLUE, mec=SURFACE, mew=2, label="Valid session")
    ax2.plot(days[~valid], rmssd[~valid], "o", ms=8, mfc=SURFACE, mec=BEFORE_GRAY, mew=1.5,
             label="Excluded (signal QC fail)")
    coef = np.polyfit(days[valid], rmssd[valid], 1)
    ax2.plot(days, np.polyval(coef, days), color=BLUE, lw=2, alpha=0.6)
    ax2.text(days[-1] + 0.3, np.polyval(coef, days[-1]), f"{coef[0]:+.1f} ms/day", va="center",
             fontsize=9, color=TEXT)
    ax2.set_xlim(0.5, 14.5)
    ax2.set_xticks(days)
    ax2.set_xlabel("Study day")
    ax2.set_ylabel("RMSSD during meditation (ms)")
    ax2.set_title("Tier 2 · Own trend (shown only with ≥ 5 valid sessions)", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8.5)
    _finish(fig, "personal_report.png", "Personalized report for one participant")


if __name__ == "__main__":
    fig_fnirs_tddr()
    fig_ppg_hrv()
    fig_eeg_dynamics()
    fig_personal_report()
