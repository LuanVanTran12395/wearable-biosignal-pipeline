from scipy.signal import welch
import numpy as np
import antropy as ant
import pywt
from antropy import spectral_entropy, higuchi_fd
from scipy import signal
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


def safe_log(x, eps=1e-8):
    return np.log(np.maximum(x, eps))

def _shannon_entropy_hist(x, bins=64, eps=1e-12):
    """Amplitude-distribution Shannon entropy (hist-based)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8:
        return 0.0
    hist, _ = np.histogram(x, bins=bins, density=False)
    p = hist.astype(np.float64)
    p = p / (p.sum() + eps)
    return float(-np.sum(p * np.log(p + eps)))


def _alpha_peak_features_psd(
    x,
    fs=244,
    alpha_band=(8, 12),
    nperseg=None,
    noverlap=None,
    prominence_rel=0.10,   # relative to max(alpha PSD)
    min_prominence=1e-12  # fallback
):
    """
    Returns 3 features:
    [alpha_peak_freq, alpha_peak_power, alpha_peak_prominence]
    If no peak found -> zeros.
    """
    x = np.asarray(x, dtype=float)
    if x.size < int(2 * fs):
        return np.zeros(3, dtype=float)

    if nperseg is None:
        nperseg = min(len(x), int(2 * fs))
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, psd = signal.welch(
        x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant"
    )

    lo, hi = alpha_band
    idx = (freqs >= lo) & (freqs <= hi)
    if not np.any(idx):
        return np.zeros(5, dtype=float)

    f_a = freqs[idx]
    p_a = psd[idx]
    if p_a.size < 3 or np.all(p_a <= 0):
        return np.zeros(5, dtype=float)

    # Find peaks in alpha PSD
    prom = max(float(prominence_rel * np.max(p_a)), float(min_prominence))
    peaks, props = signal.find_peaks(p_a, prominence=prom)

    if peaks.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    best_i = int(peaks[np.argmax(p_a[peaks])])
    alpha_pf = float(f_a[best_i])
    alpha_pp = float(p_a[best_i])
    alpha_pr = float(props["prominences"][np.argmax(p_a[peaks])])


    return np.array([alpha_pf, alpha_pp, alpha_pr], dtype=float)

# Function to extract single-channel features
def extract_dwt_features(data, fs = 244, wavelet='db4', level=5):
    coeffs = pywt.wavedec(data, wavelet, level=level)

    dwt_features = []

    for i, c in enumerate(coeffs):

        if len(c) == 0:
            dwt_features.extend([0] * 20)
            continue

        # ----- Entropy -----
        dwt_features.append(np.sum(np.log(c**2 + 1e-8)))                                 # wavelet log energy
        dwt_features.append(_shannon_entropy_hist(c))                                    # wavelet shannon entropy
        dwt_features.append(spectral_entropy(c, sf=fs, method="welch", normalize=True))  # wavelet spectral entropy

        # Wavelet entropy (prob-based)
        p = (c**2) / (np.sum(c**2) + 1e-8)
        dwt_features.append(-np.sum(p * np.log(p + 1e-8)))

        # Fractal dimension (optional)
        dwt_features.append(higuchi_fd(c))

    return np.array(dwt_features, dtype = float)



def bandpower(data, fs, band):
    fmin, fmax = band

    # auto-fix nperseg
    nperseg = min(256, len(data))

    f, Pxx = signal.welch(data, fs=fs, nperseg=nperseg)
    idx = (f >= fmin) & (f <= fmax)
    return np.sum(Pxx[idx])

def compute_plv(x, y):
    phase_x = np.angle(signal.hilbert(x))
    phase_y = np.angle(signal.hilbert(y))
    return np.abs(np.mean(np.exp(1j*(phase_x - phase_y))))


# Function to extract cross-channel features
def extract_cross_features(x, y, fs, bandpass_AF3_alpha, bandpass_AF4_alpha, bandpass_AF3_beta, bandpass_AF4_beta):

    features = []

    # ======== 3. Entropy difference ========
    hx = entropy(np.abs(x) + 1e-8)
    hy = entropy(np.abs(y) + 1e-8)
    features.append(hx - hy)
    features.append(hx + hy)   # joint entropy-like

    # ======== 4. Band-power based cross features ========
    alpha_x = bandpower(x, fs, (8, 12))
    alpha_y = bandpower(y, fs, (8, 12))
    beta_x  = bandpower(x, fs, (13, 30))
    beta_y  = bandpower(y, fs, (13, 30))
    theta_x = bandpower(x, fs, (4, 7))
    theta_y = bandpower(y, fs, (4, 7))

    # --- Alpha asymmetry ---
    FAA = np.log(alpha_y + 1e-8) - np.log(alpha_x + 1e-8)
    features.append(FAA)

    # --- Frontal Engagement Index ---
    FEI = (beta_x + beta_y) / (alpha_x + alpha_y + theta_x + theta_y + 1e-8)
    features.append(FEI)

    # ======== 5. Coherence ========
    f, Cxy = signal.coherence(x, y, fs=fs, nperseg=min(256, max(len(x), len(y))))

    # Band-limited coherence
    alpha_coh = np.mean(Cxy[(f >= 8) & (f <= 12)])
    beta_coh  = np.mean(Cxy[(f >= 13) & (f <= 30)])

    features.extend([alpha_coh, beta_coh])

    # ======== 6. PLV ========
    plv_alpha = compute_plv(
        bandpass_AF3_alpha.filter_data(x),
        bandpass_AF4_alpha.filter_data(y)
    )
    plv_beta = compute_plv(
        bandpass_AF3_beta.filter_data(x),
        bandpass_AF4_beta.filter_data(y)
    )

    features.extend([plv_alpha, plv_beta])

    return np.array(features)