"""1/f noise generation via spectral shaping (FFT → shape → IFFT).

Produces beat-level timing deviations with controllable spectral exponent β.
β = 0: white noise, β = 1: pink (1/f) noise, β = 2: brown (1/f²) noise.
Target for human timing: β ≈ 0.85–1.0 (Hennig 2014 PNAS).
"""

import numpy as np


def generate_1f_noise(n_samples, beta=0.9, seed=None):
    """Generate 1/f^β noise via spectral shaping.

    Args:
        n_samples: Number of samples to generate.
        beta: Spectral exponent. 0=white, 0.9=pink (human timing), 2=brown.
        seed: Random seed for reproducibility.

    Returns:
        numpy array of n_samples values, zero-mean, unit variance.
    """
    rng = np.random.default_rng(seed)

    # Generate white noise in frequency domain
    n_fft = n_samples
    white_spectrum = rng.standard_normal(n_fft) + 1j * rng.standard_normal(n_fft)

    # Shape the spectrum: divide by f^(β/2) so PSD ~ 1/f^β
    freqs = np.fft.fftfreq(n_fft)
    # Avoid division by zero at DC
    freqs[0] = 1.0
    shaping = np.abs(freqs) ** (beta / 2.0)
    shaped_spectrum = white_spectrum / shaping
    # Zero out DC to ensure zero mean
    shaped_spectrum[0] = 0.0

    # Inverse FFT to get time-domain signal
    noise = np.fft.ifft(shaped_spectrum).real

    # Normalize to zero mean, unit variance
    noise = noise - np.mean(noise)
    std = np.std(noise)
    if std > 0:
        noise = noise / std

    return noise


def generate_pink_drift(n_beats, sd_ms=10.0, beta=0.9, seed=None):
    """Generate 1/f noise scaled to a target standard deviation in milliseconds.

    Args:
        n_beats: Number of beats.
        sd_ms: Target standard deviation in milliseconds.
        beta: Spectral exponent (default 0.9, Hennig 2014).
        seed: Random seed.

    Returns:
        numpy array of drift values in milliseconds, length n_beats.
    """
    noise = generate_1f_noise(n_beats, beta=beta, seed=seed)
    return noise * sd_ms


def measure_psd_slope(signal):
    """Measure the spectral exponent β of a signal via log-log PSD regression.

    Args:
        signal: 1D numpy array.

    Returns:
        Tuple of (beta, freqs, psd) where beta is the measured exponent.
    """
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    psd = np.abs(spectrum) ** 2 / n
    freqs = np.fft.rfftfreq(n)

    # Exclude DC and Nyquist, fit in log-log space
    mask = freqs > 0
    log_f = np.log10(freqs[mask])
    log_psd = np.log10(psd[mask] + 1e-30)

    # Linear regression
    coeffs = np.polyfit(log_f, log_psd, 1)
    measured_beta = -coeffs[0]  # PSD ~ f^(-β), so slope = -β

    return measured_beta, freqs[mask], psd[mask]
