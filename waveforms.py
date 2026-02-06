"""Macro waveform generators for phrase-level timing arcs.

Each function takes beat indices and returns drift in milliseconds.
All waveforms operate at the phrase level (bars), producing deterministic
timing arcs that model how real musicians shape phrases.
"""

import numpy as np


# ─── Sine ────────────────────────────────────────────────────────────────

def sine_drift(n_beats, phrase_length_beats, amplitude_ms):
    """Simple sinusoidal phrase arc.

    drift(t) = A * sin(2π * t / P)
    Symmetric acceleration and deceleration.
    """
    t = np.arange(n_beats)
    return amplitude_ms * np.sin(2 * np.pi * t / phrase_length_beats)


# ─── Todd Parabolic Arch ────────────────────────────────────────────────

def _todd_arch(phase, asymmetry=1.3):
    """Single Todd arch value. phase: 0→1 within a phrase. Returns -1 to +1.

    Accelerando for first 40% (negative drift = ahead of grid),
    ritardando for last 60% (positive drift = behind grid).
    Asymmetry > 1 makes ritardando dominate, matching measured pianist data.
    """
    midpoint = 0.4
    if phase < midpoint:
        x = phase / midpoint
        return -np.sin(x * np.pi / 2) ** (1.0 / asymmetry)
    else:
        x = (phase - midpoint) / (1 - midpoint)
        return np.sin(x * np.pi / 2) ** asymmetry


def todd_drift(n_beats, phrase_length_beats, amplitude_ms, asymmetry=1.3):
    """Todd's asymmetric parabolic arch (1985/1992).

    Gradual accelerando → sharper ritardando at phrase boundaries.
    Best match to measured piano performance data.
    """
    t = np.arange(n_beats)
    phases = (t % phrase_length_beats) / phrase_length_beats
    drift = np.array([_todd_arch(p, asymmetry) for p in phases])
    return drift * amplitude_ms


# ─── Friberg Power-Function Ritardando ──────────────────────────────────

def friberg_drift(n_beats, phrase_length_beats, amplitude_ms, q=2.5):
    """Friberg & Sundberg (1999) power-function ritardando.

    Steady for first 60% of phrase, power-law deceleration for last 40%.
    Derived from the kinematics of stopping runners.
    q = 2.0–3.0, default 2.5.
    """
    t = np.arange(n_beats)
    phases = (t % phrase_length_beats) / phrase_length_beats
    drift = np.zeros(n_beats)
    steady_end = 0.6

    for i, phase in enumerate(phases):
        if phase < steady_end:
            drift[i] = 0.0
        else:
            rit_phase = (phase - steady_end) / (1.0 - steady_end)
            drift[i] = amplitude_ms * (rit_phase ** q)

    return drift


# ─── Sawtooth (Rounded Falloff) ─────────────────────────────────────────

def sawtooth_drift(n_beats, phrase_length_beats, amplitude_ms, sharpness=5.0):
    """Gradual linear acceleration → sharp exponential snap-back.

    Captures the common "lean forward, snap back" feel.
    Sharpness controls how abrupt the deceleration is.
    """
    t = np.arange(n_beats)
    phases = (t % phrase_length_beats) / phrase_length_beats
    drift = np.zeros(n_beats)
    transition = 0.8
    exp_denom = 1.0 - np.exp(-sharpness)

    for i, phase in enumerate(phases):
        if phase < transition:
            drift[i] = -(phase / transition)
        else:
            snap_phase = (phase - transition) / (1 - transition)
            drift[i] = -1.0 + 2.0 * (1 - np.exp(-sharpness * snap_phase)) / exp_denom

    return drift * amplitude_ms


# ─── Golden Ratio Quasiperiodic ─────────────────────────────────────────

def golden_drift(n_beats, phrase_length_beats, amplitude_ms):
    """Quasiperiodic modulation via golden-ratio-related periods.

    Two sinusoidal components with periods related by φ = (1+√5)/2.
    Never exactly repeats. Locally periodic, globally aperiodic.
    From Ong (2020) quasiperiodic music framework.
    """
    phi = (1 + np.sqrt(5)) / 2
    period_1 = phrase_length_beats
    period_2 = phrase_length_beats * phi
    amp_1 = amplitude_ms
    amp_2 = amplitude_ms / phi

    t = np.arange(n_beats)
    return (amp_1 * np.sin(2 * np.pi * t / period_1) +
            amp_2 * np.sin(2 * np.pi * t / period_2))


# ─── Logistic Map (Edge of Chaos) ───────────────────────────────────────

def chaos_drift(n_beats, phrase_length_beats, amplitude_ms, r=3.85, x0=0.4,
                asymmetry=1.3):
    """Deterministic chaos via logistic map, smoothed with Todd arches.

    Each phrase gets a drift amplitude from x_{n+1} = r·x_n·(1−x_n).
    The value is applied as the peak of a Todd arch within that phrase,
    preventing discontinuous jumps between phrases.

    r=3.57: onset of chaos. r=4.0: fully chaotic. r≈3.85: complex with
    occasional periodic windows — "patterns with inexactitude" (Pressing 1988).
    """
    n_phrases = int(np.ceil(n_beats / phrase_length_beats))

    # Generate chaotic sequence
    x = np.zeros(n_phrases)
    x[0] = x0
    for i in range(1, n_phrases):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    chaos_amplitudes = (2.0 * x - 1.0) * amplitude_ms

    # Apply Todd arch shape within each phrase
    t = np.arange(n_beats)
    phrases = t // phrase_length_beats
    phases = (t % phrase_length_beats) / phrase_length_beats
    drift = np.zeros(n_beats)
    for i in range(n_beats):
        p_idx = min(int(phrases[i]), n_phrases - 1)
        arch_val = _todd_arch(phases[i], asymmetry)
        drift[i] = chaos_amplitudes[p_idx] * abs(arch_val) * np.sign(arch_val)

    return drift


# ─── Prime Superposition ────────────────────────────────────────────────

def prime_drift(n_beats, beats_per_bar, amplitude_ms):
    """Superimposed sine waves at prime-number bar periods.

    Primes: [3, 5, 7, 11, 13]. Amplitudes scale as 1/p.
    Combined period = 3×5×7×11×13 = 15,015 bars ≈ 500 min at 120 BPM 4/4.
    Inspired by Messiaen's technique for non-repetition.
    """
    primes = [3, 5, 7, 11, 13]
    t = np.arange(n_beats)
    drift = np.zeros(n_beats)
    for p in primes:
        period_beats = p * beats_per_bar
        amp = amplitude_ms / p
        drift += amp * np.sin(2 * np.pi * t / period_beats)

    return drift


# ─── Hierarchical Multi-Level ───────────────────────────────────────────

# Level amplitudes: each doubling of period gets ~√2 more amplitude
HIERARCHY_LEVELS = [
    (4, 0.5),    # 4-bar: 50% of base amplitude
    (8, 0.75),   # 8-bar: 75%
    (16, 1.0),   # 16-bar: 100%
    (32, 1.3),   # 32-bar: 130%
]


def apply_hierarchy(waveform_fn, n_beats, beats_per_bar, amplitude_ms,
                    **waveform_kwargs):
    """Apply a waveform at multiple hierarchical phrase levels simultaneously.

    macro_drift(t) = Σ waveform(t, period=L_k, amplitude=A_k) for each level k

    Args:
        waveform_fn: One of the drift functions (sine_drift, todd_drift, etc.)
        n_beats: Total number of beats.
        beats_per_bar: Beats per bar (e.g., 4 for 4/4).
        amplitude_ms: Base amplitude in ms (scaled by level multipliers).
        **waveform_kwargs: Extra args passed to waveform_fn.

    Returns:
        Combined drift array in ms.
    """
    drift = np.zeros(n_beats)
    for level_bars, amp_mult in HIERARCHY_LEVELS:
        phrase_beats = level_bars * beats_per_bar
        level_amp = amplitude_ms * amp_mult
        # Each waveform_fn has (n_beats, phrase_length_beats, amplitude_ms, ...)
        drift += waveform_fn(n_beats, phrase_beats, level_amp, **waveform_kwargs)

    return drift


# ─── Dispatch ────────────────────────────────────────────────────────────

WAVEFORM_REGISTRY = {
    'sine': sine_drift,
    'todd': todd_drift,
    'friberg': friberg_drift,
    'sawtooth': sawtooth_drift,
    'golden': golden_drift,
    'chaos': chaos_drift,
    'prime': prime_drift,
}


def get_macro_drift(name, n_beats, phrase_length_beats, beats_per_bar,
                    amplitude_ms, hierarchy=False, **kwargs):
    """Get macro drift for a named waveform.

    Args:
        name: Waveform name (sine, todd, friberg, sawtooth, golden, chaos, prime).
        n_beats: Total beats.
        phrase_length_beats: Phrase length in beats (for single-level mode).
        beats_per_bar: Beats per bar.
        amplitude_ms: Peak amplitude in ms.
        hierarchy: If True, use multi-level (4/8/16/32 bar) mode.
        **kwargs: Waveform-specific params (asymmetry, q, r, x0, sharpness).

    Returns:
        Drift array in ms, length n_beats.
    """
    if name == 'prime':
        # Prime always uses its own bar-based periods
        return prime_drift(n_beats, beats_per_bar, amplitude_ms)

    fn = WAVEFORM_REGISTRY[name]

    # Filter kwargs to only those the function accepts
    import inspect
    sig = inspect.signature(fn)
    valid_params = set(sig.parameters.keys()) - {'n_beats', 'phrase_length_beats',
                                                   'amplitude_ms', 'beats_per_bar'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    if hierarchy:
        return apply_hierarchy(fn, n_beats, beats_per_bar, amplitude_ms,
                               **filtered_kwargs)
    else:
        return fn(n_beats, phrase_length_beats, amplitude_ms, **filtered_kwargs)
