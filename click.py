"""Click sound synthesis and placement for metronome audio generation.

Generates short percussive clicks via sine bursts with exponential decay.
Downbeats get a lower, louder click; regular beats get a higher, softer click.
"""

import numpy as np


def generate_click(frequency=1000, duration_ms=15, sr=44100, amplitude=0.8):
    """Synthesize a short percussive click.

    Sine burst with fast exponential decay — clean, distinct, and musical.

    Args:
        frequency: Pitch in Hz.
        duration_ms: Total duration in milliseconds.
        sr: Sample rate.
        amplitude: Peak amplitude (0.0–1.0).

    Returns:
        numpy array of audio samples.
    """
    n_samples = int(sr * duration_ms / 1000)
    t = np.arange(n_samples) / sr
    envelope = np.exp(-t * 1000 / duration_ms * 3)
    click = amplitude * np.sin(2 * np.pi * frequency * t) * envelope
    return click


def make_click_sounds(sr=44100):
    """Generate the two standard click sounds.

    Returns:
        Tuple of (downbeat_click, regular_click).
    """
    downbeat = generate_click(frequency=800, duration_ms=20, sr=sr, amplitude=0.9)
    regular = generate_click(frequency=1200, duration_ms=12, sr=sr, amplitude=0.7)
    return downbeat, regular


def place_click(output, click, sample_pos):
    """Place a click sound into the output buffer at a given sample position.

    Handles bounds checking — clips to buffer boundaries.

    Args:
        output: Output audio buffer (modified in-place).
        click: Click audio samples.
        sample_pos: Starting sample position.
    """
    if sample_pos < 0 or sample_pos >= len(output):
        return
    end = min(sample_pos + len(click), len(output))
    n = end - sample_pos
    output[sample_pos:end] += click[:n]


def render_clicks(nominal_times_s, drift_ms, beats_per_bar, sr=44100,
                  duration_s=None):
    """Render a complete click track from beat times and drift values.

    Args:
        nominal_times_s: Array of nominal beat times in seconds.
        drift_ms: Array of drift values in milliseconds per beat.
        beats_per_bar: Number of beats per bar (e.g., 4 for 4/4).
        sr: Sample rate.
        duration_s: Total duration in seconds (computed from last beat if None).

    Returns:
        numpy array of audio samples (mono, float64).
    """
    n_beats = len(nominal_times_s)
    if duration_s is None:
        duration_s = nominal_times_s[-1] + 1.0

    output_samples = int((duration_s + 1.0) * sr)
    output = np.zeros(output_samples)
    downbeat_click, regular_click = make_click_sounds(sr)

    for i in range(n_beats):
        actual_time = nominal_times_s[i] + drift_ms[i] / 1000.0
        sample_pos = int(actual_time * sr)
        is_downbeat = (i % beats_per_bar) == 0
        click = downbeat_click if is_downbeat else regular_click
        place_click(output, click, sample_pos)

    # Trim trailing silence (keep 0.5s padding after last click)
    last_beat_sample = int((nominal_times_s[-1] + drift_ms[-1] / 1000.0) * sr)
    trim_end = min(last_beat_sample + int(0.5 * sr), len(output))
    output = output[:trim_end]

    # Normalize to prevent clipping
    peak = np.max(np.abs(output))
    if peak > 0.95:
        output = output * 0.95 / peak

    return output
