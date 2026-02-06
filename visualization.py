"""Visualization: drift curve plots and PSD validation.

Generates:
- drift_curves.png: 3×3 grid of drift curves for all 9 metronomes
- psd_validation.png: log-log PSD of generated 1/f noise with reference slope
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from noise import measure_psd_slope


def plot_drift_curves(results, beats_per_bar, phrase_length_bars, output_path):
    """Plot 3×3 grid of drift curves for all metronome configurations.

    Args:
        results: Dict of {name: {'total_drift_ms': array, 'macro_drift_ms': array,
                 'pink_drift_ms': array, 'params': dict}}.
        beats_per_bar: Beats per bar.
        phrase_length_bars: Phrase length in bars.
        output_path: Path to save the PNG.
    """
    names = ['rigid', 'pink', 'sine_pink', 'todd_pink', 'friberg_pink',
             'sawtooth_pink', 'golden_pink', 'chaos_pink', 'prime_pink']

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Fractal Metronome Suite — Drift Curves', fontsize=16,
                 fontweight='bold', y=0.98)

    phrase_length_beats = phrase_length_bars * beats_per_bar

    for idx, name in enumerate(names):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        if name not in results:
            ax.set_visible(False)
            continue

        data = results[name]
        n_beats = len(data['total_drift_ms'])
        beats = np.arange(n_beats)

        # Total drift (blue)
        ax.plot(beats, data['total_drift_ms'], color='#2196F3', linewidth=0.5,
                alpha=0.8, label='Total drift')

        # Macro component only (red dashed)
        if data['macro_drift_ms'] is not None and np.any(data['macro_drift_ms'] != 0):
            ax.plot(beats, data['macro_drift_ms'], color='#F44336',
                    linewidth=1.5, linestyle='--', alpha=0.9, label='Macro arc')

        # ±1 SD band of pink component (gray shading)
        if data['pink_drift_ms'] is not None:
            sd = np.std(data['pink_drift_ms'])
            ax.axhspan(-sd, sd, color='gray', alpha=0.1, label=f'±1σ ({sd:.1f} ms)')

        # Bar lines (light vertical gridlines)
        for b in range(0, n_beats, beats_per_bar):
            ax.axvline(b, color='gray', linewidth=0.2, alpha=0.3)

        # Phrase boundaries (darker vertical lines)
        for b in range(0, n_beats, phrase_length_beats):
            ax.axvline(b, color='gray', linewidth=0.6, alpha=0.5)

        # Zero line
        ax.axhline(0, color='black', linewidth=0.3, alpha=0.5)

        # Labels
        display_name = name.replace('_', ' ').title()
        total_sd = np.std(data['total_drift_ms'])
        drift_range = data['total_drift_ms'].max() - data['total_drift_ms'].min()
        ax.set_title(f'{idx}. {display_name}\nSD={total_sd:.1f}ms  Range={drift_range:.1f}ms',
                     fontsize=10)
        ax.set_xlabel('Beat', fontsize=8)
        ax.set_ylabel('Drift (ms)', fontsize=8)
        ax.tick_params(labelsize=7)

        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')
        elif data['macro_drift_ms'] is not None and np.any(data['macro_drift_ms'] != 0):
            ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved drift curves → {output_path}')


def plot_psd_validation(pink_noise_signal, target_beta, output_path):
    """Plot log-log PSD of generated 1/f noise with reference slope.

    Args:
        pink_noise_signal: The raw 1/f noise array (unit variance).
        target_beta: Target spectral exponent.
        output_path: Path to save the PNG.
    """
    measured_beta, freqs, psd = measure_psd_slope(pink_noise_signal)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    log_f = np.log10(freqs)
    log_psd = np.log10(psd + 1e-30)

    # Actual PSD
    ax.scatter(log_f, log_psd, s=3, alpha=0.5, color='#2196F3', label='Measured PSD')

    # Reference line with target slope
    ref_line = -target_beta * log_f + np.mean(log_psd + target_beta * log_f)
    ax.plot(log_f, ref_line, 'r--', linewidth=2,
            label=f'Target slope −{target_beta:.2f}')

    # Measured fit line
    fit_line = -measured_beta * log_f + np.mean(log_psd + measured_beta * log_f)
    ax.plot(log_f, fit_line, 'g-', linewidth=1.5, alpha=0.8,
            label=f'Measured slope −{measured_beta:.2f}')

    ax.set_xlabel('log₁₀(frequency)', fontsize=12)
    ax.set_ylabel('log₁₀(PSD)', fontsize=12)
    ax.set_title(f'PSD Validation: Target β={target_beta:.2f}, '
                 f'Measured β={measured_beta:.2f}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved PSD validation → {output_path}')

    return measured_beta
