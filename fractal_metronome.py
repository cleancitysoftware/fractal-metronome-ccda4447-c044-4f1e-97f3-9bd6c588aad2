#!/usr/bin/env python3
"""Fractal Metronome Suite — CLI tool for generating humanized click tracks.

9 metronome configurations exploring 1/f fractal noise, phrase-level arcs,
quasiperiodic structures, and deterministic chaos in musical timing.

Usage:
    python fractal_metronome.py --bpm 125 --metronome all --visualize --validate
    python fractal_metronome.py --bpm 120 --metronome todd_pink --duration 300
"""

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

from noise import generate_1f_noise, generate_pink_drift, measure_psd_slope
from waveforms import get_macro_drift
from click import render_clicks
from presets import METRONOME_CONFIGS, ALL_METRONOME_NAMES, PRESETS


MAX_DRIFT_MS = 40.0  # Safety clamp


def parse_time_sig(ts_str):
    """Parse a time signature string like '4/4' into (numerator, denominator)."""
    parts = ts_str.split('/')
    if len(parts) != 2:
        raise ValueError(f"Invalid time signature: {ts_str}")
    return int(parts[0]), int(parts[1])


def generate_metronome(name, config, bpm, duration_s, beats_per_bar,
                        phrase_length_bars, pink_sd, pink_exponent,
                        arc_amplitude, intensity, hierarchy,
                        friberg_q, chaos_r, seed):
    """Generate drift arrays for a single metronome configuration.

    Returns:
        Dict with 'total_drift_ms', 'macro_drift_ms', 'pink_drift_ms',
        'nominal_times_s', 'params'.
    """
    beat_period = 60.0 / bpm
    n_beats = int(duration_s / beat_period)
    nominal_times = np.arange(n_beats) * beat_period
    phrase_length_beats = phrase_length_bars * beats_per_bar

    # Scale by intensity
    effective_pink_sd = pink_sd * intensity
    effective_arc_amp = arc_amplitude * intensity

    # ── Micro layer: 1/f noise ──
    pink_drift_ms = None
    if config['has_pink']:
        pink_drift_ms = generate_pink_drift(n_beats, sd_ms=effective_pink_sd,
                                             beta=pink_exponent, seed=seed)

    # ── Macro layer: phrase arc ──
    macro_drift_ms = None
    if config['has_macro']:
        waveform_name = config['macro_waveform']
        waveform_kwargs = {}
        if waveform_name == 'todd':
            waveform_kwargs['asymmetry'] = 1.3
        elif waveform_name == 'friberg':
            waveform_kwargs['q'] = friberg_q
        elif waveform_name == 'sawtooth':
            waveform_kwargs['sharpness'] = 5.0
        elif waveform_name == 'chaos':
            waveform_kwargs['r'] = chaos_r
            waveform_kwargs['x0'] = 0.4
            waveform_kwargs['asymmetry'] = 1.3

        macro_drift_ms = get_macro_drift(
            waveform_name, n_beats, phrase_length_beats, beats_per_bar,
            effective_arc_amp, hierarchy=hierarchy, **waveform_kwargs
        )

    # ── Combine layers ──
    total_drift_ms = np.zeros(n_beats)
    if pink_drift_ms is not None:
        total_drift_ms += pink_drift_ms
    if macro_drift_ms is not None:
        total_drift_ms += macro_drift_ms

    # Safety clamp
    total_drift_ms = np.clip(total_drift_ms, -MAX_DRIFT_MS, MAX_DRIFT_MS)

    # Fill in None arrays with zeros for uniform handling
    if pink_drift_ms is None:
        pink_drift_ms = np.zeros(n_beats)
    if macro_drift_ms is None:
        macro_drift_ms = np.zeros(n_beats)

    return {
        'total_drift_ms': total_drift_ms,
        'macro_drift_ms': macro_drift_ms,
        'pink_drift_ms': pink_drift_ms,
        'nominal_times_s': nominal_times,
        'n_beats': n_beats,
        'params': {
            'name': name,
            'bpm': bpm,
            'duration_s': duration_s,
            'beats_per_bar': beats_per_bar,
            'phrase_length_bars': phrase_length_bars,
            'pink_sd_ms': effective_pink_sd if config['has_pink'] else None,
            'pink_exponent': pink_exponent if config['has_pink'] else None,
            'arc_amplitude_ms': effective_arc_amp if config['has_macro'] else None,
            'hierarchy': hierarchy,
            'intensity': intensity,
        }
    }


def write_wav(result, beats_per_bar, output_path, sr=44100):
    """Render clicks and write WAV file."""
    audio = render_clicks(
        result['nominal_times_s'],
        result['total_drift_ms'],
        beats_per_bar,
        sr=sr
    )
    sf.write(output_path, audio, sr, subtype='PCM_16')
    duration = len(audio) / sr
    print(f'  Wrote {output_path} ({duration:.1f}s, {len(audio)} samples)')


def compute_stats(result):
    """Compute statistics for the report."""
    stats = {
        'measured_total_sd_ms': float(np.std(result['total_drift_ms'])),
        'drift_range_ms': [float(result['total_drift_ms'].min()),
                           float(result['total_drift_ms'].max())],
    }
    if result['params']['pink_sd_ms'] is not None:
        stats['measured_pink_sd_ms'] = float(np.std(result['pink_drift_ms']))
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Fractal Metronome Suite — generate humanized click tracks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fractal_metronome.py --bpm 125 --metronome all --visualize --validate
  python fractal_metronome.py --bpm 120 --metronome todd_pink --duration 300
  python fractal_metronome.py --bpm 130 --metronome chaos_pink --chaos-r 3.95 --intensity 1.5
        """
    )
    parser.add_argument('--bpm', type=float, default=125,
                        help='Tempo in BPM (default: 125)')
    parser.add_argument('--duration', type=float, default=120,
                        help='Duration in seconds (default: 120)')
    parser.add_argument('--time-sig', type=str, default='4/4',
                        help='Time signature (default: 4/4)')
    parser.add_argument('--metronome', type=str, default='pink',
                        choices=ALL_METRONOME_NAMES + ['all'],
                        help='Metronome configuration (default: pink)')
    parser.add_argument('--phrase-length', type=int, default=8,
                        help='Phrase length in bars (default: 8)')
    parser.add_argument('--intensity', type=float, default=1.0,
                        help='Scales all drift magnitudes (default: 1.0)')
    parser.add_argument('--pink-sd', type=float, default=10.0,
                        help='1/f noise SD in ms (default: 10.0)')
    parser.add_argument('--pink-exponent', type=float, default=0.9,
                        help='Spectral exponent β (default: 0.9)')
    parser.add_argument('--arc-amplitude', type=float, default=12.0,
                        help='Phrase arc peak amplitude in ms (default: 12.0)')
    parser.add_argument('--hierarchy', action='store_true',
                        help='Enable multi-level phrase arcs (4/8/16/32 bars)')
    parser.add_argument('--friberg-q', type=float, default=2.5,
                        help='Friberg curvature parameter (default: 2.5)')
    parser.add_argument('--chaos-r', type=float, default=3.85,
                        help='Logistic map parameter (default: 3.85)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Master random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='./metronomes/',
                        help='Output directory (default: ./metronomes/)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate drift curve plots')
    parser.add_argument('--validate', action='store_true',
                        help='Run PSD validation on generated noise')
    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                        help='Apply a genre preset (overrides sd/amplitude/intensity)')

    args = parser.parse_args()

    # Apply preset if specified
    if args.preset:
        p = PRESETS[args.preset]
        print(f'Applying preset: {args.preset} — {p["description"]}')
        args.pink_sd = p['pink_sd']
        args.arc_amplitude = p['arc_amplitude']
        args.intensity = p['intensity']

    beats_per_bar, _ = parse_time_sig(args.time_sig)
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which metronomes to generate
    if args.metronome == 'all':
        names = ALL_METRONOME_NAMES
    else:
        names = [args.metronome]

    print(f'Fractal Metronome Suite')
    print(f'  BPM: {args.bpm}  Duration: {args.duration}s  Time sig: {args.time_sig}')
    print(f'  Phrase: {args.phrase_length} bars  Intensity: {args.intensity}')
    print(f'  Pink SD: {args.pink_sd}ms  β: {args.pink_exponent}  Arc amp: {args.arc_amplitude}ms')
    print(f'  Seed: {args.seed}  Hierarchy: {args.hierarchy}')
    print(f'  Generating: {", ".join(names)}')
    print()

    results = {}
    raw_pink_noise = None  # For PSD validation

    for name in names:
        config = METRONOME_CONFIGS[name]
        idx = config['index']
        print(f'[{idx}] {name}: {config["description"]}')

        # Each metronome gets a derived seed so they share the same 1/f texture
        # but macro waveforms are deterministic (no seed needed)
        result = generate_metronome(
            name, config,
            bpm=args.bpm,
            duration_s=args.duration,
            beats_per_bar=beats_per_bar,
            phrase_length_bars=args.phrase_length,
            pink_sd=args.pink_sd,
            pink_exponent=args.pink_exponent,
            arc_amplitude=args.arc_amplitude,
            intensity=args.intensity,
            hierarchy=args.hierarchy,
            friberg_q=args.friberg_q,
            chaos_r=args.chaos_r,
            seed=args.seed,
        )

        # Save raw pink noise for PSD validation (from first pink-enabled config)
        if config['has_pink'] and raw_pink_noise is None:
            raw_pink_noise = generate_1f_noise(result['n_beats'],
                                                beta=args.pink_exponent,
                                                seed=args.seed)

        results[name] = result

        # Write WAV
        filename = f'{idx:02d}_{name}_{int(args.bpm)}bpm.wav'
        wav_path = os.path.join(args.output_dir, filename)
        write_wav(result, beats_per_bar, wav_path)

        stats = compute_stats(result)
        print(f'    Total SD: {stats["measured_total_sd_ms"]:.1f}ms  '
              f'Range: [{stats["drift_range_ms"][0]:.1f}, {stats["drift_range_ms"][1]:.1f}]ms')

        # Store stats for report
        result['stats'] = stats
        print()

    # ── Visualization ──
    if args.visualize:
        print('Generating visualizations...')
        from visualization import plot_drift_curves
        curves_path = os.path.join(args.output_dir, 'drift_curves.png')
        plot_drift_curves(results, beats_per_bar, args.phrase_length, curves_path)

    # ── PSD Validation ──
    if args.validate and raw_pink_noise is not None:
        print('Running PSD validation...')
        from visualization import plot_psd_validation
        psd_path = os.path.join(args.output_dir, 'psd_validation.png')
        measured_beta = plot_psd_validation(raw_pink_noise, args.pink_exponent,
                                            psd_path)
        print(f'  Target β: {args.pink_exponent:.2f}  '
              f'Measured β: {measured_beta:.2f}  '
              f'Error: {abs(measured_beta - args.pink_exponent):.3f}')
        print()

    # ── Report JSON ──
    beat_period = 60.0 / args.bpm
    n_beats = int(args.duration / beat_period)
    n_bars = n_beats // beats_per_bar

    report = {
        'bpm': args.bpm,
        'duration_s': args.duration,
        'n_beats': n_beats,
        'n_bars': n_bars,
        'time_signature': args.time_sig,
        'seed': args.seed,
        'phrase_length_bars': args.phrase_length,
        'intensity': args.intensity,
        'hierarchy': args.hierarchy,
        'configurations': {}
    }

    for name, result in results.items():
        cfg_report = dict(result['params'])
        cfg_report.update(result['stats'])
        # Remove non-serializable stuff
        cfg_report.pop('name', None)
        report['configurations'][name] = cfg_report

    if args.validate and raw_pink_noise is not None:
        beta_measured, _, _ = measure_psd_slope(raw_pink_noise)
        report['psd_validation'] = {
            'target_beta': args.pink_exponent,
            'measured_beta': float(beta_measured),
            'error': float(abs(beta_measured - args.pink_exponent)),
        }

    report_path = os.path.join(args.output_dir, 'metronome_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Saved report → {report_path}')
    print('Done.')


if __name__ == '__main__':
    main()
