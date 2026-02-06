"""Parameter presets for different musical contexts.

Each preset adjusts the intensity, SD, and arc amplitude to match
the typical timing characteristics of the genre.
"""

PRESETS = {
    'edm': {
        'description': 'EDM/House: tight grid, subtle humanization (40% rule)',
        'pink_sd': 6.0,
        'arc_amplitude': 7.0,
        'intensity': 0.5,
    },
    'classical': {
        'description': 'Classical piano: expressive rubato, strong phrase arcs',
        'pink_sd': 12.0,
        'arc_amplitude': 18.0,
        'intensity': 1.2,
    },
    'jazz': {
        'description': 'Jazz: moderate swing feel, organic drift',
        'pink_sd': 10.0,
        'arc_amplitude': 12.0,
        'intensity': 1.0,
    },
    'rock': {
        'description': 'Rock/Pop: moderate tightness, natural feel',
        'pink_sd': 8.0,
        'arc_amplitude': 8.0,
        'intensity': 0.7,
    },
}

# Metronome configuration registry
METRONOME_CONFIGS = {
    'rigid': {
        'index': 0,
        'has_pink': False,
        'has_macro': False,
        'macro_waveform': None,
        'description': 'Perfect isochronous grid. Baseline for A/B comparison.',
    },
    'pink': {
        'index': 1,
        'has_pink': True,
        'has_macro': False,
        'macro_waveform': None,
        'description': 'Pure Hennig model. Beat-to-beat 1/f noise, β ≈ 0.9.',
    },
    'sine_pink': {
        'index': 2,
        'has_pink': True,
        'has_macro': True,
        'macro_waveform': 'sine',
        'description': 'Smooth sinusoidal phrase arc + 1/f noise.',
    },
    'todd_pink': {
        'index': 3,
        'has_pink': True,
        'has_macro': True,
        'macro_waveform': 'todd',
        'description': 'Asymmetric parabolic arch (Todd 1985) + 1/f noise.',
    },
    'friberg_pink': {
        'index': 4,
        'has_pink': True,
        'has_macro': True,
        'macro_waveform': 'friberg',
        'description': 'Power-function ritardando (Friberg 1999) + 1/f noise.',
    },
    'sawtooth_pink': {
        'index': 5,
        'has_pink': True,
        'has_macro': True,
        'macro_waveform': 'sawtooth',
        'description': 'Linear accel → sharp decel + 1/f noise.',
    },
    'golden_pink': {
        'index': 6,
        'has_pink': True,
        'has_macro': True,
        'macro_waveform': 'golden',
        'description': 'Golden-ratio quasiperiodic modulation + 1/f noise.',
    },
    'chaos_pink': {
        'index': 7,
        'has_pink': True,
        'has_macro': True,
        'macro_waveform': 'chaos',
        'description': 'Logistic map edge-of-chaos + 1/f noise.',
    },
    'prime_pink': {
        'index': 8,
        'has_pink': True,
        'has_macro': True,
        'macro_waveform': 'prime',
        'description': 'Superimposed prime-period sine waves + 1/f noise.',
    },
}

ALL_METRONOME_NAMES = sorted(METRONOME_CONFIGS.keys(),
                              key=lambda k: METRONOME_CONFIGS[k]['index'])
