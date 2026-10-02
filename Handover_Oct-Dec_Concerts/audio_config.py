"""Shared thresholds and defaults for the Windows detection-only monitor."""

import os

# Point to models folder in the same directory as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "audio_results")

# Audio
CAPTURE_RATE = 48000
TARGET_RATE = 16000

# Detection thresholds
YAMNET_THRESHOLD = 0.2
SINGING_THRESHOLD = 0.5

# Detection windows
YAMNET_WINDOW_SEC = 0.96
YAMNET_HOP_SEC = 0.48
YAMNET_MUSIC_CLASS = 132

# Stability
STABILITY_COUNT = 3

# Plotting
MIN_SCORE_DEFAULT = 0.1
