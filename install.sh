#!/bin/bash
set -euo pipefail
echo "Setting up VibeBot..."

# Create conda env
if conda run -n vibebot python --version >/dev/null 2>&1; then
    echo "Using existing conda env: vibebot"
else
    conda create -n vibebot python=3.12 -y
fi
eval "$(conda shell.bash hook)"
conda activate vibebot

# Install Python deps
PIP_NO_CACHE_DIR=1 PYTHONNOUSERSITE=1 python -m pip install --no-user -r requirements.txt

# Check FFmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "WARNING: FFmpeg not found. Music playback requires FFmpeg."
    echo "Install with: sudo apt install ffmpeg"
fi

# Config
if [ ! -f config.yaml ]; then
    cp config.example.yaml config.yaml
    echo "Created config.yaml from template. Edit it with your Discord token."
fi

echo "Done. Activate with: conda activate vibebot"
echo "Run with: python -m src.bot"
