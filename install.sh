#!/bin/bash
set -e
echo "Setting up VibeBot..."

# Create conda env
conda create -n vibebot python=3.12 -y
eval "$(conda shell.bash hook)"
conda activate vibebot

# Install Python deps
pip install -r requirements.txt

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
