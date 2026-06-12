#!/bin/bash
set -euo pipefail
echo "Setting up VibeBot..."

# Install Python deps with uv (creates .venv from pyproject.toml + uv.lock)
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install it from https://docs.astral.sh/uv/"
    exit 1
fi
uv sync

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

echo "Done."
echo "Run with: uv run python -m src.bot"
