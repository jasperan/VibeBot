# VibeBot Design Document

**Date:** 2026-03-28
**Status:** Approved

## Overview

VibeBot is a Discord voice bot that combines real-time voice conversation (via Microsoft VibeVoice) with YouTube music playback. Pure voice interaction for conversation, slash commands for music control. Mutually exclusive modes: voice or music, never both.

## Architecture

Monolith. Single Python process (discord.py) calls three external services over HTTP/WebSocket:

```
Discord <-> VibeBot (discord.py) <-> VibeVoice ASR (vLLM server, port 8000)
                                 <-> vLLM Qwen3.5:9B (port 8010)
                                 <-> VibeVoice TTS (WebSocket, port 8001)
                                 <-> yt-dlp (subprocess)
```

## Project Structure

```
VibeBot/
├── src/
│   ├── bot.py              # VibeBotClient, startup, cog loader
│   ├── voice_pipeline.py   # ASR -> LLM -> TTS orchestration
│   ├── asr_client.py       # VibeVoice ASR (vLLM OpenAI-compatible endpoint)
│   ├── tts_client.py       # VibeVoice Realtime TTS (WebSocket streaming)
│   ├── llm_client.py       # vLLM Qwen3.5:9B (OpenAI-compatible endpoint)
│   ├── audio.py            # Discord AudioSource/AudioSink adapters (PCM conversion)
│   └── cogs/
│       ├── __init__.py
│       ├── voice.py        # VoiceCog: /join, /leave, /listen (toggle)
│       └── music.py        # MusicCog: /play, /skip, /queue, /pause, /resume, /stop, /np
├── config.example.yaml
├── config.yaml             # (gitignored, shares Discord token with LeagueSpy)
├── requirements.txt
├── install.sh
├── README.md
├── docs/plans/
└── tests/
```

## Voice Pipeline

When `/listen` is toggled on:

1. **CAPTURE**: discord.py VoiceClient receives Opus packets per user
2. **DECODE**: Opus -> PCM (48kHz stereo) -> resample to 16kHz mono WAV
3. **VAD**: webrtcvad splits audio into utterances (~300ms silence threshold)
4. **ASR**: Send utterance to VibeVoice ASR vLLM endpoint (POST /v1/chat/completions)
5. **LLM**: Send transcript to Qwen3.5:9B vLLM (POST /v1/chat/completions, think: false)
6. **TTS**: Stream LLM response to VibeVoice Realtime WebSocket -> receive 24kHz PCM chunks
7. **UPSAMPLE**: 24kHz mono -> 48kHz stereo PCM
8. **PLAY**: Feed PCM into discord.py VoiceClient as AudioSource

Key details:
- VAD uses `webrtcvad` (lightweight, no GPU). Aggressiveness level 2.
- Rolling context window of last 5 exchanges kept in memory for conversational continuity.
- Streaming TTS: first sentence starts playing before full LLM response completes.
- One speaker at a time: first complete utterance wins, overlapping audio dropped.
- Max utterance length: 30 seconds (truncated before ASR).

## Music System

Slash commands only (no voice commands for music):

- `/play <query>` - yt-dlp searches YouTube, FFmpeg streams audio to Discord
- `/queue` - shows current queue (in-memory deque)
- `/skip` - stops current track, plays next
- `/pause` / `/resume` - toggles FFmpeg stream
- `/stop` - clears queue, disconnects
- `/np` - shows now-playing (title, duration, requester)

Queue is in-memory (Python deque). Lost on restart. No database.

## Mutual Exclusion

Voice and music modes cannot run simultaneously:
- `/listen` checks `MusicCog.is_playing` -> rejects if true
- `/play` checks `VoiceCog.is_listening` -> rejects if true
- `/leave` clears both modes
- Both cogs share an `asyncio.Lock` on the bot instance

## Configuration

```yaml
discord:
  token: "BOT_TOKEN"  # same token as LeagueSpy

voice:
  asr_url: "http://localhost:8000/v1"
  tts_url: "ws://localhost:8001/ws/tts"
  vad_aggressiveness: 2
  silence_threshold_ms: 300
  context_window: 5

llm:
  base_url: "http://localhost:8010/v1"
  model: "qwen3.5:9b"
  max_tokens: 150
  system_prompt: "You are VibeBot, a friendly voice assistant in a Discord voice channel. Keep responses concise and conversational, 1-2 sentences max."

music:
  max_queue_size: 50
  default_volume: 0.5
```

Service startup order:
1. VibeVoice ASR vLLM server (port 8000)
2. VibeVoice Realtime TTS server (port 8001)
3. vLLM Qwen3.5:9B (port 8010)
4. VibeBot

## Error Handling

**Voice pipeline:**
- ASR/LLM down: log warning, stay in listening mode, don't respond
- TTS down: fall back to posting text response in channel (degraded mode)
- Empty transcript from ASR: skip LLM call entirely
- User speaks >30s: truncate before ASR

**Music:**
- yt-dlp fails: ephemeral "Couldn't find that track" message
- FFmpeg dies mid-song: auto-skip to next with brief error message
- Bot disconnected from voice: clear all state, reset queue

## Dependencies

- `discord.py[voice]` (opus/nacl for voice)
- `httpx` (async HTTP for ASR + LLM)
- `websockets` (TTS WebSocket client)
- `webrtcvad` (voice activity detection)
- `yt-dlp` (YouTube audio extraction)
- `PyYAML` (config)
- `numpy` (PCM resampling)

## Tech Stack

- Python 3.12+, conda env `vibebot`
- discord.py 2.7+
- VibeVoice ASR 7B + Realtime TTS 0.5B (Microsoft, MIT license)
- vLLM + Qwen3.5:9B (conversation LLM)
- FFmpeg (music streaming)
- webrtcvad (voice activity detection)
