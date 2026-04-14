# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

VibeBot is a Discord voice bot with two modes:
- **Voice mode** (`/listen`): real-time speech-to-text -> LLM -> text-to-speech pipeline with streaming playback and barge-in support
- **Music mode** (`/play`): YouTube audio streaming with queue management
- **Voice-commanded music**: in voice mode, say "play Bohemian Rhapsody" and the LLM triggers music via tool calling

Voice mode uses three external GPU services (ASR, TTS, LLM) that are lazy-started on `/listen` and shut down on `/leave` to free VRAM.

## Commands

```bash
# Setup
bash install.sh                    # Creates conda env, installs deps, copies config template
conda activate vibebot

# Run
python -m src.bot                  # Start the bot (reads config.yaml)

# Tests
pytest tests/ -v                          # Full suite (69 tests)
pytest tests/test_voice_pipeline.py -v    # Single module
pytest tests/test_e2e_real.py -v -s       # Real E2E against live Ollama (requires running Ollama)
```

## Architecture

```
Discord audio (48kHz stereo)
  -> AudioSink: stereo_to_mono + resample to 16kHz
  -> VAD (webrtcvad): detect speech boundaries via silence threshold
  -> Barge-in: if user speaks while bot is responding, stop TTS immediately
  -> VoicePipeline.process_utterance():
       ASRClient.transcribe()     POST PCM-as-WAV-base64 to VibeVoice ASR
       LLMClient.generate()       POST to LLM with optional tool definitions
         -> tool calls: queued as pending actions (music control)
         -> text: routed to TTS
       TTSClient.synthesize()     WebSocket stream to VibeVoice TTS
       resample 24kHz mono -> 48kHz stereo
  -> Streaming playback: PCMStreamSource starts playing immediately, chunks fed as they arrive
  -> Pending actions: after TTS drains, execute music commands from tool calls
```

**Key design decisions:**
- ServiceManager spawns ASR/TTS/LLM as subprocesses, polls health endpoints until ready
- LLM client supports two backends: `"openai"` (vLLM with `chat_template_kwargs`) and `"ollama"` (native `/api/chat` with `think: false`)
- Streaming TTS: playback starts before all audio arrives (silence fills gaps)
- Barge-in: listen loop runs `_handle_utterance` as a fire-and-forget task; VAD detecting speech sets `_is_responding = False` to interrupt
- Voice-commanded music: LLM receives `MUSIC_TOOLS` (play_music, skip_song, stop_music, set_volume); tool calls are queued and executed after TTS finishes
- Conversation context is a sliding window (default 5 turns = 10 messages)

## Source Layout

- `src/bot.py` -- entry point, loads config, registers VoiceCog + MusicCog + AdminCog
- `src/cogs/voice.py` -- `/join`, `/leave`, `/listen`, `/recap`, `/personality` + AudioSink + VAD + barge-in + tool execution
- `src/cogs/music.py` -- `/play`, `/skip`, `/queue`, `/pause`, `/resume`, `/stop`, `/np`, `/volume` + public `voice_play/voice_skip/voice_stop` methods
- `src/cogs/admin.py` -- `/status` service health dashboard
- `src/voice_pipeline.py` -- orchestrates ASR -> LLM (with tools) -> TTS, conversation summary
- `src/asr_client.py` -- HTTP client, wraps PCM in WAV+base64 for VibeVoice ASR
- `src/llm_client.py` -- dual-backend client (OpenAI-compatible + Ollama native), `LLMResponse` dataclass with text + tool_calls
- `src/tts_client.py` -- WebSocket client, streams PCM chunks from VibeVoice Realtime
- `src/audio.py` -- numpy-based PCM resampling, stereo/mono conversion, PCMStreamSource
- `src/service_manager.py` -- subprocess lifecycle, health polling, graceful shutdown

## Slash Commands

| Command | Description |
|---------|------------|
| `/join` | Join user's voice channel |
| `/leave` | Disconnect, stop services, free GPU |
| `/listen` | Toggle voice conversation mode (lazy-starts services) |
| `/recap` | LLM-generated summary of conversation so far |
| `/personality` | Switch bot personality (default, pirate, shakespeare, sarcastic, zen, hype) |
| `/play <query>` | Search YouTube and play/queue track |
| `/skip` | Skip current song |
| `/queue` | Show queue |
| `/pause` / `/resume` | Pause/resume playback |
| `/stop` | Stop music, clear queue |
| `/np` | Show now playing |
| `/volume <0-100>` | Set music volume |
| `/status` | Service health with latency |

## Config

`config.yaml` (gitignored, created from `config.example.yaml`). Key sections: `discord.token`, `voice.*` (ASR/TTS/VAD params), `llm.*` (model/prompt/backend), `music.*` (queue/volume), `services.*` (subprocess commands + health URLs).

The `llm.backend` field controls LLM API format: `"openai"` (default, for vLLM) or `"ollama"` (native API with thinking disabled).

**Ollama gotcha**: when `llm.backend = "ollama"`, ServiceManager does NOT manage the LLM process (no entry in `services.llm` is needed). Start Ollama separately before launching the bot: `ollama serve`.

## External Dependencies

**VibeVoice** (Microsoft) must be cloned and installed separately -- it is not on PyPI:
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice && pip install -e ".[streamingtts,vllm]"
```
Set the `cwd` in `config.yaml` `services.asr` / `services.tts` to point at this checkout.

**FFmpeg** is required for music playback (yt-dlp streams through it). Install system-wide:
```bash
sudo apt install ffmpeg   # Debian/Ubuntu
```
`install.sh` warns if it's missing but does not abort.

## Gotchas

- **Voice and music are mutually exclusive.** Stop one before starting the other. `/listen` will conflict with active music playback and vice versa.
- **GPU required for ASR and TTS.** VibeVoice models run on CUDA. The bot itself is CPU-only.
- **ServiceManager startup is sequential.** ASR then TTS then LLM (if managed), each polled until healthy. Startup timeout defaults to 120s per service -- increase in config if models are slow to load.
- `webrtcvad-wheels` (not `webrtcvad`) is the required package; the plain `webrtcvad` wheel breaks on Python 3.12.

## Audio Format Constants

- Discord captures/plays: 48kHz, stereo, 16-bit PCM (3840 bytes per 20ms frame)
- ASR expects: 16kHz, mono, 16-bit PCM (wrapped in WAV)
- TTS outputs: 24kHz, mono, 16-bit PCM
- VAD frame: 30ms at 16kHz = 960 bytes
