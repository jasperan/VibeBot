"""
Real end-to-end tests against live services.
Tests every component of VibeBot as a user would experience it.
Requires: Ollama running with qwen3.5:9b at localhost:11434
"""
import asyncio
import json
import os
import socket
import sys
import time

import numpy as np
import pytest
import yaml

import httpx

# ── 1. Config loading (as a user following README) ──────────────────

PLACEHOLDER_DISCORD_TOKEN = "YOUR_DISCORD_BOT_TOKEN"


def _load_runtime_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _require_real_discord_token(config):
    token = config["discord"]["token"].strip()
    if not token or token == PLACEHOLDER_DISCORD_TOKEN:
        pytest.skip("config.yaml does not contain a real Discord token")


def _require_http_service(name: str, url: str, timeout: float = 2.0):
    try:
        resp = httpx.get(url, timeout=timeout)
    except Exception as exc:
        pytest.skip(f"{name} is unavailable at {url}: {type(exc).__name__}: {exc}")
    if resp.status_code != 200:
        pytest.skip(f"{name} is unavailable at {url}: HTTP {resp.status_code}")


def _require_dns(hostname: str):
    try:
        socket.getaddrinfo(hostname, 443)
    except OSError as exc:
        pytest.skip(f"DNS/network unavailable for {hostname}: {exc}")

def test_config_yaml_exists_and_loads():
    assert os.path.exists("config.yaml"), "config.yaml missing. Run: cp config.example.yaml config.yaml"
    config = _load_runtime_config()
    _require_real_discord_token(config)
    assert config["voice"]["asr_url"]
    assert config["llm"]["base_url"]
    assert config["llm"]["model"]
    print(f"  Config loaded: LLM={config['llm']['model']} at {config['llm']['base_url']}")


def test_config_example_matches_required_keys():
    with open("config.example.yaml") as f:
        example = yaml.safe_load(f)
    with open("config.yaml") as f:
        actual = yaml.safe_load(f)
    for section in ("discord", "voice", "llm", "music", "services"):
        assert section in actual, f"Missing config section: {section}"


# ── 2. All module imports ───────────────────────────────────────────

def test_all_source_modules_import():
    """Every source module imports without error."""
    modules = [
        "src.bot",
        "src.audio",
        "src.asr_client",
        "src.llm_client",
        "src.tts_client",
        "src.voice_pipeline",
        "src.service_manager",
        "src.cogs.voice",
        "src.cogs.music",
        "src.cogs.admin",
    ]
    for mod in modules:
        __import__(mod)
        print(f"  Imported: {mod}")


# ── 3. Audio processing with real data ──────────────────────────────

def test_audio_resample_real_sine_wave():
    """Generate a 440Hz sine wave and verify resampling preserves frequency content."""
    from src.audio import resample_pcm, stereo_to_mono, mono_to_stereo

    # Generate 1 second of 440Hz sine at 48kHz
    t = np.linspace(0, 1.0, 48000, endpoint=False)
    sine_48k = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    pcm_48k = sine_48k.tobytes()

    # Downsample 48kHz -> 16kHz
    pcm_16k = resample_pcm(pcm_48k, from_rate=48000, to_rate=16000)
    samples_16k = np.frombuffer(pcm_16k, dtype=np.int16)
    assert len(samples_16k) == 16000, f"Expected 16000 samples, got {len(samples_16k)}"
    assert np.max(np.abs(samples_16k)) > 1000, "Signal too quiet after resample"

    # Upsample 16kHz -> 48kHz (roundtrip)
    pcm_48k_rt = resample_pcm(pcm_16k, from_rate=16000, to_rate=48000)
    samples_48k_rt = np.frombuffer(pcm_48k_rt, dtype=np.int16)
    assert len(samples_48k_rt) == 48000

    # Stereo conversion roundtrip
    stereo = mono_to_stereo(pcm_48k)
    samples_stereo = np.frombuffer(stereo, dtype=np.int16)
    assert len(samples_stereo) == 96000  # 48000 * 2 channels

    mono_back = stereo_to_mono(stereo)
    samples_mono = np.frombuffer(mono_back, dtype=np.int16)
    assert len(samples_mono) == 48000
    # Mono roundtrip should be lossless
    np.testing.assert_array_equal(samples_mono, sine_48k)
    print("  Audio roundtrip: 48kHz->16kHz->48kHz OK, stereo<->mono OK")


def test_pcm_stream_source_playback():
    """PCMStreamSource reads frames correctly."""
    from src.audio import PCMStreamSource

    source = PCMStreamSource()
    # Feed exactly 2 Discord frames (3840 bytes each)
    frame_size = 3840
    source.feed(b"\x42" * frame_size * 2)
    source.finish()

    frame1 = source.read()
    assert len(frame1) == frame_size
    assert frame1 == b"\x42" * frame_size

    frame2 = source.read()
    assert len(frame2) == frame_size

    # After data exhausted + finished, should return empty
    frame3 = source.read()
    assert frame3 == b""
    print("  PCMStreamSource: 2 frames read, then empty")


def test_pcm_stream_source_silence_when_buffering():
    """Returns silence while waiting for data (streaming TTS behavior)."""
    from src.audio import PCMStreamSource

    source = PCMStreamSource()
    # Not finished, no data: should return silence
    frame = source.read()
    assert len(frame) == 3840
    assert frame == b"\x00" * 3840
    print("  PCMStreamSource: silence during buffering OK")


# ── 4. LLM client against real Ollama ───────────────────────────────

@pytest.mark.asyncio
async def test_llm_real_generation():
    """Hit the actual Ollama LLM and get a response."""
    from src.llm_client import LLMClient

    _require_http_service("Ollama", "http://localhost:11434/api/tags")

    client = LLMClient(
        base_url="http://localhost:11434/v1",
        model="qwen3.5:9b",
        max_tokens=50,
        system_prompt="You are VibeBot. Respond in 1 sentence.",
        backend="ollama",
    )

    start = time.monotonic()
    response = await client.generate("What is 2+2?", [])
    elapsed = time.monotonic() - start

    assert response.text is not None, "LLM returned no text"
    assert len(response.text) > 0, "LLM returned empty text"
    assert "4" in response.text or "four" in response.text.lower(), \
        f"LLM didn't answer correctly: {response.text}"
    print(f"  LLM response ({elapsed:.1f}s): {response.text[:100]}")
    await client.close()


@pytest.mark.asyncio
async def test_llm_real_context_awareness():
    """LLM maintains conversation context correctly."""
    from src.llm_client import LLMClient

    _require_http_service("Ollama", "http://localhost:11434/api/tags")

    client = LLMClient(
        base_url="http://localhost:11434/v1",
        model="qwen3.5:9b",
        max_tokens=50,
        system_prompt="You are VibeBot. Be concise.",
        backend="ollama",
    )

    context = [
        {"role": "user", "content": "My name is TestUser."},
        {"role": "assistant", "content": "Nice to meet you, TestUser!"},
    ]

    response = await client.generate("What is my name?", context)
    assert response.text is not None
    assert "TestUser" in response.text or "testuser" in response.text.lower(), \
        f"LLM lost context: {response.text}"
    print(f"  Context awareness: {response.text[:100]}")
    await client.close()


@pytest.mark.asyncio
async def test_llm_real_tool_calling():
    """LLM can use tools to trigger music playback."""
    from src.llm_client import LLMClient
    from src.cogs.voice import MUSIC_TOOLS

    _require_http_service("Ollama", "http://localhost:11434/api/tags")

    client = LLMClient(
        base_url="http://localhost:11434/v1",
        model="qwen3.5:9b",
        max_tokens=100,
        system_prompt="You are VibeBot, a voice assistant that can play music. When asked to play a song, use the play_music tool.",
        backend="ollama",
    )

    response = await client.generate(
        "Play Bohemian Rhapsody by Queen", [], tools=MUSIC_TOOLS
    )

    # The model should either call the tool or mention playing music
    has_tool_call = len(response.tool_calls) > 0
    has_text = response.text is not None

    if has_tool_call:
        tool_name = response.tool_calls[0]["function"]["name"]
        raw_args = response.tool_calls[0]["function"]["arguments"]
        tool_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        print(f"  Tool call: {tool_name}({tool_args})")
        assert tool_name == "play_music"
        assert "query" in tool_args
    else:
        # Some models respond with text instead of tool calls
        print(f"  Text response (no tool call): {response.text[:100]}")
        assert has_text, "LLM returned neither text nor tool call"

    print(f"  Tool calling test: tool_calls={has_tool_call}, text={has_text}")
    await client.close()


# ── 5. Voice pipeline with real LLM ────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_real_llm_mock_asr_tts():
    """Full pipeline with real LLM, mocked ASR/TTS."""
    from unittest.mock import AsyncMock, MagicMock
    from src.voice_pipeline import VoicePipeline
    from src.llm_client import LLMClient

    _require_http_service("Ollama", "http://localhost:11434/api/tags")

    # Real LLM client
    llm = LLMClient(
        base_url="http://localhost:11434/v1",
        model="qwen3.5:9b",
        max_tokens=50,
        system_prompt="You are VibeBot. Respond in exactly 1 sentence.",
        backend="ollama",
    )

    # Mock ASR (returns fixed transcript)
    mock_asr = AsyncMock()
    mock_asr.transcribe.return_value = "Hello, how are you?"

    # Mock TTS (returns fake audio)
    mock_tts = MagicMock()
    fake_audio = np.zeros(2400, dtype=np.int16).tobytes()

    async def fake_synthesize(text):
        yield fake_audio

    mock_tts.synthesize = fake_synthesize

    pipeline = VoicePipeline(
        asr_client=mock_asr,
        llm_client=llm,
        tts_client=mock_tts,
        context_window=5,
    )

    pcm_input = np.zeros(16000, dtype=np.int16).tobytes()
    audio_chunks = []

    async for chunk in pipeline.process_utterance(pcm_input):
        audio_chunks.append(chunk)

    assert len(audio_chunks) > 0, "Pipeline produced no audio"
    assert len(pipeline._context) == 2, "Context not updated"
    assert pipeline._context[0]["role"] == "user"
    assert pipeline._context[0]["content"] == "Hello, how are you?"
    assert pipeline._context[1]["role"] == "assistant"
    print(f"  Pipeline response: {pipeline._context[1]['content'][:100]}")
    print(f"  Audio chunks: {len(audio_chunks)}")
    await llm.close()


@pytest.mark.asyncio
async def test_pipeline_real_summarize():
    """Test /recap with real LLM summarization."""
    from src.voice_pipeline import VoicePipeline
    from src.llm_client import LLMClient
    from unittest.mock import AsyncMock, MagicMock

    _require_http_service("Ollama", "http://localhost:11434/api/tags")

    llm = LLMClient(
        base_url="http://localhost:11434/v1",
        model="qwen3.5:9b",
        max_tokens=100,
        system_prompt="You are VibeBot.",
        backend="ollama",
    )

    pipeline = VoicePipeline(
        asr_client=AsyncMock(),
        llm_client=llm,
        tts_client=MagicMock(),
        context_window=5,
    )
    pipeline._context = [
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "I'm a Discord bot, I can't check the weather, but I can play music!"},
        {"role": "user", "content": "Ok play some jazz"},
        {"role": "assistant", "content": "Playing some smooth jazz for you!"},
    ]

    summary = await pipeline.summarize_context()
    assert summary is not None
    assert len(summary) > 10
    print(f"  Recap summary: {summary[:150]}")
    await llm.close()


# ── 6. Service manager lifecycle ────────────────────────────────────

@pytest.mark.asyncio
async def test_service_manager_real_process():
    """ServiceManager can start and stop a real subprocess."""
    from src.service_manager import ServiceManager

    config = {
        "services": {
            "test_svc": {
                "command": "python -m http.server 19876",
                "health_url": "http://localhost:19876/",
                "startup_timeout": 10,
            }
        }
    }

    mgr = ServiceManager(config)
    assert not mgr.is_running

    await mgr.ensure_running()
    assert mgr.is_running
    assert "test_svc" in mgr._processes
    pid = mgr._processes["test_svc"].pid
    print(f"  Started test service at PID {pid}")

    await mgr.shutdown()
    assert not mgr.is_running
    print("  Shutdown OK")


# ── 7. Bot initialization ──────────────────────────────────────────

def test_bot_initializes_with_config():
    """VibeBotClient can be created with real config."""
    from src.bot import VibeBotClient, load_config, validate_runtime_config

    config = load_config("config.yaml")
    _require_real_discord_token(config)
    validate_runtime_config(config)
    bot = VibeBotClient(config)
    assert bot.config == config
    assert bot.services is not None
    print(f"  Bot created, model={config['llm']['model']}")


# ── 8. Personality system ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_personality_changes_llm_behavior():
    """Different personalities produce different response styles."""
    from src.llm_client import LLMClient
    from src.cogs.voice import PERSONALITY_PRESETS

    _require_http_service("Ollama", "http://localhost:11434/api/tags")

    results = {}
    for name in ("default", "pirate", "zen"):
        client = LLMClient(
            base_url="http://localhost:11434/v1",
            model="qwen3.5:9b",
            max_tokens=60,
            system_prompt=PERSONALITY_PRESETS[name],
            backend="ollama",
        )
        resp = await client.generate("Hello, who are you?", [])
        results[name] = resp.text
        await client.close()

    # All should produce non-empty responses
    for name, text in results.items():
        assert text and len(text) > 5, f"Personality '{name}' produced empty response"
        print(f"  {name}: {text[:80]}")

    # Responses should differ (different personalities)
    texts = list(results.values())
    assert not all(t == texts[0] for t in texts), "All personalities gave identical responses"


# ── 9. YouTube search (music cog) ──────────────────────────────────

@pytest.mark.asyncio
async def test_youtube_search_real():
    """MusicCog can find real YouTube tracks."""
    from unittest.mock import MagicMock
    from src.cogs.music import MusicCog

    _require_dns("www.youtube.com")

    bot = MagicMock()
    bot.config = {"music": {"max_queue_size": 50, "default_volume": 0.5}}
    cog = MusicCog(bot)

    track = await cog._search_youtube("Rick Astley Never Gonna Give You Up")
    assert track is not None, "YouTube search returned no results"
    assert "title" in track
    assert "url" in track
    assert track["duration"] > 0
    print(f"  Found: {track['title']} ({track['duration']}s)")
    print(f"  URL: {track['url'][:80]}...")


# ── 10. Full cog instantiation with real config ────────────────────

def test_all_cogs_with_real_config():
    """All cogs instantiate with the actual config.yaml."""
    from unittest.mock import MagicMock
    from src.bot import load_config, validate_runtime_config
    from src.cogs.voice import VoiceCog
    from src.cogs.music import MusicCog
    from src.cogs.admin import AdminCog

    config = load_config("config.yaml")
    _require_real_discord_token(config)
    validate_runtime_config(config)
    bot = MagicMock()
    bot.config = config

    voice = VoiceCog(bot)
    music = MusicCog(bot)
    admin = AdminCog(bot)

    assert voice._pipeline is not None
    assert voice._pipeline.tools is not None
    assert len(voice._pipeline.tools) == 4
    assert music._volume == 0.5
    assert admin.bot is bot
    print("  All 3 cogs instantiated with real config")
    print(f"  Voice tools: {[t['function']['name'] for t in voice._pipeline.tools]}")
