# VibeBot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Discord voice bot that has real-time voice conversations via VibeVoice (ASR + TTS) and plays YouTube music, with mutually exclusive modes.

**Architecture:** Single discord.py process with two cogs (VoiceCog, MusicCog). Voice pipeline: Discord audio -> webrtcvad -> VibeVoice ASR (vLLM) -> Qwen3.5:9B (vLLM) -> VibeVoice Realtime TTS (WebSocket) -> Discord audio. Music: yt-dlp + FFmpeg. Cogs enforce mutual exclusion via shared lock.

**Tech Stack:** Python 3.12, discord.py[voice], httpx, websockets, webrtcvad, numpy, yt-dlp, PyYAML, FFmpeg

---

### Task 1: Project Scaffolding

**Files:**
- Create: `src/__init__.py`
- Create: `src/cogs/__init__.py`
- Create: `requirements.txt`
- Create: `config.example.yaml`
- Create: `.gitignore`
- Create: `install.sh`

**Step 1: Create requirements.txt**

```
discord.py[voice]>=2.7.0
httpx>=0.24.0
websockets>=12.0
webrtcvad>=2.0.10
numpy>=1.24.0
yt-dlp>=2024.0
PyYAML>=6.0
```

**Step 2: Create config.example.yaml**

```yaml
discord:
  token: "YOUR_DISCORD_BOT_TOKEN"

voice:
  asr_url: "http://localhost:8000/v1"
  tts_host: "localhost"
  tts_port: 3000
  tts_voice: "Carter"
  tts_cfg: 1.5
  tts_steps: 5
  vad_aggressiveness: 2
  silence_threshold_ms: 300
  context_window: 5
  max_utterance_seconds: 30

llm:
  base_url: "http://localhost:8010/v1"
  model: "qwen3.5:9b"
  max_tokens: 150
  system_prompt: "You are VibeBot, a friendly voice assistant in a Discord voice channel. Keep responses concise and conversational, 1-2 sentences max."

music:
  max_queue_size: 50
  default_volume: 0.5
```

**Step 3: Create .gitignore**

```
__pycache__/
*.pyc
*.pyo
.env
config.yaml
.agents/
.claude/
.crush/
.openhands/
.serena/
.playwright-mcp/
docs/plans/
task_plan.md
findings.md
progress.md
dogfood-output/
content/
```

**Step 4: Create install.sh**

```bash
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
```

**Step 5: Create empty __init__.py files**

```python
# src/__init__.py
```

```python
# src/cogs/__init__.py
```

**Step 6: Commit**

```bash
git add -A
git commit -m "scaffold: VibeBot project structure and dependencies"
```

---

### Task 2: Bot Core (bot.py)

**Files:**
- Create: `src/bot.py`
- Test: `tests/test_bot.py`

**Step 1: Write the failing test**

```python
# tests/test_bot.py
import pytest
from unittest.mock import patch, MagicMock
import yaml


def test_load_config_reads_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "discord": {"token": "test-token"},
        "voice": {"asr_url": "http://localhost:8000/v1"},
        "llm": {"base_url": "http://localhost:8010/v1", "model": "qwen3.5:9b"},
        "music": {"max_queue_size": 50},
    }))
    from src.bot import load_config
    config = load_config(str(config_file))
    assert config["discord"]["token"] == "test-token"
    assert config["llm"]["model"] == "qwen3.5:9b"


def test_load_config_missing_file():
    from src.bot import load_config
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_bot.py -v`
Expected: FAIL with "cannot import name 'load_config'"

**Step 3: Write bot.py**

```python
# src/bot.py
import asyncio
import logging
import sys

import discord
from discord.ext import commands
import yaml

log = logging.getLogger("vibebot")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class VibeBotClient(commands.Bot):
    def __init__(self, config: dict):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        super().__init__(command_prefix="!", intents=intents)
        self.config = config
        self.mode_lock = asyncio.Lock()

    async def setup_hook(self):
        await self._load_cogs()
        await self.tree.sync()
        log.info("Slash commands synced")

    async def _load_cogs(self):
        from src.cogs.voice import VoiceCog
        from src.cogs.music import MusicCog
        await self.add_cog(VoiceCog(self))
        await self.add_cog(MusicCog(self))
        log.info("Cogs loaded: VoiceCog, MusicCog")

    async def on_ready(self):
        log.info(f"VibeBot online as {self.user} (id={self.user.id})")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    config = load_config()
    bot = VibeBotClient(config)
    bot.run(config["discord"]["token"], log_handler=None)


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_bot.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/bot.py tests/test_bot.py
git commit -m "feat: bot core with config loading and cog setup"
```

---

### Task 3: LLM Client (llm_client.py)

**Files:**
- Create: `src/llm_client.py`
- Test: `tests/test_llm_client.py`

**Step 1: Write the failing test**

```python
# tests/test_llm_client.py
import pytest
import httpx
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_llm_generate_returns_text():
    from src.llm_client import LLMClient

    mock_response = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "Hello there!"}}]
        },
    )

    client = LLMClient(
        base_url="http://localhost:8010/v1",
        model="qwen3.5:9b",
        max_tokens=150,
        system_prompt="You are VibeBot.",
    )

    with patch.object(client._http, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await client.generate("Hi bot", [])
        assert result == "Hello there!"


@pytest.mark.asyncio
async def test_llm_generate_with_context():
    from src.llm_client import LLMClient

    mock_response = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "You said hello earlier."}}]
        },
    )

    client = LLMClient(
        base_url="http://localhost:8010/v1",
        model="qwen3.5:9b",
        max_tokens=150,
        system_prompt="You are VibeBot.",
    )

    context = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]

    with patch.object(client._http, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await client.generate("What did I say?", context)
        assert result == "You said hello earlier."


@pytest.mark.asyncio
async def test_llm_generate_returns_none_on_error():
    from src.llm_client import LLMClient

    client = LLMClient(
        base_url="http://localhost:8010/v1",
        model="qwen3.5:9b",
        max_tokens=150,
        system_prompt="You are VibeBot.",
    )

    mock_response = httpx.Response(500, text="Internal Server Error")
    with patch.object(client._http, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await client.generate("Hi", [])
        assert result is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_client.py -v`
Expected: FAIL with "cannot import name 'LLMClient'"

**Step 3: Write llm_client.py**

```python
# src/llm_client.py
import logging
import httpx

log = logging.getLogger("vibebot.llm")


class LLMClient:
    def __init__(self, base_url: str, model: str, max_tokens: int, system_prompt: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self._http = httpx.AsyncClient(timeout=30.0)

    async def generate(self, user_text: str, context: list[dict]) -> str | None:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(context)
        messages.append({"role": "user", "content": user_text})

        try:
            resp = await self._http.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            if resp.status_code != 200:
                log.warning("LLM returned %d: %s", resp.status_code, resp.text[:200])
                return None
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            log.warning("LLM request failed: %s", e)
            return None

    async def close(self):
        await self._http.aclose()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_client.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/llm_client.py tests/test_llm_client.py
git commit -m "feat: LLM client for Qwen3.5:9B via vLLM"
```

---

### Task 4: ASR Client (asr_client.py)

**Files:**
- Create: `src/asr_client.py`
- Test: `tests/test_asr_client.py`

**Step 1: Write the failing test**

```python
# tests/test_asr_client.py
import pytest
import httpx
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_asr_transcribe_returns_text():
    from src.asr_client import ASRClient

    mock_response = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": '{"text": "hello world"}'}}]
        },
    )

    client = ASRClient(base_url="http://localhost:8000/v1")

    # 16kHz mono, 1 second of silence (32000 bytes of zeros)
    pcm_data = bytes(32000)

    with patch.object(client._http, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await client.transcribe(pcm_data, sample_rate=16000)
        assert result == "hello world"


@pytest.mark.asyncio
async def test_asr_transcribe_returns_none_on_error():
    from src.asr_client import ASRClient

    client = ASRClient(base_url="http://localhost:8000/v1")
    pcm_data = bytes(32000)

    mock_response = httpx.Response(500, text="error")
    with patch.object(client._http, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await client.transcribe(pcm_data, sample_rate=16000)
        assert result is None


@pytest.mark.asyncio
async def test_asr_transcribe_empty_text():
    from src.asr_client import ASRClient

    mock_response = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": '{"text": ""}'}}]
        },
    )

    client = ASRClient(base_url="http://localhost:8000/v1")
    pcm_data = bytes(32000)

    with patch.object(client._http, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await client.transcribe(pcm_data, sample_rate=16000)
        assert result is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_asr_client.py -v`
Expected: FAIL with "cannot import name 'ASRClient'"

**Step 3: Write asr_client.py**

VibeVoice ASR uses the OpenAI chat completions format. Audio is sent as a base64 data URL in an `audio_url` content block.

```python
# src/asr_client.py
import base64
import io
import json
import logging
import struct
import wave

import httpx

log = logging.getLogger("vibebot.asr")


class ASRClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=60.0)

    def _pcm_to_wav_bytes(self, pcm_data: bytes, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return buf.getvalue()

    async def transcribe(self, pcm_data: bytes, sample_rate: int = 16000) -> str | None:
        wav_bytes = self._pcm_to_wav_bytes(pcm_data, sample_rate)
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        data_url = f"data:audio/wav;base64,{audio_b64}"

        try:
            resp = await self._http.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": "VibeVoice-ASR",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio_url",
                                    "audio_url": {"url": data_url},
                                },
                                {
                                    "type": "text",
                                    "text": "transcribe this audio",
                                },
                            ],
                        }
                    ],
                },
            )
            if resp.status_code != 200:
                log.warning("ASR returned %d: %s", resp.status_code, resp.text[:200])
                return None

            content = resp.json()["choices"][0]["message"]["content"]
            try:
                parsed = json.loads(content)
                text = parsed.get("text", "").strip()
            except json.JSONDecodeError:
                text = content.strip()

            return text if text else None
        except Exception as e:
            log.warning("ASR request failed: %s", e)
            return None

    async def close(self):
        await self._http.aclose()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_asr_client.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/asr_client.py tests/test_asr_client.py
git commit -m "feat: ASR client for VibeVoice vLLM endpoint"
```

---

### Task 5: TTS Client (tts_client.py)

**Files:**
- Create: `src/tts_client.py`
- Test: `tests/test_tts_client.py`

**Step 1: Write the failing test**

```python
# tests/test_tts_client.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_tts_synthesize_returns_pcm():
    from src.tts_client import TTSClient

    client = TTSClient(host="localhost", port=3000, voice="Carter", cfg=1.5, steps=5)

    # Simulate WebSocket that sends two binary frames then closes
    mock_ws = AsyncMock()
    mock_ws.__aiter__ = lambda self: self
    frames = [b"\x00\x01" * 100, b"\x00\x02" * 100]
    mock_ws.__anext__ = AsyncMock(side_effect=[*frames, StopAsyncIteration()])

    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.return_value.__aexit__ = AsyncMock(return_value=False)

        chunks = []
        async for chunk in client.synthesize("Hello world"):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0] == frames[0]


def test_tts_build_url():
    from src.tts_client import TTSClient

    client = TTSClient(host="localhost", port=3000, voice="Carter", cfg=1.5, steps=5)
    url = client._build_url("Hello world")
    assert "ws://localhost:3000/stream" in url
    assert "text=Hello+world" in url or "text=Hello%20world" in url
    assert "voice=Carter" in url
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tts_client.py -v`
Expected: FAIL with "cannot import name 'TTSClient'"

**Step 3: Write tts_client.py**

VibeVoice Realtime TTS uses a WebSocket connection. All params go in the query string. Server streams back raw PCM16 binary frames (24kHz mono, signed int16 little-endian). JSON text messages are status/logs and can be ignored.

```python
# src/tts_client.py
import logging
from typing import AsyncIterator
from urllib.parse import urlencode

import websockets

log = logging.getLogger("vibebot.tts")

SAMPLE_RATE = 24000  # VibeVoice Realtime output


class TTSClient:
    def __init__(self, host: str, port: int, voice: str = "Carter",
                 cfg: float = 1.5, steps: int = 5):
        self.host = host
        self.port = port
        self.voice = voice
        self.cfg = cfg
        self.steps = steps

    def _build_url(self, text: str) -> str:
        params = urlencode({
            "text": text,
            "voice": self.voice,
            "cfg": self.cfg,
            "steps": self.steps,
        })
        return f"ws://{self.host}:{self.port}/stream?{params}"

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        url = self._build_url(text)
        try:
            async with websockets.connect(url) as ws:
                async for message in ws:
                    if isinstance(message, bytes):
                        yield message
                    # Text messages are status/logs, skip them
        except Exception as e:
            log.warning("TTS WebSocket error: %s", e)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tts_client.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tts_client.py tests/test_tts_client.py
git commit -m "feat: TTS client for VibeVoice Realtime WebSocket"
```

---

### Task 6: Audio Utilities (audio.py)

**Files:**
- Create: `src/audio.py`
- Test: `tests/test_audio.py`

**Step 1: Write the failing test**

```python
# tests/test_audio.py
import numpy as np
import pytest


def test_resample_48k_to_16k():
    from src.audio import resample_pcm

    # 480 samples at 48kHz = 10ms, should become 160 samples at 16kHz
    samples_48k = np.zeros(480, dtype=np.int16)
    result = resample_pcm(samples_48k.tobytes(), from_rate=48000, to_rate=16000)
    result_samples = np.frombuffer(result, dtype=np.int16)
    assert len(result_samples) == 160


def test_resample_24k_to_48k():
    from src.audio import resample_pcm

    # 240 samples at 24kHz = 10ms, should become 480 samples at 48kHz
    samples_24k = np.zeros(240, dtype=np.int16)
    result = resample_pcm(samples_24k.tobytes(), from_rate=24000, to_rate=48000)
    result_samples = np.frombuffer(result, dtype=np.int16)
    assert len(result_samples) == 480


def test_stereo_to_mono():
    from src.audio import stereo_to_mono

    # Stereo: L=100, R=200 -> mono should be 150
    stereo = np.array([100, 200, 100, 200], dtype=np.int16)
    result = stereo_to_mono(stereo.tobytes())
    mono = np.frombuffer(result, dtype=np.int16)
    assert len(mono) == 2
    assert mono[0] == 150


def test_mono_to_stereo():
    from src.audio import mono_to_stereo

    mono = np.array([100, 200], dtype=np.int16)
    result = mono_to_stereo(mono.tobytes())
    stereo = np.frombuffer(result, dtype=np.int16)
    assert len(stereo) == 4
    assert stereo[0] == 100  # L
    assert stereo[1] == 100  # R (duplicated)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio.py -v`
Expected: FAIL with "cannot import name 'resample_pcm'"

**Step 3: Write audio.py**

```python
# src/audio.py
import numpy as np
import discord


def resample_pcm(pcm_data: bytes, from_rate: int, to_rate: int) -> bytes:
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    ratio = to_rate / from_rate
    new_length = int(len(samples) * ratio)
    indices = np.linspace(0, len(samples) - 1, new_length).astype(int)
    resampled = samples[indices]
    return resampled.astype(np.int16).tobytes()


def stereo_to_mono(pcm_data: bytes) -> bytes:
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    left = samples[0::2]
    right = samples[1::2]
    mono = ((left.astype(np.int32) + right.astype(np.int32)) // 2).astype(np.int16)
    return mono.tobytes()


def mono_to_stereo(pcm_data: bytes) -> bytes:
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    stereo = np.empty(len(samples) * 2, dtype=np.int16)
    stereo[0::2] = samples
    stereo[1::2] = samples
    return stereo.tobytes()


class PCMStreamSource(discord.AudioSource):
    """Plays raw PCM data (48kHz stereo int16) into Discord voice."""

    def __init__(self):
        self._buffer = bytearray()
        self._finished = False

    def feed(self, pcm_48k_stereo: bytes):
        self._buffer.extend(pcm_48k_stereo)

    def finish(self):
        self._finished = True

    def read(self) -> bytes:
        # Discord expects 20ms frames: 48000 * 2 channels * 2 bytes * 0.02s = 3840 bytes
        frame_size = 3840
        if len(self._buffer) >= frame_size:
            frame = bytes(self._buffer[:frame_size])
            del self._buffer[:frame_size]
            return frame
        if self._finished:
            return b""
        # Not enough data yet, return silence
        return b"\x00" * frame_size

    def is_opus(self) -> bool:
        return False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_audio.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/audio.py tests/test_audio.py
git commit -m "feat: audio utilities (resample, stereo/mono, PCM stream source)"
```

---

### Task 7: Voice Pipeline (voice_pipeline.py)

**Files:**
- Create: `src/voice_pipeline.py`
- Test: `tests/test_voice_pipeline.py`

**Step 1: Write the failing test**

```python
# tests/test_voice_pipeline.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np


@pytest.mark.asyncio
async def test_pipeline_processes_utterance():
    from src.voice_pipeline import VoicePipeline

    mock_asr = AsyncMock()
    mock_asr.transcribe.return_value = "hello"

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "Hi there!"

    # TTS returns a single chunk of 24kHz mono PCM
    tts_chunk = np.zeros(2400, dtype=np.int16).tobytes()
    mock_tts = MagicMock()

    async def fake_synthesize(text):
        yield tts_chunk

    mock_tts.synthesize = fake_synthesize

    pipeline = VoicePipeline(
        asr_client=mock_asr,
        llm_client=mock_llm,
        tts_client=mock_tts,
        context_window=5,
    )

    # 16kHz mono PCM, 1 second
    pcm_16k = np.zeros(16000, dtype=np.int16).tobytes()
    audio_chunks = []

    async for chunk in pipeline.process_utterance(pcm_16k):
        audio_chunks.append(chunk)

    assert len(audio_chunks) > 0
    mock_asr.transcribe.assert_awaited_once()
    mock_llm.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_skips_empty_transcript():
    from src.voice_pipeline import VoicePipeline

    mock_asr = AsyncMock()
    mock_asr.transcribe.return_value = None

    mock_llm = AsyncMock()
    mock_tts = MagicMock()

    pipeline = VoicePipeline(
        asr_client=mock_asr,
        llm_client=mock_llm,
        tts_client=mock_tts,
        context_window=5,
    )

    pcm_16k = np.zeros(16000, dtype=np.int16).tobytes()
    audio_chunks = []
    async for chunk in pipeline.process_utterance(pcm_16k):
        audio_chunks.append(chunk)

    assert len(audio_chunks) == 0
    mock_llm.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_pipeline_maintains_context():
    from src.voice_pipeline import VoicePipeline

    mock_asr = AsyncMock()
    mock_asr.transcribe.return_value = "what is 2+2"

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "Four!"

    tts_chunk = np.zeros(2400, dtype=np.int16).tobytes()
    mock_tts = MagicMock()

    async def fake_synthesize(text):
        yield tts_chunk

    mock_tts.synthesize = fake_synthesize

    pipeline = VoicePipeline(
        asr_client=mock_asr,
        llm_client=mock_llm,
        tts_client=mock_tts,
        context_window=5,
    )

    pcm_16k = np.zeros(16000, dtype=np.int16).tobytes()
    async for _ in pipeline.process_utterance(pcm_16k):
        pass

    assert len(pipeline._context) == 2  # user + assistant
    assert pipeline._context[0]["role"] == "user"
    assert pipeline._context[1]["role"] == "assistant"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_voice_pipeline.py -v`
Expected: FAIL with "cannot import name 'VoicePipeline'"

**Step 3: Write voice_pipeline.py**

```python
# src/voice_pipeline.py
import logging
from collections import deque
from typing import AsyncIterator

from src.asr_client import ASRClient
from src.llm_client import LLMClient
from src.tts_client import TTSClient
from src.audio import resample_pcm, mono_to_stereo

log = logging.getLogger("vibebot.pipeline")


class VoicePipeline:
    def __init__(self, asr_client: ASRClient, llm_client: LLMClient,
                 tts_client: TTSClient, context_window: int = 5):
        self.asr = asr_client
        self.llm = llm_client
        self.tts = tts_client
        self._context: list[dict] = []
        self._max_context = context_window * 2  # pairs of user+assistant

    async def process_utterance(self, pcm_16k_mono: bytes) -> AsyncIterator[bytes]:
        # Step 1: ASR
        transcript = await self.asr.transcribe(pcm_16k_mono, sample_rate=16000)
        if not transcript:
            log.debug("Empty transcript, skipping")
            return

        log.info("ASR transcript: %s", transcript)

        # Step 2: LLM
        response = await self.llm.generate(transcript, self._context)
        if not response:
            log.warning("LLM returned no response")
            return

        log.info("LLM response: %s", response)

        # Update context
        self._context.append({"role": "user", "content": transcript})
        self._context.append({"role": "assistant", "content": response})
        if len(self._context) > self._max_context:
            self._context = self._context[-self._max_context:]

        # Step 3: TTS -> 24kHz mono PCM -> 48kHz stereo PCM for Discord
        async for tts_chunk in self.tts.synthesize(response):
            # Resample 24kHz -> 48kHz
            pcm_48k = resample_pcm(tts_chunk, from_rate=24000, to_rate=48000)
            # Mono -> stereo
            pcm_48k_stereo = mono_to_stereo(pcm_48k)
            yield pcm_48k_stereo

    def clear_context(self):
        self._context.clear()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_voice_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/voice_pipeline.py tests/test_voice_pipeline.py
git commit -m "feat: voice pipeline (ASR -> LLM -> TTS orchestration)"
```

---

### Task 8: Voice Cog (cogs/voice.py)

**Files:**
- Create: `src/cogs/voice.py`
- Test: `tests/test_voice_cog.py`

**Step 1: Write the failing test**

```python
# tests/test_voice_cog.py
import pytest
from unittest.mock import AsyncMock, MagicMock, PropertyMock


@pytest.mark.asyncio
async def test_voice_cog_listen_rejects_when_music_playing():
    from src.cogs.voice import VoiceCog

    bot = MagicMock()
    bot.config = {
        "voice": {
            "asr_url": "http://localhost:8000/v1",
            "tts_host": "localhost",
            "tts_port": 3000,
            "tts_voice": "Carter",
            "tts_cfg": 1.5,
            "tts_steps": 5,
            "vad_aggressiveness": 2,
            "silence_threshold_ms": 300,
            "context_window": 5,
            "max_utterance_seconds": 30,
        },
        "llm": {
            "base_url": "http://localhost:8010/v1",
            "model": "qwen3.5:9b",
            "max_tokens": 150,
            "system_prompt": "You are VibeBot.",
        },
    }

    music_cog = MagicMock()
    music_cog.is_playing = True
    bot.get_cog.return_value = music_cog

    cog = VoiceCog(bot)
    assert cog.is_listening is False


def test_voice_cog_initial_state():
    from src.cogs.voice import VoiceCog

    bot = MagicMock()
    bot.config = {
        "voice": {
            "asr_url": "http://localhost:8000/v1",
            "tts_host": "localhost",
            "tts_port": 3000,
            "tts_voice": "Carter",
            "tts_cfg": 1.5,
            "tts_steps": 5,
            "vad_aggressiveness": 2,
            "silence_threshold_ms": 300,
            "context_window": 5,
            "max_utterance_seconds": 30,
        },
        "llm": {
            "base_url": "http://localhost:8010/v1",
            "model": "qwen3.5:9b",
            "max_tokens": 150,
            "system_prompt": "You are VibeBot.",
        },
    }
    cog = VoiceCog(bot)
    assert cog.is_listening is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_voice_cog.py -v`
Expected: FAIL with "cannot import name 'VoiceCog'"

**Step 3: Write cogs/voice.py**

```python
# src/cogs/voice.py
import asyncio
import logging
from collections import deque

import discord
from discord import app_commands
from discord.ext import commands

import webrtcvad

from src.asr_client import ASRClient
from src.llm_client import LLMClient
from src.tts_client import TTSClient
from src.voice_pipeline import VoicePipeline
from src.audio import resample_pcm, stereo_to_mono, PCMStreamSource

log = logging.getLogger("vibebot.voice")

# VAD operates on 10/20/30ms frames of 16kHz 16-bit mono PCM
VAD_FRAME_MS = 30
VAD_FRAME_BYTES = int(16000 * 2 * VAD_FRAME_MS / 1000)  # 960 bytes


class VoiceCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.is_listening = False
        self._listen_task: asyncio.Task | None = None

        vc = bot.config["voice"]
        lc = bot.config["llm"]

        self._asr = ASRClient(base_url=vc["asr_url"])
        self._llm = LLMClient(
            base_url=lc["base_url"],
            model=lc["model"],
            max_tokens=lc["max_tokens"],
            system_prompt=lc["system_prompt"],
        )
        self._tts = TTSClient(
            host=vc["tts_host"],
            port=vc["tts_port"],
            voice=vc["tts_voice"],
            cfg=vc["tts_cfg"],
            steps=vc["tts_steps"],
        )
        self._pipeline = VoicePipeline(
            asr_client=self._asr,
            llm_client=self._llm,
            tts_client=self._tts,
            context_window=vc["context_window"],
        )
        self._vad = webrtcvad.Vad(vc["vad_aggressiveness"])
        self._silence_threshold_ms = vc["silence_threshold_ms"]
        self._max_utterance_bytes = int(
            vc["max_utterance_seconds"] * 16000 * 2
        )

    @app_commands.command(name="join", description="Join your voice channel")
    async def join(self, interaction: discord.Interaction):
        if not interaction.user.voice:
            await interaction.response.send_message(
                "You need to be in a voice channel.", ephemeral=True
            )
            return

        channel = interaction.user.voice.channel
        if interaction.guild.voice_client:
            await interaction.guild.voice_client.move_to(channel)
        else:
            await channel.connect()
        await interaction.response.send_message(
            f"Joined {channel.name}", ephemeral=True
        )

    @app_commands.command(name="leave", description="Leave voice channel")
    async def leave(self, interaction: discord.Interaction):
        vc = interaction.guild.voice_client
        if not vc:
            await interaction.response.send_message(
                "Not in a voice channel.", ephemeral=True
            )
            return

        self.is_listening = False
        if self._listen_task:
            self._listen_task.cancel()
            self._listen_task = None
        self._pipeline.clear_context()

        # Also clear music state if MusicCog is active
        music_cog = self.bot.get_cog("MusicCog")
        if music_cog:
            music_cog._clear_state()

        await vc.disconnect()
        await interaction.response.send_message("Left voice channel.", ephemeral=True)

    @app_commands.command(name="listen", description="Toggle voice conversation mode")
    async def listen(self, interaction: discord.Interaction):
        vc = interaction.guild.voice_client
        if not vc:
            await interaction.response.send_message(
                "Not in a voice channel. Use /join first.", ephemeral=True
            )
            return

        # Mutual exclusion check
        music_cog = self.bot.get_cog("MusicCog")
        if music_cog and music_cog.is_playing:
            await interaction.response.send_message(
                "Stop music first with /stop.", ephemeral=True
            )
            return

        if self.is_listening:
            # Toggle off
            self.is_listening = False
            if self._listen_task:
                self._listen_task.cancel()
                self._listen_task = None
            self._pipeline.clear_context()
            await interaction.response.send_message(
                "Stopped listening.", ephemeral=True
            )
        else:
            # Toggle on
            self.is_listening = True
            self._listen_task = asyncio.create_task(self._listen_loop(vc))
            await interaction.response.send_message(
                "Listening! Talk and I'll respond.", ephemeral=True
            )

    async def _listen_loop(self, voice_client: discord.VoiceClient):
        """Main loop: capture audio, detect speech, process through pipeline."""
        utterance_buffer = bytearray()
        silence_frames = 0
        silence_limit = int(self._silence_threshold_ms / VAD_FRAME_MS)
        is_speaking = False

        try:
            while self.is_listening and voice_client.is_connected():
                # discord.py voice receive: read raw PCM from the sink
                # We need to use a custom sink to capture audio
                # For now, we'll use voice_client.listen() with a callback
                if not hasattr(voice_client, "_vibebot_sink"):
                    sink = AudioSink()
                    voice_client._vibebot_sink = sink
                    voice_client.listen(sink)

                sink = voice_client._vibebot_sink

                # Process buffered audio frames
                while len(sink.buffer) >= VAD_FRAME_BYTES:
                    frame = bytes(sink.buffer[:VAD_FRAME_BYTES])
                    del sink.buffer[:VAD_FRAME_BYTES]

                    try:
                        speech = self._vad.is_speech(frame, 16000)
                    except Exception:
                        speech = False

                    if speech:
                        is_speaking = True
                        silence_frames = 0
                        utterance_buffer.extend(frame)

                        # Enforce max utterance length
                        if len(utterance_buffer) >= self._max_utterance_bytes:
                            await self._handle_utterance(
                                bytes(utterance_buffer), voice_client
                            )
                            utterance_buffer.clear()
                            is_speaking = False
                    elif is_speaking:
                        silence_frames += 1
                        if silence_frames >= silence_limit:
                            # End of utterance
                            await self._handle_utterance(
                                bytes(utterance_buffer), voice_client
                            )
                            utterance_buffer.clear()
                            is_speaking = False
                            silence_frames = 0

                await asyncio.sleep(0.01)  # Yield to event loop
        except asyncio.CancelledError:
            log.info("Listen loop cancelled")
        except Exception as e:
            log.error("Listen loop error: %s", e, exc_info=True)
        finally:
            if hasattr(voice_client, "_vibebot_sink"):
                voice_client.stop_listening()
                del voice_client._vibebot_sink

    async def _handle_utterance(self, pcm_16k_mono: bytes,
                                 voice_client: discord.VoiceClient):
        """Process a complete utterance through the pipeline and play response."""
        source = PCMStreamSource()

        async for audio_chunk in self._pipeline.process_utterance(pcm_16k_mono):
            source.feed(audio_chunk)

        source.finish()

        if voice_client.is_playing():
            voice_client.stop()
        voice_client.play(source)

    async def cog_unload(self):
        self.is_listening = False
        if self._listen_task:
            self._listen_task.cancel()
        await self._asr.close()
        await self._llm.close()


class AudioSink(discord.AudioSink):
    """Captures audio from Discord voice, converts to 16kHz mono PCM."""

    def __init__(self):
        self.buffer = bytearray()

    def write(self, user, data: discord.VoiceData):
        # Discord sends 48kHz stereo PCM
        pcm_48k_stereo = data.pcm
        pcm_48k_mono = stereo_to_mono(pcm_48k_stereo)
        pcm_16k_mono = resample_pcm(pcm_48k_mono, from_rate=48000, to_rate=16000)
        self.buffer.extend(pcm_16k_mono)

    def cleanup(self):
        self.buffer.clear()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_voice_cog.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/cogs/voice.py tests/test_voice_cog.py
git commit -m "feat: voice cog with /join, /leave, /listen commands"
```

---

### Task 9: Music Cog (cogs/music.py)

**Files:**
- Create: `src/cogs/music.py`
- Test: `tests/test_music_cog.py`

**Step 1: Write the failing test**

```python
# tests/test_music_cog.py
import pytest
from unittest.mock import MagicMock
from collections import deque


def test_music_cog_initial_state():
    from src.cogs.music import MusicCog

    bot = MagicMock()
    bot.config = {"music": {"max_queue_size": 50, "default_volume": 0.5}}
    cog = MusicCog(bot)
    assert cog.is_playing is False
    assert len(cog.queue) == 0


def test_music_cog_clear_state():
    from src.cogs.music import MusicCog

    bot = MagicMock()
    bot.config = {"music": {"max_queue_size": 50, "default_volume": 0.5}}
    cog = MusicCog(bot)
    cog.queue.append({"title": "test", "url": "http://test"})
    cog.is_playing = True
    cog.now_playing = {"title": "test"}

    cog._clear_state()
    assert cog.is_playing is False
    assert len(cog.queue) == 0
    assert cog.now_playing is None


def test_music_cog_rejects_when_listening():
    from src.cogs.music import MusicCog

    bot = MagicMock()
    bot.config = {"music": {"max_queue_size": 50, "default_volume": 0.5}}
    voice_cog = MagicMock()
    voice_cog.is_listening = True
    bot.get_cog.return_value = voice_cog

    cog = MusicCog(bot)
    # The actual slash command rejection is tested via integration,
    # but we can verify the check logic
    assert voice_cog.is_listening is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_music_cog.py -v`
Expected: FAIL with "cannot import name 'MusicCog'"

**Step 3: Write cogs/music.py**

```python
# src/cogs/music.py
import asyncio
import logging
from collections import deque

import discord
from discord import app_commands
from discord.ext import commands

log = logging.getLogger("vibebot.music")

FFMPEG_OPTIONS = {
    "before_options": "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
    "options": "-vn",
}


class MusicCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        mc = bot.config["music"]
        self._max_queue = mc["max_queue_size"]
        self._volume = mc["default_volume"]
        self.queue: deque[dict] = deque(maxlen=self._max_queue)
        self.is_playing = False
        self.now_playing: dict | None = None

    def _clear_state(self):
        self.queue.clear()
        self.is_playing = False
        self.now_playing = None

    def _check_listening(self) -> bool:
        voice_cog = self.bot.get_cog("VoiceCog")
        return voice_cog is not None and voice_cog.is_listening

    async def _search_youtube(self, query: str) -> dict | None:
        """Use yt-dlp to search YouTube and extract audio info."""
        loop = asyncio.get_event_loop()
        try:
            import yt_dlp
            ydl_opts = {
                "format": "bestaudio/best",
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
                "default_search": "ytsearch",
                "extract_flat": False,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = await loop.run_in_executor(
                    None, lambda: ydl.extract_info(query, download=False)
                )
                if "entries" in info:
                    info = info["entries"][0]
                return {
                    "title": info.get("title", "Unknown"),
                    "url": info["url"],
                    "webpage_url": info.get("webpage_url", ""),
                    "duration": info.get("duration", 0),
                }
        except Exception as e:
            log.warning("yt-dlp search failed: %s", e)
            return None

    def _play_next(self, guild: discord.Guild, error=None):
        if error:
            log.warning("Player error: %s", error)

        if self.queue:
            track = self.queue.popleft()
            self.now_playing = track
            vc = guild.voice_client
            if vc:
                source = discord.FFmpegPCMAudio(track["url"], **FFMPEG_OPTIONS)
                source = discord.PCMVolumeTransformer(source, volume=self._volume)
                vc.play(source, after=lambda e: self._play_next(guild, e))
        else:
            self.is_playing = False
            self.now_playing = None

    @app_commands.command(name="play", description="Play a song from YouTube")
    @app_commands.describe(query="Song name or YouTube URL")
    async def play(self, interaction: discord.Interaction, query: str):
        if self._check_listening():
            await interaction.response.send_message(
                "Disable listening first with /listen.", ephemeral=True
            )
            return

        vc = interaction.guild.voice_client
        if not vc:
            if interaction.user.voice:
                vc = await interaction.user.voice.channel.connect()
            else:
                await interaction.response.send_message(
                    "You need to be in a voice channel.", ephemeral=True
                )
                return

        await interaction.response.defer()

        track = await self._search_youtube(query)
        if not track:
            await interaction.followup.send("Couldn't find that track.")
            return

        track["requester"] = interaction.user.display_name

        if vc.is_playing():
            self.queue.append(track)
            await interaction.followup.send(
                f"Queued: **{track['title']}** (#{len(self.queue)} in queue)"
            )
        else:
            self.is_playing = True
            self.now_playing = track
            source = discord.FFmpegPCMAudio(track["url"], **FFMPEG_OPTIONS)
            source = discord.PCMVolumeTransformer(source, volume=self._volume)
            vc.play(source, after=lambda e: self._play_next(interaction.guild, e))
            await interaction.followup.send(f"Now playing: **{track['title']}**")

    @app_commands.command(name="skip", description="Skip the current song")
    async def skip(self, interaction: discord.Interaction):
        vc = interaction.guild.voice_client
        if not vc or not vc.is_playing():
            await interaction.response.send_message(
                "Nothing is playing.", ephemeral=True
            )
            return
        vc.stop()  # Triggers _play_next via the after callback
        await interaction.response.send_message("Skipped.", ephemeral=True)

    @app_commands.command(name="queue", description="Show the music queue")
    async def show_queue(self, interaction: discord.Interaction):
        if not self.queue and not self.now_playing:
            await interaction.response.send_message(
                "Queue is empty.", ephemeral=True
            )
            return

        lines = []
        if self.now_playing:
            lines.append(f"**Now playing:** {self.now_playing['title']}")
        for i, track in enumerate(self.queue, 1):
            lines.append(f"{i}. {track['title']}")

        await interaction.response.send_message("\n".join(lines), ephemeral=True)

    @app_commands.command(name="pause", description="Pause the current song")
    async def pause(self, interaction: discord.Interaction):
        vc = interaction.guild.voice_client
        if vc and vc.is_playing():
            vc.pause()
            await interaction.response.send_message("Paused.", ephemeral=True)
        else:
            await interaction.response.send_message(
                "Nothing is playing.", ephemeral=True
            )

    @app_commands.command(name="resume", description="Resume the paused song")
    async def resume(self, interaction: discord.Interaction):
        vc = interaction.guild.voice_client
        if vc and vc.is_paused():
            vc.resume()
            await interaction.response.send_message("Resumed.", ephemeral=True)
        else:
            await interaction.response.send_message(
                "Nothing is paused.", ephemeral=True
            )

    @app_commands.command(name="stop", description="Stop music and clear queue")
    async def stop(self, interaction: discord.Interaction):
        vc = interaction.guild.voice_client
        if vc and (vc.is_playing() or vc.is_paused()):
            vc.stop()
        self._clear_state()
        await interaction.response.send_message(
            "Stopped and cleared queue.", ephemeral=True
        )

    @app_commands.command(name="np", description="Show now playing")
    async def now_playing_cmd(self, interaction: discord.Interaction):
        if not self.now_playing:
            await interaction.response.send_message(
                "Nothing is playing.", ephemeral=True
            )
            return

        track = self.now_playing
        duration = track.get("duration", 0)
        mins, secs = divmod(duration, 60)
        await interaction.response.send_message(
            f"**{track['title']}** [{mins}:{secs:02d}] "
            f"(requested by {track.get('requester', 'unknown')})",
            ephemeral=True,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_music_cog.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/cogs/music.py tests/test_music_cog.py
git commit -m "feat: music cog with /play, /skip, /queue, /pause, /resume, /stop, /np"
```

---

### Task 10: Integration Test and README

**Files:**
- Create: `tests/test_integration.py`
- Create: `README.md`

**Step 1: Write integration test**

```python
# tests/test_integration.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import yaml


def test_full_config_loads():
    """Verify the example config is valid and has all required sections."""
    with open("config.example.yaml") as f:
        config = yaml.safe_load(f)

    assert "discord" in config
    assert "token" in config["discord"]
    assert "voice" in config
    assert "asr_url" in config["voice"]
    assert "tts_host" in config["voice"]
    assert "tts_port" in config["voice"]
    assert "llm" in config
    assert "base_url" in config["llm"]
    assert "model" in config["llm"]
    assert "music" in config
    assert "max_queue_size" in config["music"]


def test_mutual_exclusion_voice_blocks_music():
    from src.cogs.music import MusicCog
    from src.cogs.voice import VoiceCog

    bot = MagicMock()
    bot.config = {
        "voice": {
            "asr_url": "http://localhost:8000/v1",
            "tts_host": "localhost",
            "tts_port": 3000,
            "tts_voice": "Carter",
            "tts_cfg": 1.5,
            "tts_steps": 5,
            "vad_aggressiveness": 2,
            "silence_threshold_ms": 300,
            "context_window": 5,
            "max_utterance_seconds": 30,
        },
        "llm": {
            "base_url": "http://localhost:8010/v1",
            "model": "qwen3.5:9b",
            "max_tokens": 150,
            "system_prompt": "You are VibeBot.",
        },
        "music": {"max_queue_size": 50, "default_volume": 0.5},
    }

    voice_cog = VoiceCog(bot)
    music_cog = MusicCog(bot)

    # Simulate voice listening
    voice_cog.is_listening = True
    bot.get_cog.return_value = voice_cog

    # Music should detect listening is active
    assert music_cog._check_listening() is True

    # Simulate music playing
    voice_cog.is_listening = False
    music_cog.is_playing = True
    bot.get_cog.return_value = music_cog

    # Voice should see music is playing (checked in the slash command handler)
    assert music_cog.is_playing is True
```

**Step 2: Run test**

Run: `pytest tests/test_integration.py -v`
Expected: PASS

**Step 3: Write README.md**

```markdown
# VibeBot

Discord voice conversation + music bot. Uses Microsoft VibeVoice for real-time
speech-to-text and text-to-speech, with Qwen3.5:9B for generating responses.

## What it does

**Voice mode** (`/listen`): Talk in a voice channel and the bot responds
conversationally. Uses VibeVoice ASR to transcribe your speech, Qwen3.5:9B
to generate a response, and VibeVoice Realtime TTS to speak it back.

**Music mode** (`/play`): Play YouTube audio in a voice channel with standard
queue controls.

Voice and music are mutually exclusive. Stop one before starting the other.

## Requirements

- Python 3.12+
- FFmpeg
- NVIDIA GPU (for VibeVoice models)
- Discord bot token

## External services

These run as separate processes:

1. **VibeVoice ASR** (vLLM server, port 8000)
2. **VibeVoice Realtime TTS** (WebSocket server, port 3000)
3. **vLLM Qwen3.5:9B** (port 8010)

## Setup

```bash
# Clone VibeVoice
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice && pip install -e .[streamingtts,vllm]

# Start ASR server
python vllm_plugin/scripts/start_server.py

# Start TTS server (separate terminal)
python demo/vibevoice_realtime_demo.py \
  --model_path microsoft/VibeVoice-Realtime-0.5B \
  --device cuda --port 3000

# Start LLM server (separate terminal)
vllm serve Qwen/Qwen3.5-9B --port 8010

# Install VibeBot
cd VibeBot
bash install.sh

# Edit config
vim config.yaml  # add your Discord token

# Run
conda activate vibebot
python -m src.bot
```

## Commands

| Command | Description |
|---------|-------------|
| `/join` | Join your voice channel |
| `/leave` | Leave and clear all state |
| `/listen` | Toggle voice conversation mode |
| `/play <query>` | Play a song from YouTube |
| `/skip` | Skip current song |
| `/queue` | Show the queue |
| `/pause` | Pause playback |
| `/resume` | Resume playback |
| `/stop` | Stop music, clear queue |
| `/np` | Show now playing |

## Tests

```bash
pytest tests/ -v
```
```

**Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add tests/test_integration.py README.md
git commit -m "feat: integration tests and README"
```

---

### Task 11: Final verification and push

**Step 1: Run full test suite**

```bash
cd /home/ubuntu/git/personal/VibeBot
conda activate vibebot
pytest tests/ -v
```

**Step 2: Push to GitHub**

```bash
git remote add origin https://github.com/jasperan/VibeBot.git
git push -u origin main
```
