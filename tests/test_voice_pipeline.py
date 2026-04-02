import pytest
from unittest.mock import AsyncMock, MagicMock
import numpy as np


def _make_llm_response(text="Hi there!", tool_calls=None):
    from src.llm_client import LLMResponse
    return LLMResponse(text=text, tool_calls=tool_calls or [])


@pytest.mark.asyncio
async def test_pipeline_processes_utterance():
    from src.voice_pipeline import VoicePipeline

    mock_asr = AsyncMock()
    mock_asr.transcribe.return_value = "hello"

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = _make_llm_response("Hi there!")

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
    mock_llm.generate.return_value = _make_llm_response("Four!")

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

    assert len(pipeline._context) == 2
    assert pipeline._context[0]["role"] == "user"
    assert pipeline._context[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_pipeline_executes_tool_calls():
    from src.voice_pipeline import VoicePipeline

    mock_asr = AsyncMock()
    mock_asr.transcribe.return_value = "play some jazz"

    tool_call = {
        "id": "call_1",
        "function": {"name": "play_music", "arguments": '{"query": "jazz"}'},
    }
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = _make_llm_response(
        "Playing some jazz!", tool_calls=[tool_call]
    )

    tts_chunk = np.zeros(2400, dtype=np.int16).tobytes()
    mock_tts = MagicMock()

    async def fake_synthesize(text):
        yield tts_chunk

    mock_tts.synthesize = fake_synthesize

    tool_executor = AsyncMock()

    pipeline = VoicePipeline(
        asr_client=mock_asr,
        llm_client=mock_llm,
        tts_client=mock_tts,
        context_window=5,
    )
    pipeline.tools = [{"type": "function", "function": {"name": "play_music"}}]
    pipeline.tool_executor = tool_executor

    pcm_16k = np.zeros(16000, dtype=np.int16).tobytes()
    async for _ in pipeline.process_utterance(pcm_16k):
        pass

    tool_executor.assert_awaited_once_with(tool_call)


@pytest.mark.asyncio
async def test_pipeline_no_tts_when_text_is_none():
    """When LLM returns only tool calls with no text, skip TTS."""
    from src.voice_pipeline import VoicePipeline

    mock_asr = AsyncMock()
    mock_asr.transcribe.return_value = "skip this song"

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = _make_llm_response(text=None, tool_calls=[
        {"function": {"name": "skip_song", "arguments": "{}"}},
    ])

    mock_tts = MagicMock()
    tts_called = False

    async def fake_synthesize(text):
        nonlocal tts_called
        tts_called = True
        yield b""

    mock_tts.synthesize = fake_synthesize

    pipeline = VoicePipeline(
        asr_client=mock_asr,
        llm_client=mock_llm,
        tts_client=mock_tts,
        context_window=5,
    )
    pipeline.tool_executor = AsyncMock()

    pcm_16k = np.zeros(16000, dtype=np.int16).tobytes()
    async for _ in pipeline.process_utterance(pcm_16k):
        pass

    assert tts_called is False


@pytest.mark.asyncio
async def test_pipeline_summarize_context():
    from src.voice_pipeline import VoicePipeline

    mock_asr = AsyncMock()
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = _make_llm_response("You discussed math.")
    mock_tts = MagicMock()

    pipeline = VoicePipeline(
        asr_client=mock_asr,
        llm_client=mock_llm,
        tts_client=mock_tts,
    )
    pipeline._context = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "Four!"},
    ]

    summary = await pipeline.summarize_context()
    assert summary == "You discussed math."


@pytest.mark.asyncio
async def test_pipeline_summarize_empty_context():
    from src.voice_pipeline import VoicePipeline

    pipeline = VoicePipeline(
        asr_client=AsyncMock(),
        llm_client=AsyncMock(),
        tts_client=MagicMock(),
    )

    summary = await pipeline.summarize_context()
    assert summary is None


@pytest.mark.asyncio
async def test_pipeline_context_property():
    from src.voice_pipeline import VoicePipeline

    pipeline = VoicePipeline(
        asr_client=AsyncMock(),
        llm_client=AsyncMock(),
        tts_client=MagicMock(),
    )
    pipeline._context = [{"role": "user", "content": "hi"}]

    ctx = pipeline.context
    assert ctx == [{"role": "user", "content": "hi"}]
    # Should be a copy
    ctx.append({"role": "test"})
    assert len(pipeline._context) == 1
