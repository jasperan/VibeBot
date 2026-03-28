import pytest
from unittest.mock import AsyncMock, MagicMock
import numpy as np


@pytest.mark.asyncio
async def test_pipeline_processes_utterance():
    from src.voice_pipeline import VoicePipeline

    mock_asr = AsyncMock()
    mock_asr.transcribe.return_value = "hello"

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "Hi there!"

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

    assert len(pipeline._context) == 2
    assert pipeline._context[0]["role"] == "user"
    assert pipeline._context[1]["role"] == "assistant"
