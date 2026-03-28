import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_tts_synthesize_returns_pcm():
    from src.tts_client import TTSClient

    client = TTSClient(host="localhost", port=3000, voice="Carter", cfg=1.5, steps=5)

    frames = [b"\x00\x01" * 100, b"\x00\x02" * 100]

    mock_ws = AsyncMock()
    mock_ws.__aiter__ = lambda self: self
    call_count = 0
    async def next_frame(self):
        nonlocal call_count
        if call_count < len(frames):
            frame = frames[call_count]
            call_count += 1
            return frame
        raise StopAsyncIteration()
    mock_ws.__anext__ = next_frame

    mock_connect = AsyncMock()
    mock_connect.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_connect.__aexit__ = AsyncMock(return_value=False)

    with patch("websockets.connect", return_value=mock_connect):
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
    assert "voice=Carter" in url
