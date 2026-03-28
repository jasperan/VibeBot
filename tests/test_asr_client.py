import pytest
import httpx
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_asr_transcribe_returns_text():
    from src.asr_client import ASRClient

    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": '{"text": "hello world"}'}}]},
    )

    client = ASRClient(base_url="http://localhost:8000/v1")
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
        json={"choices": [{"message": {"content": '{"text": ""}'}}]},
    )

    client = ASRClient(base_url="http://localhost:8000/v1")
    pcm_data = bytes(32000)

    with patch.object(client._http, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await client.transcribe(pcm_data, sample_rate=16000)
        assert result is None
