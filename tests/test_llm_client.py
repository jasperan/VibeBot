import pytest
import httpx
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_llm_generate_returns_text():
    from src.llm_client import LLMClient

    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "Hello there!"}}]},
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
        json={"choices": [{"message": {"content": "You said hello earlier."}}]},
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
