import pytest
import httpx
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_llm_generate_returns_response():
    from src.llm_client import LLMClient, LLMResponse

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
        assert isinstance(result, LLMResponse)
        assert result.text == "Hello there!"
        assert result.tool_calls == []
        assert bool(result) is True


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
        assert result.text == "You said hello earlier."


@pytest.mark.asyncio
async def test_llm_generate_returns_empty_on_error():
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
        assert result.text is None
        assert bool(result) is False


@pytest.mark.asyncio
async def test_llm_generate_with_tools():
    from src.llm_client import LLMClient

    mock_response = httpx.Response(
        200,
        json={
            "choices": [{
                "message": {
                    "content": "Sure, playing that for you!",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "play_music",
                            "arguments": '{"query": "Bohemian Rhapsody"}',
                        },
                    }],
                }
            }]
        },
    )

    client = LLMClient(
        base_url="http://localhost:8010/v1",
        model="qwen3.5:9b",
        max_tokens=150,
        system_prompt="You are VibeBot.",
    )

    tools = [{
        "type": "function",
        "function": {
            "name": "play_music",
            "description": "Play a song",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }]

    with patch.object(client._http, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
        result = await client.generate("Play Bohemian Rhapsody", [], tools=tools)
        assert result.text == "Sure, playing that for you!"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "play_music"

        # Verify tools were sent in payload
        call_args = mock_post.call_args
        payload = call_args.kwargs["json"] if "json" in call_args.kwargs else call_args[1]["json"]
        assert "tools" in payload
        assert payload["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_llm_generate_tool_call_only():
    """LLM returns tool call with no text content."""
    from src.llm_client import LLMClient

    mock_response = httpx.Response(
        200,
        json={
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "skip_song",
                            "arguments": "{}",
                        },
                    }],
                }
            }]
        },
    )

    client = LLMClient(
        base_url="http://localhost:8010/v1",
        model="qwen3.5:9b",
        max_tokens=150,
        system_prompt="You are VibeBot.",
    )

    with patch.object(client._http, "post", new_callable=AsyncMock, return_value=mock_response):
        result = await client.generate("Skip this song", [])
        assert result.text is None
        assert len(result.tool_calls) == 1
        assert bool(result) is True  # tool_calls make it truthy


@pytest.mark.asyncio
async def test_llm_response_bool():
    from src.llm_client import LLMResponse

    assert bool(LLMResponse()) is False
    assert bool(LLMResponse(text="hi")) is True
    assert bool(LLMResponse(tool_calls=[{"id": "1"}])) is True
    assert bool(LLMResponse(text="hi", tool_calls=[{"id": "1"}])) is True
