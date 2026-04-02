import pytest
from unittest.mock import MagicMock, AsyncMock, patch


def _make_bot():
    bot = MagicMock()
    bot.config = {
        "services": {
            "asr": {
                "command": "echo asr",
                "health_url": "http://localhost:8000/v1/models",
                "startup_timeout": 120,
            },
            "tts": {
                "command": "echo tts",
                "health_url": "ws://localhost:3000/stream",
                "startup_timeout": 120,
            },
            "llm": {
                "command": "echo llm",
                "health_url": "http://localhost:8010/v1/models",
                "startup_timeout": 120,
            },
        },
    }
    bot.guilds = [MagicMock()]
    bot.voice_clients = []
    bot.services._processes = {}

    voice_cog = MagicMock()
    voice_cog.is_listening = False
    voice_cog._current_personality = "default"

    music_cog = MagicMock()
    music_cog.is_playing = False

    def get_cog(name):
        if name == "VoiceCog":
            return voice_cog
        if name == "MusicCog":
            return music_cog
        return None

    bot.get_cog = get_cog
    return bot


def test_admin_cog_init():
    from src.cogs.admin import AdminCog
    bot = _make_bot()
    cog = AdminCog(bot)
    assert cog.bot is bot


@pytest.mark.asyncio
async def test_check_health_http_success():
    from src.cogs.admin import AdminCog
    import httpx

    bot = _make_bot()
    cog = AdminCog(bot)

    mock_resp = httpx.Response(200)
    with patch("src.cogs.admin.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await cog._check_health("http://localhost:8000/health")
        assert result is True


@pytest.mark.asyncio
async def test_check_health_http_failure():
    from src.cogs.admin import AdminCog

    bot = _make_bot()
    cog = AdminCog(bot)

    with patch("src.cogs.admin.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Connection refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await cog._check_health("http://localhost:8000/health")
        assert result is False
