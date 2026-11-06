import pytest
from unittest.mock import MagicMock, AsyncMock, patch
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


def test_music_cog_check_listening():
    from src.cogs.music import MusicCog
    bot = MagicMock()
    bot.config = {"music": {"max_queue_size": 50, "default_volume": 0.5}}
    voice_cog = MagicMock()
    voice_cog.is_listening = True
    bot.get_cog.return_value = voice_cog
    cog = MusicCog(bot)
    assert cog._check_listening() is True


def test_set_volume():
    from src.cogs.music import MusicCog
    bot = MagicMock()
    bot.config = {"music": {"max_queue_size": 50, "default_volume": 0.5}}
    bot.voice_clients = []
    cog = MusicCog(bot)

    cog.set_volume(0.8)
    assert cog._volume == 0.8

    # Clamps to bounds
    cog.set_volume(1.5)
    assert cog._volume == 1.0

    cog.set_volume(-0.3)
    assert cog._volume == 0.0


def test_set_volume_applies_to_live_source():
    import discord
    from src.cogs.music import MusicCog
    bot = MagicMock()
    bot.config = {"music": {"max_queue_size": 50, "default_volume": 0.5}}
    cog = MusicCog(bot)

    source = MagicMock(spec=discord.PCMVolumeTransformer)
    vc = MagicMock()
    vc.source = source
    bot.voice_clients = [vc]

    cog.set_volume(0.3)
    assert source.volume == 0.3


@pytest.mark.asyncio
async def test_voice_play_queues_when_playing():
    from src.cogs.music import MusicCog
    bot = MagicMock()
    bot.config = {"music": {"max_queue_size": 50, "default_volume": 0.5}}
    cog = MusicCog(bot)

    guild = MagicMock()
    vc = MagicMock()
    vc.is_playing.return_value = True
    guild.voice_client = vc

    track = {"title": "Test Song", "url": "http://test", "webpage_url": "", "duration": 180}
    with patch.object(cog, "_search_youtube", new_callable=AsyncMock, return_value=track):
        await cog.voice_play(guild, "test song")

    assert len(cog.queue) == 1
    assert cog.queue[0]["title"] == "Test Song"
    assert cog.queue[0]["requester"] == "VibeBot (voice)"


@pytest.mark.asyncio
async def test_voice_play_starts_when_idle():
    from src.cogs.music import MusicCog
    bot = MagicMock()
    bot.config = {"music": {"max_queue_size": 50, "default_volume": 0.5}}
    cog = MusicCog(bot)

    guild = MagicMock()
    vc = MagicMock()
    vc.is_playing.return_value = False
    guild.voice_client = vc

    track = {"title": "Test Song", "url": "http://test", "webpage_url": "", "duration": 180}
    with patch.object(cog, "_search_youtube", new_callable=AsyncMock, return_value=track):
        await cog.voice_play(guild, "test song")

    assert cog.is_playing is True
    assert cog.now_playing["title"] == "Test Song"
    vc.play.assert_called_once()


@pytest.mark.asyncio
async def test_voice_skip():
    from src.cogs.music import MusicCog
    bot = MagicMock()
    bot.config = {"music": {"max_queue_size": 50, "default_volume": 0.5}}
    cog = MusicCog(bot)

    guild = MagicMock()
    vc = MagicMock()
    vc.is_playing.return_value = True
    guild.voice_client = vc

    await cog.voice_skip(guild)
    vc.stop.assert_called_once()


@pytest.mark.asyncio
async def test_voice_stop():
    from src.cogs.music import MusicCog
    bot = MagicMock()
    bot.config = {"music": {"max_queue_size": 50, "default_volume": 0.5}}
    cog = MusicCog(bot)
    cog.is_playing = True
    cog.now_playing = {"title": "Song"}

    guild = MagicMock()
    vc = MagicMock()
    vc.is_playing.return_value = True
    vc.is_paused.return_value = False
    guild.voice_client = vc

    await cog.voice_stop(guild)
    vc.stop.assert_called_once()
    assert cog.is_playing is False
    assert cog.now_playing is None


@pytest.mark.asyncio
async def test_voice_play_no_results():
    from src.cogs.music import MusicCog
    bot = MagicMock()
    bot.config = {"music": {"max_queue_size": 50, "default_volume": 0.5}}
    cog = MusicCog(bot)

    guild = MagicMock()
    guild.voice_client = MagicMock()

    with patch.object(cog, "_search_youtube", new_callable=AsyncMock, return_value=None):
        await cog.voice_play(guild, "nonexistent song xyz")

    assert len(cog.queue) == 0
    assert cog.is_playing is False


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8000/v1/models",
    "https://169.254.169.254/latest/meta-data/",
    "file:///etc/hostname",
    "ftp://example.com/x",
    "//evil.example/x",
    "http://youtube.com@evil.example/x",
    "https://youtube.com.evil.example/watch?v=x",
])
def test_as_search_query_rejects_non_youtube_urls(url):
    from src.cogs.music import _as_search_query
    with pytest.raises(ValueError):
        _as_search_query(url)


def test_as_search_query_allows_youtube_urls_and_wraps_search_phrases():
    from src.cogs.music import _as_search_query
    assert _as_search_query("https://www.youtube.com/watch?v=abc") == "https://www.youtube.com/watch?v=abc"
    assert _as_search_query("https://youtu.be/abc") == "https://youtu.be/abc"
    assert _as_search_query("Re: something good") == "ytsearch1:Re: something good"
    assert _as_search_query("127.0.0.1:8000/x") == "ytsearch1:127.0.0.1:8000/x"


@pytest.mark.asyncio
async def test_search_youtube_does_not_resolve_loopback_url():
    from src.cogs.music import MusicCog
    bot = MagicMock()
    bot.config = {"music": {"max_queue_size": 50, "default_volume": 0.5}}
    cog = MusicCog(bot)
    with patch("yt_dlp.YoutubeDL") as ydl:
        assert await cog._search_youtube("http://127.0.0.1:8000/v1/models") is None
    ydl.assert_not_called()
