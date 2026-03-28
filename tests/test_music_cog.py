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


def test_music_cog_check_listening():
    from src.cogs.music import MusicCog
    bot = MagicMock()
    bot.config = {"music": {"max_queue_size": 50, "default_volume": 0.5}}
    voice_cog = MagicMock()
    voice_cog.is_listening = True
    bot.get_cog.return_value = voice_cog
    cog = MusicCog(bot)
    assert cog._check_listening() is True
