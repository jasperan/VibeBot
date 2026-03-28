import pytest
from unittest.mock import MagicMock


def _make_bot_config():
    return {
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


def test_voice_cog_initial_state():
    from src.cogs.voice import VoiceCog
    bot = MagicMock()
    bot.config = _make_bot_config()
    cog = VoiceCog(bot)
    assert cog.is_listening is False


def test_voice_cog_listen_rejects_when_music_playing():
    from src.cogs.voice import VoiceCog
    bot = MagicMock()
    bot.config = _make_bot_config()
    music_cog = MagicMock()
    music_cog.is_playing = True
    bot.get_cog.return_value = music_cog
    cog = VoiceCog(bot)
    assert cog.is_listening is False
