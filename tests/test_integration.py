import pytest
from unittest.mock import MagicMock
import yaml


def test_full_config_loads():
    with open("config.example.yaml") as f:
        config = yaml.safe_load(f)
    assert "discord" in config
    assert "token" in config["discord"]
    assert "voice" in config
    assert "asr_url" in config["voice"]
    assert "tts_host" in config["voice"]
    assert "tts_port" in config["voice"]
    assert "llm" in config
    assert "base_url" in config["llm"]
    assert "model" in config["llm"]
    assert "music" in config
    assert "max_queue_size" in config["music"]
    assert "services" in config


def test_mutual_exclusion_voice_blocks_music():
    from src.cogs.music import MusicCog
    from src.cogs.voice import VoiceCog

    bot = MagicMock()
    bot.config = {
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
        "music": {"max_queue_size": 50, "default_volume": 0.5},
    }

    voice_cog = VoiceCog(bot)
    music_cog = MusicCog(bot)

    voice_cog.is_listening = True
    bot.get_cog.return_value = voice_cog
    assert music_cog._check_listening() is True

    voice_cog.is_listening = False
    music_cog.is_playing = True
    bot.get_cog.return_value = music_cog
    assert music_cog.is_playing is True


def test_all_cogs_instantiate():
    """All three cogs can be created with valid config."""
    from src.cogs.voice import VoiceCog
    from src.cogs.music import MusicCog
    from src.cogs.admin import AdminCog

    bot = MagicMock()
    bot.config = {
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
        "music": {"max_queue_size": 50, "default_volume": 0.5},
        "services": {
            "asr": {"health_url": "http://localhost:8000/v1/models"},
            "llm": {"health_url": "http://localhost:8010/v1/models"},
            "tts": {"health_url": "ws://localhost:3000/stream"},
        },
    }

    voice = VoiceCog(bot)
    music = MusicCog(bot)
    admin = AdminCog(bot)

    assert voice.is_listening is False
    assert music.is_playing is False
    assert admin.bot is bot


def test_voice_tools_wired_to_pipeline():
    """Voice cog wires music tools into the pipeline."""
    from src.cogs.voice import VoiceCog, MUSIC_TOOLS

    bot = MagicMock()
    bot.config = {
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

    cog = VoiceCog(bot)
    assert cog._pipeline.tools is MUSIC_TOOLS
    assert cog._pipeline.tool_executor is not None

    # Verify all 4 tools are defined
    tool_names = {t["function"]["name"] for t in MUSIC_TOOLS}
    assert tool_names == {"play_music", "skip_song", "stop_music", "set_volume"}
