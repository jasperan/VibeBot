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
    assert cog._is_responding is False
    assert cog._current_personality == "default"


def test_voice_cog_listen_rejects_when_music_playing():
    from src.cogs.voice import VoiceCog
    bot = MagicMock()
    bot.config = _make_bot_config()
    music_cog = MagicMock()
    music_cog.is_playing = True
    bot.get_cog.return_value = music_cog
    cog = VoiceCog(bot)
    assert cog.is_listening is False


def test_personality_presets_exist():
    from src.cogs.voice import PERSONALITY_PRESETS
    assert "default" in PERSONALITY_PRESETS
    assert "pirate" in PERSONALITY_PRESETS
    assert "shakespeare" in PERSONALITY_PRESETS
    assert "sarcastic" in PERSONALITY_PRESETS
    assert "zen" in PERSONALITY_PRESETS
    assert "hype" in PERSONALITY_PRESETS
    for name, prompt in PERSONALITY_PRESETS.items():
        assert len(prompt) > 10, f"Preset '{name}' prompt too short"


def test_personality_switch():
    from src.cogs.voice import VoiceCog, PERSONALITY_PRESETS
    bot = MagicMock()
    bot.config = _make_bot_config()
    cog = VoiceCog(bot)

    assert cog._current_personality == "default"
    cog._current_personality = "pirate"
    cog._llm.system_prompt = PERSONALITY_PRESETS["pirate"]
    assert "pirate" in cog._llm.system_prompt.lower()


def test_music_tools_defined():
    from src.cogs.voice import MUSIC_TOOLS
    tool_names = {t["function"]["name"] for t in MUSIC_TOOLS}
    assert "play_music" in tool_names
    assert "skip_song" in tool_names
    assert "stop_music" in tool_names
    assert "set_volume" in tool_names


def test_pipeline_has_tools_wired():
    from src.cogs.voice import VoiceCog, MUSIC_TOOLS
    bot = MagicMock()
    bot.config = _make_bot_config()
    cog = VoiceCog(bot)
    assert cog._pipeline.tools == MUSIC_TOOLS
    assert cog._pipeline.tool_executor is not None
