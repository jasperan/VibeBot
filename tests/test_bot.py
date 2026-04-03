import pytest
from unittest.mock import patch, MagicMock
import yaml


def test_load_config_reads_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "discord": {"token": "test-token"},
        "voice": {"asr_url": "http://localhost:8000/v1"},
        "llm": {"base_url": "http://localhost:8010/v1", "model": "qwen3.5:9b"},
        "music": {"max_queue_size": 50},
    }))
    from src.bot import load_config
    config = load_config(str(config_file))
    assert config["discord"]["token"] == "test-token"
    assert config["llm"]["model"] == "qwen3.5:9b"


def test_load_config_missing_file():
    from src.bot import load_config
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_validate_runtime_config_accepts_real_token():
    from src.bot import validate_runtime_config

    config = {"discord": {"token": "real-token"}}
    validate_runtime_config(config)


def test_validate_runtime_config_rejects_placeholder_token():
    from src.bot import validate_runtime_config

    config = {"discord": {"token": "YOUR_DISCORD_BOT_TOKEN"}}
    with pytest.raises(ValueError, match="Discord bot token"):
        validate_runtime_config(config)


def test_validate_runtime_config_rejects_missing_token():
    from src.bot import validate_runtime_config

    with pytest.raises(ValueError, match="Discord bot token"):
        validate_runtime_config({"discord": {}})
