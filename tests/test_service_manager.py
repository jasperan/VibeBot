import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import subprocess


def test_service_manager_initial_state():
    from src.service_manager import ServiceManager

    mgr = ServiceManager({})
    assert mgr.is_running is False


def test_service_manager_empty_config():
    from src.service_manager import ServiceManager

    mgr = ServiceManager({"services": {}})
    assert mgr.is_running is False


@pytest.mark.asyncio
async def test_service_manager_ensure_running_starts_processes():
    from src.service_manager import ServiceManager

    config = {
        "services": {
            "llm": {
                "command": "echo hello",
                "health_url": "http://localhost:8010/v1/models",
                "startup_timeout": 5,
            }
        }
    }
    mgr = ServiceManager(config)

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.pid = 12345

    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
         patch.object(mgr, "_wait_for_health", new_callable=AsyncMock):
        await mgr.ensure_running()

    assert mgr.is_running is True
    mock_popen.assert_called_once()


@pytest.mark.asyncio
async def test_service_manager_ensure_running_idempotent():
    from src.service_manager import ServiceManager

    mgr = ServiceManager({"services": {}})
    await mgr.ensure_running()
    assert mgr.is_running is True

    # Calling again should be a no-op
    await mgr.ensure_running()
    assert mgr.is_running is True


@pytest.mark.asyncio
async def test_service_manager_shutdown():
    from src.service_manager import ServiceManager

    mgr = ServiceManager({"services": {}})
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.pid = 12345
    mgr._processes["llm"] = mock_proc
    mgr._running = True

    await mgr.shutdown()

    assert mgr.is_running is False
    assert len(mgr._processes) == 0
    mock_proc.terminate.assert_called_once()


@pytest.mark.asyncio
async def test_service_manager_shutdown_kills_on_timeout():
    from src.service_manager import ServiceManager

    mgr = ServiceManager({"services": {}})
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.pid = 12345
    mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 10), None]
    mgr._processes["llm"] = mock_proc
    mgr._running = True

    await mgr.shutdown()

    mock_proc.terminate.assert_called_once()
    mock_proc.kill.assert_called_once()
