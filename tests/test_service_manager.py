import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class FakeProcess:
    def __init__(self, *, pid: int = 12345, exit_on_terminate: bool = True,
                 exit_on_kill: bool = True):
        self.pid = pid
        self.returncode = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self._exit_on_terminate = exit_on_terminate
        self._exit_on_kill = exit_on_kill

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        if self._exit_on_terminate:
            self.returncode = 0

    def kill(self):
        self.kill_calls += 1
        if self._exit_on_kill:
            self.returncode = -9

    def killpg(self, pgid, sig):
        """Stand-in for os.killpg: signal 0 probes group liveness."""
        import signal as _signal
        assert pgid == self.pid
        if sig == 0:
            if self.returncode is not None:
                raise ProcessLookupError
        elif sig == _signal.SIGTERM:
            self.terminate_calls += 1
            if self._exit_on_terminate:
                self.returncode = 0
        elif sig == _signal.SIGKILL:
            self.kill_calls += 1
            if self._exit_on_kill:
                self.returncode = -9


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
    proc = FakeProcess()
    mgr._processes["llm"] = proc
    mgr._running = True

    with patch("src.service_manager.os.killpg", side_effect=proc.killpg):
        await mgr.shutdown()

    assert mgr.is_running is False
    assert len(mgr._processes) == 0
    assert proc.terminate_calls == 1
    assert proc.kill_calls == 0

@pytest.mark.asyncio
async def test_service_manager_shutdown_kills_on_timeout():
    from src.service_manager import ServiceManager

    mgr = ServiceManager({"services": {}})
    proc = FakeProcess(exit_on_terminate=False, exit_on_kill=True)
    mgr._processes["llm"] = proc
    mgr._running = True

    with patch("src.service_manager.SHUTDOWN_TIMEOUT_SECONDS", 0.01), \
         patch("src.service_manager.SHUTDOWN_POLL_INTERVAL_SECONDS", 0.001), \
         patch("src.service_manager.os.killpg", side_effect=proc.killpg):
        await mgr.shutdown()

    assert proc.terminate_calls == 1
    assert proc.kill_calls == 1


@pytest.mark.asyncio
async def test_service_manager_shutdown_terminates_shell_children(tmp_path):
    """Regression: shell=True wrapper pid != service pid; the whole group must die."""
    import os
    import time
    from src.service_manager import ServiceManager

    pidfile = tmp_path / "child.pid"
    # `sleep` runs as a background child of the shell wrapper, not the wrapper itself.
    cmd = f"sleep 60 & echo $! > {pidfile}; wait"
    mgr = ServiceManager({"services": {"svc": {"command": cmd}}})

    await mgr.ensure_running()
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text().strip():
            break
        time.sleep(0.05)
    child_pid = int(pidfile.read_text())
    os.kill(child_pid, 0)  # alive

    await mgr.shutdown()

    for _ in range(50):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("service child process survived shutdown()")
