import asyncio
import logging
import signal
import subprocess

import httpx

log = logging.getLogger("vibebot.services")


class ServiceManager:
    """Manages external service subprocesses (ASR, TTS, LLM). Lazy-starts on demand."""

    def __init__(self, config: dict):
        self._services_config = config.get("services", {})
        self._processes: dict[str, subprocess.Popen] = {}
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def ensure_running(self) -> None:
        """Start all services if not already running. Blocks until healthy."""
        if self._running:
            return

        for name, svc in self._services_config.items():
            if name in self._processes and self._processes[name].poll() is None:
                log.info("Service %s already running (pid %d)", name, self._processes[name].pid)
                continue

            log.info("Starting service: %s", name)
            proc = subprocess.Popen(
                svc["command"],
                shell=True,
                cwd=svc.get("cwd"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
            )
            self._processes[name] = proc
            log.info("Service %s started (pid %d)", name, proc.pid)

        # Wait for all services to become healthy
        for name, svc in self._services_config.items():
            health_url = svc.get("health_url", "")
            timeout = svc.get("startup_timeout", 120)
            if health_url:
                await self._wait_for_health(name, health_url, timeout)

        self._running = True
        log.info("All services ready")

    async def _wait_for_health(self, name: str, url: str, timeout: int) -> None:
        """Poll a health endpoint until it responds or timeout."""
        if url.startswith("ws://") or url.startswith("wss://"):
            await self._wait_for_ws_health(name, url, timeout)
            return

        async with httpx.AsyncClient(timeout=5.0) as client:
            elapsed = 0
            interval = 2
            while elapsed < timeout:
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        log.info("Service %s is healthy", name)
                        return
                except Exception:
                    pass
                await asyncio.sleep(interval)
                elapsed += interval

        raise TimeoutError(f"Service {name} did not become healthy within {timeout}s")

    async def _wait_for_ws_health(self, name: str, url: str, timeout: int) -> None:
        """Probe a WebSocket endpoint until it accepts connections."""
        import websockets

        elapsed = 0
        interval = 2
        while elapsed < timeout:
            try:
                async with websockets.connect(url, close_timeout=2) as ws:
                    log.info("Service %s is healthy (WebSocket)", name)
                    return
            except Exception:
                pass
            await asyncio.sleep(interval)
            elapsed += interval

        raise TimeoutError(f"Service {name} (WebSocket) did not become healthy within {timeout}s")

    async def shutdown(self) -> None:
        """Terminate all managed service processes."""
        for name, proc in self._processes.items():
            if proc.poll() is None:
                log.info("Stopping service %s (pid %d)", name, proc.pid)
                proc.terminate()
                try:
                    await asyncio.to_thread(proc.wait, timeout=10)
                except subprocess.TimeoutExpired:
                    log.warning("Service %s did not stop, killing", name)
                    proc.kill()
                    await asyncio.to_thread(proc.wait)
                log.info("Service %s stopped", name)

        self._processes.clear()
        self._running = False
        log.info("All services stopped")
