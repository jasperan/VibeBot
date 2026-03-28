import logging
from typing import AsyncIterator
from urllib.parse import urlencode

import websockets

log = logging.getLogger("vibebot.tts")

SAMPLE_RATE = 24000


class TTSClient:
    def __init__(self, host: str, port: int, voice: str = "Carter",
                 cfg: float = 1.5, steps: int = 5):
        self.host = host
        self.port = port
        self.voice = voice
        self.cfg = cfg
        self.steps = steps

    def _build_url(self, text: str) -> str:
        params = urlencode({
            "text": text,
            "voice": self.voice,
            "cfg": self.cfg,
            "steps": self.steps,
        })
        return f"ws://{self.host}:{self.port}/stream?{params}"

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        url = self._build_url(text)
        try:
            async with websockets.connect(url) as ws:
                async for message in ws:
                    if isinstance(message, bytes):
                        yield message
        except Exception as e:
            log.warning("TTS WebSocket error: %s", e)
