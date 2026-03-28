import base64
import io
import json
import logging
import wave

import httpx

log = logging.getLogger("vibebot.asr")


class ASRClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=60.0)

    def _pcm_to_wav_bytes(self, pcm_data: bytes, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return buf.getvalue()

    async def transcribe(self, pcm_data: bytes, sample_rate: int = 16000) -> str | None:
        wav_bytes = self._pcm_to_wav_bytes(pcm_data, sample_rate)
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        data_url = f"data:audio/wav;base64,{audio_b64}"

        try:
            resp = await self._http.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": "VibeVoice-ASR",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio_url", "audio_url": {"url": data_url}},
                                {"type": "text", "text": "transcribe this audio"},
                            ],
                        }
                    ],
                },
            )
            if resp.status_code != 200:
                log.warning("ASR returned %d: %s", resp.status_code, resp.text[:200])
                return None

            content = resp.json()["choices"][0]["message"]["content"]
            try:
                parsed = json.loads(content)
                text = parsed.get("text", "").strip()
            except json.JSONDecodeError:
                text = content.strip()

            return text if text else None
        except Exception as e:
            log.warning("ASR request failed: %s", e)
            return None

    async def close(self):
        await self._http.aclose()
