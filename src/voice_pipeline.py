import logging
from typing import AsyncIterator

from src.asr_client import ASRClient
from src.llm_client import LLMClient
from src.tts_client import TTSClient
from src.audio import resample_pcm, mono_to_stereo

log = logging.getLogger("vibebot.pipeline")


class VoicePipeline:
    def __init__(self, asr_client: ASRClient, llm_client: LLMClient,
                 tts_client: TTSClient, context_window: int = 5):
        self.asr = asr_client
        self.llm = llm_client
        self.tts = tts_client
        self._context: list[dict] = []
        self._max_context = context_window * 2

    async def process_utterance(self, pcm_16k_mono: bytes) -> AsyncIterator[bytes]:
        transcript = await self.asr.transcribe(pcm_16k_mono, sample_rate=16000)
        if not transcript:
            log.debug("Empty transcript, skipping")
            return

        log.info("ASR transcript: %s", transcript)

        response = await self.llm.generate(transcript, self._context)
        if not response:
            log.warning("LLM returned no response")
            return

        log.info("LLM response: %s", response)

        self._context.append({"role": "user", "content": transcript})
        self._context.append({"role": "assistant", "content": response})
        if len(self._context) > self._max_context:
            self._context = self._context[-self._max_context:]

        async for tts_chunk in self.tts.synthesize(response):
            pcm_48k = resample_pcm(tts_chunk, from_rate=24000, to_rate=48000)
            pcm_48k_stereo = mono_to_stereo(pcm_48k)
            yield pcm_48k_stereo

    def clear_context(self):
        self._context.clear()
