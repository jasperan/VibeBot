import logging
from typing import AsyncIterator, Callable, Awaitable

from src.asr_client import ASRClient
from src.llm_client import LLMClient, LLMResponse
from src.tts_client import TTSClient
from src.audio import resample_pcm, mono_to_stereo

log = logging.getLogger("vibebot.pipeline")

ToolExecutor = Callable[[dict], Awaitable[None]]


class VoicePipeline:
    def __init__(self, asr_client: ASRClient, llm_client: LLMClient,
                 tts_client: TTSClient, context_window: int = 5):
        self.asr = asr_client
        self.llm = llm_client
        self.tts = tts_client
        self._context: list[dict] = []
        self._max_context = context_window * 2
        self.tools: list[dict] | None = None
        self.tool_executor: ToolExecutor | None = None

    async def process_utterance(self, pcm_16k_mono: bytes) -> AsyncIterator[bytes]:
        transcript = await self.asr.transcribe(pcm_16k_mono, sample_rate=16000)
        if not transcript:
            log.debug("Empty transcript, skipping")
            return

        log.info("ASR transcript: %s", transcript)

        response = await self.llm.generate(transcript, self._context, tools=self.tools)
        if not response:
            log.warning("LLM returned no response")
            return

        # Execute any tool calls (voice-commanded music, etc.)
        if response.tool_calls and self.tool_executor:
            for call in response.tool_calls:
                try:
                    await self.tool_executor(call)
                except Exception as e:
                    log.warning("Tool execution failed: %s", e)

        text = response.text
        log.info("LLM response: %s", text)

        self._context.append({"role": "user", "content": transcript})
        if text:
            self._context.append({"role": "assistant", "content": text})
        if len(self._context) > self._max_context:
            self._context = self._context[-self._max_context:]

        if text:
            async for tts_chunk in self.tts.synthesize(text):
                pcm_48k = resample_pcm(tts_chunk, from_rate=24000, to_rate=48000)
                pcm_48k_stereo = mono_to_stereo(pcm_48k)
                yield pcm_48k_stereo

    @property
    def context(self) -> list[dict]:
        return list(self._context)

    async def summarize_context(self) -> str | None:
        """Use LLM to summarize the conversation so far."""
        if not self._context:
            return None
        response = await self.llm.generate(
            "Summarize this conversation concisely in 2-3 sentences.",
            self._context,
        )
        return response.text

    def clear_context(self):
        self._context.clear()
