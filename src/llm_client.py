import json
import logging
import uuid
from dataclasses import dataclass, field

import httpx

log = logging.getLogger("vibebot.llm")


@dataclass
class LLMResponse:
    """Structured response from the LLM, supporting both text and tool calls."""
    text: str | None = None
    tool_calls: list[dict] = field(default_factory=list)

    def __bool__(self):
        return bool(self.text) or bool(self.tool_calls)


class LLMClient:
    """LLM client supporting both OpenAI-compatible (vLLM) and Ollama backends.

    Backend is set via the `backend` config key:
      - "openai" (default): uses /v1/chat/completions with chat_template_kwargs
      - "ollama": uses /api/chat with think=false for lower latency
    """

    def __init__(self, base_url: str, model: str, max_tokens: int,
                 system_prompt: str, backend: str = "openai"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.backend = backend
        self._http = httpx.AsyncClient(timeout=60.0)

    async def generate(self, user_text: str, context: list[dict],
                       tools: list[dict] | None = None) -> LLMResponse:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(context)
        messages.append({"role": "user", "content": user_text})

        if self.backend == "ollama":
            return await self._generate_ollama(messages, tools)
        return await self._generate_openai(messages, tools)

    async def _generate_openai(self, messages: list[dict],
                               tools: list[dict] | None) -> LLMResponse:
        """OpenAI-compatible endpoint (vLLM, LiteLLM, etc.)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            resp = await self._http.post(
                f"{self.base_url}/chat/completions", json=payload,
            )
            if resp.status_code != 200:
                log.warning("LLM returned %d: %s", resp.status_code, resp.text[:200])
                return LLMResponse()

            msg = resp.json()["choices"][0]["message"]
            text = msg.get("content")
            tool_calls = msg.get("tool_calls", [])
            return LLMResponse(text=text, tool_calls=tool_calls)
        except Exception as e:
            log.warning("LLM request failed: %s", e)
            return LLMResponse()

    async def _generate_ollama(self, messages: list[dict],
                               tools: list[dict] | None) -> LLMResponse:
        """Ollama native API with thinking disabled for low-latency voice."""
        # Strip /v1 suffix if present to get native API base
        base = self.base_url
        if base.endswith("/v1"):
            base = base[:-3]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"num_predict": self.max_tokens},
        }
        if tools:
            payload["tools"] = tools

        try:
            resp = await self._http.post(f"{base}/api/chat", json=payload)
            if resp.status_code != 200:
                log.warning("Ollama returned %d: %s", resp.status_code, resp.text[:200])
                return LLMResponse()

            data = resp.json()
            msg = data.get("message", {})
            text = msg.get("content") or None

            # Ollama returns tool calls differently
            tool_calls = []
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": json.dumps(func.get("arguments", {})),
                    },
                })

            return LLMResponse(text=text, tool_calls=tool_calls)
        except Exception as e:
            log.warning("Ollama request failed: %s: %s", type(e).__name__, e)
            return LLMResponse()

    async def close(self):
        await self._http.aclose()
