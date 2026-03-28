import logging
import httpx

log = logging.getLogger("vibebot.llm")


class LLMClient:
    def __init__(self, base_url: str, model: str, max_tokens: int, system_prompt: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self._http = httpx.AsyncClient(timeout=30.0)

    async def generate(self, user_text: str, context: list[dict]) -> str | None:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(context)
        messages.append({"role": "user", "content": user_text})

        try:
            resp = await self._http.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            if resp.status_code != 200:
                log.warning("LLM returned %d: %s", resp.status_code, resp.text[:200])
                return None
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            log.warning("LLM request failed: %s", e)
            return None

    async def close(self):
        await self._http.aclose()
