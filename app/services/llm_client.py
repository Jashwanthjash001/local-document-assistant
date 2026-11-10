"""
LLM client — Stage 5.

Thin client for a **local** OpenAI-compatible chat endpoint (llama.cpp server
from edge-ai-client). Uses httpx directly so there's no extra dependency and it's
easy to mock in tests.

Hard rules:
- The base URL is always local (config). We NEVER fall back to a cloud model and
  NEVER send document content to an external API.
- Errors are normalized to typed exceptions so the API layer can respond with a
  real service error instead of pretending a question was answered.
"""
from __future__ import annotations

import httpx

from app import config


class LLMError(Exception):
    """Base class for LLM client failures."""


class LLMConnectionError(LLMError):
    """Could not reach the local LLM server."""


class LLMTimeoutError(LLMError):
    """The LLM server did not respond in time."""


class LLMResponseError(LLMError):
    """The LLM server returned an error status or an unparseable response."""


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant's text.

        Raises an LLMError subclass on any failure.
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
            "max_tokens": config.LLM_MAX_TOKENS if max_tokens is None else max_tokens,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"LLM request timed out after {self.timeout}s") from exc
        except httpx.RequestError as exc:
            # Connection refused, DNS, etc. — the local server is probably down.
            raise LLMConnectionError(
                f"Could not reach local LLM at {self.base_url}: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise LLMResponseError(
                f"LLM server returned HTTP {response.status_code}: {response.text[:200]}"
            )

        # Don't assume the body shape — validate it and fail loudly if it's wrong.
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(f"Unexpected LLM response shape: {exc}") from exc

        if not isinstance(content, str):
            raise LLMResponseError("LLM response content was not text.")
        return content


# --- module singleton ------------------------------------------------------
_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient(
            base_url=config.LLM_BASE_URL,
            model=config.LLM_MODEL,
            api_key=config.LLM_API_KEY,
            timeout=config.LLM_TIMEOUT,
        )
    return _client
