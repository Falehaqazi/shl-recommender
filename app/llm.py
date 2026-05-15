"""
LLM client. Single entry point so the rest of the codebase doesn't care
which provider is live.

Design notes (interview defense):

- OpenRouter is PRIMARY because it gives us free access to Llama-3.3-70B
  with a reasonable per-day rate budget. The API is OpenAI-compatible,
  so we just talk to it with httpx — no new SDK dependency.

- Groq is FALLBACK. If OpenRouter rate-limits us (429) or has an outage,
  Groq's free tier kicks in. We accept Groq's heavier rate limits as a
  fallback-only cost.

- Gemini is a SECOND fallback if both above fail. Optional; only used
  when GEMINI_API_KEY is set.

- All calls request JSON mode. Free-text replies still go through json
  mode by wrapping in {"reply": "..."} so we never have to parse
  markdown fences or fight model preambles.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx

from app.config import settings

log = logging.getLogger(__name__)


# OpenRouter free model that works well for our task. The :free suffix is
# critical -- it routes to the free provider pool.
OPENROUTER_DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class LLMError(RuntimeError):
    """Raised when all providers fail."""


class LLMClient:
    def __init__(self) -> None:
        self._openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self._openrouter_model = os.getenv(
            "OPENROUTER_MODEL", OPENROUTER_DEFAULT_MODEL
        ).strip()

        self._groq_client = None
        if settings.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(
                    api_key=settings.groq_api_key,
                    timeout=8.0,
                )
            except ImportError:
                log.warning("groq package not installed; skipping Groq fallback")

        self._gemini_configured = False
        if settings.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.gemini_api_key)
                self._gemini_configured = True
            except ImportError:
                log.warning("google-generativeai not installed; skipping Gemini")

        # HTTP client reused across calls for connection pooling.
        self._http = httpx.Client(timeout=15.0)

        if not (self._openrouter_key or self._groq_client or self._gemini_configured):
            log.error(
                "No LLM provider configured. Set OPENROUTER_API_KEY, GROQ_API_KEY, "
                "or GEMINI_API_KEY."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_json(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> dict[str, Any]:
        """Call the LLM and return parsed JSON.

        Tries providers in order:
            1. OpenRouter (free Llama 3.3 70B)
            2. Groq (free Llama variants)
            3. Gemini Flash

        Raises LLMError only if ALL configured providers fail.
        """
        errors: list[str] = []

        # --- OpenRouter (primary) ---
        if self._openrouter_key:
            try:
                return self._call_openrouter(system, messages, temperature, max_tokens)
            except (httpx.HTTPError, json.JSONDecodeError, LLMError, ValueError) as e:
                log.warning("OpenRouter call failed: %s. Trying Groq.", str(e)[:200])
                errors.append(f"openrouter: {e}")

        # --- Groq (fallback 1) ---
        if self._groq_client is not None:
            try:
                return self._call_groq(system, messages, temperature, max_tokens)
            except Exception as e:
                log.warning("Groq fallback failed: %s. Trying Gemini.", str(e)[:200])
                errors.append(f"groq: {e}")

        # --- Gemini (fallback 2) ---
        if self._gemini_configured:
            try:
                return self._call_gemini(system, messages, temperature, max_tokens)
            except Exception as e:
                log.error("Gemini fallback also failed: %s", str(e)[:200])
                errors.append(f"gemini: {e}")

        raise LLMError(f"All providers failed: {errors}")

    # ------------------------------------------------------------------
    # OpenRouter (primary)
    # ------------------------------------------------------------------

    def _call_openrouter(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        body = {
            "model": self._openrouter_model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://shl-recommender.local",
            "X-Title": "SHL Assessment Recommender",
        }

        t0 = time.monotonic()
        r = self._http.post(f"{OPENROUTER_BASE}/chat/completions", json=body, headers=headers)
        latency = time.monotonic() - t0

        if r.status_code == 429:
            raise LLMError(f"OpenRouter rate-limited (429) after {latency:.1f}s")
        if r.status_code >= 400:
            raise LLMError(
                f"OpenRouter HTTP {r.status_code}: {r.text[:200]}"
            )

        data = r.json()
        if "choices" not in data or not data["choices"]:
            raise LLMError(f"OpenRouter returned no choices: {data}")
        content = data["choices"][0]["message"].get("content") or "{}"
        log.debug("openrouter call %.2fs", latency)

        # Some free providers ignore response_format and wrap JSON in
        # markdown. Strip fences defensively.
        content = _strip_json_fences(content)
        return json.loads(content)

    # ------------------------------------------------------------------
    # Groq (fallback 1)
    # ------------------------------------------------------------------

    def _call_groq(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        assert self._groq_client is not None
        t0 = time.monotonic()
        resp = self._groq_client.chat.completions.create(
            model=settings.primary_model,
            messages=[{"role": "system", "content": system}, *messages],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        log.debug("groq call %.2fs", time.monotonic() - t0)
        content = resp.choices[0].message.content or "{}"
        return json.loads(_strip_json_fences(content))

    # ------------------------------------------------------------------
    # Gemini (fallback 2)
    # ------------------------------------------------------------------

    def _call_gemini(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        import google.generativeai as genai
        model = genai.GenerativeModel(
            model_name=settings.fallback_model,
            system_instruction=system,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "response_mime_type": "application/json",
            },
        )
        history = []
        for m in messages[:-1]:
            history.append(
                {
                    "role": "user" if m["role"] == "user" else "model",
                    "parts": [m["content"]],
                }
            )
        chat = model.start_chat(history=history)
        resp = chat.send_message(messages[-1]["content"])
        return json.loads(_strip_json_fences(resp.text))


def _strip_json_fences(s: str) -> str:
    """Some free models wrap JSON in ```json ... ``` blocks despite our
    response_format request. Strip those defensively."""
    s = s.strip()
    if s.startswith("```"):
        # Drop opening fence (possibly ```json)
        s = s.split("\n", 1)[-1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


# Singleton -- instantiated once at import.
llm_client = LLMClient()
