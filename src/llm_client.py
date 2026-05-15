from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, Optional

from openai import OpenAI

try:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
except Exception:  # pragma: no cover - older SDKs
    APIConnectionError = APITimeoutError = APIStatusError = RateLimitError = Exception  # type: ignore


def _load_dotenv_if_available() -> None:
    """
    Loads .env if python-dotenv is installed.
    Safe no-op if not installed.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


_load_dotenv_if_available()

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (check .env or environment variables).")

    global _client
    if _client is None:
        _client = OpenAI(api_key=api_key)
    return _client


def _extract_retry_after(exc: Exception) -> Optional[float]:
    """Extract Retry-After header from an OpenAI SDK exception, if present."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None) or {}
    for header in ("Retry-After", "retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        value = headers.get(header) if hasattr(headers, "get") else None
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        if status is not None and (status == 429 or 500 <= int(status) < 600):
            return True
    # Empty-output retries are handled separately via RuntimeError.
    if isinstance(exc, RuntimeError) and "empty output_text" in str(exc).lower():
        return True
    return False


def generate_text(
    user_prompt: str,
    *,
    system_prompt: str = "You are a helpful assistant.",
    api_endpoint: str = "responses",
    model: str = "gpt-5.4",
    temperature: Optional[float] = 0.0,
    max_output_tokens: int = 1500,
    reasoning_effort: Optional[str] = None,
    retries: int = 5,
    retry_backoff_s: float = 2.0,
    max_backoff_s: float = 60.0,
) -> str:
    """
    Minimal LLM call wrapper using the OpenAI Responses API.

    - user_prompt: the main prompt/content
    - system_prompt: system instructions
    - api_endpoint: currently supports "responses"
    - model: e.g. "gpt-5.4"
    - temperature: set None to omit (some models/settings may reject it)
    - max_output_tokens: output budget
    - reasoning_effort: optional Responses API reasoning effort, e.g. "high"
    - retries: total number of attempts on retryable errors
    - retry_backoff_s: base backoff; exponential with jitter, honors Retry-After
    """
    client = _get_client()
    if api_endpoint != "responses":
        raise ValueError(f"Unsupported api_endpoint: {api_endpoint}. Only 'responses' is implemented.")

    input_items = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req: Dict[str, Any] = {
                "model": model,
                "input": input_items,
                "max_output_tokens": max_output_tokens,
            }

            # Temperature is optional; omit if None.
            if temperature is not None:
                req["temperature"] = float(temperature)
            if reasoning_effort is not None:
                req["reasoning"] = {"effort": reasoning_effort}

            resp = client.responses.create(**req)
            text = (resp.output_text or "").strip()
            if not text:
                raise RuntimeError("Model returned empty output_text.")
            return text

        except Exception as e:
            last_err = e

            # If model rejects temperature, drop it and retry immediately.
            msg = str(e).lower()
            if temperature is not None and "temperature" in msg and ("not supported" in msg or "unsupported" in msg):
                temperature = None
                continue

            if attempt >= retries or not _is_retryable(e):
                break

            # Honor Retry-After header if present; otherwise exponential backoff with jitter.
            retry_after = _extract_retry_after(e)
            if retry_after is not None and retry_after > 0:
                sleep_s = min(retry_after, max_backoff_s)
            else:
                sleep_s = min(retry_backoff_s * (2 ** (attempt - 1)), max_backoff_s)
                sleep_s += random.uniform(0, sleep_s * 0.25)  # jitter
            time.sleep(sleep_s)

    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}") from last_err
