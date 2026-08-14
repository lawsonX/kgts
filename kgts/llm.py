"""LLM abstraction layer: protocol, mock client, LiteLLM client, budget/cache.

The whole pipeline talks to an ``LLMClient``; tests and the offline quickstart
use ``MockLLM`` so nothing requires an API key.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class BudgetExceeded(RuntimeError):
    """Raised when the configured LLM call / cost budget is exhausted."""


@runtime_checkable
class LLMClient(Protocol):
    """Minimal contract every LLM backend must satisfy."""

    model: str

    def complete(self, prompt: str, *, temperature: float = 0.7, **kw: Any) -> str:
        """Return the raw completion text."""
        ...

    def complete_json(self, prompt: str, *, temperature: float = 0.2, **kw: Any) -> Any:
        """Complete and parse the reply as JSON (raises ValueError on bad JSON)."""
        ...


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from an LLM reply (handles ```json fences)."""
    text = text.strip()
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # find the outermost JSON object or array
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in LLM reply: {text[:200]!r}")


class MockLLM:
    """Deterministic offline LLM for tests and the quickstart demo.

    ``script`` maps a substring of the prompt to a reply (str or JSON-able).
    The first matching key (in insertion order) wins. Unmatched prompts get a
    generic JSON reply derived from ``default``.
    """

    model = "mock-llm"

    def __init__(self, script: dict[str, Any] | None = None, default: Any = None):
        self.script = dict(script or {})
        self.default = default if default is not None else {}
        self.calls: list[str] = []  # prompt log, for assertions

    def _reply(self, prompt: str) -> Any:
        self.calls.append(prompt)
        for key, value in self.script.items():
            if key in prompt:
                return value
        return self.default

    def complete(self, prompt: str, *, temperature: float = 0.7, **kw: Any) -> str:
        reply = self._reply(prompt)
        return reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)

    def complete_json(self, prompt: str, *, temperature: float = 0.2, **kw: Any) -> Any:
        reply = self._reply(prompt)
        if isinstance(reply, str):
            return extract_json(reply)
        return reply


class LiteLLMClient:
    """LLMClient over any OpenAI-compatible endpoint, via litellm (optional dep)."""

    def __init__(self, model: str, api_base: str | None = None, **default_kw: Any):
        try:
            import litellm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "LiteLLMClient requires the `llm` extra: pip install 'kgts[llm]'"
            ) from e
        self.model = model
        self.api_base = api_base
        self.default_kw = default_kw
        self.last_cost = 0.0  # USD cost of the most recent call (for ManagedLLM)

    def complete(self, prompt: str, *, temperature: float = 0.7, **kw: Any) -> str:
        import litellm

        resp = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            api_base=self.api_base,
            **{**self.default_kw, **kw},
        )
        try:
            self.last_cost = float(litellm.completion_cost(completion_response=resp) or 0.0)
        except Exception:
            self.last_cost = 0.0  # unknown pricing: budget stays call-based
        return resp.choices[0].message.content or ""

    def complete_json(self, prompt: str, *, temperature: float = 0.2, **kw: Any) -> Any:
        return extract_json(self.complete(prompt, temperature=temperature, **kw))


class ManagedLLM:
    """Cross-cutting wrapper: budget hard-cap, on-disk cache, rate limiting.

    Wraps any ``LLMClient`` and is what pipeline stages actually receive.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        max_calls: int | None = None,
        max_cost_usd: float | None = None,
        rpm: int | None = None,
        cache_dir: str | Path | None = None,
        max_retries: int = 4,
        retry_backoff: float = 20.0,
    ):
        self.client = client
        self.model = getattr(client, "model", "unknown")
        self.max_calls = max_calls
        self.max_cost_usd = max_cost_usd
        self.rpm = rpm
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_calls = 0
        self.total_cost = 0.0
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._last_call_ts = 0.0

    # -- internals ---------------------------------------------------------
    def _cache_key(self, prompt: str, temperature: float, kind: str) -> str:
        raw = f"{self.model}|{kind}|{temperature}|{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _throttle(self) -> None:
        if not self.rpm:
            return
        min_gap = 60.0 / self.rpm
        wait = min_gap - (time.monotonic() - self._last_call_ts)
        if wait > 0:
            time.sleep(wait)

    def _call(self, prompt: str, temperature: float, kind: str, **kw: Any) -> str:
        key = self._cache_key(prompt, temperature, kind)
        cache_file = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cache_file and cache_file.exists():
            return json.loads(cache_file.read_text())["reply"]
        if self.max_calls is not None and self.n_calls >= self.max_calls:
            raise BudgetExceeded(
                f"LLM call budget exhausted ({self.max_calls}); raise budget.max_llm_calls"
            )
        if self.max_cost_usd is not None and self.total_cost >= self.max_cost_usd:
            raise BudgetExceeded(
                f"LLM cost budget exhausted (${self.total_cost:.4f} >= "
                f"${self.max_cost_usd}); raise budget.max_cost_usd"
            )
        self._throttle()
        reply = self._complete_with_retry(prompt, temperature, **kw)
        self._last_call_ts = time.monotonic()
        self.n_calls += 1
        self.total_cost += float(getattr(self.client, "last_cost", 0.0) or 0.0)
        if cache_file:
            cache_file.write_text(json.dumps({"reply": reply}, ensure_ascii=False))
        return reply

    # transient provider errors worth backing off for (matched by class name
    # so this stays client-agnostic): 429s, timeouts, connection drops, 5xx
    _RETRYABLE = ("RateLimit", "Timeout", "APIConnection", "ServiceUnavailable", "InternalServer")

    def _complete_with_retry(self, prompt: str, temperature: float, **kw: Any) -> str:
        attempt = 0
        while True:
            try:
                return self.client.complete(prompt, temperature=temperature, **kw)
            except Exception as exc:
                retryable = any(k in type(exc).__name__ for k in self._RETRYABLE)
                if not retryable or attempt >= self.max_retries:
                    raise
                delay = self.retry_backoff * (2**attempt)
                time.sleep(delay)
                attempt += 1

    # -- public API (mirrors LLMClient) ------------------------------------
    def complete(self, prompt: str, *, temperature: float = 0.7, **kw: Any) -> str:
        return self._call(prompt, temperature, "text", **kw)

    def complete_json(self, prompt: str, *, temperature: float = 0.2, **kw: Any) -> Any:
        return extract_json(self._call(prompt, temperature, "json", **kw))
