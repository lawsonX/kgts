"""Offline tests for the LLM layer (ManagedLLM budget/cache behavior)."""

import pytest

from kgts.llm import BudgetExceeded, ManagedLLM, MockLLM


class _CostlyMock(MockLLM):
    """MockLLM that reports a fixed cost per call, like LiteLLMClient does."""

    def __init__(self, cost: float, **kw):
        super().__init__(**kw)
        self.last_cost = cost


def test_cost_budget_stops_calls(tmp_path):
    llm = ManagedLLM(_CostlyMock(0.5, default={}), max_cost_usd=1.0)
    llm.complete("a")
    llm.complete("b")  # total_cost now 1.0 -> next call must fail
    assert llm.total_cost == 1.0
    with pytest.raises(BudgetExceeded, match="cost budget"):
        llm.complete("c")
    assert llm.n_calls == 2


def test_call_budget_and_cache(tmp_path):
    inner = MockLLM(default={"k": "v"})
    llm = ManagedLLM(inner, max_calls=1, cache_dir=tmp_path)
    first = llm.complete("same prompt")
    with pytest.raises(BudgetExceeded):
        llm.complete("different prompt")
    # cached reply does not consume budget
    assert llm.complete("same prompt") == first
    assert llm.n_calls == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


class RateLimitError(Exception):
    pass  # class name is what ManagedLLM matches on


class _FlakyMock(MockLLM):
    def __init__(self, fail_times: int, **kw):
        super().__init__(**kw)
        self.fail_times = fail_times
        self.attempts = 0

    def complete(self, prompt, *, temperature=0.7, **kw2):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise RateLimitError("429")
        return super().complete(prompt, temperature=temperature, **kw2)


def test_transient_errors_are_retried(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    llm = ManagedLLM(_FlakyMock(2, default={"ok": True}), retry_backoff=0.01)
    assert llm.complete_json("q") == {"ok": True}
    assert llm.client.attempts == 3  # two failures + one success


def test_retry_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    llm = ManagedLLM(_FlakyMock(99, default={}), max_retries=2, retry_backoff=0.01)
    with pytest.raises(RateLimitError):
        llm.complete("q")
    assert llm.client.attempts == 3  # initial + 2 retries


def test_non_retryable_errors_fail_fast():
    llm = ManagedLLM(_FlakyMock(0, default={}))
    llm.client.fail_times = 0

    class _Bad(MockLLM):
        def complete(self, p, *, temperature=0.7, **kw):
            raise ValueError("bad request")  # not in the retryable list

    llm2 = ManagedLLM(_Bad(default={}))
    with pytest.raises(ValueError):
        llm2.complete("q")
