"""Offline tests for Stage E (verify) using MockLLM."""

import pytest

from kgts.config import VerifyConfig
from kgts.llm import MockLLM
from kgts.models import Material, SampleBundle, SampleIntent, SourceType, Task, VerifyResult
from kgts.verify.answer_match import AnswerMatchVerifier
from kgts.verify.pipeline import verify_task
from kgts.verify.rubric import RubricJudge


def _task(
    answer: str = "",
    materials: tuple[str, ...] = ("m_1",),
    verifier: str | None = "answer_match",
    verify_result: VerifyResult = VerifyResult.UNVERIFIED,
) -> Task:
    bundle = SampleBundle(nodes=["n1"], intent=SampleIntent.DEPTH)
    return Task(
        task_type="atomic_qa",
        sample_bundle=bundle,
        materials=list(materials),
        question="What is a GPU kernel?",
        answer=answer,
        rubric=["accurate", "grounded"],
        verifier=verifier,
        verify_result=verify_result,
    )


def _materials() -> dict[str, Material]:
    return {"m_1": Material(id="m_1", source_type=SourceType.LOCAL, title="t", snippet="s")}


# --------------------------------------------------------- AnswerMatch


def test_answer_match_pass_when_citing():
    task = _task(answer="A kernel is a device function [m_1].")
    result, score, note = AnswerMatchVerifier().verify(task, _materials())
    assert result == VerifyResult.PASS
    assert score == 1.0
    assert "m_1" in note or "cites" in note


def test_answer_match_fail_without_citation():
    task = _task(answer="A kernel is a device function.")
    result, score, note = AnswerMatchVerifier().verify(task, _materials())
    assert result == VerifyResult.FAIL
    assert score == 0.0
    assert note


def test_answer_match_fail_empty_answer():
    result, score, _ = AnswerMatchVerifier().verify(_task(answer="  "), _materials())
    assert result == VerifyResult.FAIL
    assert score == 0.0


# ---------------------------------------------------------- RubricJudge


def test_rubric_judge_pass():
    llm = MockLLM(default={"scores": [0.9, 0.8], "rationale": "good"})
    result, score, note = RubricJudge(llm, pass_score=0.7).verify(_task(answer="a"), _materials())
    assert result == VerifyResult.PASS
    assert score == pytest.approx(0.85)
    assert note == "good"


def test_rubric_judge_fail_below_threshold():
    llm = MockLLM(default={"scores": [0.2, 0.4], "rationale": "weak"})
    result, score, _ = RubricJudge(llm, pass_score=0.7).verify(_task(answer="a"), _materials())
    assert result == VerifyResult.FAIL
    assert score == pytest.approx(0.3)


def test_rubric_judge_bad_reply_fails():
    result, score, note = RubricJudge(MockLLM(default={"nope": 1})).verify(
        _task(answer="a"), _materials()
    )
    assert result == VerifyResult.FAIL
    assert score == 0.0
    assert note


# ------------------------------------------------------------- verify_task


def test_verify_task_dispatches_answer_match():
    task = _task(answer="Grounded answer [m_1].")
    out = verify_task(task, _materials(), MockLLM(), VerifyConfig())
    assert out.verify_result == VerifyResult.PASS


def test_verify_task_sft_only_passthrough():
    task = _task(answer="anything", verifier=None, verify_result=VerifyResult.SFT_ONLY)
    llm = MockLLM()
    out = verify_task(task, _materials(), llm, VerifyConfig())
    assert out.verify_result == VerifyResult.SFT_ONLY
    assert llm.calls == []  # no judge call for unverifiable tasks


def test_verify_task_rubric_judge_dispatch():
    llm = MockLLM(default={"scores": [1.0, 1.0], "rationale": "ok"})
    task = _task(answer="a", verifier="rubric_judge")
    out = verify_task(task, _materials(), llm, VerifyConfig())
    assert out.verify_result == VerifyResult.PASS


def test_verify_task_unknown_verifier_falls_back():
    llm = MockLLM(default={"scores": [0.9], "rationale": "ok"})
    task = _task(answer="a", verifier="mystery")
    out = verify_task(task, _materials(), llm, VerifyConfig(fallback="rubric_judge"))
    assert out.verify_result == VerifyResult.PASS


def test_verify_task_unknown_verifier_and_fallback_downgrades():
    task = _task(answer="a", verifier="mystery")
    out = verify_task(task, _materials(), MockLLM(), VerifyConfig(fallback="also_unknown"))
    assert out.verify_result == VerifyResult.SFT_ONLY
