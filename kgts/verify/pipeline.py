"""Stage E entry point used by the orchestrator."""

from __future__ import annotations

from typing import Any

from kgts.config import VerifyConfig
from kgts.models import Material, Task, VerifyResult
from kgts.verify.answer_match import AnswerMatchVerifier
from kgts.verify.base import Verifier
from kgts.verify.rubric import RubricJudge


def _make_verifier(name: str, llm: Any, config: VerifyConfig) -> Verifier | None:
    if name == "answer_match":
        return AnswerMatchVerifier()
    if name == "rubric_judge":
        return RubricJudge(llm, config.rubric_judge.pass_score)
    return None


def verify_task(
    task: Task, materials_by_id: dict[str, Material], llm: Any, config: VerifyConfig
) -> Task:
    """Verify one task and return it (orchestrator contract).

    Tasks marked SFT_ONLY (no verifier registered) pass through unchanged.
    Unknown verifier names fall back to ``config.fallback``; if the fallback is
    also unknown the task is downgraded to SFT_ONLY.
    """
    if task.verify_result == VerifyResult.SFT_ONLY or task.verifier is None:
        return task
    verifier = _make_verifier(task.verifier, llm, config)
    if verifier is None:
        verifier = _make_verifier(config.fallback, llm, config)
    if verifier is None:
        task.verify_result = VerifyResult.SFT_ONLY
        return task
    result, _score, _note = verifier.verify(task, materials_by_id)
    task.verify_result = result
    return task
