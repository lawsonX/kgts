"""Generic fallback verifier: an LLM judges the answer against the task rubric."""

from __future__ import annotations

from typing import Any

from kgts.models import Material, Task, VerifyResult


class RubricJudge:
    """LLM-as-judge over the task's rubric items; PASS if mean >= pass_score."""

    name = "rubric_judge"

    def __init__(self, llm: Any, pass_score: float = 0.7) -> None:
        self.llm = llm
        self.pass_score = pass_score

    def verify(
        self, task: Task, materials_by_id: dict[str, Material]
    ) -> tuple[VerifyResult, float, str]:
        prompt = self._build_prompt(task, materials_by_id)
        try:
            reply = self.llm.complete_json(prompt)
        except ValueError as e:
            return VerifyResult.FAIL, 0.0, f"judge reply not parseable: {e}"
        if not isinstance(reply, dict) or not isinstance(reply.get("scores"), list):
            return VerifyResult.FAIL, 0.0, "judge reply lacks a 'scores' list"
        if not reply["scores"]:
            return VerifyResult.FAIL, 0.0, "judge returned an empty 'scores' list"
        try:
            scores = [float(s) for s in reply["scores"]]
        except (TypeError, ValueError):
            return VerifyResult.FAIL, 0.0, "judge scores are not numeric"
        mean = sum(scores) / len(scores)
        rationale = str(reply.get("rationale", "")).strip()
        result = VerifyResult.PASS if mean >= self.pass_score else VerifyResult.FAIL
        return result, mean, rationale or f"mean rubric score {mean:.2f}"

    def _build_prompt(self, task: Task, materials_by_id: dict[str, Material]) -> str:
        lines = [
            "You are verifying a generated training task against its rubric.",
            "",
            f"Question: {task.question}",
            f"Answer: {task.answer}",
            "",
            "Rubric items:",
        ]
        lines += [f"{i}. {item}" for i, item in enumerate(task.rubric, 1)] or ["1. (no rubric)"]
        lines += ["", "Materials:"]
        for mid in task.materials:
            m = materials_by_id.get(mid)
            if m is not None:
                lines.append(f"[{m.id}] {m.title}: {m.snippet}")
        lines += [
            "",
            "Score how well the answer satisfies each rubric item, from 0 to 1.",
            'Reply with JSON only: {"scores": [float, ...], "rationale": str}.',
        ]
        return "\n".join(lines)
