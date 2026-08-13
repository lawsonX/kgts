"""Self-consistency verifier for generated QA tasks.

Heuristic (kept honest and simple): these tasks carry their own reference
answer produced from retrieved materials, so there is no external gold answer
to match against. Instead we check the answer is self-consistent with its
grounding: after normalization (lowercase, whitespace/punctuation collapsed)
it must be non-empty and explicitly cite at least one of the material IDs
listed in ``task.materials``. An answer that cites nothing is very likely
ungrounded (or dropped its citations), so it FAILs.
"""

from __future__ import annotations

import re

from kgts.models import Material, Task, VerifyResult

_WS_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def _normalize(text: str) -> str:
    return _WS_PUNCT_RE.sub(" ", text.lower()).strip()


class AnswerMatchVerifier:
    name = "answer_match"

    def verify(
        self, task: Task, materials_by_id: dict[str, Material]
    ) -> tuple[VerifyResult, float, str]:
        answer = _normalize(task.answer)
        if not answer:
            return VerifyResult.FAIL, 0.0, "empty answer"
        if not task.materials:
            return VerifyResult.FAIL, 0.0, "task lists no materials to cite"
        cited = [mid for mid in task.materials if _normalize(mid) in answer]
        if cited:
            note = f"answer cites {len(cited)}/{len(task.materials)} material IDs"
            return VerifyResult.PASS, 1.0, note
        return VerifyResult.FAIL, 0.0, "answer cites none of the task's material IDs"
