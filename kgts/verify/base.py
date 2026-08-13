"""Stage E verifier protocol."""

from __future__ import annotations

from typing import Protocol

from kgts.models import Material, Task, VerifyResult


class Verifier(Protocol):
    """A task-level verifier (design doc section 8)."""

    name: str

    def verify(
        self, task: Task, materials_by_id: dict[str, Material]
    ) -> tuple[VerifyResult, float, str]:
        """Return (result, score in [0, 1], human-readable note)."""
        ...
