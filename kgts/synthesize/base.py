"""Stage D plugin base: TaskType ABC + registry (design doc section 7)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from kgts.models import Material, SampleBundle, Task


class MaterialSpec(BaseModel):
    """What a task type needs from the retrieved material pool."""

    min_docs: int = 1
    prefers: list[str] = Field(default_factory=list)


class TaskType(ABC):
    """A synthesizable task type. Implementations must be deterministic apart
    from the LLM call itself, and raise ValueError on an unusable LLM reply."""

    name: str
    requires: MaterialSpec = MaterialSpec()
    verifier_name: str | None = None

    @abstractmethod
    def generate(
        self, llm: Any, bundle: SampleBundle, materials: list[Material], style: dict
    ) -> Task:
        """Generate one grounded task from ``materials``."""


_REGISTRY: dict[str, type[TaskType]] = {}


def register_task_type(name: str):
    """Class decorator: register a TaskType under ``name``."""

    def deco(cls: type[TaskType]) -> type[TaskType]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_task_type(name: str) -> TaskType:
    """Instantiate the registered task type ``name``."""
    try:
        return _REGISTRY[name]()
    except KeyError:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown task type {name!r}; registered: {valid}") from None


def list_task_types() -> list[str]:
    return sorted(_REGISTRY)
