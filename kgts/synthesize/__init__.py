"""Stage D: task synthesis. Importing this package registers the built-in task types."""

from kgts.synthesize import builtin as builtin  # noqa: F401 -- registers built-ins
from kgts.synthesize.base import (
    MaterialSpec,
    TaskType,
    get_task_type,
    list_task_types,
    register_task_type,
)
from kgts.synthesize.synthesizer import Synthesizer

__all__ = [
    "MaterialSpec",
    "Synthesizer",
    "TaskType",
    "get_task_type",
    "list_task_types",
    "register_task_type",
]
