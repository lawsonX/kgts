"""Built-in task types (v0.3 set, design doc section 7). Registered on import.

All built-ins share one generation flow: build a prompt with node labels +
ancestor paths, the style dict, and the materials listed as
``[material_id] title: snippet``; the prompt forces the model to cite material
IDs in the answer; the JSON reply becomes a Task.
"""

from __future__ import annotations

from typing import Any

from kgts.models import Material, SampleBundle, Task, VerifyResult
from kgts.synthesize.base import MaterialSpec, TaskType, register_task_type

_CITE_RULE = (
    "Every key fact in the answer MUST cite the material it came from, using "
    "the material ID in square brackets exactly as listed above (e.g. [m_ab12cd34ef56])."
)


def _node_label(bundle: SampleBundle, node_id: str) -> str:
    path = bundle.ancestor_paths.get(node_id) or []
    return path[-1] if path else node_id


def _build_prompt(
    instruction: str, bundle: SampleBundle, materials: list[Material], style: dict
) -> str:
    language = style.get("language", "en")
    lines = [
        "You are generating a training task grounded in retrieved materials.",
        "",
        f"Task type instruction: {instruction}",
        "",
        "Concepts (with ancestor paths from the knowledge graph):",
    ]
    for nid in bundle.nodes:
        path = bundle.ancestor_paths.get(nid) or []
        crumbs = " > ".join(path) if path else nid
        lines.append(f"- {_node_label(bundle, nid)} (path: {crumbs})")
    lines += [
        "",
        f"Style: language={language}, length={style.get('length', 'medium')}, "
        f"difficulty={style.get('difficulty', 'mixed')}. "
        f"Write the question and the answer in language '{language}'.",
        "",
        "Materials:",
    ]
    for m in materials:
        lines.append(f"[{m.id}] {m.title}: {m.snippet}")
    lines += [
        "",
        _CITE_RULE,
        "",
        'Reply with JSON only: {"question": str, "answer": str, "rubric": [str, ...]}.',
    ]
    return "\n".join(lines)


def _generate(
    llm: Any,
    task_type_name: str,
    verifier_name: str | None,
    instruction: str,
    bundle: SampleBundle,
    materials: list[Material],
    style: dict,
) -> Task:
    """Shared generation flow for all built-in task types."""
    prompt = _build_prompt(instruction, bundle, materials, style)
    reply = llm.complete_json(prompt)
    if not isinstance(reply, dict):
        raise ValueError(f"LLM reply is not a JSON object: {reply!r}")
    question, answer = reply.get("question"), reply.get("answer")
    if not question or not answer:
        raise ValueError("LLM reply lacks a non-empty 'question' or 'answer'")
    rubric = reply.get("rubric") or []
    if not isinstance(rubric, list):
        rubric = [str(rubric)]
    return Task(
        task_type=task_type_name,
        sample_bundle=bundle,
        materials=[m.id for m in materials],
        prompt=prompt,
        question=str(question),
        answer=str(answer),
        rubric=[str(r) for r in rubric],
        verifier=verifier_name,
        verify_result=VerifyResult.UNVERIFIED,
        style=dict(style),
    )


class _Builtin(TaskType):
    """Base for built-ins: subclasses only declare spec/verifier/instruction."""

    instruction: str = ""

    def generate(
        self, llm: Any, bundle: SampleBundle, materials: list[Material], style: dict
    ) -> Task:
        return _generate(
            llm, self.name, self.verifier_name, self.instruction, bundle, materials, style
        )


@register_task_type("atomic_qa")
class AtomicQA(_Builtin):
    """Single-fact QA grounded in the materials of one node."""

    requires = MaterialSpec(min_docs=1)
    verifier_name = "answer_match"
    instruction = (
        "Write a question about a single fact that is directly stated in one "
        "of the materials; the answer cites that material."
    )


@register_task_type("aggregated_qa")
class AggregatedQA(_Builtin):
    """QA whose answer must aggregate information across >= 2 material IDs."""

    requires = MaterialSpec(min_docs=2)
    verifier_name = "answer_match"
    instruction = (
        "Write a question whose answer must aggregate facts from at least two "
        "different materials; cite every material used."
    )


@register_task_type("multihop_qa")
class MultiHopQA(_Builtin):
    """Question requiring multi-hop reasoning over multiple concepts."""

    requires = MaterialSpec(min_docs=2)
    verifier_name = "answer_match"
    instruction = (
        "Write a question that requires chaining reasoning over multiple "
        "concepts from the knowledge graph (multi-hop); the answer must cite "
        "the materials supporting each hop."
    )


@register_task_type("grounded_summary")
class GroundedSummary(_Builtin):
    """Summary of the materials with citations (no dedicated verifier)."""

    requires = MaterialSpec(min_docs=1)
    verifier_name = None
    instruction = (
        "Ask for a concise summary of the materials; every claim in the answer "
        "must cite the material ID it comes from."
    )


@register_task_type("comparative_analysis")
class ComparativeAnalysis(_Builtin):
    """Compare viewpoints across materials (no dedicated verifier)."""

    requires = MaterialSpec(min_docs=2)
    verifier_name = None
    instruction = (
        "Ask for a comparison of the viewpoints or approaches in at least two "
        "materials; the answer must contrast them and cite their material IDs."
    )
