"""Stage E export: SFT/RL JSONL formats plus a full-lineage manifest (design §8).

Context injection: exported rows embed the task's cited materials (truncated
to a char budget) so the trainee actually sees the context the question
refers to -- a question that points at unseen materials trains hallucination.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from kgts.models import Material, Run, Task, VerifyResult

DEFAULT_CONTEXT_BUDGET = 4000  # chars, shared by all materials of one task


def _context_block(
    task: Task, materials_by_id: dict[str, Material], char_budget: int
) -> str:
    """Render the task's cited materials as a context block ('' if none)."""
    zh = str(task.style.get("language", "")).lower().startswith("zh")
    header = "材料：" if zh else "Materials:"
    question = "问题" if zh else "Question"
    parts: list[str] = []
    used = 0
    for mid in task.materials:
        m = materials_by_id.get(mid)
        if m is None:
            continue
        text = (m.text or m.snippet).strip()
        if not text:
            continue
        remaining = char_budget - used
        if remaining <= 0:
            break
        text = text[:remaining]
        used += len(text)
        parts.append(f"[{m.id}] {m.title}\n{text}".strip())
    if not parts:
        return ""
    return f"{header}\n" + "\n\n".join(parts) + f"\n\n{question}："


def export_sft(
    tasks: list[Task],
    materials_by_id: dict[str, Material] | None = None,
    *,
    include_context: bool = True,
    char_budget: int = DEFAULT_CONTEXT_BUDGET,
) -> list[dict]:
    """SFT messages format; keeps tasks that passed or are SFT-only (no verifier).

    With ``materials_by_id`` given and ``include_context``, the user message is
    ``<context block><question>`` so grounded questions are answerable as shipped.
    """
    rows = []
    for t in tasks:
        if t.verify_result not in (VerifyResult.PASS, VerifyResult.SFT_ONLY):
            continue
        context = ""
        if include_context and materials_by_id:
            context = _context_block(t, materials_by_id, char_budget)
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": f"{context}{t.question}"},
                    {"role": "assistant", "content": t.answer},
                ],
                "task_id": t.id,
                "task_type": t.task_type,
            }
        )
    return rows


def export_rl(
    tasks: list[Task],
    materials_by_id: dict[str, Material] | None = None,
    *,
    include_context: bool = True,
    char_budget: int = DEFAULT_CONTEXT_BUDGET,
) -> list[dict]:
    """RL format: prompt + context + rubric + verifier hook; requires PASS + verifier."""
    rows = []
    for t in tasks:
        if t.verify_result != VerifyResult.PASS or t.verifier is None:
            continue
        context = ""
        if include_context and materials_by_id:
            context = _context_block(t, materials_by_id, char_budget)
        rows.append(
            {
                "task_id": t.id,
                "prompt": t.question,
                "context": context,
                "rubric": t.rubric,
                "verifier": t.verifier,
                "env": None,
            }
        )
    return rows


def write_export(
    tasks: list[Task],
    fmt: str,
    out_path: str | Path,
    materials_by_id: dict[str, Material] | None = None,
    *,
    include_context: bool = True,
) -> int:
    """Write one JSON object per line; returns the number of rows written."""
    fmt = fmt.lower()
    if fmt == "sft":
        rows = export_sft(tasks, materials_by_id, include_context=include_context)
    elif fmt == "rl":
        rows = export_rl(tasks, materials_by_id, include_context=include_context)
    else:
        raise ValueError(f"unknown export format {fmt!r}; expected 'sft' or 'rl'")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def write_manifest(
    run: Run,
    tasks: list[Task],
    materials: list[Material],
    config_hash: str,
    out_path: str | Path,
) -> None:
    """Write manifest.json: run info, aggregate counts, and per-task lineage."""
    by_task_type = Counter(t.task_type for t in tasks)
    by_verify = Counter(t.verify_result.value for t in tasks)
    by_source = Counter(m.source_type.value for m in materials)
    manifest = {
        "run": {
            "id": run.id,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "config_hash": run.config_hash or config_hash,
            "stage_stats": run.stage_stats,
            "llm_usage": run.llm_usage,
        },
        "config_hash": config_hash,
        "counts": {
            "tasks": len(tasks),
            "materials": len(materials),
            "by_task_type": dict(by_task_type),
            "by_verify_result": dict(by_verify),
            "by_source_type": dict(by_source),
        },
        "lineage": [
            {
                "task_id": t.id,
                "task_type": t.task_type,
                "nodes": t.sample_bundle.nodes,
                "ancestor_paths": t.sample_bundle.ancestor_paths,
                "material_ids": t.materials,
                "verifier": t.verifier,
                "verify_result": t.verify_result.value,
            }
            for t in tasks
        ],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
