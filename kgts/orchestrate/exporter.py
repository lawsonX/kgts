"""Stage E export: SFT/RL JSONL formats plus a full-lineage manifest (design §8)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from kgts.models import Material, Run, Task, VerifyResult


def export_sft(tasks: list[Task]) -> list[dict]:
    """SFT messages format; keeps tasks that passed or are SFT-only (no verifier)."""
    rows = []
    for t in tasks:
        if t.verify_result not in (VerifyResult.PASS, VerifyResult.SFT_ONLY):
            continue
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": t.question},
                    {"role": "assistant", "content": t.answer},
                ],
                "task_id": t.id,
                "task_type": t.task_type,
            }
        )
    return rows


def export_rl(tasks: list[Task]) -> list[dict]:
    """RL format: prompt + rubric + verifier hook; requires PASS and a verifier."""
    rows = []
    for t in tasks:
        if t.verify_result != VerifyResult.PASS or t.verifier is None:
            continue
        rows.append(
            {
                "task_id": t.id,
                "prompt": t.question,
                "rubric": t.rubric,
                "verifier": t.verifier,
                "env": None,
            }
        )
    return rows


def write_export(tasks: list[Task], fmt: str, out_path: str | Path) -> int:
    """Write one JSON object per line; returns the number of rows written."""
    fmt = fmt.lower()
    if fmt == "sft":
        rows = export_sft(tasks)
    elif fmt == "rl":
        rows = export_rl(tasks)
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
