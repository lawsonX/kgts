"""Dataset-level evaluation report (design §8, Stage E).

Five sections per run: coverage, duplication, diversity, quality, provenance.
Pure functions over in-memory artifacts -- fully offline.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from itertools import combinations

from kgts.graph.store import GraphStore
from kgts.models import Material, NodeStatus, Run, Task, VerifyResult

_JACCARD_SAMPLE_CAP = 200  # bound the pairwise cost on large runs


def _ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    """Token n-gram set; falls back to unigrams for very short questions."""
    tokens = text.lower().split()
    if len(tokens) < n:
        return {(t,) for t in tokens}
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _coverage(store: GraphStore, tasks: list[Task]) -> dict:
    nodes = store.nodes()
    sampled = {nid for t in tasks for nid in t.sample_bundle.nodes}
    per_level: dict[str, dict[str, int]] = defaultdict(lambda: {"nodes": 0, "sampled": 0})
    for node in nodes:
        bucket = per_level[str(node.level)]
        bucket["nodes"] += 1
        if node.id in sampled:
            bucket["sampled"] += 1
    atomic = [n for n in nodes if n.status == NodeStatus.ATOMIC]
    never_sampled = sum(1 for n in atomic if n.id not in sampled)
    return {
        "n_nodes": len(nodes),
        "n_sampled_nodes": len(sampled & {n.id for n in nodes}),
        "per_level": dict(sorted(per_level.items(), key=lambda kv: int(kv[0]))),
        "long_tail_ratio": (never_sampled / len(atomic)) if atomic else 0.0,
    }


def _duplication(tasks: list[Task]) -> dict:
    sample = tasks
    if len(tasks) > _JACCARD_SAMPLE_CAP:
        sample = random.Random(42).sample(tasks, _JACCARD_SAMPLE_CAP)
    gram_sets = [_ngrams(t.question) for t in sample]
    scores = []
    for a, b in combinations(gram_sets, 2):
        union = a | b
        scores.append(len(a & b) / len(union) if union else 0.0)
    use_counts = Counter(mid for t in tasks for mid in set(t.materials))
    used = sum(1 for c in use_counts.values() if c >= 1)
    reused = sum(1 for c in use_counts.values() if c > 1)
    return {
        "pairwise_jaccard_mean": (sum(scores) / len(scores)) if scores else 0.0,
        "n_pairs": len(scores),
        "material_reuse_rate": (reused / used) if used else 0.0,
    }


def _diversity(tasks: list[Task]) -> dict:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t in tasks:
        matrix[t.task_type][str(t.sample_bundle.level)] += 1
    return {
        "matrix": {
            tt: dict(sorted(lv.items(), key=lambda kv: int(kv[0])))
            for tt, lv in matrix.items()
        },
        "total": len(tasks),
    }


def _quality(tasks: list[Task]) -> dict:
    def counts(ts: list[Task]) -> dict[str, int]:
        c = Counter(t.verify_result.value for t in ts)
        return {r.value: c.get(r.value, 0) for r in VerifyResult}

    by_type: dict[str, list[Task]] = defaultdict(list)
    for t in tasks:
        by_type[t.task_type].append(t)
    return {
        "overall": counts(tasks),
        "by_task_type": {tt: counts(ts) for tt, ts in sorted(by_type.items())},
    }


def _provenance(store: GraphStore, tasks: list[Task], materials: list[Material]) -> dict:
    material_ids = {m.id for m in materials}
    broken_ids = []
    for t in tasks:
        materials_ok = all(mid in material_ids for mid in t.materials)
        nodes_ok = all(nid in store for nid in t.sample_bundle.nodes)
        if not (materials_ok and nodes_ok):
            broken_ids.append(t.id)
    return {
        "completeness": (len(tasks) - len(broken_ids)) / len(tasks) if tasks else 1.0,
        "n_broken": len(broken_ids),
        "broken_ids": broken_ids,
    }


def generate_report(
    store: GraphStore,
    tasks: list[Task],
    materials: list[Material],
    run: Run | None = None,
) -> dict:
    """Compute the five-section evaluation report for one run."""
    return {
        "run_id": run.id if run else None,
        "coverage": _coverage(store, tasks),
        "duplication": _duplication(tasks),
        "diversity": _diversity(tasks),
        "quality": _quality(tasks),
        "provenance": _provenance(store, tasks, materials),
    }


def render_markdown(report: dict) -> str:
    """Render the report dict as compact, human-readable Markdown."""
    lines = ["# KGTS run report", ""]
    if report.get("run_id"):
        lines += [f"Run: `{report['run_id']}`", ""]

    cov = report["coverage"]
    lines += ["## Coverage", ""]
    lines += [
        f"- nodes: {cov['n_nodes']} (sampled: {cov['n_sampled_nodes']})",
        f"- long-tail ratio (unsampled atomic / atomic): {cov['long_tail_ratio']:.3f}",
        "",
        "| level | nodes | sampled |",
        "| --- | ---: | ---: |",
    ]
    for level, c in cov["per_level"].items():
        lines.append(f"| {level} | {c['nodes']} | {c['sampled']} |")

    dup = report["duplication"]
    lines += [
        "",
        "## Duplication",
        "",
        f"- pairwise question 3-gram Jaccard mean: {dup['pairwise_jaccard_mean']:.4f}"
        f" (over {dup['n_pairs']} pairs)",
        f"- material reuse rate (used by >1 task): {dup['material_reuse_rate']:.3f}",
    ]

    div = report["diversity"]
    levels = sorted({lv for m in div["matrix"].values() for lv in m}, key=int)
    lines += ["", "## Diversity (task_type x level)", ""]
    lines.append("| task_type | " + " | ".join(levels) + " | total |")
    lines.append("| --- | " + " | ".join("---:" for _ in levels) + " | ---: |")
    for tt, m in sorted(div["matrix"].items()):
        row = [str(m.get(lv, 0)) for lv in levels]
        lines.append(f"| {tt} | " + " | ".join(row) + f" | {sum(m.values())} |")

    qual = report["quality"]
    lines += ["", "## Quality (verify_result counts)", ""]
    lines.append("| task_type | pass | fail | sft_only | unverified |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    ov = qual["overall"]
    lines.append(
        f"| **overall** | {ov['pass']} | {ov['fail']} | {ov['sft_only']} | {ov['unverified']} |"
    )
    for tt, c in qual["by_task_type"].items():
        lines.append(f"| {tt} | {c['pass']} | {c['fail']} | {c['sft_only']} | {c['unverified']} |")

    prov = report["provenance"]
    lines += [
        "",
        "## Provenance",
        "",
        f"- completeness: {prov['completeness']:.3f}",
        f"- broken tasks: {prov['n_broken']}",
    ]
    if prov["broken_ids"]:
        lines.append(f"- broken ids: {', '.join(prov['broken_ids'])}")
    lines.append("")
    return "\n".join(lines)
