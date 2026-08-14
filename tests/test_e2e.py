"""End-to-end offline smoke test: full pipeline over a tiny local corpus.

Runs build -> sample -> retrieve -> synthesize -> verify -> export -> report
with a deterministic MockLLM and a tmp_path workdir; no network, no API keys.

The mock explorer returns ``candidates: []``, so the graph stays at exactly
seeds + their human-given first layer, which makes the expected node count
deterministic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from kgts.config import (
    BudgetConfig,
    Config,
    ExportConfig,
    LLMConfig,
    LocalSourceConfig,
    RetrieveConfig,
    RunConfig,
    SampleConfig,
    SeedSpec,
    SynthesizeConfig,
)
from kgts.graph.store import GraphStore
from kgts.llm import MockLLM
from kgts.orchestrate.runner import run_pipeline
from kgts.orchestrate.store import ArtifactStore

# One JSON reply covering every key any stage parses (see stage docstrings):
# explorer: definition/candidates/material_estimate; aligner: verdict/canonical;
# synthesizer: question/answer/rubric; rubric judge: scores/rationale.
DEFAULT_REPLY = {
    "definition": "A mock definition from the offline test LLM.",
    "candidates": [],
    "material_estimate": 8,
    "verdict": "distinct",
    "matched_label": None,
    "canonical": "",
    "question": "What do the materials say about this concept?",
    "answer": "An answer without citations (overridden for synthesis prompts).",
    "rubric": ["The answer is grounded in the cited materials."],
    "scores": [1.0],
    "rationale": "Mock judge reply.",
}

_MATERIAL_ID_RE = re.compile(r"\[(m_[0-9a-f]{12})\]")

CORPUS = {
    "sorting.txt": (
        "Sorting algorithms rearrange records into a specified order. Quicksort "
        "partitions the array around a pivot and recurses on each side, giving "
        "average quadratic-to-linearithmic behavior depending on pivot choice; "
        "merge sort splits the array in half, sorts both halves, and merges them "
        "in guaranteed linearithmic time. Comparison-based sorting cannot beat "
        "n log n comparisons in the worst case, while counting sort exploits "
        "integer keys to run in linear time."
    ),
    "routing.txt": (
        "Routing in computer networks selects paths for packets across links. "
        "Distance-vector protocols such as RIP let routers exchange tables of "
        "distances to destinations, while link-state protocols such as OSPF "
        "flood topology information so every router can compute shortest paths "
        "with Dijkstra's algorithm. Between autonomous systems, BGP exchanges "
        "reachability information driven by policy rather than pure distance."
    ),
    "parsing.txt": (
        "Parsing in compilers turns a flat token stream into a syntax tree. "
        "A lexer first groups characters into tokens, and the parser checks the "
        "token stream against a context-free grammar. Top-down parsers such as "
        "recursive descent predict productions from the leftmost derivation, "
        "while bottom-up LR parsers shift tokens onto a stack and reduce handles. "
        "Ambiguous grammars must be resolved before a deterministic parser works."
    ),
}

SEEDS = [
    SeedSpec(label="Algorithms", children=["Sorting", "Graph Traversal"]),
    SeedSpec(label="Computer Networks", children=["Routing", "Congestion Control"]),
    SeedSpec(label="Compilers", children=["Parsing", "Code Generation"]),
]
N_NODES = 3 + 6  # seeds + first-layer children (mock proposes no candidates)
N_SAMPLES = 6


class E2EMockLLM(MockLLM):
    """MockLLM whose synthesized answers cite the material IDs from the prompt,
    so answer_match can PASS offline (IDs only exist after retrieval)."""

    def complete_json(self, prompt: str, *, temperature: float = 0.2, **kw) -> object:
        if "You are generating a training task" in prompt:
            ids = list(dict.fromkeys(_MATERIAL_ID_RE.findall(prompt)))
            cited = " and ".join(f"[{mid}]" for mid in ids[:2])
            reply = dict(DEFAULT_REPLY)
            reply["answer"] = f"Grounded mock answer citing {cited}."
            return reply
        return super().complete_json(prompt, temperature=temperature, **kw)


def _make_config(tmp_path: Path) -> Config:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in CORPUS.items():
        (corpus / name).write_text(text, encoding="utf-8")
    return Config(
        run=RunConfig(name="e2e", workdir=str(tmp_path / "workdir"), seed=7),
        seeds=SEEDS,
        llm=LLMConfig(model="mock", cache=False),
        budget=BudgetConfig(max_llm_calls=500, max_nodes=100),
        sample=SampleConfig(n_samples=N_SAMPLES),
        retrieve=RetrieveConfig(
            sources=["local"], local=LocalSourceConfig(paths=[str(corpus)])
        ),
        synthesize=SynthesizeConfig(
            style={"language": "en", "length": "short", "difficulty": "mixed"}
        ),
        export=ExportConfig(formats=["sft", "rl"], out_dir=str(tmp_path / "output")),
    )


def test_full_pipeline_offline(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    llm = E2EMockLLM(default=DEFAULT_REPLY)
    # resume=True on a fresh tmp workdir: stages compute once, then chained
    # upstream stage calls reuse the checkpoints just written
    run_pipeline(config, resume=True, llm=llm)

    workdir = Path(config.run.workdir)
    out_dir = Path(config.export.out_dir)

    # Stage A: graph checkpoint with exactly seeds + first-layer children
    graph_db = workdir / "graph.db"
    assert graph_db.exists()
    store = GraphStore.load(graph_db)
    assert len(store) == N_NODES

    # Stage B: exactly n_samples bundles
    bundles = json.loads((workdir / "bundles.json").read_text())
    assert len(bundles) == N_SAMPLES

    # Stage C: materials retrieved from the local corpus
    materials = json.loads((workdir / "materials.json").read_text())
    assert materials, "expected non-empty materials from the local corpus"

    # Stage D: one task per bundle (synthesized or explicit reject-sample)
    artifacts = ArtifactStore(workdir / "artifacts.db")
    tasks = artifacts.load_tasks()
    assert len(tasks) == len(bundles)

    # Provenance: every material a task cites exists in the artifact store
    store_ids = {m.id for m in artifacts.load_materials()}
    assert store_ids
    for task in tasks:
        missing = set(task.materials) - store_ids
        assert not missing, f"task {task.id} cites unknown materials: {missing}"

    # Stage E exports + report
    assert (out_dir / "tasks_sft.jsonl").exists()
    assert (out_dir / "tasks_rl.jsonl").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["counts"]["tasks"] == len(tasks)
    assert manifest["counts"]["materials"] == len(store_ids)
    assert (out_dir / "report.md").exists()
    assert (out_dir / "report.json").exists()


def test_material_stats_write_back(tmp_path: Path) -> None:
    """Stage C feedback: real material counts land in node.stats, and
    material-sufficient nodes are re-judged ATOMIC (design 6.3 loop)."""

    config = _make_config(tmp_path)
    run_pipeline(config, resume=True, llm=E2EMockLLM(default=DEFAULT_REPLY))
    store = GraphStore.load(Path(config.run.workdir) / "graph.db")
    counts = {n.label: n.stats.n_materials for n in store.nodes()}
    assert any(c > 0 for c in counts.values()), "no material counts written back"
    # (the atomic re-judgment itself is covered precisely in test_orchestrate)
