"""Offline end-to-end quickstart: run the full KGTS pipeline with a MockLLM.

Run from anywhere (paths are resolved relative to the repo root):

    python3 examples/quickstart_offline.py

What it does:
  1. loads configs/seeds/cs_small.yaml (3 seeds + human-given first layer),
  2. redirects the workdir and export dir into examples/output/ (gitignored),
  3. runs all stages with a deterministic offline MockLLM,
  4. prints graph/sample/material/task/verify/export statistics.

The mock explorer returns ``candidates: []``, so the knowledge graph stays at
seeds + their first layer -- exactly what a fast offline smoke demo wants.
"""

from __future__ import annotations

import re
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `kgts`

from kgts.config import load_config
from kgts.graph.store import GraphStore
from kgts.llm import MockLLM
from kgts.orchestrate.runner import graph_db_path, run_pipeline
from kgts.orchestrate.store import ArtifactStore

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "examples" / "output"
CONFIG_PATH = REPO_ROOT / "configs" / "seeds" / "cs_small.yaml"

# One JSON reply containing every key any pipeline stage parses:
#   explorer (build):      definition / candidates / material_estimate
#   aligner judge (build): verdict / matched_label / canonical
#   synthesizer:           question / answer / rubric
#   rubric judge (verify): scores / rationale
# candidates: [] keeps the graph at seeds + first layer; material_estimate >=
# build.atomicity.min_materials marks every node atomic.
DEFAULT_REPLY = {
    "definition": "A mock definition supplied by the offline demo LLM.",
    "candidates": [],
    "material_estimate": 8,
    "verdict": "distinct",
    "matched_label": None,
    "canonical": "",
    "question": "What do the retrieved materials say about this concept?",
    "answer": "The materials describe the concept, but cite no source.",
    "rubric": ["The answer is grounded in the cited materials."],
    "scores": [1.0],
    "rationale": "Mock judge: all rubric items satisfied.",
}

_MATERIAL_ID_RE = re.compile(r"\[(m_[0-9a-f]{12})\]")
_SYNTH_MARKER = "You are generating a training task"


class QuickstartLLM(MockLLM):
    """MockLLM whose synthesized answers cite the real material IDs.

    The answer_match verifier requires the answer to cite at least one of the
    task's material IDs, and those IDs only exist after the retrieve stage, so
    a static default reply cannot know them. Synthesis prompts list materials
    as ``[m_...] title: snippet``; this override parses the IDs out of the
    prompt and cites them, which is the honest offline equivalent of what a
    real LLM is instructed to do. Everything else falls through to
    ``default``.
    """

    def complete_json(self, prompt: str, *, temperature: float = 0.2, **kw) -> object:
        if _SYNTH_MARKER in prompt:
            ids = list(dict.fromkeys(_MATERIAL_ID_RE.findall(prompt)))
            cited = " and ".join(f"[{mid}]" for mid in ids[:2])
            reply = dict(DEFAULT_REPLY)
            reply["answer"] = (
                f"According to {cited}, the concept is grounded in the "
                "retrieved materials, which define it and describe how it works."
            )
            return reply
        return super().complete_json(prompt, temperature=temperature, **kw)


def main() -> None:
    config = load_config(CONFIG_PATH)
    # redirect runtime artifacts and exports into examples/output/
    config.run.workdir = str(OUT_DIR / "workdir")
    config.export.out_dir = str(OUT_DIR)
    # corpus paths are cwd-relative in the config; anchor them at the repo root
    config.retrieve.local.paths = [str(REPO_ROOT / p) for p in config.retrieve.local.paths]

    # start from a clean slate (only this script's own gitignored output dir),
    # then run with resume=True so chained stages reuse each other's
    # checkpoints instead of recomputing upstream stages
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    llm = QuickstartLLM(default=DEFAULT_REPLY)
    run = run_pipeline(config, resume=True, llm=llm)

    # ---- gather results from the checkpoints / artifact store --------------
    store = GraphStore.load(graph_db_path(config))
    artifacts = ArtifactStore(Path(config.run.workdir) / "artifacts.db")
    tasks = artifacts.load_tasks()
    materials = artifacts.load_materials()
    n_bundles = len({t.sample_bundle.id for t in tasks})
    by_verify = Counter(t.verify_result.value for t in tasks)
    by_type = Counter(t.task_type for t in tasks)
    export_counts = run.stage_stats.get("export_counts", {})
    out_dir = Path(config.export.out_dir)

    # ---- report -------------------------------------------------------------
    print("KGTS offline quickstart (mock LLM)")
    print(f"  config:            {CONFIG_PATH.relative_to(REPO_ROOT)}")
    print(f"  workdir:           {config.run.workdir}")
    print(f"  graph:             {len(store)} nodes, {len(store.edges())} edges")
    print(f"  sample bundles:    {n_bundles}")
    print(f"  materials:         {len(materials)}")
    print(f"  tasks:             {len(tasks)}  by type: {dict(by_type)}")
    print(f"  verify results:    {dict(by_verify)}")
    print(f"  export counts:     {export_counts}")
    print(f"  mock LLM calls:    {len(llm.calls)}")
    print(f"  report:            {out_dir / 'report.md'}")
    print(f"  manifest:          {out_dir / 'manifest.json'}")
    sft_path = out_dir / "tasks_sft.jsonl"
    if sft_path.exists():
        print(f"  sft export:        {sft_path}")


if __name__ == "__main__":
    main()
