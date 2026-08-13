# AGENTS.md — orientation for AI coding agents

## What this is

KGTS (Knowledge-Graph-Guided Task Synthesis) is an open-source
implementation of the K3-style pipeline: an agent expands a concept DAG
from seeds, a sampler schedules picks over the DAG, a retriever anchors
picks to materials via ancestor-path queries, task-type plugins synthesize
grounded tasks, and verifiers plus a dataset report close the loop. The
graph is the control plane; synthesis is plugins. Apache-2.0, Python 3.11+.

## Layout

- `kgts/models.py` — all pydantic data models (Node/Edge/Material/Task/
  SampleBundle/AlignDecision/Run). Single source of truth for the schema.
- `kgts/llm.py` — `LLMClient` protocol, `LiteLLMClient`, `MockLLM`,
  `ManagedLLM` (budget/cache/rate limit).
- `kgts/config.py` — YAML → typed config; every field has a default.
- `kgts/cli.py` — typer CLI; thin wrapper over `orchestrate/runner.py`.
- `kgts/graph/store.py` — `GraphStore`: NetworkX DAG + SQLite, alias index,
  cycle/level invariants, review flags.
- `kgts/build/` — Stage A: `expand.py` (loop), `explorer.py`, `aligner.py`,
  `atomicity.py`.
- `kgts/sample/` — Stage B: `sampler.py` (breadth/depth/joint),
  `prioritizer.py`.
- `kgts/retrieve/` — Stage C: `query.py`, `sources.py`, `postprocess.py`,
  `retriever.py`.
- `kgts/synthesize/` — Stage D: `base.py` (TaskType + registry),
  `builtin.py`, `synthesizer.py`.
- `kgts/verify/` — Stage E: `base.py`, `answer_match.py`, `rubric.py`,
  `pipeline.py`.
- `kgts/eval/report.py` — dataset-level report (offline, pure functions).
- `kgts/orchestrate/` — `runner.py` (stage wiring + checkpoints),
  `store.py` (ArtifactStore), `exporter.py` (SFT/RL JSONL + manifest).
- `kgts/ui/app.py` — Gradio viewer (optional extra, lazily imported).
- `configs/`, `examples/`, `tests/`, `docs/` — see README.

## Commands

```bash
pip install -e ".[dev]"
pytest            # offline suite, must stay green
ruff check .      # line-length 100, rules E/F/I/UP/B
```

## Invariants (do not break)

1. **DAG acyclicity** — the `is_subconcept` subgraph must stay acyclic;
   enforcement lives in `GraphStore.add_edge` (raises `CycleError`).
2. **Provenance chain** — every task must trace
   `Task → SampleBundle → Node → Material → Run` (`run.config_hash`
   included). The exporter's `manifest.json` and the report's
   `provenance.completeness` metric assume this.
3. **Offline tests** — tests never touch the network or API keys; use
   `MockLLM`. No test may require the `llm` or `ui` extras.
4. **No new heavy dependencies** — core deps are networkx, pydantic,
   pyyaml, typer. Anything else needs prior discussion (open an issue).

## Frozen cross-module contracts

These signatures are imported across stage boundaries (orchestrator ↔
stages, tests ↔ stages). Do not change them casually; a change requires
updating all call sites and a note in the PR:

- `kgts.build.expand.expand_graph(seeds, llm, store, config, artifact_store=None)`
- `kgts.sample.sampler.sample_bundles(store, config, seed=42)`
- `kgts.retrieve.retriever.Retriever.retrieve(store, bundle)`
- `kgts.synthesize.synthesizer.Synthesizer.synthesize(store, bundle, materials)`
- `kgts.verify.pipeline.verify_task(task, materials_by_id, llm, config)`

Also stable-by-convention: `build_sources(config)` in `retrieve/sources.py`,
the `TaskType`/`MaterialSource`/`Verifier`/`Prioritizer` protocols, and the
checkpoint file names in the workdir (`graph.db`, `bundles.json`,
`materials.json`, `artifacts.db`, `llm_cache/`).

## Conventions

- English everywhere; pydantic models for anything persisted; stdlib-first.
- Errors: network sources raise `RuntimeError` on failure; one bad node
  must never kill the expansion loop (record to `store.review_flags`).
- Docs live in `docs/`; keep `docs/fidelity.md` honest when changing
  mechanism behavior.
