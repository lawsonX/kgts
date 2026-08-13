# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-13

Initial public release. Implements the v0.1–v0.3 scope of the design:
the full five-stage pipeline is end-to-end runnable, offline or against any
OpenAI-compatible endpoint.

### Added

- **Stage A (build)**: queue-driven recursive graph expansion —
  `ExplorerAgent` (multi-round sub-concept discovery), two-stage `Aligner`
  (alias/embedding recall + LLM ternary judge), `AtomicityJudge` (three
  signals + depth cap), DAG persistence with cycle and multi-parent guards
  (`NetworkX` + SQLite, `graph.db`).
- **Stage B (sample)**: breadth / depth / joint sampling operators,
  inverse-frequency long-tail weighting via the `Prioritizer` protocol,
  per-run coverage pass and per-node quotas, deterministic seeding.
- **Stage C (retrieve)**: `QueryBuilder` with ancestor-path disambiguation
  and per-source query shapes; material sources `local` (pure-python
  TF-IDF), `web` (Tavily), `github`, `arxiv`; post-processing (license
  whitelist, simhash/URL dedup, relevance rerank).
- **Stage D (synthesize)**: `TaskType` plugin registry with five built-in
  types — `atomic_qa`, `aggregated_qa`, `multihop_qa`, `grounded_summary`,
  `comparative_analysis` — with material-citation enforcement and explicit
  reject samples for under-supplied bundles.
- **Stage E (verify/eval)**: `Verifier` protocol, `answer_match` and
  `rubric_judge` verifiers, `sft_only` marking for tasks without a
  verifier; dataset-level report (coverage, duplication, diversity,
  quality, provenance completeness).
- **Export**: SFT (`messages`) and RL (`prompt + rubric + verifier`)
  JSONL plus `manifest.json` with full per-task lineage.
- **Orchestration**: per-stage checkpoints (`graph.db`, `bundles.json`,
  `materials.json`, `artifacts.db`), idempotent resume, `Run` audit records
  with config hash; LLM budget hard-cap, on-disk cache, and rate limiting
  (`ManagedLLM`).
- **LLM layer**: `LLMClient` protocol, `LiteLLMClient` (optional `llm`
  extra), deterministic offline `MockLLM` for tests and demos.
- **CLI**: `kgts build | sample | retrieve | synth | export | report | run
  | graph | review | serve` (typer).
- **UI**: Gradio workdir viewer (optional `ui` extra): graph, runs, review
  queue.
- **Offline quickstart** (`examples/quickstart_offline.py`) and example
  configs (`configs/default.yaml`, `configs/seeds/*.yaml`).
- Offline test suite (`pytest`, MockLLM only) and ruff CI.

[0.1.0]: https://github.com/kgts-project/kgts/releases/tag/v0.1.0
