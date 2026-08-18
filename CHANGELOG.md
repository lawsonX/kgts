# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `kgts.api` stable consumption surface + `docs/consuming.md` +
  `skills/kgts-graph-consumption/SKILL.md`: graph creation/consumption is now
  deliberately decoupled from dataset generation, so external synthesis
  agents can consume the DAG / bundles / materials directly.
- Graph cards (`graph_card.json` authoritative + `.md` rendered):
  auto-generated after every graph-mutating stage; idempotent when
  unchanged, revision + delta history on expansion/merges/material
  write-back, rename-aware; notes section computed from the graph's actual
  state. `kgts graph --card` regenerates on demand.
- `build.atomicity.max_depth` (default 6): the depth fuse is now
  configurable — with deepest-first expansion a sane floor keeps the budget
  from piling up as description-less leaves at the hard cap.
- `build.queue_policy` (bfs|dfs|balanced): the expansion frontier's pop
  order decides which queued nodes the budget explores. `balanced` is
  seed-fair depth-first (rotates seed subtrees, dives deepest within each) —
  plain global deepest-first degenerated in practice: one seed's subtree
  took 577/600 nodes and the hard cap piled up description-less leaves.
- `export.min_quality` (default 0 = off): generic quality gate — materials
  below the threshold leave the injected context, and tasks with no
  adequately-grounded material left leave the export entirely.
- Export context injection (`export.include_context`, default on): exported
  SFT rows embed the cited materials in the user message and RL rows gain a
  `context` field, so grounded questions are answerable as shipped.
- Generation prompts now carry explicit answerability rules (no references to
  context the solver cannot see); reject-sample wording no longer points at
  unshipped materials.
- **CorpusAdapterAgent** (`kgts/retrieve/ingest.py`): agentic input-data
  compatibility — samples any local corpus, infers a declarative
  `ExtractionSpec` via LLM, verifies it on real records with one
  self-correction round, caches it to `workdir/corpus_spec.json`, falls back
  to built-in heuristics. No LLM-generated code is ever executed.
  New CLI: `kgts ingest --config C [--save]`; new config:
  `retrieve.local.adapter: auto|heuristic`.
- `retrieve.verify_ssl` config (default `true`) for HTTP material sources —
  the documented escape hatch behind TLS-inspecting corporate proxies.
- `README.zh-CN.md` (Simplified Chinese) and a plainer English README.
- General-knowledge (Chinese) example: `configs/seeds/general_small.yaml`
  + `examples/corpus_zh/`.
- `build.max_children_per_node` config (default 6) to cap expansion fan-out.
- Material feedback loop (design 6.3): Stage C writes real per-node
  material counts back into `node.stats.n_materials`, re-judges atomicity,
  and re-persists `graph.db` — expansion stop signals now use real numbers.
- UI: interactive graph view (pyvis/vis-network, inline JS so it renders
  offline; color = level, thick border = atomic, dashed = is_related) and a
  Tasks browser tab with type/result filters and per-task provenance.
- LLM resilience: transient provider errors (429/timeout/5xx, matched by
  class name) retry with exponential backoff; new config
  `llm.max_retries` / `llm.retry_backoff`.
- `kgts graph --export dot|json` and `kgts review --queue all|align|parents`.
- Local corpus: `.jsonl` heuristic parsing (text/content/body fields, OCR
  dict dumps), chunk cosine norms precomputed at index time.

### Fixed (found by debugging against a real LLM endpoint, DeepSeek-V4-Flash)
- Expansion loop: atomicity is now judged **before** exploration, so an
  atomic verdict can no longer orphan already-committed children; nodes
  whose exploration yields no new candidates become `ATOMIC`.
- The explorer's `material_estimate` (an LLM self-report; observed values
  like 50000) is no longer written into `node.stats.n_materials` — the
  material-sufficiency signal only counts materials actually retrieved.
- Node budget is now also enforced inside the per-node commit batch.
- Retrieval scoring now uses a CJK-aware tokenizer (ASCII-only tokenization
  made Chinese corpora retrieve nothing, producing only reject samples).
- `run_pipeline(resume=False)` no longer synthesizes every bundle twice
  (stage re-entry through `stage_verify`).
- `sample.prioritizer` values other than `inverse_frequency` now raise a
  clear "not wired yet" error instead of being silently ignored.
- `budget.max_cost_usd` is now enforced: LiteLLMClient tracks per-call cost
  via litellm.completion_cost and ManagedLLM raises BudgetExceeded at the cap.
- Aligner embeddings now use the shared CJK-aware tokenizer: with the old
  ASCII-only tokenizer every Chinese candidate scored 0 similarity and
  short-circuited to DISTINCT, so near-duplicates survived without ever
  reaching the LLM judge (observed at 300-node scale).
- Node ids keep CJK characters readable (`slugify` is now unicode-aware).
- Web source: a non-JSON 200 response (proxy/filter block page) now raises
  a clear error instead of a bare JSONDecodeError.

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
