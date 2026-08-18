# Architecture

This document walks the codebase module by module. The guiding split is
**control plane / data plane**: the knowledge DAG decides *what* to
synthesize and retrieve; task-type plugins decide *how* to synthesize.

## Pipeline stages and frozen contracts

Stages communicate through checkpoint files in the run workdir and through
a small set of cross-module function signatures. These signatures are
**frozen contracts** — they are imported across stage boundaries (and by
tests); do not change them without updating all call sites:

| Contract | Location | Signature |
|---|---|---|
| Graph expansion | `kgts/build/expand.py` | `expand_graph(seeds, llm, store, config, artifact_store=None) -> GraphStore` |
| Sampling | `kgts/sample/sampler.py` | `sample_bundles(store, config, seed=42) -> list[SampleBundle]` |
| Retrieval | `kgts/retrieve/retriever.py` | `Retriever.retrieve(store, bundle) -> list[Material]` |
| Synthesis | `kgts/synthesize/synthesizer.py` | `Synthesizer.synthesize(store, bundle, materials) -> Task \| None` |
| Verification | `kgts/verify/pipeline.py` | `verify_task(task, materials_by_id, llm, config) -> Task` |
| Source factory | `kgts/retrieve/sources.py` | `build_sources(config) -> dict[str, MaterialSource]` |

The orchestrator (`kgts/orchestrate/runner.py`) wires the stages via
`stage_build / stage_sample / stage_retrieve / stage_synthesize /
stage_verify / stage_export / stage_report` and `run_pipeline`. Sibling
stage modules are imported lazily inside each stage function, so the
runner module imports cleanly on its own. The CLI (`kgts/cli.py`) is a
thin typer wrapper over these stage functions.

## Module walkthrough

### `kgts/models.py` — data model

Every artifact is a pydantic model with provenance. Key types:

- `Node` — concept node: deterministic id (`make_node_id`: slug + sha1 of
  the canonical label), `aliases`, `level` (shortest-path depth from
  nearest seed), `status` (`expanding | atomic | merged | frozen`),
  `provenance` (list of `Ref`), `stats` (`NodeStats`: `n_materials`,
  `synth_success_rate`, `times_sampled`, `ece`, `child_material_overlap`).
- `Edge` — `parent → child`, `relation` (`is_subconcept` default,
  `is_related` lateral), `confidence`.
- `Material` — `source_type` (`web | local | github | arxiv`), `uri`/`path`,
  `snippet`, `license` (mandatory to record for web), `quality_score`,
  `linked_nodes`.
- `SampleBundle` — sampler output: `nodes`, `ancestor_paths` (node → label
  path; Stage C's disambiguation context), `level`, `intent`
  (`breadth | depth | joint`).
- `Task` — `task_type`, embedded `sample_bundle`, `materials` (ids),
  `question`/`answer`/`rubric`, `verifier`, `verify_result`
  (`pass | fail | sft_only | unverified`), `style`, `run_id`.
- `AlignDecision` — persisted alignment judgment (verdict, matched node,
  canonical suggestion, judge model, prompt hash).
- `Run` — audit unit: `config_hash`, `stage_stats`, `llm_usage`.

**Provenance invariant**: every task must trace
`Task → SampleBundle → Node → Material → Run → config_hash`. The exporter
writes this lineage into `manifest.json`; the report measures
`provenance.completeness`.

### `kgts/config.py` — configuration

`load_config(path)` parses YAML into typed pydantic settings (`Config`).
Every field has a default (see `configs/default.yaml` for the annotated
full schema): `run` (name/workdir/seed), `seeds`, `llm`
(model/cheap_model/api_base/rpm/cache), `budget` (max_llm_calls,
max_cost_usd, max_nodes, max_explore_rounds), `build` (align/atomicity/
max_parents), `sample` (mixture/n_samples/quotas/joint/prioritizer),
`retrieve` (sources/per_node_materials/web/local/postprocess),
`synthesize` (task_types/type_weights/style/insufficient_material),
`verify` (fallback/rubric_judge), `export` (formats/out_dir).
`config_hash()` fingerprints a run's configuration for reproduction.

### `kgts/llm.py` — LLM abstraction

- `LLMClient` protocol: `complete(prompt, ...) -> str` and
  `complete_json(prompt, ...) -> Any`.
- `LiteLLMClient` — any OpenAI-compatible endpoint via litellm (optional
  `llm` extra).
- `MockLLM` — deterministic offline client (prompt-substring → scripted
  reply); used by all tests and the offline quickstart. Selected when
  `llm.model` starts with `"mock"`.
- `ManagedLLM` — the wrapper stages actually receive: hard call budget
  (`BudgetExceeded` when `budget.max_llm_calls` is hit), on-disk cache
  (`workdir/llm_cache/`, keyed by model+prompt+temperature), and rpm
  throttling.

### `kgts/graph/store.py` — GraphStore

NetworkX `DiGraph` in memory, SQLite persistence (`graph.db`). Write-side
invariants:

- `add_edge` rejects `is_subconcept` cycles (`CycleError`);
- levels recomputed per edge insert; `child.level <= parent.level` is a
  soft violation → `review_flags`;
- `merge()` folds duplicates into a canonical node (alias index +
  child re-pointing); `merged` nodes are excluded from `nodes()`;
- `find()` = exact lookup by canonical label or alias; `ancestor_path()` =
  shortest root→node label path over the `is_subconcept` view.

### `kgts/build/` — Stage A

- `explorer.py` `ExplorerAgent.explore(node, ancestor_path) -> NodeBrief`:
  round 1 = definition + candidates + material estimate; later rounds ask
  for missing specifics; early stop when a round adds nothing; candidates
  deduped by normalized label.
- `aligner.py` `Aligner.align(candidate_label, parent) -> AlignDecision`:
  stage 1 = exact alias lookup + hashed bag-of-words embedding recall
  (dim 256, cosine, `recall_top_k`); below `embed_threshold` with no exact
  match → `DISTINCT` without an LLM call; stage 2 = LLM ternary judge.
- `atomicity.py` `AtomicityJudge.is_atomic(node)`: three signals from
  `node.stats` (materials, synth success, child overlap) + the configurable depth fuse `atomicity.max_depth`.
- `expand.py` `expand_graph(...)`: queue-driven loop (`_Frontier` with
  `build.queue_policy` = bfs|dfs|balanced — the pop order decides which
  queued nodes the budget actually explores) —
  `atomicity gate → explore → align each candidate → commit`; seeds and
  their human-given first layer initialize the queue. Committed children
  are always enqueued (an atomic verdict never orphans them); nodes whose
  exploration yields no new children become `ATOMIC`; fan-out is capped by
  `build.max_children_per_node` (default 6) and the node budget is also
  checked inside the commit batch. The explorer's `material_estimate` is an
  untrusted LLM self-report and is never written into `node.stats` — the
  material signal only counts materials actually retrieved by Stage C.
  `BudgetExceeded` stops the loop gracefully; any other per-node error is
  recorded in `store.review_flags` and the loop continues. Commit applies
  verdicts: `EQUIVALENT` folds the label into the matched node's aliases;
  `RELATED` adds an `is_related` edge; `DISTINCT` creates the node and
  attaches it under `max_parents` + cycle guards.

### `kgts/sample/` — Stage B

- `sampler.py` `sample_bundles(store, config, seed)`: builds three pools —
  breadth (level ≤ median), depth (atomic nodes, fallback: leaves), joint
  (sibling groups of size 2..`max_group_size`) — splits `n_samples` per
  `mixture`, then draws with a coverage pass (one pick per eligible node
  first) followed by weighted fill. Deterministic for a fixed seed;
  `times_sampled` is updated on the graph for audit.
- `prioritizer.py` — `Prioritizer` protocol (`weight(node) -> float`) and
  `InverseFrequencyPrioritizer` (`1 / (1 + alpha * times_sampled)`). ECE
  plug-in point; see `docs/plugins.md`.

### `kgts/retrieve/` — Stage C

- `query.py` `QueryBuilder.build(label, ancestor_path, intent,
  source_type, group_labels)`: leaf + 1–2 nearest ancestors, per-source
  query shapes (web/github/arxiv/local), plus a combined `AND` query for
  JOINT bundles.
- `sources.py` — `MaterialSource` protocol and four built-ins
  (`LocalCorpusSource` pure-python TF-IDF over chunked .txt/.md/.jsonl;
  `WebSearchSource` Tavily via urllib; `GitHubSource`; `ArxivSource`), all
  stdlib-only. HTTP sources share `_make_ssl_context(verify_ssl)` (config `retrieve.verify_ssl`) for TLS-inspecting proxies. `build_sources()` instantiates from `retrieve.sources`.
- `postprocess.py` — license whitelist, dedup (URL normalization or
  simhash), token-overlap rerank against node descriptions.
- `ingest.py` — CorpusAdapterAgent (agentic input compatibility): samples
  the corpus (one file per extension, binaries skipped), has the LLM infer a
  declarative `ExtractionSpec`, verifies it on real records with one
  self-correction round, and caches it to `workdir/corpus_spec.json`;
  `LocalCorpusSource(spec=...)` then accepts any file format. The spec
  interpreter is container-tolerant on `text_fields` (dict/list values are
  joined) because LLMs routinely confuse field kinds.
- `text.py` — shared tokenizer for all retrieval scoring: ASCII words plus
  CJK character unigrams/bigrams. (An earlier ASCII-only tokenizer made
  Chinese corpora retrieve nothing; keep every scorer on this one tokenizer.)
- `retriever.py` `Retriever.retrieve(store, bundle)`: per node × per
  source query fan-out; if every source raised, the first error is
  re-raised (no silent empty results); results are license-filtered,
  deduped, reranked.

### `kgts/synthesize/` — Stage D

- `base.py` — `TaskType` ABC (`requires: MaterialSpec`, `verifier_name`,
  `generate(llm, bundle, materials, style) -> Task`), the
  `@register_task_type` decorator, and `get_task_type` / `list_task_types`.
- `builtin.py` — five built-ins (`atomic_qa`, `aggregated_qa`,
  `multihop_qa`, `grounded_summary`, `comparative_analysis`) sharing one
  flow: prompt with node labels + ancestor paths + style + materials
  listed as `[id] title: snippet`; the reply JSON must cite material IDs.
- `synthesizer.py` `Synthesizer.synthesize(store, bundle, materials)`:
  picks a type (JOINT → `multihop_qa` if enabled, else weighted random,
  seeded per bundle), enforces `requires.min_docs` (under-supplied bundles
  → explicit reject sample or drop per `insufficient_material`), marks
  verifier-less tasks `SFT_ONLY`.

### `kgts/verify/` — Stage E (task level)

- `base.py` — `Verifier` protocol: `verify(task, materials_by_id) ->
  (VerifyResult, score, note)`.
- `answer_match.py` — self-consistency check: non-empty answer citing at
  least one of the task's material IDs.
- `rubric.py` — LLM judge against the task's rubric
  (`verify.rubric_judge.pass_score`).
- `pipeline.py` `verify_task(...)`: `SFT_ONLY`/verifier-less tasks pass
  through; unknown verifier names fall back to `verify.fallback`; if that
  is also unknown the task is downgraded to `SFT_ONLY`.

### `kgts/eval/report.py` — Stage E (dataset level)

Pure offline functions producing five sections: **coverage** (per-level
sampled-node counts, long-tail ratio over atomic nodes), **duplication**
(pairwise question n-gram Jaccard, capped sampling; material reuse),
**diversity** (task type × level), **quality** (verifier pass rates),
**provenance** (completeness of the lineage chain). Rendered to
`report.json` + `report.md`.

### `kgts/orchestrate/` — wiring, checkpoints, export

- `runner.py` — stage functions + `run_pipeline`; `CheckpointError` when an
  upstream checkpoint is missing (the CLI turns this into a clean exit
  with a hint).
- `store.py` `ArtifactStore` — SQLite (`artifacts.db`) for materials,
  tasks, align verdicts, runs; idempotent `INSERT OR REPLACE`.
- `exporter.py` — `tasks_sft.jsonl` (messages format; PASS + SFT_ONLY
  rows), `tasks_rl.jsonl` (prompt + rubric + verifier; PASS with verifier
  only), `manifest.json` (run info, aggregate counts, per-task lineage).

### `kgts/ui/app.py` — Gradio viewer

Optional `ui` extra, imported lazily. Read-only workdir viewer: nodes,
edges, runs, review queue (`kgts serve --config ...`).

## Checkpoint and resume design

Each stage reads the previous stage's checkpoint and writes its own, so
any stage is independently re-runnable and interrupted runs resume where
they stopped (`--resume` is the default; `--no-resume` rebuilds).

| File (under `run.workdir`) | Written by | Contents |
|---|---|---|
| `graph.db` | Stage A (`stage_build`) | GraphStore SQLite: nodes + edges |
| `bundles.json` | Stage B (`stage_sample`) | list of `SampleBundle` |
| `materials.json` | Stage C (`stage_retrieve`) | list of `Material` |
| `artifacts.db` | Stages C/D/E | materials, tasks, align verdicts, runs tables |
| `llm_cache/` | `ManagedLLM` | one JSON file per cached completion |

Resume semantics: Stage A reloads `graph.db`; Stage B reloads
`bundles.json`; Stage C reloads `materials.json`; Stage D skips bundles
that already produced a task (matched by bundle id); Stage E verifies only
tasks still `UNVERIFIED`. Exports and reports (`export.out_dir`:
`tasks_*.jsonl`, `manifest.json`, `report.json`, `report.md`) are
rewritten wholesale from `artifacts.db`.
