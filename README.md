# KGTS — Knowledge-Graph-Guided Task Synthesis

[![CI](https://github.com/kgts-project/kgts/actions/workflows/ci.yml/badge.svg)](https://github.com/kgts-project/kgts/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python >=3.11](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/)

KGTS is an open-source implementation of the knowledge-graph-guided task
synthesis pipeline described in the Kimi K3 technical report (Moonshot AI):
an agent recursively expands a concept DAG from a handful of coarse seeds,
a sampler schedules node and node-group picks over that DAG, a retriever
anchors each pick to real materials using ancestor-path-disambiguated
queries, pluggable task types synthesize grounded training tasks, and
verifiers plus a dataset-level report close the loop. The core idea KGTS
productizes is **the knowledge graph as the control plane for data
synthesis** — the graph decides *what* to synthesize and *what* to retrieve;
how each task is generated is a plugin concern.

## Why KGTS

Existing open-source projects each cover one segment of the pipeline; none
cover all of it:

| Project | What it does | What is missing vs KGTS |
|---|---|---|
| GraphRAG / KGGen / RAKG / AutoSchemaKG | text → flat entity KG | flat graph only; no coarse→fine DAG, no synthesis scheduling |
| InstructLab | taxonomy-driven synthetic data generation | taxonomy is hand-maintained, not agent-expanded |
| GraphGen | KG-guided QA synthesis + blind-spot (ECE) weighting | no hierarchy, no web retrieval, no task-type decoupling |
| HippoRAG 2 / LightRAG | subgraph-anchored retrieval | built for QA, not for data scheduling |
| AgentInstruct | agentic synthesis flows | paper only, no open code |

KGTS is not another GraphRAG and not another doc2qa tool. Its differentiators:

- **Recursive agentic DAG construction** — explore → align → commit →
  atomicity-check, queue-driven until the budget runs out.
- **Schedulable sampling** — breadth / depth / joint operators with
  long-tail (inverse-frequency) weighting, over a hierarchical DAG.
- **Ancestor-context retrieval** — queries are disambiguated by the node's
  path from the root, per material source (web / local / GitHub / arXiv).
- **Task-type plugins** — synthesis is decoupled from the graph; every task
  type declares its material requirements, generator, and verifier.
- **Full provenance** — every exported example traces back
  `Task → SampleBundle → Node → Material → Run → config_hash`.

## Architecture

```text
                 Seeds Config (coarse seeds + domain declaration)
                                  │
   ┌──────────────────────────────▼───────────────────────────────┐
   │ Stage A  build/     Explorer → Aligner → Commit → Atomicity  │
   │                     queue-driven expansion until budget out  │
   └──────────────────────────────┬───────────────────────────────┘
                          Knowledge DAG (NetworkX + SQLite)
   ┌──────────────────────────────▼───────────────────────────────┐
   │ Stage B  sample/    breadth | depth | joint operators        │
   │                     inverse-frequency weighting → bundles    │
   └──────────────────────────────┬───────────────────────────────┘
   ┌──────────────────────────────▼───────────────────────────────┐
   │ Stage C  retrieve/  QueryBuilder (ancestor-path queries)     │
   │                     sources: local | web | github | arxiv    │
   │                     postprocess: license → dedup → rerank    │
   └──────────────────────────────┬───────────────────────────────┘
   ┌──────────────────────────────▼───────────────────────────────┐
   │ Stage D  synthesize/ TaskType plugin registry                │
   │                      atomic_qa | aggregated_qa | multihop_qa │
   │                      grounded_summary | comparative_analysis │
   └──────────────────────────────┬───────────────────────────────┘
   ┌──────────────────────────────▼───────────────────────────────┐
   │ Stage E  verify/ + eval/ + orchestrate/                      │
   │          verifiers → JSONL (SFT/RL) + manifest.json          │
   │          + coverage/duplication/diversity/provenance report  │
   └──────────────────────────────────────────────────────────────┘
   Cross-cutting: orchestrate/ (checkpoints, resume, budget)
                  llm.py (LiteLLM / MockLLM, budget + cache + rate limit)
                  ui/ (Gradio workdir viewer)
```

## Install

```bash
pip install kgts                # core (no LLM/UI deps)
pip install "kgts[llm]"         # + LiteLLM for real endpoints
pip install "kgts[ui]"          # + Gradio viewer
pip install "kgts[all]"         # everything incl. dev tools
```

From source:

```bash
git clone https://github.com/kgts-project/kgts.git
cd kgts
pip install -e ".[all]"
```

Requires Python 3.11+.

## Quickstart

### Offline demo (no API key)

`examples/quickstart_offline.py` runs the whole pipeline — seeds → graph →
bundles → materials → tasks → report — against `MockLLM` and a local toy
corpus, fully offline and deterministic:

```bash
python examples/quickstart_offline.py
```

### Real run

1. Define coarse seeds (the human-given coordinate system; the first layer
   under each seed is given, not auto-generated). Example seed configs:
   `configs/seeds/cs_small.yaml`, `configs/seeds/medical_small.yaml`,
   `configs/seeds/legal_small.yaml`.
2. Point `llm.model` in your config at any OpenAI-compatible endpoint
   (via LiteLLM), e.g.:

   ```yaml
   llm:
     model: gpt-4o-mini      # or any LiteLLM model string
     api_base: null
   budget:
     max_llm_calls: 2000
   retrieve:
     sources: [local, web]   # web needs TAVILY_API_KEY
   ```

   Start from `configs/default.yaml`, which documents every field.
3. Run the pipeline (per-stage checkpoints; re-running resumes):

   ```bash
   kgts run --config configs/seeds/cs_small.yaml
   kgts report --config configs/seeds/cs_small.yaml
   kgts serve --config configs/seeds/cs_small.yaml   # Gradio viewer
   ```

Exports land in `export.out_dir`: `tasks_sft.jsonl`, `tasks_rl.jsonl`,
`manifest.json`, `report.md` / `report.json`.

## CLI

All commands take `--config/-c PATH` (required).

| Command | Flags | What it does |
|---|---|---|
| `kgts build` | `--cheap-mode`, `--resume/--no-resume` | Stage A: expand the DAG → `graph.db` |
| `kgts sample` | `-n N`, `--resume/--no-resume` | Stage B: sample bundles → `bundles.json` |
| `kgts retrieve` | `--resume/--no-resume` | Stage C: bundle → materials → `materials.json` |
| `kgts synth` | `--types a,b`, `--resume/--no-resume` | Stage D+E: synthesize tasks, then verify |
| `kgts export` | `--format sft\|rl`, `--out DIR` | write JSONL per format + `manifest.json` |
| `kgts report` | `--run ID` | coverage/duplication/diversity/quality/provenance report |
| `kgts run` | `--resume/--no-resume` | full pipeline end to end, per-stage resume |
| `kgts graph` | `--stats` | inspect `graph.db` (node/edge counts, level histogram) |
| `kgts review` | — | print the human review queue (soft-constraint violations) |
| `kgts serve` | `--port 7860` | launch the Gradio workdir viewer |

Any single-stage command re-runs on top of existing checkpoints in the
workdir (`kgts sample` works once `kgts build` has produced `graph.db`, etc.).

## Repository layout

```text
kgts/
├── kgts/
│   ├── models.py       # Node/Edge/Material/Task/SampleBundle/AlignDecision/Run
│   ├── llm.py          # LLMClient protocol, LiteLLMClient, MockLLM, ManagedLLM
│   ├── config.py       # YAML -> typed pydantic settings
│   ├── cli.py          # typer CLI (see table above)
│   ├── graph/          # GraphStore: NetworkX DAG + SQLite, cycle/level invariants
│   ├── build/          # ExplorerAgent, Aligner, AtomicityJudge, expansion loop
│   ├── sample/         # breadth/depth/joint operators, Prioritizer interface
│   ├── retrieve/       # QueryBuilder, MaterialSource plugins, postprocess
│   ├── synthesize/     # TaskType registry + built-in task types
│   ├── verify/         # Verifier protocol, answer_match, rubric_judge
│   ├── eval/           # dataset-level report (coverage/dup/diversity/...)
│   ├── orchestrate/    # stage runner, checkpoints, exporter, artifact store
│   └── ui/             # Gradio workdir viewer (optional extra)
├── configs/            # default.yaml + example seeds (cs/medical/legal)
├── examples/           # quickstart_offline.py (MockLLM, no API key)
├── tests/              # offline test suite (pytest, MockLLM only)
└── docs/               # architecture, fidelity-to-K3, plugins, cost model
```

## Roadmap

| Version | Scope | Status |
|---|---|---|
| v0.1 | Stage A expansion loop, DAG storage, Gradio graph browsing | implemented in 0.1.0 |
| v0.2 | Stage B+C: three sampling operators, ancestor-path queries, web+local retrieval, lineage | implemented in 0.1.0 |
| v0.3 | Stage D QA task types + summary/compare, rubric verifier, SFT export, auto report | implemented in 0.1.0 |
| v0.4 | `coding_task` / `data_analysis` types, sandbox verifier, ECE long-tail plugin, RL-format hardening | planned |
| v0.5 | `agentic_workflow` tasks, downstream SFT ablation (KG-guided vs random sampling), Neo4j/Graphiti backend, human-review workflow | planned |

GitHub and arXiv sources are already available behind `retrieve.sources`;
the ECE prioritizer hooks into the existing `Prioritizer` protocol
(see `docs/plugins.md`).

## Relationship to the K3 report

KGTS is an **independent open-source implementation** of mechanisms
described publicly in the Kimi K3 technical report. Where the report does
not disclose details — the atomicity criterion, alignment thresholds, the
task-type selection mapping, all default parameters — KGTS uses its own
engineering choices, which are marked as such in
[docs/fidelity.md](docs/fidelity.md). KGTS is not affiliated with, endorsed
by, or derived from code by Moonshot AI.

RL training itself is out of scope: KGTS produces verifiable tasks and RL
export rows (`prompt + rubric + verifier hook`); the training loop belongs
to external frameworks (LLaMA-Factory, verl, ...).

## Corpus compliance

You are responsible for the licensing of the training data you synthesize
and the materials you retrieve. KGTS records a `license` field on every
material; web retrieval defaults to a whitelist filter
(`retrieve.postprocess.license_mode: whitelist`), but this is a heuristic,
not legal advice. The local corpus source (`retrieve.sources: [local]`) is
the compliance-first path: point it at material you have rights to.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The most welcome contributions are
new task-type plugins and material sources — see
[docs/plugins.md](docs/plugins.md).

## License

[Apache License 2.0](LICENSE)
