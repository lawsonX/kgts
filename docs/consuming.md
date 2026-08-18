# Consuming KGTS from your own synthesis agent

KGTS separates **graph creation & consumption** from **dataset generation**.
Production-grade task authoring — clarifying requirements with the user, trial
production, diversity control — belongs to your own agent with your own tools.
KGTS gives that agent three things: a knowledge DAG, sampling bundles, and
anchored materials. This page is the contract.

## What you get, and where
Every DAG ships with a **graph card** (`graph_card.json` authoritative +
`graph_card.md` rendered) regenerated after each graph-mutating stage:
idempotent when unchanged, revision-bumped with delta history on expansion
or material write-back, rename-aware. Read it first — its notes section
reflects the graph's actual state.



After a run, the workdir (config `run.workdir`) contains:

| File | Content | Format |
|---|---|---|
| `graph.db` | the knowledge DAG: nodes (label/aliases/level/status/stats), edges (coarse→fine) | SQLite |
| `bundles.json` | sampled `SampleBundle`s: node ids + ancestor paths + intent | JSON |
| `materials.json` | retrieved materials with `linked_nodes`, `text`, `license`, `quality_score` | JSON |
| `graph_card.json` / `.md` | auto-generated graph card: name, revision, stats, build history, usage notes | JSON + MD |
| `corpus_spec.json` | the CorpusAdapterAgent's extraction spec for the local corpus | JSON |
| `artifacts.db` | tasks / align verdicts / runs (if KGTS synthesis also ran) | SQLite |
| `<out_dir>/graph.json` / `graph.dot` | `kgts graph --export` output | JSON / DOT |

The JSON files are the language-agnostic interface: any agent, any language.

## Level 1 — files only

```python
import json
bundles = json.loads(open(".kgts/bundles.json").read())
for b in bundles:
    print(b["intent"], b["nodes"], b["ancestor_paths"])
```

`ancestor_paths` maps each node id to its root→node label path — use it to
disambiguate terms in your generation prompts (that is what KGTS does
internally for retrieval queries).

## Level 2 — CLI

```bash
kgts graph --config C --export json     # dump the DAG as JSON
kgts graph --config C --stats           # quick histogram
kgts sample --config C -n 500           # (re)sample bundles into bundles.json
kgts retrieve --config C                # refresh materials.json for the bundles
```

## Level 3 — Python API

```python
from kgts.api import load_graph, load_sample_bundles, load_materials

store = load_graph(".kgts")
for node in store.atomic_nodes():                 # terminal, fine-grained nodes
    print(node.label, node.level, node.stats.n_materials)

bundles = load_sample_bundles(".kgts")            # what to synthesize about
materials = load_materials(".kgts")               # grounded text with provenance
by_node = {}
for m in materials:
    for nid in m.linked_nodes:
        by_node.setdefault(nid, []).append(m)

for b in bundles:
    mats = [m for nid in b.nodes for m in by_node.get(nid, [])]
    # ... hand b.ancestor_paths + mats to YOUR generator ...
```

## Contract notes for generator authors

- **Ground rules for questions**: a question must be answerable from what you
  ship with it. If you reference materials, ship them (KGTS's own exporter
  injects a truncated context block; see `export.include_context`). Never
  produce "根据材料…" questions with no materials attached — that trains
  hallucination.
- **Quality gate**: KGTS's own exporter supports `export.min_quality` —
  materials below it leave the context, tasks left ungrounded are dropped.
  Apply the same idea in your generator (OCR junk pages score low).
- **Provenance**: keep the chain `task → bundle.id → node ids → material ids`.
  If you write your own export, preserve these ids so the data stays auditable.
- **Sampling is yours to redo**: `bundles.json` is a suggestion. Re-run
  `kgts sample` with a different mixture, or ignore bundles entirely and walk
  the DAG yourself — the graph is the product, the sampler is a convenience.
- **Node status**: `atomic` = fine-grained terminal; `expanding` = can grow in
  a future build pass; `merged` = alias-index only, do not sample.
- A ready-made agent skill lives in `skills/kgts-graph-consumption/SKILL.md`.
