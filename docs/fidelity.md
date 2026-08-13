# Fidelity to the K3 report

KGTS is an **independent open-source implementation** of mechanisms
described publicly in the Kimi K3 technical report (Moonshot AI). It is not
affiliated with Moonshot AI and contains no Moonshot code.

The K3 report describes the pipeline at a mechanism level but does not
disclose most parameters, prompts, or stopping criteria. Where KGTS fills
such a gap, the choice is **original engineering** and is marked as such
below. Confidence levels mean:

- **high** — the mechanism is stated in the report and KGTS implements it
  as described;
- **mechanism high / parameters original** — the mechanism is stated, but
  every threshold, prompt, and default is KGTS's own;
- **original** — the report does not disclose the mechanism; KGTS's
  version is a from-scratch engineering choice;
- **partial** — only part of the reported scope is implemented.

Summary table (details in the subsections below):

| K3 mechanism | KGTS implementation | Confidence |
|---|---|---|
| Predefined coarse seeds | `seeds` config (`SeedSpec`) | high |
| Agent multi-round exploration | `ExplorerAgent` | high |
| Equivalence/related check before commit | `Aligner` (two-stage) | mechanism high / parameters original |
| Coarse→fine edges, acyclic DAG | `GraphStore` invariants | high |
| "Atomic enough" stopping | `AtomicityJudge` (three signals) | original |
| Layered sampling, single or joint | `sample_bundles` (breadth/depth/joint) | high |
| Node + ancestor context in web queries | `QueryBuilder` | high |
| Fetch articles/blogs/repos and assemble | `retrieve/sources.py` + postprocess | high (adds local source) |
| Per-instance task-type selection | `TaskType` registry + `_pick_type` | mechanism high / mapping original |
| Knowledge/Coding/Vision task families | 5 QA-family built-ins | partial |
| Verifiable environments, independent verifiers | `Verifier` protocol + 2 verifiers | interface only |
| Long-tail / underrepresented coverage | `InverseFrequencyPrioritizer` | mechanism aligned |

## 1. Predefined coarse seeds — high

**K3**: expansion starts from predefined coarse-grained seeds.

**KGTS**: seeds are declared in the run config (`seeds:` list of
`SeedSpec`, see `kgts/config.py` and `configs/seeds/*.yaml`). Each seed
carries a human-written description and a human-given first layer
(`children`). Seeds and their first layer are inserted as provenance
`seed` / `human` refs in `kgts/build/expand.py`. KGTS deliberately does
not auto-generate the first layer: the seeds are the domain coordinate
system.

## 2. Agent multi-round exploration — high

**K3**: an agent explores a node over multiple retrieval/reading rounds to
discover sub-concepts.

**KGTS**: `ExplorerAgent` (`kgts/build/explorer.py`). Round 1 asks for a
definition, candidate sub-concepts, and a material-availability estimate;
rounds 2..`budget.max_explore_rounds` (default 4) ask for additional, more
specific candidates and stop early when a round adds nothing new.
Candidates are merged across rounds and deduped by normalized label.

*Original*: the exact prompt shapes, the round structure, and the
early-stop rule.

## 3. Equivalence/related check before commit — mechanism high / parameters original

**K3**: before a candidate enters the graph, existing equivalent or
related nodes are checked and reused.

**KGTS**: `Aligner` (`kgts/build/aligner.py`), two stages. Stage 1 (cheap):
exact normalized label/alias lookup plus a hashed bag-of-words embedding
(dim 256, md5 buckets, cosine) over `label + description + aliases`;
candidates below `build.align.embed_threshold` (default 0.75) with no
exact match are declared `DISTINCT` without spending an LLM call. Stage 2:
an LLM judge returns a ternary verdict `equivalent / related / distinct`
with a canonical-label suggestion. Every judgment is persisted as an
`AlignDecision` (with judge model and prompt hash) for human review and
prompt iteration; `kgts/cli.py`'s `kgts review` prints the review queue.

*Original*: the embedding scheme, `recall_top_k` (8), `embed_threshold`,
the judge prompt, and the canonicalization rule.

## 4. Coarse→fine edges, acyclic DAG — high

**K3**: edges point from coarser to finer concepts; the structure is a DAG.

**KGTS**: `GraphStore.add_edge` (`kgts/graph/store.py`) rejects any
`is_subconcept` edge that would create a cycle (`CycleError`). Levels are
recomputed as shortest-path depth from the nearest seed over
`is_subconcept` edges; `child.level > parent.level` is a *soft*
constraint — multi-parent shortcuts are legitimate, so violations are
recorded in `review_flags` instead of being rejected. `merged` nodes serve
only the alias index and are excluded from sampling. `is_related` edges
are excluded from level computation. Nodes may have multiple parents up to
`build.max_parents` (default 3); excess goes to the review queue.

## 5. "Atomic enough" stopping — original (not disclosed in the report)

**K3**: expansion stops when nodes are "sufficiently atomic"; the
criterion is not disclosed.

**KGTS**: `AtomicityJudge` (`kgts/build/atomicity.py`) combines three
signals, all read from `node.stats` so the audit trail stays on the node:

- material sufficiency: `n_materials >= build.atomicity.min_materials` (5);
- synthesizability: `synth_success_rate >= min_synth_success` (0.7), when
  measured;
- marginal benefit: `child_material_overlap < max_child_material_overlap`
  (0.5), when measured.

A node is atomic iff all available signals pass; `DEPTH_CAP = 6` is a hard
stop. All thresholds and the signal set itself are KGTS engineering
choices — tune them per domain.

## 6. Layered sampling, single or joint — high

**K3**: nodes are sampled at different hierarchy levels, individually or
as related groups.

**KGTS**: `sample_bundles` (`kgts/sample/sampler.py`) with three operators
mixed per `sample.mixture` (default `{breadth: 0.2, depth: 0.6,
joint: 0.2}`):

- **breadth** — nodes at or below the median level (overview tasks);
- **depth** — `atomic` nodes, falling back to DAG leaves (long tail);
- **joint** — sibling groups sharing a common `is_subconcept` parent,
  size 2..`sample.joint.max_group_size`.

Each `SampleBundle` carries the full ancestor path of every node (Stage C
needs it) and an intent label for auditing. Sampling is deterministic for a
fixed `run.seed`.

*Original*: the operator pool definitions; `sample.joint.k_hop` exists in
the config schema but k-hop subgraph sampling is not yet implemented
(sibling groups only).

## 7. Node + ancestor context in web queries — high

**K3**: retrieval queries are built from the node plus its ancestor
context.

**KGTS**: `QueryBuilder.build` (`kgts/retrieve/query.py`) combines the leaf
label with its 1–2 nearest ancestor labels (disambiguating e.g. "kernel"
under OS vs GPU vs ML) and shapes queries per source: web gets
keyword + tutorial/guide forms, GitHub gets repo-search syntax with
`topic:`, arXiv gets a quoted term query, local gets the full path string
for TF-IDF matching. JOINT bundles additionally get a `label AND label`
group query.

*Original*: the per-source query shapes and the "1–2 nearest ancestors"
window.

## 8. Fetch articles/blogs/repos and assemble — high (plus a local source)

**K3**: materials (articles, blogs, code repositories) are fetched and
assembled for the sampled nodes.

**KGTS**: the `MaterialSource` protocol (`kgts/retrieve/sources.py`) with
four built-ins: `local` (chunked .txt/.md/.jsonl + pure-python TF-IDF),
`web` (Tavily), `github` (repo search), `arxiv`. Post-processing
(`kgts/retrieve/postprocess.py`): license whitelist filter, cross-source
dedup (URL normalization or 64-bit simhash), and relevance rerank against
node descriptions. Every material records its `license` and
`linked_nodes`.

*Original*: the local-corpus source as a first-class peer of web (K3 is
web-centric), the postprocess pipeline, and per-node material pooling
(`retrieve.per_node_materials`).

## 9. Per-instance task-type selection — mechanism high / mapping original

**K3**: a task type is chosen per synthesized instance.

**KGTS**: the `TaskType` registry (`kgts/synthesize/base.py`) plus
`Synthesizer._pick_type` (`kgts/synthesize/synthesizer.py`): JOINT bundles
prefer `multihop_qa` when enabled; otherwise the type is drawn from
`synthesize.task_types` weighted by `synthesize.type_weights`
(uniform by default), seeded per bundle for reproducibility.

*Original*: the joint→multihop rule and the weighted-random mapping.

## 10. Knowledge/Coding/Vision task families — partial

**K3**: synthesized tasks span knowledge, coding, vision, and more.

**KGTS 0.1.0** ships the knowledge family: `atomic_qa`, `aggregated_qa`,
`multihop_qa`, `grounded_summary`, `comparative_analysis`
(`kgts/synthesize/builtin.py`). All enforce material-citation in answers;
under-supplied bundles produce explicit reject ("insufficient
information") samples or are dropped, per
`synthesize.insufficient_material`. Coding and data-analysis types are
planned for v0.4; vision is not scheduled. The plugin API
(`docs/plugins.md`) is the supported way to add more.

## 11. Verifiable environments, independent verifiers — interface only

**K3**: tasks come with verifiable environments and independent verifiers
(hardened against reward hacking).

**KGTS**: the `Verifier` protocol (`kgts/verify/base.py`), a
self-consistency `answer_match` verifier (checks grounding citations), and
an LLM `rubric_judge` fallback (`verify.fallback`). Tasks without a
verifier are marked `sft_only` and excluded from RL export
(`kgts/orchestrate/exporter.py`). Sandbox execution, hidden tests, and
anti-reward-hacking mechanisms are v0.4+ work; **the RL training loop
itself is out of scope** — KGTS emits the `prompt + rubric + verifier`
rows, training belongs to external frameworks.

## 12. Long-tail / underrepresented coverage — mechanism aligned

**K3**: sampling actively covers long-tail knowledge instead of letting
high-frequency knowledge dominate.

**KGTS**: `InverseFrequencyPrioritizer` (`kgts/sample/prioritizer.py`) —
weight `1 / (1 + alpha * times_sampled)`, with a coverage pass that gives
every eligible node one pick before repeats and `quotas.max_per_node`
capping repeats. The `Prioritizer` protocol is the hook for a GraphGen-style
ECE (expected-calibration blind-spot) plugin, planned for v0.4; `Node.stats.ece`
is reserved for it. Note: in 0.1.0 the sampler always uses
`InverseFrequencyPrioritizer`; the `sample.prioritizer` config value is
accepted but not yet wired to a plugin loader.
