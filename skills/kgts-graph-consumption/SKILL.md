# Skill: Consuming a KGTS knowledge graph for data synthesis

Use this when the user has (or wants) a KGTS-produced knowledge graph and asks
you to synthesize training data from it. KGTS builds and serves the graph;
YOU are the generator — requirement clarification, trial production, and
diversity control are your job.

## What KGTS gives you

A run workdir (ask the user for its path, or the config that produced it):

- `graph.db` — the knowledge DAG (SQLite). Read it via `kgts.api.load_graph`.
- `bundles.json` — sampled bundles: `nodes`, `ancestor_paths`, `intent`, `level`.
- `materials.json` — retrieved texts with `linked_nodes`, `license`, `quality_score`.
- CLI alternative: `kgts graph --config C --export json`, `kgts sample`, `kgts retrieve`.

## Workflow

1. **Clarify requirements with the user first**: domain focus (which subtrees?),
   task types, difficulty/length/language mix, volume, target format (SFT/RL),
   whether materials must be shipped with each task. Do NOT skip this to start
   generating — trial a small batch and confirm before scaling.
2. **Inspect the graph**:
   ```python
   from kgts.api import load_graph
   store = load_graph(WORKDIR)
   [(n.label, n.level, n.status.value) for n in store.nodes()]
   ```
   Prefer `atomic` nodes for specific tasks, shallow nodes for overviews.
3. **Pick what to synthesize about**: read `bundles.json`, or sample yourself
   (`kgts sample -n N`), or walk the DAG directly. Use `ancestor_paths` in your
   prompts to disambiguate terms.
4. **Gather materials**: from `materials.json` via `linked_nodes`; or re-run
   `kgts retrieve` for fresh bundles. Respect `license` — default policy is
   whitelist-only for web materials.
5. **Generate, then self-check**:
   - Every question must be answerable from what you ship with it. If it says
     "根据材料…", the materials MUST accompany the task (context block).
   - Never reference context the solver cannot see ("上文", "该文档").
   - Keep provenance: `task → bundle.id → node ids → material ids`.
6. **Trial production first**: generate ~20 samples, show the user a handful,
   adjust prompts, then scale. Check type × level diversity against the plan.

## Anti-patterns (learned from real failures)

- Generating "根据材料/根据上文" questions WITHOUT shipping the materials.
- Trusting node `stats` you didn't produce: `n_materials` is real only after a
  retrieve pass; treat 0 as "unknown", not "no materials exist".
- Sampling `merged` nodes (alias-index only) — filter on `status`.
