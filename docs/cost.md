# Cost model

Graph expansion (Stage A) is where KGTS spends LLM calls; everything else
is an order of magnitude cheaper. This document explains where calls go,
which knobs control them, and how to estimate a run's cost. All numbers
below are **illustrative** — plug in your endpoint's actual prices; KGTS
ships no benchmarks.

## Where LLM calls go

| Stage | LLM calls | Driver |
|---|---|---|
| A explore | 1..`budget.max_explore_rounds` per expanded node | refinement stops early when a round yields no new candidates |
| A align judge | ≤ 1 per candidate sub-concept | skipped entirely when recall similarity < `build.align.embed_threshold` and no exact alias match (the cheap path) |
| A atomicity | 0 | rule-based, reads `node.stats` |
| B sample | 0 | deterministic |
| C retrieve | 0 in 0.1.0 | rerankers `cross_encoder`/`llm` are config names; the built-in fallback is token overlap |
| D synthesize | 1 per bundle | `Synthesizer` generates one task per sample bundle |
| E rubric_judge | 1 per task routed to the rubric judge | only tasks whose verifier is `rubric_judge` (or fall back to it) |
| E answer_match | 0 | heuristic, no LLM |

Expansion dominates because it runs per node **and** per candidate:

```text
llm_calls ≈ Σ_nodes explore_rounds(n)            # 1..max_explore_rounds, early-stopped
          + Σ_nodes judged_candidates(n)         # candidates not short-circuited by embed_threshold
          + n_samples                            # one synthesis call per bundle
          + n_rubric_judged                      # tasks verified via rubric_judge
```

For a 300-node graph averaging 2 explore rounds and 3 judged candidates
per node, plus 1000 samples: roughly `300·(2+3) + 1000 ≈ 2500` calls —
which is why the default `budget.max_llm_calls` is 2000 and why you should
set it before the first real run.

Token cost per call varies most with the align judge (recalled-node list
in the prompt) and synthesis (materials in the prompt). Illustrative
formula:

```text
cost_usd ≈ calls × (avg_input_tokens × price_in + avg_output_tokens × price_out) / 1e6
```

Example, with purely illustrative figures — 1500 input / 300 output tokens
per call at $0.15 / $0.60 per 1M tokens (check your provider's page for
real prices): `2500 × (1500×0.15 + 300×0.60) / 1e6 ≈ $1.01`. Treat this as
an order-of-magnitude sanity check, not a measurement.

## Knobs

- `budget.max_llm_calls` (default 2000) — **enforced hard cap**:
  `ManagedLLM` raises `BudgetExceeded`; the expansion loop catches it and
  stops gracefully with the graph checkpointed, so a capped run still
  yields a usable DAG. Enforced in 0.1.0.
- `budget.max_cost_usd` (default 10.0) — declared in the config schema but
  **not enforced in 0.1.0** (no per-call cost accounting yet). Do not rely
  on it; use `max_llm_calls`.
- `budget.max_nodes` (default 500) — caps the graph size, bounding the
  total explore+align work.
- `budget.max_explore_rounds` (default 4) — per-node explore ceiling;
  2–3 is often enough at deeper levels.
- `kgts build --cheap-mode` — uses `llm.cheap_model` instead of
  `llm.model` for the whole build stage, so a small model explores while
  (in a split-run setup) a larger model can be reserved for align/synth.
  Requires `llm.cheap_model` to be set.
- `llm.cache: true` (default) — on-disk completion cache in
  `workdir/llm_cache/`; identical prompts (model + prompt + temperature)
  are free on re-runs, which makes `--resume` and config iteration cheap.
- `build.align.embed_threshold` (default 0.75) — the short-circuit:
  candidates whose best recall similarity falls below it (with no exact
  alias match) are declared `DISTINCT` without a judge call. Raising it
  saves calls at the risk of duplicate nodes; lowering it buys precision.
- `llm.rpm` (default 60) — throttling; affects wall time, not cost.

## Practical guidance

- Iterate offline first: `python examples/quickstart_offline.py` and any
  config with `llm.model: mock-...` run the full pipeline at zero cost.
- Keep `llm.cache` on and reuse the same `run.workdir` while tuning —
  re-runs only pay for prompts that actually changed.
- Budget by phase: a tight `max_llm_calls` during `kgts build`, then
  relax it for `kgts synth` (synthesis is one call per bundle and easy to
  predict: `sample.n_samples`).
- Prefer the local source (`retrieve.sources: [local]`) while iterating;
  web retrieval adds API costs and rate limits outside KGTS's budget
  accounting.
