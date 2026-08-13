# Contributing to KGTS

Thanks for your interest. The most valuable contributions are new task-type
plugins, new material sources, and verifier improvements — the pipeline
core is deliberately small.

## Dev setup

```bash
git clone https://github.com/kgts-project/kgts.git
cd kgts
pip install -e ".[dev]"
```

Run the checks (both must pass before a PR):

```bash
ruff check .
pytest
```

The test suite is fully offline: every test uses `MockLLM`
(`kgts/llm.py`) and never touches the network or an API key. Keep it that
way — new tests must not require credentials, network access, or optional
extras (`llm`, `ui`).

## Code style

- Python 3.11+, line length 100 (enforced by ruff, see `pyproject.toml`).
- Type hints on all public functions; pydantic models for anything that
  crosses a stage boundary or gets persisted.
- All code, comments, and docstrings in English.
- Stdlib-first: the core depends only on `networkx`, `pydantic`, `pyyaml`,
  `typer`. Do not add a dependency without discussing it in an issue first.

## Adding a plugin

Full walkthroughs with working example code live in
[docs/plugins.md](docs/plugins.md). Short version:

- **Task type** — subclass `TaskType` (`kgts/synthesize/base.py`), declare
  `requires` (a `MaterialSpec`) and optionally `verifier_name`, implement
  `generate(llm, bundle, materials, style) -> Task`, and decorate with
  `@register_task_type("your_name")`. Built-in examples:
  `kgts/synthesize/builtin.py`.
- **Material source** — implement the `MaterialSource` protocol
  (`kgts/retrieve/sources.py`): `search(queries, budget) -> list[Material]`.
  Record `license` on web materials. Network sources must raise
  `RuntimeError` on failure rather than returning nothing silently.
- **Verifier** — implement the `Verifier` protocol (`kgts/verify/base.py`)
  and register it in `kgts/verify/pipeline.py` (`_make_verifier`).
- **Prioritizer** — implement the `Prioritizer` protocol
  (`kgts/sample/prioritizer.py`); this is where an ECE/blind-spot plugin
  hooks in.

## Invariants and frozen contracts

Some rules keep the pipeline auditable; PRs that break them will be
rejected:

- The `is_subconcept` graph is a DAG — `GraphStore.add_edge` rejects cycles.
- The provenance chain `Task → SampleBundle → Node → Material → Run` must
  stay complete for every exported example.
- These cross-module signatures are imported by the orchestrator and other
  stages; do not change them without a design discussion:
  `expand_graph` (`kgts/build/expand.py`), `sample_bundles`
  (`kgts/sample/sampler.py`), `Retriever.retrieve`
  (`kgts/retrieve/retriever.py`), `Synthesizer.synthesize`
  (`kgts/synthesize/synthesizer.py`), `verify_task`
  (`kgts/verify/pipeline.py`).

## PR guidelines

- One concern per PR; keep diffs reviewable.
- Add or update tests for behavior changes; keep `pytest` green.
- Update docs (`docs/`, README) when you change user-visible behavior,
  config fields, or CLI flags.
- Reference the design rationale in the commit message when a change
  touches fidelity to the K3 mechanisms (see `docs/fidelity.md`).

## Scope notes

Out of scope for this repository: the RL training loop itself, model
training frameworks (export formats target LLaMA-Factory/verl instead),
and serving infrastructure. If a PR needs one of those, it probably belongs
in a downstream project that consumes KGTS exports.
