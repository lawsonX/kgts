# Plugins

KGTS keeps the pipeline core small and pushes variation into four plugin
points: task types, material sources, verifiers, and prioritizers. This
document shows working code for each.

## Task types (`kgts/synthesize/base.py`)

A task type subclasses `TaskType`, declares what materials it needs, and
implements one method:

```python
# my_package/definition_qa.py
from typing import Any

from kgts.models import Material, SampleBundle, Task, VerifyResult
from kgts.synthesize.base import MaterialSpec, TaskType, register_task_type


@register_task_type("definition_qa")
class DefinitionQA(TaskType):
    """Ask for a definition grounded in the retrieved materials."""

    requires = MaterialSpec(min_docs=1)     # minimum materials to attempt generation
    verifier_name = "answer_match"          # or None -> task is marked sft_only

    def generate(
        self, llm: Any, bundle: SampleBundle, materials: list[Material], style: dict
    ) -> Task:
        labels = ", ".join(
            (bundle.ancestor_paths.get(nid) or [nid])[-1] for nid in bundle.nodes
        )
        listing = "\n".join(f"[{m.id}] {m.title}: {m.snippet}" for m in materials)
        prompt = (
            f"Write one 'what is' question and answer about: {labels}.\n"
            f"Answer language: {style.get('language', 'en')}.\n"
            f"Ground every key fact in a material and cite it as [material_id].\n\n"
            f"Materials:\n{listing}\n\n"
            'Reply with JSON only: {"question": str, "answer": str, "rubric": [str]}.'
        )
        reply = llm.complete_json(prompt)
        if not isinstance(reply, dict) or not reply.get("question") or not reply.get("answer"):
            raise ValueError(f"unusable LLM reply: {reply!r}")  # counted as synth failure
        return Task(
            task_type=self.name,
            sample_bundle=bundle,
            materials=[m.id for m in materials],
            prompt=prompt,
            question=str(reply["question"]),
            answer=str(reply["answer"]),
            rubric=[str(r) for r in reply.get("rubric") or []],
            verifier=self.verifier_name,
            verify_result=VerifyResult.UNVERIFIED,
            style=dict(style),
        )
```

Rules of the contract:

- Be deterministic apart from the LLM call itself.
- Raise `ValueError` on an unusable LLM reply — the orchestrator counts it
  as a synthesis failure and moves on; do not return a half-built `Task`.
- Set `verifier_name = None` when the type has no automatic check; the
  task is then marked `sft_only` (exported for SFT, excluded from RL).

Registration happens at import time. For a plugin shipped inside the repo,
import it from `kgts/synthesize/__init__.py` alongside `builtin`. For an
external package, import your module before running the pipeline (or ship
it as a small wrapper package that depends on `kgts`). Enable it in the
config:

```yaml
synthesize:
  task_types: [atomic_qa, multihop_qa, definition_qa]
  type_weights: {definition_qa: 2.0}   # empty = uniform
```

Notes on type selection (`Synthesizer._pick_type`): JOINT bundles use
`multihop_qa` when it is enabled; otherwise the type is drawn from
`synthesize.task_types` weighted by `type_weights`, seeded per bundle id.

Study `kgts/synthesize/builtin.py` first — the five built-ins share one
prompt/generation flow you can reuse.

## Material sources (`kgts/retrieve/sources.py`)

A source implements the `MaterialSource` protocol:

```python
class MaterialSource(Protocol):
    def search(self, queries: list[str], budget: int) -> list[Material]: ...
```

Example — a minimal in-house JSONL document store:

```python
import json
from pathlib import Path

from kgts.models import Material, SourceType


class JsonlStoreSource:
    """Serves pre-chunked documents from a JSONL file (one doc per line)."""

    def __init__(self, path: str) -> None:
        self._docs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

    def search(self, queries: list[str], budget: int) -> list[Material]:
        terms = {t for q in queries for t in q.lower().split()}
        hits = [
            d for d in self._docs
            if terms & set((d["title"] + " " + d["text"]).lower().split())
        ]
        return [
            Material(
                source_type=SourceType.LOCAL,
                uri=d.get("uri"),
                title=d["title"],
                snippet=d["text"][:200],
                text=d["text"],
                license=d.get("license", "unknown"),
            )
            for d in hits[:budget]
        ]
```

Contract rules:

- Return **at most** `budget` materials. `queries` are already
  disambiguated by the `QueryBuilder` (leaf label + nearest ancestors,
  shaped per source type).
- Always record `license`; web-derived materials without one should use
  `"unknown"` (the whitelist filter keeps them but downstream compliance
  review needs the field populated).
- Raise `RuntimeError` on infrastructure failure (missing API key, HTTP
  error, parse failure) — never return an empty list to mask an outage.
  The retriever tolerates one source failing as long as another succeeds.

To wire a source into the orchestrator, add it to the `factories` map in
`build_sources` (`kgts/retrieve/sources.py`) under a new name and list that
name in `retrieve.sources`. Keep built-ins stdlib-only; a source that
needs a heavy dependency belongs behind an optional extra.

## Verifiers (`kgts/verify/base.py`)

```python
from kgts.models import Material, Task, VerifyResult


class ExactNumericVerifier:
    name = "exact_numeric"

    def verify(
        self, task: Task, materials_by_id: dict[str, Material]
    ) -> tuple[VerifyResult, float, str]:
        expected = (task.style.get("expected") or "").strip()
        got = task.answer.strip()
        if expected and got == expected:
            return VerifyResult.PASS, 1.0, "exact match"
        return VerifyResult.FAIL, 0.0, f"expected {expected!r}, got {got!r}"
```

Register new verifier names in `_make_verifier` (`kgts/verify/pipeline.py`).
Unknown names fall back to `verify.fallback` (default `rubric_judge`); if
the fallback is unknown too, the task is downgraded to `sft_only`.

## Prioritizers (`kgts/sample/prioritizer.py`)

```python
from typing import Protocol
from kgts.models import Node


class Prioritizer(Protocol):
    def weight(self, node: Node) -> float: ...
```

Higher weight = more likely to be sampled. The built-in
`InverseFrequencyPrioritizer` implements `1 / (1 + alpha * times_sampled)`.

**Where an ECE plugin hooks in.** A GraphGen-style blind-spot prioritizer
would (a) probe the target trainee model per node, (b) write the resulting
error/blind-spot score into the reserved `Node.stats.ece` field, and (c)
return a weight that combines inverse frequency with that score:

```python
class ECEPrioritizer:
    def __init__(self, alpha: float = 1.0, ece_weight: float = 1.0) -> None:
        self.alpha, self.ece_weight = alpha, ece_weight

    def weight(self, node: Node) -> float:
        base = 1.0 / (1.0 + self.alpha * node.stats.times_sampled)
        return base * (1.0 + self.ece_weight * (node.stats.ece or 0.0))
```

Status in 0.1.0: `sample_bundles` (`kgts/sample/sampler.py`) instantiates
`InverseFrequencyPrioritizer` directly — the `sample.prioritizer` config
field is accepted but not yet wired to a plugin loader. Making the
prioritizer pluggable in the sampler is the intended v0.4 change and is a
good first contribution; keep the `sample_bundles(store, config, seed)`
signature unchanged when you do it.
