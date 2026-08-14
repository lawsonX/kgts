"""Stage D driver: pick a task type, check material sufficiency, generate."""

from __future__ import annotations

import random
from typing import Any

from kgts.config import SynthesizeConfig
from kgts.graph.store import GraphStore
from kgts.models import Material, SampleBundle, SampleIntent, Task, VerifyResult
from kgts.synthesize.base import get_task_type

_REJECT_ANSWERS = {
    "zh": "根据现有检索到的材料，信息不足，无法可靠回答该问题。",
    "en": (
        "The retrieved materials do not contain enough information "
        "to answer this question reliably."
    ),
}


class Synthesizer:
    def __init__(self, llm: Any, config: SynthesizeConfig, seed: int = 42) -> None:
        self.llm = llm
        self.config = config
        self.seed = seed

    def synthesize(
        self, store: GraphStore, bundle: SampleBundle, materials: list[Material]
    ) -> Task | None:
        """Generate one task for ``bundle``; None means dropped/failed."""
        name = self._pick_type(bundle)
        task_type = get_task_type(name)
        if len(materials) < task_type.requires.min_docs:
            if self.config.insufficient_material == "drop":
                return None
            return self._reject_sample(store, bundle, materials, name)
        try:
            task = task_type.generate(self.llm, bundle, materials, self.config.style)
        except ValueError:
            return None  # unusable LLM reply; upstream counts this as a synth failure
        if task.verifier is None:
            task.verify_result = VerifyResult.SFT_ONLY
        return task

    def _pick_type(self, bundle: SampleBundle) -> str:
        enabled = list(self.config.task_types)
        if not enabled:
            raise ValueError("synthesize.task_types is empty; nothing to generate")
        if bundle.intent == SampleIntent.JOINT and "multihop_qa" in enabled:
            return "multihop_qa"
        weights = [self.config.type_weights.get(t, 1.0) for t in enabled]
        rng = random.Random(self.seed + hash(bundle.id))
        return rng.choices(enabled, weights=weights, k=1)[0]

    def _reject_sample(
        self, store: GraphStore, bundle: SampleBundle, materials: list[Material], name: str
    ) -> Task:
        """Explicit reject-answer sample for under-supplied bundles (section 7)."""
        language = str(self.config.style.get("language", "en")).lower()
        if language not in _REJECT_ANSWERS:
            language = "en"
        labels = [self._label(store, bundle, nid) for nid in bundle.nodes]
        topic = "、".join(labels) if language == "zh" else ", ".join(labels)
        question = (
            f"请介绍：{topic}。"
            if language == "zh"
            else f"Explain: {topic}."
        )
        return Task(
            task_type=name,
            sample_bundle=bundle,
            materials=[m.id for m in materials],
            question=question,
            answer=_REJECT_ANSWERS[language],
            verifier=None,
            verify_result=VerifyResult.SFT_ONLY,
            style=dict(self.config.style),
        )

    @staticmethod
    def _label(store: GraphStore, bundle: SampleBundle, node_id: str) -> str:
        path = bundle.ancestor_paths.get(node_id) or []
        if path:
            return path[-1]
        try:
            return store.get(node_id).label
        except KeyError:
            return node_id
