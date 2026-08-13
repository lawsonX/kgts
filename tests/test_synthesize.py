"""Offline tests for Stage D (synthesize) using MockLLM."""

import pytest

import kgts.synthesize  # noqa: F401 -- registers the built-in task types
from kgts.config import SynthesizeConfig
from kgts.graph.store import GraphStore
from kgts.llm import MockLLM
from kgts.models import Material, Node, SampleBundle, SampleIntent, SourceType, VerifyResult
from kgts.synthesize.base import get_task_type, list_task_types
from kgts.synthesize.synthesizer import Synthesizer


def _bundle(intent: SampleIntent = SampleIntent.DEPTH, nodes=("n1",)) -> SampleBundle:
    paths = {"n1": ["CS", "GPU programming", "kernel"], "n2": ["CS", "GPU programming", "CUDA"]}
    return SampleBundle(
        id="sb_test",
        nodes=list(nodes),
        ancestor_paths={n: paths[n] for n in nodes},
        intent=intent,
    )


def _materials(n: int = 2) -> list[Material]:
    return [
        Material(
            id=f"m_{i}",
            source_type=SourceType.LOCAL,
            title=f"doc{i}",
            snippet=f"snippet {i}",
            text=f"text {i}",
        )
        for i in range(n)
    ]


def _store() -> GraphStore:
    store = GraphStore()
    store.add_node(Node(id="n1", label="kernel"))
    store.add_node(Node(id="n2", label="CUDA"))
    return store


_GOOD_REPLY = {
    "question": "What is a GPU kernel? [m_0]",
    "answer": "A GPU kernel is a function that runs on the device [m_0].",
    "rubric": ["answer cites its materials"],
}


# ---------------------------------------------------------------- registry


def test_registry_lists_five_builtins():
    assert list_task_types() == [
        "aggregated_qa",
        "atomic_qa",
        "comparative_analysis",
        "grounded_summary",
        "multihop_qa",
    ]


def test_get_task_type_unknown_lists_registered():
    with pytest.raises(ValueError, match="atomic_qa"):
        get_task_type("nope")


# ------------------------------------------------------------- synthesize


def test_synthesize_returns_task_with_materials_and_verifier():
    llm = MockLLM(default=_GOOD_REPLY)
    cfg = SynthesizeConfig(task_types=["atomic_qa"])
    task = Synthesizer(llm, cfg).synthesize(_store(), _bundle(), _materials(1))
    assert task is not None
    assert task.task_type == "atomic_qa"
    assert task.materials == ["m_0"]
    assert task.verifier == "answer_match"
    assert task.verify_result == VerifyResult.UNVERIFIED
    assert task.question == _GOOD_REPLY["question"]
    # prompt must list materials and demand citations
    assert "[m_0]" in task.prompt
    assert "cite" in task.prompt.lower()


def test_joint_bundle_prefers_multihop():
    llm = MockLLM(default=_GOOD_REPLY)
    cfg = SynthesizeConfig()  # all five built-ins enabled
    bundle = _bundle(intent=SampleIntent.JOINT, nodes=("n1", "n2"))
    task = Synthesizer(llm, cfg).synthesize(_store(), bundle, _materials(2))
    assert task is not None
    assert task.task_type == "multihop_qa"


def test_no_verifier_type_marked_sft_only():
    llm = MockLLM(default=_GOOD_REPLY)
    cfg = SynthesizeConfig(task_types=["grounded_summary"])
    task = Synthesizer(llm, cfg).synthesize(_store(), _bundle(), _materials(1))
    assert task is not None
    assert task.verifier is None
    assert task.verify_result == VerifyResult.SFT_ONLY


# ---------------------------------------------- insufficient material paths


def test_insufficient_material_drop_returns_none():
    llm = MockLLM(default=_GOOD_REPLY)
    cfg = SynthesizeConfig(task_types=["aggregated_qa"], insufficient_material="drop")
    task = Synthesizer(llm, cfg).synthesize(_store(), _bundle(), _materials(1))
    assert task is None
    assert llm.calls == []  # no LLM call wasted


def test_insufficient_material_reject_sample():
    llm = MockLLM()
    cfg = SynthesizeConfig(
        task_types=["aggregated_qa"],
        insufficient_material="reject_sample",
        style={"language": "en"},
    )
    task = Synthesizer(llm, cfg).synthesize(_store(), _bundle(), _materials(1))
    assert task is not None
    assert task.task_type == "aggregated_qa"
    assert task.verifier is None
    assert task.verify_result == VerifyResult.SFT_ONLY
    assert "kernel" in task.question
    assert "not contain enough information" in task.answer
    assert llm.calls == []


def test_bad_llm_reply_returns_none():
    llm = MockLLM(default={"foo": "bar"})  # lacks question/answer
    cfg = SynthesizeConfig(task_types=["atomic_qa"])
    task = Synthesizer(llm, cfg).synthesize(_store(), _bundle(), _materials(1))
    assert task is None
