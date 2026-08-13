"""Offline tests for Stage A (recursive agentic graph expansion) — MockLLM only."""

from kgts.build.aligner import Aligner
from kgts.build.atomicity import AtomicityJudge
from kgts.build.expand import expand_graph
from kgts.build.explorer import ExplorerAgent
from kgts.config import AlignConfig, AtomicityConfig, Config, SeedSpec
from kgts.graph.store import GraphStore
from kgts.llm import ManagedLLM, MockLLM
from kgts.models import AlignVerdict, Node, NodeStatus


def _brief(definition: str, candidates: list[str], estimate: int) -> dict:
    return {
        "definition": definition,
        "candidates": [{"label": c} for c in candidates],
        "material_estimate": estimate,
    }


# ------------------------------------------------------------------ explorer
def test_explorer_parses_brief_and_dedups_candidates():
    llm = MockLLM(
        script={
            "Photosynthesis": {
                "definition": "Conversion of light energy into chemical energy.",
                "candidates": [
                    {"label": "Light Reactions", "rationale": "core stage"},
                    {"label": "Calvin Cycle"},
                    {"label": "light reactions"},  # duplicate after normalization
                ],
                "material_estimate": 12,
            }
        }
    )
    agent = ExplorerAgent(llm)
    brief = agent.explore(Node.create("Photosynthesis"), ["Biology", "Photosynthesis"])
    assert brief.definition.startswith("Conversion")
    assert brief.material_estimate == 12
    assert [c.label for c in brief.candidates] == ["Light Reactions", "Calvin Cycle"]
    # round 2 returns the same payload -> no new candidates -> early stop
    assert len(llm.calls) == 2


def test_explorer_refinement_round_adds_new_candidates():
    llm = MockLLM(
        script={
            # refine prompts ask for "additional subconcepts"; checked first
            "additional subconcepts": {"candidates": [{"label": "Stomata"}]},
            "Photosynthesis": _brief("d", ["Calvin Cycle"], 5),
        }
    )
    agent = ExplorerAgent(llm, max_rounds=4)
    brief = agent.explore(Node.create("Photosynthesis"), ["Biology"])
    assert [c.label for c in brief.candidates] == ["Calvin Cycle", "Stomata"]
    # round 1 + one productive refine + one no-op refine -> early stop
    assert len(llm.calls) == 3


# ------------------------------------------------------------------- aligner
def test_aligner_distinct_short_circuit_without_llm_call():
    store = GraphStore()
    parent = store.add_node(
        Node.create("Quantum Entanglement", description="correlated quantum states")
    )
    llm = MockLLM(
        default={"verdict": "equivalent", "matched_label": "Quantum Entanglement",
                 "canonical": "x"}
    )
    aligner = Aligner(llm, store, AlignConfig())
    decision = aligner.align("Photosynthesis", parent)
    assert decision.verdict == AlignVerdict.DISTINCT
    assert decision.matched_node is None
    assert llm.calls == []  # below embed_threshold -> no judge call spent


def test_aligner_equivalent_via_scripted_judge():
    store = GraphStore()
    existing = store.add_node(Node.create("Neural Networks", aliases=["Neural Nets"]))
    llm = MockLLM(
        default={"verdict": "equivalent", "matched_label": "Neural Networks",
                 "canonical": "Neural Networks"}
    )
    aligner = Aligner(llm, store, AlignConfig(embed_threshold=0.3))
    decision = aligner.align("Neural Nets", existing)
    assert decision.verdict == AlignVerdict.EQUIVALENT
    assert decision.matched_node == existing.id
    assert decision.canonical_suggestion == "Neural Networks"
    assert decision.judge_model == "mock-llm"
    assert len(decision.prompt_hash) == 12
    assert len(llm.calls) == 1


# ----------------------------------------------------------------- atomicity
def test_atomicity_judge_signals_and_depth_cap():
    judge = AtomicityJudge(AtomicityConfig(min_materials=5))
    node = Node.create("X")
    node.stats.n_materials = 3
    assert not judge.is_atomic(node)  # insufficient materials
    node.stats.n_materials = 5
    assert judge.is_atomic(node)  # other signals unknown -> pass
    node.stats.synth_success_rate = 0.2
    assert not judge.is_atomic(node)
    node.stats.synth_success_rate = None
    node.stats.child_material_overlap = 0.9
    assert not judge.is_atomic(node)  # too much overlap -> no marginal benefit
    node.level = 6
    assert judge.is_atomic(node)  # depth safety cap overrides


# --------------------------------------------------------------------- expand
class _FakeArtifactStore:
    """Duck-typed artifact store recording persisted AlignDecisions."""

    def __init__(self) -> None:
        self.align_decisions: list = []

    def save_align_decision(self, decision) -> None:
        self.align_decisions.append(decision)


def _expand_script() -> dict:
    return {
        # judge replies (checked first: these labels also appear in prompts)
        "Sorting": {"verdict": "distinct", "matched_label": None, "canonical": "Sorting"},
        "Graph Algorithms": {"verdict": "distinct", "matched_label": None,
                             "canonical": "Graph Algorithms"},
        "DNA Sequencing": {"verdict": "distinct", "matched_label": None,
                           "canonical": "DNA Sequencing"},
        # explore briefs for the human-given first layer
        "Algorithms": _brief("Study of algorithms.", ["Sorting", "Graph Algorithms"], 3),
        "Genetics": _brief("Study of heredity.", ["DNA Sequencing"], 3),
    }


def test_expand_graph_end_to_end():
    seeds = [
        SeedSpec(label="Computer Science", children=["Algorithms"]),
        SeedSpec(label="Biology", children=["Genetics"]),
    ]
    llm = MockLLM(
        script=_expand_script(),
        default={"definition": "", "candidates": [], "material_estimate": 0},
    )
    store = GraphStore()
    artifacts = _FakeArtifactStore()
    out = expand_graph(seeds, llm, store, Config(), artifact_store=artifacts)

    assert out is store
    assert len(store) == 7  # 2 seeds + 2 first-layer + 3 explored children
    assert store.find("Computer Science").level == 0
    assert store.find("Biology").level == 0
    algo = store.find("Algorithms")
    assert algo.description == "Study of algorithms."
    assert algo.stats.n_materials == 3
    for label in ("Sorting", "Graph Algorithms"):
        child = store.find(label)
        assert child is not None and child.level == 2
        assert store.parents(child.id) == [algo.id]
    assert store.find("DNA Sequencing") is not None
    # no cycle was ever attempted
    assert not any(f["kind"] == "cycle_prevented" for f in store.review_flags)
    # every AlignDecision was persisted
    assert len(artifacts.align_decisions) == 3
    assert all(d.verdict == AlignVerdict.DISTINCT for d in artifacts.align_decisions)


def test_expand_graph_budget_exceeded_breaks_gracefully():
    seeds = [
        SeedSpec(label="CS", children=["AI"]),
        SeedSpec(label="Bio", children=["Genetics"]),
    ]
    inner = MockLLM(default=_brief("d", ["X"], 0))
    llm = ManagedLLM(inner, max_calls=1)  # 1 call -> refine round raises BudgetExceeded
    store = GraphStore()
    out = expand_graph(seeds, llm, store, Config())
    assert out is store
    # seeds + first layer were inserted before the loop broke; no exception escaped
    assert len(store) == 4
    assert store.find("CS").status == NodeStatus.EXPANDING
