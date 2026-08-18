"""Offline tests for Stage A (recursive agentic graph expansion) — MockLLM only."""

import pytest

from kgts.build.aligner import Aligner
from kgts.build.atomicity import AtomicityJudge
from kgts.build.expand import expand_graph
from kgts.build.explorer import ExplorerAgent
from kgts.config import AlignConfig, AtomicityConfig, Config, SeedSpec
from kgts.graph.store import GraphStore
from kgts.llm import ManagedLLM, MockLLM
from kgts.models import AlignVerdict, Edge, Node, NodeStatus


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
    # the explorer's material_estimate is an LLM self-report and must NOT be
    # trusted into node stats (real runs showed inflated values like 50000)
    assert algo.stats.n_materials == 0
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


def test_aligner_chinese_near_duplicates_reach_judge():
    """Regression: CJK labels must produce embedding tokens; otherwise every
    Chinese candidate scores 0 and short-circuits to DISTINCT without the
    LLM judge (observed: 危险品识别与处置 vs 危险物品识别与处置)."""
    from kgts.build.aligner import Aligner, _cosine, _hash_embed

    sim = _cosine(_hash_embed("危险品识别与处置"), _hash_embed("危险物品识别与处置"))
    assert sim > 0.5, f"Chinese near-duplicates should be similar, got {sim}"

    store = GraphStore()
    store.add_node(Node.create("危险品识别与处置"))
    judge = MockLLM(default={"verdict": "equivalent", "matched_label": "危险品识别与处置",
                             "canonical": "危险品识别与处置"})
    aligner = Aligner(judge, store, AlignConfig(embed_threshold=0.75))
    decision = aligner.align("危险物品识别与处置", store.get(next(iter(store._nodes))))
    assert judge.calls, "similar Chinese labels must reach the LLM judge"
    assert decision.verdict == AlignVerdict.EQUIVALENT


def test_slugify_keeps_cjk():
    from kgts.models import slugify

    assert slugify("犯罪现场勘查") == "犯罪现场勘查"
    assert slugify("GPU Programming") == "gpu-programming"


# ------------------------------------------------------------ queue policy
def _leveled_store():
    """S0(0) -> A(1) -> B(2) plus another root S1(0)."""
    store = GraphStore()
    s0 = store.add_node(Node.create("S0"))
    a = store.add_node(Node.create("A"))
    b = store.add_node(Node.create("B"))
    store.add_node(Node.create("S1"))
    store.add_edge(Edge(parent=s0.id, child=a.id))
    store.add_edge(Edge(parent=a.id, child=b.id))
    return store


def test_frontier_pop_order_per_policy():
    from kgts.build.expand import _Frontier

    store = _leveled_store()
    by_label = {n.label: n.id for n in store.nodes()}

    def order(policy):
        f = _Frontier(store, policy)
        for label in ("S0", "S1", "A", "B"):
            f.push(by_label[label])
        out = []
        while len(f):
            out.append(store.get(f.pop()).label)
        return out

    assert order("bfs") == ["S0", "S1", "A", "B"]  # FIFO
    assert order("dfs") == ["B", "A", "S1", "S0"]  # LIFO
    # seed-fair depth-first: rotate over seed buckets, deepest within each
    assert order("balanced") == ["B", "S1", "A", "S0"]
    f = _Frontier(store, "bfs")
    f.push(by_label["S0"])
    f.push(by_label["S0"])  # dedup
    assert len(f) == 1
    with pytest.raises(ValueError, match="queue_policy"):
        _Frontier(store, "spiral")


def _exploration_order(llm: MockLLM) -> list[str]:
    """Labels of explored nodes in order (first-round explorer prompts)."""
    out = []
    for call in llm.calls:
        if "Concept: " in call:
            label = call.split("Concept: ", 1)[1].split("\n", 1)[0]
            if not out or out[-1] != label:
                out.append(label)
    return out


def _policy_script():
    def brief(label):
        return {"definition": "d", "candidates": [{"label": label}],
                "material_estimate": 0}

    return {
        "Concept: S1\n": brief("B1"),
        "Concept: B1\n": brief("C1"),
        "Concept: C1\n": {"definition": "d", "candidates": []},
        "Concept: S2\n": brief("D2"),
        "Concept: D2\n": brief("E2"),
        "Concept: E2\n": {"definition": "d", "candidates": []},
    }


def _expand_order_with(policy: str) -> list[str]:
    seeds = [SeedSpec(label="S1", children=["A1"]), SeedSpec(label="S2", children=["A2"])]
    cfg = Config()
    cfg.budget.max_nodes = 30
    cfg.build.queue_policy = policy
    llm = MockLLM(
        script=_policy_script(),
        default={"definition": "", "candidates": [], "material_estimate": 0},
    )
    expand_graph(seeds, llm, GraphStore(), cfg)
    return _exploration_order(llm)


def test_queue_policy_bfs_is_fifo():
    order = _expand_order_with("bfs")
    assert order == ["S1", "A1", "S2", "A2", "B1", "D2", "C1", "E2"]


def test_queue_policy_balanced_is_seed_fair_depth_first():
    order = _expand_order_with("balanced")
    # L1 children first (deepest at start), then both seeds get their turn
    # BEFORE any branch dives, then the two chains interleave: B1, D2, C1, E2
    assert order == ["A1", "A2", "S1", "S2", "B1", "D2", "C1", "E2"]
    # the fairness property: no seed's chain dives twice before the other's first dive
    assert order.index("D2") < order.index("C1")
