"""Offline tests for Stage B (sampling scheduler)."""

import pytest

from kgts.config import SampleConfig
from kgts.graph.store import GraphStore
from kgts.models import Edge, Node, NodeStatus, Relation, SampleIntent
from kgts.sample.prioritizer import InverseFrequencyPrioritizer
from kgts.sample.sampler import sample_bundles

_MIXTURE = {"breadth": 0.25, "depth": 0.5, "joint": 0.25}


def _fixture_store() -> GraphStore:
    """8 nodes, levels 0-3, one atomic branch:

    CS -> {AI -> {ML*, NLP*}, Systems -> {Databases -> Indexing*, OS*}}  (* = atomic)
    """
    store = GraphStore()

    def add(label: str, parent: str | None = None, status: NodeStatus = NodeStatus.EXPANDING):
        node = store.add_node(Node.create(label, status=status))
        if parent is not None:
            store.add_edge(
                Edge(parent=store.find(parent).id, child=node.id,
                     relation=Relation.IS_SUBCONCEPT)
            )
        return node

    add("CS")
    add("AI", "CS")
    add("Systems", "CS")
    add("ML", "AI", NodeStatus.ATOMIC)
    add("NLP", "AI", NodeStatus.ATOMIC)
    add("Databases", "Systems")
    add("OS", "Systems", NodeStatus.ATOMIC)
    add("Indexing", "Databases", NodeStatus.ATOMIC)
    return store


def test_sample_bundles_mixture_counts_and_metadata():
    store = _fixture_store()
    config = SampleConfig(n_samples=12, mixture=dict(_MIXTURE))
    bundles = sample_bundles(store, config, seed=7)

    assert len(bundles) == 12  # mixture counts sum to n_samples
    by_intent = {i: [b for b in bundles if b.intent == i] for i in SampleIntent}
    assert len(by_intent[SampleIntent.BREADTH]) == 3
    assert len(by_intent[SampleIntent.DEPTH]) == 6
    assert len(by_intent[SampleIntent.JOINT]) == 3

    atomic_ids = {n.id for n in store.atomic_nodes()}
    for b in by_intent[SampleIntent.DEPTH]:
        assert len(b.nodes) == 1 and b.nodes[0] in atomic_ids
    for b in by_intent[SampleIntent.BREADTH]:
        assert len(b.nodes) == 1 and store.get(b.nodes[0]).level <= 2
    for b in by_intent[SampleIntent.JOINT]:
        assert 2 <= len(b.nodes) <= config.joint.max_group_size
        # members are siblings: all share one common is_subconcept parent
        assert any(set(b.nodes) <= set(store.children(p.id)) for p in store.nodes())

    for b in bundles:
        assert b.level == max(store.get(nid).level for nid in b.nodes)
        for nid in b.nodes:
            path = b.ancestor_paths[nid]
            assert path and path[-1] == store.get(nid).label

    total_picks = sum(len(b.nodes) for b in bundles)
    assert sum(n.stats.times_sampled for n in store.nodes()) == total_picks
    # coverage pass: the 3 breadth picks hit 3 distinct nodes
    assert len({b.nodes[0] for b in by_intent[SampleIntent.BREADTH]}) == 3


def test_inverse_frequency_weight_decreases_with_times_sampled():
    prio = InverseFrequencyPrioritizer(alpha=1.0)
    node = Node.create("X")
    assert prio.weight(node) == 1.0
    node.stats.times_sampled = 3
    assert prio.weight(node) == 0.25
    node.stats.times_sampled = 9
    assert prio.weight(node) < 0.25


def test_per_node_cap_respected():
    store = _fixture_store()
    config = SampleConfig(n_samples=40)
    config.quotas.max_per_node = 2
    bundles = sample_bundles(store, config, seed=1)
    assert bundles  # pools saturate before n_samples is reached; that's fine
    for n in store.nodes():
        assert n.stats.times_sampled <= 2


def test_sample_bundles_deterministic_with_same_seed():
    config = SampleConfig(n_samples=12, mixture=dict(_MIXTURE))
    first = sample_bundles(_fixture_store(), config, seed=123)
    second = sample_bundles(_fixture_store(), config, seed=123)
    key = lambda bs: [(b.intent, sorted(b.nodes)) for b in bs]  # noqa: E731
    assert key(first) == key(second)


def test_depth_pool_falls_back_to_leaves_when_no_atomic():
    store = _fixture_store()
    for n in store.nodes():
        n.status = NodeStatus.EXPANDING
    config = SampleConfig(n_samples=4, mixture={"breadth": 0.0, "depth": 1.0, "joint": 0.0})
    bundles = sample_bundles(store, config, seed=5)
    leaves = {n.id for n in store.nodes() if not store.children(n.id)}
    assert len(bundles) == 4
    assert all(b.intent == SampleIntent.DEPTH for b in bundles)
    assert all(b.nodes[0] in leaves for b in bundles)


def test_unknown_prioritizer_raises():
    from kgts.config import SampleConfig

    store = _fixture_store()
    with pytest.raises(ValueError, match="prioritizer"):
        sample_bundles(store, SampleConfig(prioritizer="ece"), seed=1)
