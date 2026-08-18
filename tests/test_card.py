"""Offline tests for the graph card (kgts/graph/card.py)."""

from kgts.config import Config, RunConfig
from kgts.graph.card import (
    CARD_JSON,
    CARD_MD,
    GraphCard,
    compute_notes,
    compute_stats,
    generate_card,
    render_markdown,
    write_card,
)
from kgts.graph.store import GraphStore
from kgts.models import Edge, Node


def _config(tmp_path, name="test-graph"):
    return Config(run=RunConfig(name=name, workdir=str(tmp_path)))


def _store_with_graph():
    store = GraphStore()
    root = store.add_node(Node.create("Root"))
    child = store.add_node(Node.create("Child", description="has a description"))
    leaf = store.add_node(Node.create("Leaf"))
    store.add_edge(Edge(parent=root.id, child=child.id))
    store.add_edge(Edge(parent=child.id, child=leaf.id))
    return store, root, child, leaf


def test_generate_first_card(tmp_path):
    store, *_ = _store_with_graph()
    card = generate_card(store, _config(tmp_path), tmp_path)
    assert card.name == "test-graph" and card.revision == 1
    assert len(card.history) == 1 and card.history[0].note == "initial"
    assert card.stats["nodes"] == 3 and card.stats["max_level"] == 2
    assert card.stats["levels"] == {"0": 1, "1": 1, "2": 1}
    assert card.stats["seeds"] == ["Root"]


def test_idempotent_when_unchanged(tmp_path):
    store, *_ = _store_with_graph()
    cfg = _config(tmp_path)
    first = write_card(store, cfg, tmp_path)
    second = generate_card(store, cfg, tmp_path)
    assert second.revision == 1 and len(second.history) == 1
    assert second.updated_at == first.updated_at


def test_incremental_expansion_bumps_revision_and_delta(tmp_path):
    store, root, child, leaf = _store_with_graph()
    cfg = _config(tmp_path)
    first = write_card(store, cfg, tmp_path)

    new = store.add_node(Node.create("Grandchild"))
    store.add_edge(Edge(parent=leaf.id, child=new.id))
    card = write_card(store, cfg, tmp_path)

    assert card.revision == 2
    assert card.created_at == first.created_at  # preserved across updates
    assert len(card.history) == 2
    assert card.history[-1].delta_nodes == 1 and card.history[-1].delta_edges == 1
    assert card.stats["max_level"] == 3
    persisted = GraphCard.model_validate_json((tmp_path / CARD_JSON).read_text())
    assert persisted.revision == 2


def test_notes_reflect_graph_state(tmp_path):
    store, *_ = _store_with_graph()
    cfg = _config(tmp_path)
    stats = compute_stats(store)
    notes = " ".join(compute_notes(store, stats, cfg))
    assert "尚未跑 Stage C 检索" in notes  # zero-materials caveat
    assert "queue_policy=bfs" in notes
    for n in store.nodes():
        n.stats.n_materials = 3
    notes2 = " ".join(compute_notes(store, compute_stats(store), cfg))
    assert "尚未跑 Stage C 检索" not in notes2  # caveat disappears once real


def test_render_and_files(tmp_path):
    store, *_ = _store_with_graph()
    card = write_card(store, _config(tmp_path), tmp_path)
    md = render_markdown(card)
    assert "rev 1" in md and "Root" in md and "Δ节点" in md
    assert (tmp_path / CARD_JSON).exists() and (tmp_path / CARD_MD).exists()
    # corrupt card file must not crash regeneration
    (tmp_path / CARD_JSON).write_text("{broken")
    recovered = generate_card(store, _config(tmp_path), tmp_path)
    assert recovered.revision == 1


def test_rename_is_recorded_without_fake_growth(tmp_path):
    store, *_ = _store_with_graph()
    write_card(store, _config(tmp_path, name="old-name"), tmp_path)
    card = write_card(store, _config(tmp_path, name="new-name"), tmp_path)
    assert card.name == "new-name" and card.revision == 2
    assert card.history[-1].note == "renamed"
    assert card.history[-1].delta_nodes == 0  # no fake growth on rename
