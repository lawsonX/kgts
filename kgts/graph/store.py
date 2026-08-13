"""Knowledge DAG storage: NetworkX in-memory graph + SQLite persistence.

Invariants enforced on write (design doc section 3):
- the ``is_subconcept`` subgraph is acyclic (checked before every edge insert);
- ``child.level > parent.level`` is a soft constraint -- violations are allowed
  (the DAG may legitimately have multi-parent shortcuts) but recorded in
  ``review_flags`` for the human review queue;
- ``merged`` nodes only serve the alias index and are excluded from sampling.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import networkx as nx

from kgts.models import Edge, Node, NodeStatus, Relation, make_node_id

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
  id TEXT PRIMARY KEY, label TEXT NOT NULL, aliases JSON NOT NULL,
  level INT NOT NULL, description TEXT, status TEXT NOT NULL,
  provenance JSON NOT NULL, stats JSON NOT NULL, embedding BLOB
);
CREATE TABLE IF NOT EXISTS edges (
  parent TEXT NOT NULL, child TEXT NOT NULL, relation TEXT NOT NULL,
  confidence REAL NOT NULL, PRIMARY KEY (parent, child, relation)
);
CREATE INDEX IF NOT EXISTS idx_edges_child ON edges(child);
"""


class CycleError(ValueError):
    pass


class GraphStore:
    def __init__(self) -> None:
        self._g = nx.DiGraph()  # edge attr: relation, confidence
        self._nodes: dict[str, Node] = {}
        self._alias_index: dict[str, str] = {}  # normalized alias -> node id
        self.review_flags: list[dict] = []  # soft-constraint violations

    # ------------------------------------------------------------------ read
    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def nodes(self, *, include_merged: bool = False) -> list[Node]:
        return [
            n
            for n in self._nodes.values()
            if include_merged or n.status != NodeStatus.MERGED
        ]

    def edges(self) -> list[Edge]:
        return [
            Edge(parent=u, child=v, relation=Relation(d["relation"]), confidence=d["confidence"])
            for u, v, d in self._g.edges(data=True)
        ]

    @staticmethod
    def _norm(label: str) -> str:
        return " ".join(label.lower().split())

    def find(self, label: str) -> Node | None:
        """Exact lookup by canonical label or any known alias."""
        node_id = self._alias_index.get(self._norm(label))
        return self._nodes.get(node_id) if node_id else None

    def parents(self, node_id: str, *, subconcept_only: bool = True) -> list[str]:
        out = []
        for u, _, d in self._g.in_edges(node_id, data=True):
            if subconcept_only and d["relation"] != Relation.IS_SUBCONCEPT.value:
                continue
            out.append(u)
        return out

    def children(self, node_id: str, *, subconcept_only: bool = True) -> list[str]:
        out = []
        for _, v, d in self._g.out_edges(node_id, data=True):
            if subconcept_only and d["relation"] != Relation.IS_SUBCONCEPT.value:
                continue
            out.append(v)
        return out

    def ancestors(self, node_id: str) -> list[str]:
        """All is_subconcept ancestors (coarse concepts above this node)."""
        sub = self._subconcept_view()
        return [a for a in nx.ancestors(sub, node_id)] if node_id in sub else []

    def ancestor_path(self, node_id: str) -> list[str]:
        """Shortest root->node path as a list of labels (disambiguation context)."""
        sub = self._subconcept_view()
        if node_id not in sub:
            return []
        roots = [n for n in sub if sub.in_degree(n) == 0]
        best: list[str] = []
        for r in roots:
            try:
                path = nx.shortest_path(sub, r, node_id)
            except nx.NetworkXNoPath:
                continue
            if not best or len(path) < len(best):
                best = path
        return [self._nodes[n].label for n in best]

    def atomic_nodes(self) -> list[Node]:
        return [n for n in self.nodes() if n.status == NodeStatus.ATOMIC]

    def _subconcept_view(self) -> nx.DiGraph:
        edges = [
            (u, v)
            for u, v, d in self._g.edges(data=True)
            if d["relation"] == Relation.IS_SUBCONCEPT.value
        ]
        view = nx.DiGraph()
        view.add_nodes_from(self._nodes)
        view.add_edges_from(edges)
        return view

    # ----------------------------------------------------------------- write
    def add_node(self, node: Node) -> Node:
        existing = self.find(node.label)
        if existing is not None:
            return existing  # idempotent: same canonical label -> same node
        self._nodes[node.id] = node
        self._g.add_node(node.id)
        self._index_aliases(node)
        return node

    def add_edge(self, edge: Edge) -> None:
        if edge.parent not in self._nodes or edge.child not in self._nodes:
            raise KeyError("edge endpoints must exist in the graph")
        if edge.relation == Relation.IS_SUBCONCEPT:
            view = self._subconcept_view()
            view.add_edge(edge.parent, edge.child)
            if not nx.is_directed_acyclic_graph(view):
                raise CycleError(f"edge {edge.parent} -> {edge.child} would create a cycle")
        self._g.add_edge(
            edge.parent,
            edge.child,
            relation=edge.relation.value,
            confidence=edge.confidence,
        )
        if edge.relation == Relation.IS_SUBCONCEPT:
            self._recompute_levels()

    def merge(self, duplicate_id: str, canonical_id: str, *, alias: str | None = None) -> None:
        """Fold ``duplicate_id`` into ``canonical_id`` (alignment EQUIVALENT)."""
        dup, canon = self._nodes[duplicate_id], self._nodes[canonical_id]
        dup.status = NodeStatus.MERGED
        extra = [alias, dup.label, *dup.aliases]
        for a in extra:
            if a and self._norm(a) != self._norm(canon.label) and a not in canon.aliases:
                canon.aliases.append(a)
                self._alias_index[self._norm(a)] = canon.id
        # re-point children of the duplicate to the canonical node
        for child in self.children(duplicate_id):
            d = self._g.edges[duplicate_id, child]
            self._g.remove_edge(duplicate_id, child)
            if not self._g.has_edge(canonical_id, child):
                self._g.add_edge(canonical_id, child, **d)
        self._recompute_levels()

    def _index_aliases(self, node: Node) -> None:
        self._alias_index[self._norm(node.label)] = node.id
        for a in node.aliases:
            self._alias_index[self._norm(a)] = node.id

    def _recompute_levels(self) -> None:
        """level = shortest-path depth from the nearest root (seed), over
        is_subconcept edges. Soft constraint violations are flagged."""
        sub = self._subconcept_view()
        roots = [n for n in sub if sub.in_degree(n) == 0]
        for nid, node in self._nodes.items():
            if node.status == NodeStatus.MERGED:
                continue
            dists = []
            for r in roots:
                try:
                    dists.append(nx.shortest_path_length(sub, r, nid))
                except nx.NetworkXNoPath:
                    continue
            node.level = min(dists) if dists else 0
        for u, v in sub.edges():
            if self._nodes[v].status == NodeStatus.MERGED:
                continue
            if self._nodes[v].level <= self._nodes[u].level:
                flag = {
                    "kind": "level_soft_violation",
                    "parent": u,
                    "child": v,
                    "parent_level": self._nodes[u].level,
                    "child_level": self._nodes[v].level,
                }
                if flag not in self.review_flags:
                    self.review_flags.append(flag)

    # ----------------------------------------------------------- persistence
    def save(self, db_path: str | Path) -> None:
        conn = sqlite3.connect(str(db_path))
        try:
            with conn:
                conn.executescript(_SCHEMA)
                conn.execute("DELETE FROM nodes")
                conn.execute("DELETE FROM edges")
                for n in self._nodes.values():
                    conn.execute(
                        "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            n.id,
                            n.label,
                            json.dumps(n.aliases, ensure_ascii=False),
                            n.level,
                            n.description,
                            n.status.value,
                            n.model_dump_json(),
                            n.stats.model_dump_json(),
                            json.dumps(n.embedding) if n.embedding else None,
                        ),
                    )
                for e in self.edges():
                    conn.execute(
                        "INSERT INTO edges VALUES (?,?,?,?)",
                        (e.parent, e.child, e.relation.value, e.confidence),
                    )
        finally:
            conn.close()

    @classmethod
    def load(cls, db_path: str | Path) -> GraphStore:
        store = cls()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(_SCHEMA)
            for row in conn.execute("SELECT id, provenance FROM nodes"):
                node = Node.model_validate_json(row[1])
                store._nodes[node.id] = node
                store._g.add_node(node.id)
                store._index_aliases(node)
            for parent, child, relation, confidence in conn.execute(
                "SELECT parent, child, relation, confidence FROM edges"
            ):
                store._g.add_edge(parent, child, relation=relation, confidence=confidence)
        finally:
            conn.close()
        store._recompute_levels()
        return store


def node_id_for(label: str) -> str:
    """Convenience re-export used by pipeline stages."""
    return make_node_id(label)
