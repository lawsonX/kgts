"""Stage C driver: build queries per node, fan out to sources, post-process."""

from __future__ import annotations

from kgts.config import RetrieveConfig
from kgts.graph.store import GraphStore
from kgts.models import Material, SampleBundle, SampleIntent, SourceType
from kgts.retrieve.postprocess import dedup, filter_license, rerank
from kgts.retrieve.query import QueryBuilder
from kgts.retrieve.sources import MaterialSource


class Retriever:
    """Retrieve and post-process materials for a sample bundle."""

    def __init__(self, sources: dict[str, MaterialSource], config: RetrieveConfig) -> None:
        self.sources = sources
        self.config = config

    def retrieve(self, store: GraphStore, bundle: SampleBundle) -> list[Material]:
        group_labels = None
        if bundle.intent == SampleIntent.JOINT:
            group_labels = [self._label(store, nid) for nid in bundle.nodes]

        collected: list[Material] = []
        errors: list[RuntimeError] = []
        succeeded = 0
        for node_id in bundle.nodes:
            label = self._label(store, node_id)
            path = bundle.ancestor_paths.get(node_id) or self._path(store, node_id)
            for name, source in self.sources.items():
                queries = QueryBuilder.build(
                    label, path, bundle.intent, SourceType(name), group_labels=group_labels
                )
                try:
                    materials = source.search(queries, self.config.per_node_materials)
                except RuntimeError as e:
                    errors.append(e)  # e.g. missing API key; other sources may still work
                    continue
                succeeded += 1
                for m in materials:
                    if node_id not in m.linked_nodes:
                        m.linked_nodes.append(node_id)
                collected.extend(materials)

        if succeeded == 0 and errors:
            raise errors[0]  # every source failed: surface the error instead of silence

        post = self.config.postprocess
        out = filter_license(collected, post.license_mode)
        out = dedup(out, post.dedup)
        if post.rerank != "none":
            # cross_encoder / llm rerankers are config-level names; the offline
            # default falls back to token-overlap rerank against node descriptions.
            description = " ".join(self._description(store, nid) for nid in bundle.nodes)
            out = rerank(out, description)
        return out

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _label(store: GraphStore, node_id: str) -> str:
        try:
            return store.get(node_id).label
        except KeyError:
            return node_id

    @staticmethod
    def _description(store: GraphStore, node_id: str) -> str:
        try:
            return store.get(node_id).description
        except KeyError:
            return ""

    @staticmethod
    def _path(store: GraphStore, node_id: str) -> list[str]:
        try:
            return store.ancestor_path(node_id)
        except KeyError:
            return []
