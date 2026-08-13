"""Stage C query construction: ancestor-context disambiguation (design doc section 6.1).

The leaf label alone is often ambiguous across domains ("kernel" in OS vs GPU
vs ML). Queries therefore combine the leaf with its 1-2 nearest ancestor
labels, and each material source gets its own query shape.
"""

from __future__ import annotations

from kgts.models import SampleIntent, SourceType


class QueryBuilder:
    """Deterministic query builder for a graph node."""

    @staticmethod
    def build(
        node_label: str,
        ancestor_path: list[str],
        intent: SampleIntent,
        source_type: SourceType,
        group_labels: list[str] | None = None,
    ) -> list[str]:
        """Build source-shaped queries for ``node_label``.

        ``ancestor_path`` is the root->node label path (the node's own label is
        dropped if present). ``group_labels`` is only used for JOINT bundles.
        """
        label = node_label.strip()
        path = [
            p.strip() for p in ancestor_path if p.strip() and p.strip().lower() != label.lower()
        ]
        context = path[-2:]  # 1-2 nearest ancestors for disambiguation
        disambiguated = " ".join([*context, label])

        if source_type == SourceType.WEB:
            queries = [disambiguated, f"{disambiguated} tutorial"]
            if context:
                queries.append(f"{label} in {context[-1]} guide")
        elif source_type == SourceType.GITHUB:
            joined = " ".join([*path, label])
            topic_src = context[-1] if context else label
            topic = "-".join(topic_src.lower().split())
            queries = [joined, f"{label} topic:{topic}"]
        elif source_type == SourceType.ARXIV:
            queries = [f'"{disambiguated}"']
        elif source_type == SourceType.LOCAL:
            queries = [" ".join([*path, label])]
        else:
            raise ValueError(f"unknown source type: {source_type!r}")

        if intent == SampleIntent.JOINT and group_labels:
            combined = " AND ".join(dict.fromkeys(g.strip() for g in group_labels if g.strip()))
            if combined:
                queries.append(combined)
        return [q for q in queries if q]
