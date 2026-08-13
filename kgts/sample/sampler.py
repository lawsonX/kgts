"""Stage B: sampling scheduler (design doc section 5).

Three operators over non-merged nodes, mixed per ``config.mixture``:

- breadth: shallow nodes (level <= median level) -> overview tasks;
- depth:   atomic nodes (fallback: leaves of the is_subconcept DAG) -> long tail;
- joint:   sibling groups sharing a common parent (size 2..max_group_size).

Weighting is inverse-frequency via ``InverseFrequencyPrioritizer``; a first
coverage pass gives each eligible node one pick before repeats (best effort,
subject to operator pools); ``quotas.max_per_node`` caps repeats per node
within one run. Deterministic for a fixed ``seed``.
"""

from __future__ import annotations

import itertools
import random
import statistics
from collections.abc import Callable

from kgts.config import SampleConfig
from kgts.graph.store import GraphStore
from kgts.models import SampleBundle, SampleIntent
from kgts.sample.prioritizer import InverseFrequencyPrioritizer

_INTENT_KEYS = ("breadth", "depth", "joint")


def sample_bundles(store: GraphStore, config: SampleConfig, seed: int = 42) -> list[SampleBundle]:
    rng = random.Random(seed)
    nodes = store.nodes()  # merged nodes are excluded (alias-index only)
    if not nodes or config.n_samples <= 0:
        return []
    prioritizer = InverseFrequencyPrioritizer(alpha=config.quotas.long_tail_alpha)

    def weight(node_id: str) -> float:
        return prioritizer.weight(store.get(node_id))

    pools = {
        SampleIntent.BREADTH: _breadth_pool(nodes),
        SampleIntent.DEPTH: _depth_pool(store, nodes),
        SampleIntent.JOINT: _joint_pool(store, nodes, config.joint.max_group_size),
    }
    quotas = _split(config.n_samples, config.mixture)

    counts: dict[str, int] = {}  # per-node picks within this run (cap enforcement)
    chosen: list[tuple[SampleIntent, tuple[str, ...]]] = []
    for intent in (SampleIntent.BREADTH, SampleIntent.DEPTH, SampleIntent.JOINT):
        for unit in _draw(rng, pools[intent], quotas[intent], weight, counts,
                          config.quotas.max_per_node):
            chosen.append((intent, unit))

    bundles: list[SampleBundle] = []
    for intent, unit in chosen:
        bundles.append(
            SampleBundle(
                nodes=list(unit),
                ancestor_paths={nid: store.ancestor_path(nid) for nid in unit},
                level=max(store.get(nid).level for nid in unit),
                intent=intent,
            )
        )
        for nid in unit:
            store.get(nid).stats.times_sampled += 1
    return bundles


# --------------------------------------------------------------------- pools
def _breadth_pool(nodes: list) -> list[tuple[str, ...]]:
    median_level = statistics.median(n.level for n in nodes)
    return [(n.id,) for n in nodes if n.level <= median_level]


def _depth_pool(store: GraphStore, nodes: list) -> list[tuple[str, ...]]:
    atomic = store.atomic_nodes()
    pool = atomic if atomic else [n for n in nodes if not store.children(n.id)]
    return [(n.id,) for n in pool]


def _joint_pool(store: GraphStore, nodes: list, max_size: int) -> list[tuple[str, ...]]:
    """Sibling groups: combinations of nodes sharing a common is_subconcept parent."""
    node_ids = {n.id for n in nodes}
    groups: set[tuple[str, ...]] = set()
    for n in nodes:
        kids = sorted(c for c in store.children(n.id) if c in node_ids)
        for size in range(2, min(max_size, len(kids)) + 1):
            groups.update(itertools.combinations(kids, size))
    return sorted(groups)


# ----------------------------------------------------------------- internals
def _split(n_samples: int, mixture: dict[str, float]) -> dict[SampleIntent, int]:
    mix = {k: max(0.0, float(mixture.get(k, 0.0))) for k in _INTENT_KEYS}
    total = sum(mix.values()) or 1.0
    n_breadth = int(n_samples * mix["breadth"] / total)
    n_depth = int(n_samples * mix["depth"] / total)
    return {
        SampleIntent.BREADTH: n_breadth,
        SampleIntent.DEPTH: n_depth,
        SampleIntent.JOINT: max(0, n_samples - n_breadth - n_depth),
    }


def _under_cap(unit: tuple[str, ...], counts: dict[str, int], cap: int) -> bool:
    return all(counts.get(nid, 0) < cap for nid in unit)


def _draw(
    rng: random.Random,
    pool: list[tuple[str, ...]],
    n_picks: int,
    weight: Callable[[str], float],
    counts: dict[str, int],
    cap: int,
) -> list[tuple[str, ...]]:
    picked: list[tuple[str, ...]] = []
    if n_picks <= 0 or not pool:
        return picked
    # coverage pass: one pick per eligible node before any repeats
    covered: set[str] = set()
    order = list(pool)
    rng.shuffle(order)
    for unit in order:
        if len(picked) >= n_picks:
            break
        if any(nid in covered for nid in unit) or not _under_cap(unit, counts, cap):
            continue
        picked.append(unit)
        for nid in unit:
            counts[nid] = counts.get(nid, 0) + 1
            covered.add(nid)
    # weighted fill for the remaining picks
    while len(picked) < n_picks:
        avail = [u for u in pool if _under_cap(u, counts, cap)]
        if not avail:
            break  # every node hit the per-node cap; quota is lost best-effort
        unit = _weighted_choice(rng, avail, [sum(weight(nid) for nid in u) for u in avail])
        picked.append(unit)
        for nid in unit:
            counts[nid] = counts.get(nid, 0) + 1
    return picked


def _weighted_choice(
    rng: random.Random, items: list[tuple[str, ...]], weights: list[float]
) -> tuple[str, ...]:
    total = sum(weights)
    if total <= 0:
        return rng.choice(items)
    threshold = rng.random() * total
    acc = 0.0
    for item, w in zip(items, weights, strict=True):
        acc += w
        if threshold <= acc:
            return item
    return items[-1]
