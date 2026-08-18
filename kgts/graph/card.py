"""Graph card: every DAG ships with a name card (design §9a).

The card is the artifact's self-description for downstream consumers (human
or agent): identity, current stats, build history, and auto-computed usage
notes. ``graph_card.json`` (authoritative, lives next to ``graph.db`` in the
workdir) is regenerated after every graph-mutating stage; ``graph_card.md``
is the rendered copy.

Incremental semantics: regeneration is idempotent when nothing changed;
when the graph did change (expansion, merges, material write-back), the
revision bumps, ``created_at`` is preserved, and the delta is appended to
the history (capped).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from kgts.config import Config
from kgts.graph.store import GraphStore
from kgts.models import utc_now

CARD_JSON = "graph_card.json"
CARD_MD = "graph_card.md"
_HISTORY_CAP = 50


class BuildEvent(BaseModel):
    at: str
    nodes: int
    edges: int
    delta_nodes: int = 0
    delta_edges: int = 0
    note: str = ""


class GraphCard(BaseModel):
    name: str
    revision: int = 1
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    config_hash: str = ""
    build_config: dict[str, Any] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    history: list[BuildEvent] = Field(default_factory=list)


def compute_stats(store: GraphStore) -> dict[str, Any]:
    nodes = store.nodes()
    levels = Counter(n.level for n in nodes)
    desc_by_level: dict[int, list[int]] = {}
    for n in nodes:
        d = desc_by_level.setdefault(n.level, [0, 0])
        d[1] += 1
        if n.description:
            d[0] += 1
    return {
        "nodes": len(nodes),
        "edges": len(store.edges()),
        "levels": {str(k): levels[k] for k in sorted(levels)},
        "max_level": max(levels, default=0),
        "status": dict(Counter(n.status.value for n in nodes)),
        "seeds": [n.label for n in nodes if n.level == 0],
        "desc_coverage": {
            str(k): f"{v[0]}/{v[1]}" for k, v in sorted(desc_by_level.items())
        },
        "nodes_with_materials": sum(1 for n in nodes if n.stats.n_materials > 0),
        "review_flags": len(store.review_flags),
    }


def compute_notes(store: GraphStore, stats: dict[str, Any], config: Config) -> list[str]:
    """Usage caveats derived from the graph's actual state (not boilerplate)."""
    notes: list[str] = []
    max_level = stats.get("max_level", 0)
    if max_level:
        notes.append(
            f"L{max_level} 叶节点按设计无描述文本（终端采样目标）；消费时直接使用 "
            "label + ancestor_path，描述不是必需项。"
        )
    if stats["nodes"] and stats["nodes_with_materials"] == 0:
        notes.append(
            "节点 stats.n_materials 全为 0：本图谱尚未跑 Stage C 检索，材料计数未知，"
            "不代表'没有材料'。"
        )
    if stats["review_flags"]:
        notes.append(f"存在 {stats['review_flags']} 条复审旗标，用 `kgts review` 查看。")
    policy = config.build.queue_policy
    notes.append(
        f"构建策略：queue_policy={policy}, max_depth={config.build.atomicity.max_depth}, "
        f"扇出上限 {config.build.max_children_per_node}。"
    )
    return notes


def generate_card(store: GraphStore, config: Config, workdir: str | Path) -> GraphCard:
    """Build the updated card for the current graph state (idempotent)."""
    workdir = Path(workdir)
    path = workdir / CARD_JSON
    stats = compute_stats(store)
    prev: GraphCard | None = None
    if path.exists():
        try:
            prev = GraphCard.model_validate_json(path.read_text())
        except Exception:
            prev = None  # corrupt card: start fresh, don't crash the pipeline

    build_config = {
        "queue_policy": config.build.queue_policy,
        "max_depth": config.build.atomicity.max_depth,
        "max_children_per_node": config.build.max_children_per_node,
        "budget_max_nodes": config.budget.max_nodes,
    }
    if (
        prev
        and prev.stats == stats
        and prev.build_config == build_config
        and prev.name == config.run.name
    ):
        return prev  # nothing changed: do not touch the file

    renamed = (
        prev is not None
        and prev.stats == stats
        and prev.build_config == build_config
        and prev.name != config.run.name
    )
    event = BuildEvent(
        at=utc_now(),
        nodes=stats["nodes"],
        edges=stats["edges"],
        delta_nodes=stats["nodes"] - (prev.stats["nodes"] if prev else 0),
        delta_edges=stats["edges"] - (prev.stats["edges"] if prev else 0),
        note="initial" if prev is None else ("renamed" if renamed else "updated"),
    )
    history = (prev.history if prev else []) + [event]
    return GraphCard(
        name=config.run.name,
        revision=(prev.revision + 1) if prev else 1,
        created_at=prev.created_at if prev else event.at,
        updated_at=event.at,
        config_hash=config.config_hash(),
        build_config=build_config,
        stats=stats,
        notes=compute_notes(store, stats, config),
        history=history[-_HISTORY_CAP:],
    )


def render_markdown(card: GraphCard) -> str:
    lines = [
        f"# {card.name}（图谱名片 rev {card.revision}）",
        "",
        "> 本文件由 KGTS 自动生成（`graph_card.json` 为权威源），请勿手改。",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 节点/边 | {card.stats['nodes']} / {card.stats['edges']} |",
        "| 层级分布 | "
        + " → ".join(f"L{k}:{v}" for k, v in card.stats["levels"].items())
        + " |",
        f"| 种子域 | {' · '.join(card.stats['seeds'])} |",
        "| 描述覆盖 | "
        + ", ".join(f"L{k} {v}" for k, v in card.stats["desc_coverage"].items())
        + " |",
        f"| 含材料节点 | {card.stats['nodes_with_materials']} |",
        f"| 构建于 | {card.created_at}（更新 {card.updated_at}） |",
        f"| config_hash | {card.config_hash} |",
        "",
        "## 使用注意",
        "",
        *[f"- {n}" for n in card.notes],
        "",
        "## 消费入口",
        "",
        "- 契约文档：`docs/consuming.md`；agent skill：`skills/kgts-graph-consumption/SKILL.md`",
        "- Python：`from kgts.api import load_graph; store = load_graph(<workdir>)`",
        "- 可视化：`kgts serve --config <config> --host 0.0.0.0 --port 7861`",
        "",
        "## 构建历史",
        "",
        "| rev | 时间 | 节点 | 边 | Δ节点 | Δ边 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i, ev in enumerate(card.history, start=1):
        lines.append(
            f"| {i} | {ev.at} | {ev.nodes} | {ev.edges} "
            f"| {ev.delta_nodes:+d} | {ev.delta_edges:+d} |"
        )
    return "\n".join(lines) + "\n"


def write_card(store: GraphStore, config: Config, workdir: str | Path) -> GraphCard:
    """Regenerate (if changed) and persist graph_card.json + graph_card.md."""
    workdir = Path(workdir)
    card = generate_card(store, config, workdir)
    json_path = workdir / CARD_JSON
    if not json_path.exists() or GraphCard.model_validate_json(
        json_path.read_text()
    ) != card:
        json_path.write_text(card.model_dump_json(indent=2))
    (workdir / CARD_MD).write_text(render_markdown(card))
    return card
