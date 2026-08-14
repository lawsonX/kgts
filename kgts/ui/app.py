"""Gradio workdir viewer (design §9): interactive graph, tasks, runs, review queue.

Gradio and pyvis are optional dependencies -- imported lazily so the rest of
the package works without them (``pip install 'kgts[ui]'``). The graph view
embeds vis-network inline so it renders fully offline.
"""

from __future__ import annotations

import json
from pathlib import Path

_LEVEL_PALETTE = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4", "#74add1", "#313695"]


def _import_gradio():
    try:
        import gradio as gr
    except ImportError as e:
        raise RuntimeError("the UI requires gradio: pip install 'kgts[ui]'") from e
    return gr


def _graph_html(workdir: Path) -> str:
    """Interactive node-link diagram of graph.db (pyvis, inline JS = offline)."""
    graph_db = workdir / "graph.db"
    if not graph_db.exists():
        return "<p>no graph.db in this workdir — run <code>kgts build</code> first.</p>"
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p>graph view needs pyvis: <code>pip install 'kgts[ui]'</code></p>"
    from kgts.graph.store import GraphStore

    store = GraphStore.load(graph_db)
    net = Network(
        height="720px", width="100%", directed=True, bgcolor="#ffffff",
        cdn_resources="in_line",
    )
    for n in store.nodes():
        color = _LEVEL_PALETTE[min(n.level, len(_LEVEL_PALETTE) - 1)]
        atomic = n.status.value == "atomic"
        title = (
            f"<b>{n.label}</b><br>level {n.level} · {n.status.value}"
            f"<br>materials={n.stats.n_materials} · sampled={n.stats.times_sampled}"
            f"<br>{n.description[:300]}"
        )
        net.add_node(
            n.id,
            label=n.label,
            title=title,
            color={"background": color, "border": "#222222" if atomic else color},
            borderWidth=3 if atomic else 1,
            size=8 + 3 * len(store.children(n.id)),
        )
    for e in store.edges():
        net.add_edge(
            e.parent, e.child, arrows="to",
            dashes=e.relation.value == "is_related", color="#999999",
        )
    net.set_options(
        json.dumps(
            {
                "interaction": {"hover": True, "tooltipDelay": 100},
                "physics": {
                    "solver": "forceAtlas2Based",
                    "forceAtlas2Based": {"gravitationalConstant": -40, "springLength": 80},
                    "stabilization": {"iterations": 200},
                },
                "edges": {"smooth": {"type": "continuous"}},
            }
        )
    )
    return net.generate_html()


def _graph_panel_html(graph_html: str) -> str:
    """Wrap the pyvis page in an iframe: gr.HTML does not execute <script>
    tags inserted via innerHTML, but an iframe srcdoc is parsed as a full
    document, so the vis-network JS runs."""
    import html as _html

    doc = _html.escape(graph_html, quote=True)
    return (
        f'<iframe srcdoc="{doc}" '
        'style="width:100%;height:780px;border:1px solid #ddd;border-radius:8px"></iframe>'
    )


def _load_view(workdir: Path):
    """Read graph.db / artifacts.db into plain table rows (missing db -> empty)."""
    nodes_rows: list[list] = []
    edges_rows: list[list] = []
    flags: list[dict] = []
    runs_rows: list[list] = []
    tasks = []

    graph_db = workdir / "graph.db"
    if graph_db.exists():
        from kgts.graph.store import GraphStore

        store = GraphStore.load(graph_db)
        for n in store.nodes():
            nodes_rows.append(
                [n.id, n.label, n.level, n.status.value, n.stats.times_sampled]
            )
        for e in store.edges():
            edges_rows.append([e.parent, e.child, e.relation.value, e.confidence])
        flags = store.review_flags

    artifacts_db = workdir / "artifacts.db"
    if artifacts_db.exists():
        from kgts.orchestrate.store import ArtifactStore

        artifacts = ArtifactStore(artifacts_db)
        for r in artifacts.list_runs():
            runs_rows.append([r.id, r.started_at, r.finished_at, r.config_hash])
        tasks = artifacts.load_tasks()

    return nodes_rows, edges_rows, runs_rows, flags, tasks


def _task_label(t) -> str:
    return f"{t.id} [{t.task_type}/{t.verify_result.value}] {t.question[:36]}"


def _task_detail(t) -> tuple[str, str]:
    md = (
        f"### {t.task_type} · {t.verify_result.value}\n\n"
        f"**Q** {t.question}\n\n**A** {t.answer}\n\n"
        f"**rubric**\n" + "\n".join(f"- {r}" for r in t.rubric)
    )
    provenance = json.dumps(
        {
            "task_id": t.id,
            "verifier": t.verifier,
            "materials": t.materials,
            "bundle_nodes": t.sample_bundle.nodes,
            "ancestor_paths": t.sample_bundle.ancestor_paths,
            "intent": t.sample_bundle.intent.value,
            "style": t.style,
            "run_id": t.run_id,
        },
        ensure_ascii=False,
        indent=2,
    )
    return md, provenance


def build_app(workdir: str | Path):
    """Build (but do not launch) the Gradio Blocks app for a run workdir."""
    gr = _import_gradio()
    workdir = Path(workdir)
    nodes_rows, edges_rows, runs_rows, flags, tasks = _load_view(workdir)
    graph_html = _graph_panel_html(_graph_html(workdir))

    with gr.Blocks(title=f"KGTS - {workdir}") as app:
        gr.Markdown(f"# KGTS workdir viewer\n`{workdir}`")
        refresh = gr.Button("Refresh", variant="primary")
        with gr.Tab("Graph"):
            gr.Markdown(
                "节点颜色 = 层级（红粗 → 蓝细）；粗边框 = atomic；虚线 = is_related。"
                "悬停看详情，拖拽/缩放浏览。"
            )
            graph_panel = gr.HTML(value=graph_html)
        with gr.Tab("Tasks"):
            type_filter = gr.Dropdown(
                choices=["all"] + sorted({t.task_type for t in tasks}),
                value="all", label="task_type",
            )
            result_filter = gr.Dropdown(
                choices=["all"] + sorted({t.verify_result.value for t in tasks}),
                value="all", label="verify_result",
            )
            task_pick = gr.Dropdown(choices=[_task_label(t) for t in tasks], label="task")
            task_md = gr.Markdown()
            task_prov = gr.Code(language="json", label="provenance")

            def _filtered(tasks_now, type_sel, result_sel):
                picked = [
                    t for t in tasks_now
                    if (type_sel == "all" or t.task_type == type_sel)
                    and (result_sel == "all" or t.verify_result.value == result_sel)
                ]
                labels = [_task_label(t) for t in picked]
                return gr.update(choices=labels, value=labels[0] if labels else None)

            def _show(tasks_now, label):
                if not label:
                    return "", ""
                task = next(t for t in tasks_now if _task_label(t) == label)
                return _task_detail(task)

            tasks_state = gr.State(tasks)
            for ctl in (type_filter, result_filter):
                ctl.change(_filtered, [tasks_state, type_filter, result_filter], task_pick)
            task_pick.change(_show, [tasks_state, task_pick], [task_md, task_prov])
        with gr.Tab("Nodes"):
            nodes_table = gr.Dataframe(
                headers=["id", "label", "level", "status", "times_sampled"],
                value=nodes_rows,
            )
        with gr.Tab("Edges"):
            edges_table = gr.Dataframe(
                headers=["parent", "child", "relation", "confidence"],
                value=edges_rows,
            )
        with gr.Tab("Runs"):
            runs_table = gr.Dataframe(
                headers=["id", "started_at", "finished_at", "config_hash"],
                value=runs_rows,
            )
        with gr.Tab("Review queue"):
            flags_json = gr.JSON(value=flags)

        def _refresh():
            rows = _load_view(workdir)
            panel = _graph_panel_html(_graph_html(workdir))
            return rows[0], rows[1], rows[2], rows[3], rows[4], panel

        refresh.click(
            _refresh,
            outputs=[nodes_table, edges_table, runs_table, flags_json, tasks_state, graph_panel],
        )
    return app


def main(workdir: str = ".kgts", port: int = 7860, host: str = "127.0.0.1") -> None:
    """Launch the viewer for ``workdir`` on the given port."""
    build_app(workdir).launch(server_name=host, server_port=port)
