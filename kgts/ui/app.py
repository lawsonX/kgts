"""Gradio workdir viewer (design §9): graph, runs, and review queue.

Gradio is an optional dependency -- imported lazily so the rest of the
package works without it (``pip install 'kgts[ui]'``).
"""

from __future__ import annotations

from pathlib import Path


def _import_gradio():
    try:
        import gradio as gr
    except ImportError as e:
        raise RuntimeError("the UI requires gradio: pip install 'kgts[ui]'") from e
    return gr


def _load_view(workdir: Path):
    """Read graph.db / artifacts.db into plain table rows (missing db -> empty)."""
    nodes_rows: list[list] = []
    edges_rows: list[list] = []
    flags: list[dict] = []
    runs_rows: list[list] = []

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

        for r in ArtifactStore(artifacts_db).list_runs():
            runs_rows.append([r.id, r.started_at, r.finished_at, r.config_hash])

    return nodes_rows, edges_rows, runs_rows, flags


def build_app(workdir: str | Path):
    """Build (but do not launch) the Gradio Blocks app for a run workdir."""
    gr = _import_gradio()
    workdir = Path(workdir)
    nodes_rows, edges_rows, runs_rows, flags = _load_view(workdir)

    with gr.Blocks(title=f"KGTS - {workdir}") as app:
        gr.Markdown(f"# KGTS workdir viewer\n`{workdir}`")
        refresh = gr.Button("Refresh", variant="primary")
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
            return _load_view(workdir)

        refresh.click(
            _refresh,
            outputs=[nodes_table, edges_table, runs_table, flags_json],
        )
    return app


def main(workdir: str = ".kgts", port: int = 7860) -> None:
    """Launch the viewer for ``workdir`` on the given port."""
    build_app(workdir).launch(server_port=port)
