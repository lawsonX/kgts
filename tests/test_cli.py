"""Offline CLI smoke tests via typer's CliRunner."""

from __future__ import annotations

from typer.testing import CliRunner

from kgts.cli import app
from kgts.graph.store import GraphStore
from kgts.models import Edge, Node

runner = CliRunner()


def _write_config(tmp_path, workdir) -> str:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"run:\n  workdir: {workdir}\n")
    return str(cfg)


def _make_graph(workdir, n_nodes: int = 3) -> None:
    store = GraphStore()
    nodes = [store.add_node(Node.create(f"Concept {i}")) for i in range(n_nodes)]
    for child in nodes[1:]:
        store.add_edge(Edge(parent=nodes[0].id, child=child.id))
    workdir.mkdir(parents=True, exist_ok=True)
    store.save(workdir / "graph.db")


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("build", "sample", "retrieve", "synth", "export", "report", "run", "graph"):
        assert cmd in result.output


def test_graph_command_prints_counts(tmp_path):
    workdir = tmp_path / ".kgts"
    _make_graph(workdir, n_nodes=4)
    cfg = _write_config(tmp_path, workdir)
    result = runner.invoke(app, ["graph", "--config", cfg, "--stats"])
    assert result.exit_code == 0, result.output
    assert "nodes: 4" in result.output
    assert "edges: 3" in result.output
    assert "level 0: 1" in result.output
    assert "review_flags: 0" in result.output


def test_graph_command_missing_checkpoint(tmp_path):
    cfg = _write_config(tmp_path, tmp_path / ".kgts")
    result = runner.invoke(app, ["graph", "--config", cfg])
    assert result.exit_code != 0


def test_sample_without_graph_fails_cleanly(tmp_path):
    cfg = _write_config(tmp_path, tmp_path / ".kgts")
    result = runner.invoke(app, ["sample", "--config", cfg])
    assert result.exit_code != 0


def test_missing_config_fails_cleanly(tmp_path):
    result = runner.invoke(app, ["graph", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code != 0
