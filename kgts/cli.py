"""KGTS command line interface (design appendix B).

All commands share ``--config``; intermediate artifacts live in the run
workdir, so any single-stage command can re-run on top of existing
checkpoints (``kgts sample`` works once ``kgts build`` has produced
``graph.db``, etc.).
"""

# typer.Option/typer.Argument in parameter defaults is the canonical typer idiom.
# ruff: noqa: B008
from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer

from kgts.config import load_config
from kgts.orchestrate.runner import (
    CheckpointError,
    graph_db_path,
    stage_build,
    stage_export,
    stage_report,
    stage_retrieve,
    stage_sample,
    stage_synthesize,
    stage_verify,
)

app = typer.Typer(help="KGTS: Knowledge-Graph-Guided Task Synthesis pipeline.")

_CONFIG = typer.Option(..., "--config", "-c", help="Path to the pipeline YAML config.")


def _config(path: Path):
    try:
        return load_config(path)
    except FileNotFoundError:
        typer.secho(f"config not found: {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None


def _guard(fn, *args, **kw):
    """Run a stage function, converting missing-checkpoint errors to clean exits."""
    try:
        return fn(*args, **kw)
    except CheckpointError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None


@app.command()
def build(
    config: Path = _CONFIG,
    cheap_mode: bool = typer.Option(
        False,
        "--cheap-mode",
        help="Use llm.cheap_model instead of llm.model for graph exploration.",
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
):
    """Stage A: expand the knowledge DAG, checkpoint to workdir/graph.db."""
    cfg = _config(config)
    if cheap_mode:
        if not cfg.llm.cheap_model:
            typer.secho("--cheap-mode needs llm.cheap_model set in the config", err=True)
            raise typer.Exit(1)
        cfg.llm.model = cfg.llm.cheap_model
    store = _guard(stage_build, cfg, resume=resume)
    typer.echo(f"graph: {len(store)} nodes, {len(store.edges())} edges -> {graph_db_path(cfg)}")


@app.command()
def sample(
    config: Path = _CONFIG,
    n: int | None = typer.Option(None, "-n", help="Override sample.n_samples."),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
):
    """Stage B: sample SampleBundles over the DAG."""
    cfg = _config(config)
    if n is not None:
        cfg.sample.n_samples = n
    _, bundles = _guard(stage_sample, cfg, resume=resume)
    typer.echo(f"sampled {len(bundles)} bundles")


@app.command()
def retrieve(
    config: Path = _CONFIG,
    resume: bool = typer.Option(True, "--resume/--no-resume"),
):
    """Stage C: retrieve materials for each sampled bundle."""
    cfg = _config(config)
    _, _, materials = _guard(stage_retrieve, cfg, resume=resume)
    typer.echo(f"retrieved {len(materials)} unique materials")


@app.command()
def ingest(
    config: Path = _CONFIG,
    save: bool = typer.Option(False, "--save", help="Cache the inferred spec into the workdir."),
):
    """Preview the CorpusAdapterAgent's extraction spec for the local corpus."""
    cfg = _config(config)
    from kgts.orchestrate.runner import make_llm
    from kgts.retrieve.ingest import CorpusAdapterAgent

    agent = CorpusAdapterAgent(make_llm(cfg, Path(cfg.run.workdir)))
    samples = agent.sample_files(cfg.retrieve.local.paths)
    if not samples:
        typer.echo("no readable corpus files found", err=True)
        raise typer.Exit(1)
    typer.echo(f"sampled {len(samples)} files: {', '.join(samples)}")
    spec = agent.infer_spec(samples)
    if spec is None:
        typer.echo("no spec inferred (heuristics will be used)")
        raise typer.Exit(1)
    typer.echo(spec.model_dump_json(indent=2))
    if save:
        from kgts.orchestrate.runner import corpus_spec_path

        path = corpus_spec_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(spec.model_dump_json())
        typer.echo(f"spec cached to {path}")


@app.command()
def synth(
    config: Path = _CONFIG,
    types: str | None = typer.Option(None, "--types", help="Comma-separated task types."),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
):
    """Stage D+E: synthesize tasks from bundles + materials, then verify them."""
    cfg = _config(config)
    if types:
        cfg.synthesize.task_types = [t.strip() for t in types.split(",") if t.strip()]
    tasks = _guard(stage_synthesize, cfg, resume=resume)
    verified = _guard(stage_verify, cfg, resume=resume)
    typer.echo(f"synthesized {len(tasks)} tasks, verified {len(verified)}")


@app.command(name="export")
def export_cmd(
    config: Path = _CONFIG,
    fmt: str | None = typer.Option(None, "--format", help="sft | rl (default: config formats)."),
    out: Path | None = typer.Option(None, "--out", help="Output dir (default: config out_dir)."),
):
    """Stage E (export): write JSONL per format plus manifest.json."""
    cfg = _config(config)
    if fmt is not None:
        if fmt not in ("sft", "rl"):
            typer.secho(f"unknown format {fmt!r}; expected sft or rl", err=True)
            raise typer.Exit(1)
        cfg.export.formats = [fmt]
    if out is not None:
        cfg.export.out_dir = str(out)
    counts = stage_export(cfg)
    for f, n in counts.items():
        typer.echo(f"exported {n} rows ({f}) -> {Path(cfg.export.out_dir) / f'tasks_{f}.jsonl'}")


@app.command()
def report(
    config: Path = _CONFIG,
    run: str | None = typer.Option(None, "--run", help="Run id (default: latest)."),
):
    """Stage E (report): coverage/duplication/diversity/quality/provenance."""
    from kgts.orchestrate.runner import artifacts_of

    cfg = _config(config)
    run_obj = None
    if run:
        run_obj = artifacts_of(cfg).load_run(run)
        if run_obj is None:
            typer.secho(f"run not found: {run}", err=True)
            raise typer.Exit(1)
    rep = _guard(stage_report, cfg, run=run_obj)
    typer.echo(
        f"report: {rep['coverage']['n_nodes']} nodes, "
        f"{rep['diversity']['total']} tasks, "
        f"provenance completeness {rep['provenance']['completeness']:.3f}"
        f" -> {Path(cfg.export.out_dir) / 'report.md'}"
    )


@app.command()
def run(
    config: Path = _CONFIG,
    resume: bool = typer.Option(True, "--resume/--no-resume"),
):
    """Run the full pipeline end to end (with per-stage resume)."""
    from kgts.orchestrate.runner import run_pipeline

    cfg = _config(config)
    run_obj = _guard(run_pipeline, cfg, resume=resume)
    typer.echo(f"run {run_obj.id} finished at {run_obj.finished_at}")
    for k, v in run_obj.stage_stats.items():
        typer.echo(f"  {k}: {v}")


@app.command()
def graph(
    config: Path = _CONFIG,
    stats: bool = typer.Option(False, "--stats", help="Print per-level histogram."),
    export: str | None = typer.Option(None, "--export", help="Export the DAG: dot|json."),
    card: bool = typer.Option(False, "--card", help="Regenerate and print the graph card."),
):
    """Inspect the knowledge DAG checkpoint (workdir/graph.db)."""
    from kgts.orchestrate.runner import load_graph

    cfg = _config(config)
    store = _guard(load_graph, cfg, "graph")
    typer.echo(f"nodes: {len(store)}")
    typer.echo(f"edges: {len(store.edges())}")
    typer.echo(f"review_flags: {len(store.review_flags)}")
    if stats:
        hist = Counter(n.level for n in store.nodes())
        for level in sorted(hist):
            typer.echo(f"  level {level}: {hist[level]}")
    if export:
        _export_graph(store, cfg, export)
    if card:
        from kgts.graph.card import render_markdown, write_card
        from kgts.orchestrate.runner import workdir_of

        typer.echo(render_markdown(write_card(store, cfg, workdir_of(cfg))))


def _export_graph(store, cfg, fmt: str) -> None:
    import json as _json

    out = Path(cfg.export.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        payload = {
            "nodes": [n.model_dump() for n in store.nodes()],
            "edges": [e.model_dump() for e in store.edges()],
        }
        path = out / "graph.json"
        path.write_text(_json.dumps(payload, ensure_ascii=False, indent=2))
    elif fmt == "dot":
        lines = ["digraph kgts {"]
        for n in store.nodes():
            label = n.label.replace('"', "'")
            lines.append(f'  "{n.id}" [label="{label}\\nL{n.level} {n.status.value}"];')
        for e in store.edges():
            style = "solid" if e.relation.value == "is_subconcept" else "dashed"
            lines.append(f'  "{e.parent}" -> "{e.child}" [style={style}];')
        lines.append("}")
        path = out / "graph.dot"
        path.write_text("\n".join(lines) + "\n")
    else:
        typer.echo(f"unknown export format {fmt!r}: dot|json", err=True)
        raise typer.Exit(1)
    typer.echo(f"graph exported -> {path}")


@app.command()
def review(
    config: Path = _CONFIG,
    queue: str = typer.Option("all", "--queue", help="all|align|parents"),
):
    """Print the human review queue: soft-constraint flags and/or align verdicts."""
    from kgts.orchestrate.runner import artifacts_of, load_graph

    cfg = _config(config)
    store = _guard(load_graph, cfg, "review")
    if queue not in ("all", "align", "parents"):
        typer.echo(f"unknown queue {queue!r}: all|align|parents", err=True)
        raise typer.Exit(1)
    if queue in ("all", "parents"):
        kinds = None if queue == "all" else {"max_parents_exceeded", "level_soft_violation"}
        flags = [f for f in store.review_flags if kinds is None or f.get("kind") in kinds]
        typer.echo(f"review flags: {len(flags)}")
        for flag in flags:
            typer.echo(f"  {flag}")
    if queue in ("all", "align"):
        decisions = artifacts_of(cfg).load_align_decisions()
        typer.echo(f"align verdicts: {len(decisions)}")
        for d in decisions[:50]:
            typer.echo(f"  [{d.verdict.value}] {d.candidate_label} -> {d.matched_node}")


@app.command()
def serve(
    config: Path = _CONFIG,
    port: int = typer.Option(7860, "--port"),
    host: str = typer.Option("127.0.0.1", "--host", help="Use 0.0.0.0 to expose on the LAN."),
):
    """Launch the Gradio workdir viewer."""
    from kgts.ui.app import main as ui_main

    cfg = _config(config)
    try:
        ui_main(workdir=cfg.run.workdir, port=port, host=host)
    except RuntimeError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
