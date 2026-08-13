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


@app.command()
def review(config: Path = _CONFIG):
    """Print the human review queue (soft-constraint violations)."""
    from kgts.orchestrate.runner import load_graph

    cfg = _config(config)
    store = _guard(load_graph, cfg, "review")
    flags = store.review_flags
    typer.echo(f"review queue: {len(flags)} flags")
    for flag in flags:
        typer.echo(f"  {flag}")


@app.command()
def serve(
    config: Path = _CONFIG,
    port: int = typer.Option(7860, "--port"),
):
    """Launch the Gradio workdir viewer."""
    from kgts.ui.app import main as ui_main

    cfg = _config(config)
    try:
        ui_main(workdir=cfg.run.workdir, port=port)
    except RuntimeError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
