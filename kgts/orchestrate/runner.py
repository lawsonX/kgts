"""Pipeline orchestration: checkpointed stages + full-run wiring (design §8/§9).

Each stage reads its inputs from the previous stage's checkpoint inside the
run workdir and writes its own checkpoint back, so every stage is
independently re-runnable via the CLI (``kgts sample`` after ``kgts build``,
etc.). Checkpoints:

- ``graph.db``       -- Stage A output (GraphStore SQLite)
- ``bundles.json``   -- Stage B output (list of SampleBundle)
- ``materials.json`` -- Stage C output (list of Material)
- ``artifacts.db``   -- materials/tasks/align_verdicts/runs tables (ArtifactStore)

Sibling stage modules (build/sample/retrieve/synthesize/verify) are imported
lazily inside each stage function so this module imports cleanly on its own.
"""

from __future__ import annotations

import json
from pathlib import Path

from kgts.config import Config
from kgts.graph.store import GraphStore
from kgts.models import (
    Material,
    Run,
    SampleBundle,
    Task,
    VerifyResult,
    utc_now,
)
from kgts.orchestrate.store import ArtifactStore

DEFAULT_STAGES = ("build", "sample", "retrieve", "synthesize", "verify", "export", "report")


class CheckpointError(RuntimeError):
    """A required upstream checkpoint is missing from the workdir."""


# --------------------------------------------------------------- path helpers
def workdir_of(config: Config) -> Path:
    w = Path(config.run.workdir)
    w.mkdir(parents=True, exist_ok=True)
    return w


def graph_db_path(config: Config) -> Path:
    return Path(config.run.workdir) / "graph.db"


def artifacts_of(config: Config) -> ArtifactStore:
    return ArtifactStore(Path(config.run.workdir) / "artifacts.db")


def _require(path: Path, stage: str, hint: str) -> Path:
    if not path.exists():
        raise CheckpointError(
            f"stage '{stage}' needs {path}; run `kgts {hint}` first (or `kgts run`)"
        )
    return path


def load_graph(config: Config, stage: str = "?") -> GraphStore:
    path = _require(graph_db_path(config), stage, "build")
    return GraphStore.load(path)


def load_bundles(config: Config, stage: str = "?") -> list[SampleBundle]:
    path = _require(Path(config.run.workdir) / "bundles.json", stage, "sample")
    return [SampleBundle.model_validate(b) for b in json.loads(path.read_text())]


def load_materials_json(config: Config, stage: str = "?") -> list[Material]:
    path = _require(Path(config.run.workdir) / "materials.json", stage, "retrieve")
    return [Material.model_validate(m) for m in json.loads(path.read_text())]


# ------------------------------------------------------------------- LLM
def make_llm(config: Config, workdir: Path):
    """Build the managed LLM client: mock for tests, LiteLLM otherwise."""
    from kgts.llm import LiteLLMClient, ManagedLLM, MockLLM

    if config.llm.model.startswith("mock"):
        client = MockLLM()
    else:
        client = LiteLLMClient(config.llm.model, api_base=config.llm.api_base)
    cache_dir = workdir / "llm_cache" if config.llm.cache else None
    return ManagedLLM(
        client,
        max_calls=config.budget.max_llm_calls,
        max_cost_usd=config.budget.max_cost_usd,
        rpm=config.llm.rpm,
        cache_dir=cache_dir,
        max_retries=config.llm.max_retries,
        retry_backoff=config.llm.retry_backoff,
    )


# ------------------------------------------------------------------ stages
def stage_build(config: Config, *, llm=None, resume: bool = True) -> GraphStore:
    """Stage A: expand the knowledge DAG (or load the existing checkpoint)."""
    workdir_of(config)
    gdb = graph_db_path(config)
    if resume and gdb.exists():
        return GraphStore.load(gdb)
    from kgts.build.expand import expand_graph

    llm = llm or make_llm(config, Path(config.run.workdir))
    store = expand_graph(
        config.seeds, llm, GraphStore(), config, artifact_store=artifacts_of(config)
    )
    store.save(gdb)
    from kgts.graph.card import write_card

    write_card(store, config, workdir_of(config))  # every DAG ships with a name card
    return store


def stage_sample(config: Config, *, resume: bool = True) -> tuple[GraphStore, list[SampleBundle]]:
    """Stage B: sample bundles over the DAG (checkpoint: bundles.json)."""
    store = load_graph(config, "sample")
    path = Path(config.run.workdir) / "bundles.json"
    if resume and path.exists():
        return store, load_bundles(config, "sample")
    from kgts.sample.sampler import sample_bundles

    bundles = sample_bundles(store, config.sample, seed=config.run.seed)
    path.write_text(json.dumps([b.model_dump() for b in bundles], ensure_ascii=False))
    return store, bundles


def stage_retrieve(
    config: Config, *, llm=None, resume: bool = True
) -> tuple[GraphStore, list[SampleBundle], dict[str, Material]]:
    """Stage C: retrieve materials per bundle (checkpoint: materials.json).

    When the local source is enabled with ``local.adapter == "auto"``, the
    CorpusAdapterAgent first infers (or loads the cached) ExtractionSpec for
    the corpus — this is the agentic input-compatibility layer.
    """
    store, bundles = stage_sample(config, resume=resume)
    path = Path(config.run.workdir) / "materials.json"
    if resume and path.exists():
        materials = load_materials_json(config, "retrieve")
    else:
        from kgts.retrieve.retriever import Retriever
        from kgts.retrieve.sources import build_sources

        local_spec = _local_spec(config, llm=llm, resume=resume)
        retriever = Retriever(
            build_sources(config.retrieve, local_spec=local_spec), config.retrieve
        )
        by_id: dict[str, Material] = {}
        for bundle in bundles:
            for m in retriever.retrieve(store, bundle):
                by_id[m.id] = m
        materials = list(by_id.values())
        path.write_text(json.dumps([m.model_dump() for m in materials], ensure_ascii=False))
    artifacts_of(config).save_materials(materials)
    _write_back_material_stats(config, store, materials)
    return store, bundles, {m.id: m for m in materials}


def _write_back_material_stats(
    config: Config, store: GraphStore, materials: list[Material]
) -> None:
    """Stage C -> Stage A feedback loop (design 6.3): real per-node material
    counts replace the (untrusted) explorer estimates in ``node.stats``, and
    nodes now proven material-sufficient are re-judged atomic. Persisted back
    to graph.db so reports and future expansion runs see real numbers.
    """
    from kgts.build.atomicity import AtomicityJudge

    counts: dict[str, int] = {}
    for m in materials:
        for nid in m.linked_nodes:
            counts[nid] = counts.get(nid, 0) + 1
    judge = AtomicityJudge(config.build.atomicity)
    changed = False
    for nid, count in counts.items():
        if nid not in store:
            continue
        node = store.get(nid)
        node.stats.n_materials = count
        if judge.is_atomic(node):
            from kgts.models import NodeStatus

            node.status = NodeStatus.ATOMIC
        changed = True
    if changed:
        store.save(graph_db_path(config))
        from kgts.graph.card import write_card

        write_card(store, config, workdir_of(config))  # material stats changed the card


def corpus_spec_path(config: Config) -> Path:
    return Path(config.run.workdir) / "corpus_spec.json"


def _local_spec(config: Config, *, llm=None, resume: bool = True):
    """Infer (or load the cached) ExtractionSpec for the local corpus."""
    if "local" not in config.retrieve.sources:
        return None
    if config.retrieve.local.adapter != "auto":
        return None
    path = corpus_spec_path(config)
    if resume and path.exists():
        from kgts.retrieve.ingest import ExtractionSpec

        return ExtractionSpec.model_validate_json(path.read_text())
    from kgts.retrieve.ingest import analyze_corpus

    llm = llm or make_llm(config, Path(config.run.workdir))
    spec = analyze_corpus(config.retrieve.local.paths, llm)
    if spec is not None:
        path.write_text(spec.model_dump_json())
    return spec


def stage_synthesize(config: Config, *, llm=None, resume: bool = True) -> list[Task]:
    """Stage D: synthesize one task per bundle from its linked materials.

    On resume, bundles that already produced a task (matched by bundle id)
    are skipped, so the stage is idempotent.
    """
    store, bundles, materials_by_id = stage_retrieve(config, resume=resume)
    artifacts = artifacts_of(config)
    existing = artifacts.load_tasks() if resume else []
    done = {t.sample_bundle.id for t in existing}
    pending = [b for b in bundles if b.id not in done]
    new_tasks: list[Task] = []
    if pending:
        from kgts.synthesize.synthesizer import Synthesizer

        llm = llm or make_llm(config, Path(config.run.workdir))
        synth = Synthesizer(llm, config.synthesize, seed=config.run.seed)
        for bundle in pending:
            node_set = set(bundle.nodes)
            mats = [m for m in materials_by_id.values() if node_set & set(m.linked_nodes)]
            task = synth.synthesize(store, bundle, mats)
            if task is not None:
                new_tasks.append(task)
        artifacts.save_tasks(new_tasks)
    return existing + new_tasks


def stage_verify(
    config: Config, *, llm=None, resume: bool = True, tasks: list[Task] | None = None
) -> list[Task]:
    """Stage E (task-level): verify synthesized tasks; idempotent on resume.

    ``tasks`` may be passed in by ``run_pipeline`` to avoid re-entering
    ``stage_synthesize`` (which would duplicate work when ``resume=False``).
    """
    if tasks is None:
        tasks = stage_synthesize(config, llm=llm, resume=resume)
    pending = [t for t in tasks if not resume or t.verify_result == VerifyResult.UNVERIFIED]
    if not pending:
        return tasks
    from kgts.verify.pipeline import verify_task

    llm = llm or make_llm(config, Path(config.run.workdir))
    artifacts = artifacts_of(config)
    materials_by_id = {m.id: m for m in artifacts.load_materials()}
    done_by_id = {t.id: t for t in tasks}
    for task in pending:
        verified = verify_task(task, materials_by_id, llm, config.verify)
        artifacts.save_task(verified)
        done_by_id[verified.id] = verified
    return list(done_by_id.values())


def stage_export(config: Config) -> dict[str, int]:
    """Stage E (export): write JSONL per configured format + manifest.json."""
    artifacts = artifacts_of(config)
    tasks = artifacts.load_tasks()
    materials = artifacts.load_materials()
    out_dir = Path(config.export.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from kgts.orchestrate.exporter import write_export, write_manifest

    counts = {}
    materials_by_id = {m.id: m for m in materials}
    for fmt in config.export.formats:
        counts[fmt] = write_export(
            tasks,
            fmt,
            out_dir / f"tasks_{fmt}.jsonl",
            materials_by_id,
            include_context=config.export.include_context,
            min_quality=config.export.min_quality,
        )
    runs = artifacts.list_runs()
    run = runs[-1] if runs else Run(config_hash=config.config_hash())
    write_manifest(run, tasks, materials, config.config_hash(), out_dir / "manifest.json")
    card_md = Path(config.run.workdir) / "graph_card.md"
    if card_md.exists():
        (out_dir / "graph_card.md").write_text(card_md.read_text())
    return counts


def stage_report(config: Config, *, run: Run | None = None) -> dict:
    """Stage E (report): dataset-level report -> report.json + report.md."""
    from kgts.eval.report import generate_report, render_markdown

    store = load_graph(config, "report")
    artifacts = artifacts_of(config)
    report = generate_report(store, artifacts.load_tasks(), artifacts.load_materials(), run)
    out_dir = Path(config.export.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out_dir / "report.md").write_text(render_markdown(report))
    return report


# --------------------------------------------------------------- full pipeline
def run_pipeline(
    config: Config,
    *,
    resume: bool = True,
    llm=None,
    stages: tuple[str, ...] = DEFAULT_STAGES,
) -> Run:
    """Run the pipeline end to end with per-stage checkpointing.

    With ``resume=True`` (default) each stage reuses its checkpoint if one
    exists, so interrupted runs pick up where they stopped. Individual stages
    are also independently re-runnable via the CLI (``kgts build``,
    ``kgts sample``, ...), which call the ``stage_*`` functions above.
    """
    workdir_of(config)  # ensure the workdir exists before any stage writes
    artifacts = artifacts_of(config)
    run = Run(config_hash=config.config_hash())
    artifacts.create_run(run)

    tasks: list[Task] = []
    if "build" in stages:
        stage_build(config, llm=llm, resume=resume)
    if "sample" in stages:
        stage_sample(config, resume=resume)
    if "retrieve" in stages:
        stage_retrieve(config, llm=llm, resume=resume)
    if "synthesize" in stages:
        tasks = stage_synthesize(config, llm=llm, resume=resume)
        run.stage_stats["n_tasks_synthesized"] = len(tasks)
    if "verify" in stages:
        tasks = stage_verify(config, llm=llm, resume=resume)
        run.stage_stats["n_passed"] = sum(
            1 for t in tasks if t.verify_result == VerifyResult.PASS
        )
    if "export" in stages:
        run.stage_stats["export_counts"] = stage_export(config)
    if "report" in stages:
        report = stage_report(config, run=run)
        run.stage_stats["report_headline"] = {
            "n_nodes": report["coverage"]["n_nodes"],
            "long_tail_ratio": report["coverage"]["long_tail_ratio"],
            "provenance_completeness": report["provenance"]["completeness"],
        }

    run.finished_at = utc_now()
    artifacts.finish_run(run)
    return run
