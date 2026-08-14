"""Offline tests for the orchestration layer: ArtifactStore, exporter, report.

No sibling stage modules (build/sample/retrieve/synthesize/verify) are used.
"""

from __future__ import annotations

import json

import pytest

from kgts.eval.report import generate_report, render_markdown
from kgts.graph.store import GraphStore
from kgts.models import (
    AlignDecision,
    AlignVerdict,
    Edge,
    Material,
    Node,
    NodeStatus,
    Run,
    SampleBundle,
    SampleIntent,
    SourceType,
    Task,
    VerifyResult,
)
from kgts.orchestrate.exporter import (
    export_rl,
    export_sft,
    write_export,
    write_manifest,
)
from kgts.orchestrate.store import ArtifactStore


# --------------------------------------------------------------- fixtures
def _material(mid: str, nodes: list[str] | None = None) -> Material:
    return Material(
        id=mid,
        source_type=SourceType.LOCAL,
        path=f"/corpus/{mid}.txt",
        text=f"text of {mid}",
        linked_nodes=nodes or [],
    )


def _task(
    tid: str,
    nodes: list[str],
    *,
    task_type: str = "atomic_qa",
    level: int = 2,
    materials: list[str] | None = None,
    verify: VerifyResult = VerifyResult.PASS,
    verifier: str | None = "exact_match",
    question: str = "what is the meaning of this concept in detail",
) -> Task:
    bundle = SampleBundle(nodes=nodes, level=level, intent=SampleIntent.DEPTH)
    return Task(
        id=tid,
        task_type=task_type,
        sample_bundle=bundle,
        materials=materials or [],
        question=question,
        answer="an answer",
        rubric=["is correct"],
        verifier=verifier,
        verify_result=verify,
    )


@pytest.fixture()
def fixture():
    """Graph with 6 nodes (3 atomic), 5 tasks, 4 materials."""
    store = GraphStore()
    root = store.add_node(Node.create("Root"))
    alpha = store.add_node(Node.create("Alpha"))
    beta = store.add_node(Node.create("Beta"))
    a1 = store.add_node(Node.create("Alpha One", status=NodeStatus.ATOMIC))
    a2 = store.add_node(Node.create("Alpha Two", status=NodeStatus.ATOMIC))
    b1 = store.add_node(Node.create("Beta One", status=NodeStatus.ATOMIC))
    store.add_edge(Edge(parent=root.id, child=alpha.id))
    store.add_edge(Edge(parent=root.id, child=beta.id))
    store.add_edge(Edge(parent=alpha.id, child=a1.id))
    store.add_edge(Edge(parent=alpha.id, child=a2.id))
    store.add_edge(Edge(parent=beta.id, child=b1.id))

    materials = [
        _material("m1", [a1.id]),
        _material("m2", [a2.id]),
        _material("m3", [b1.id]),
        _material("m4", [a1.id]),
    ]
    tasks = [
        _task("t1", [a1.id], materials=["m1"], question="what is alpha one exactly"),
        _task("t2", [a1.id], materials=["m1", "m4"], question="what is alpha one exactly"),
        _task("t3", [a2.id], materials=["m2"], question="how does alpha two differ here"),
        _task(
            "t4",
            [a2.id],
            task_type="multihop_qa",
            materials=["m2"],
            verify=VerifyResult.FAIL,
            verifier=None,
            question="why does alpha two behave like that",
        ),
        _task(
            "t5",
            [a1.id, a2.id],
            task_type="aggregated_qa",
            level=1,
            materials=["m4"],
            verify=VerifyResult.SFT_ONLY,
            verifier=None,
            question="compare the alpha children together now",
        ),
    ]
    return store, tasks, materials, (root, alpha, beta, a1, a2, b1)


# --------------------------------------------------------- ArtifactStore
class TestArtifactStore:
    def test_material_roundtrip(self, tmp_path):
        store = ArtifactStore(tmp_path / "artifacts.db")
        m = _material("m1", ["n_x"])
        store.save_material(m)
        store.save_materials([_material("m2"), _material("m3")])
        all_mats = store.load_materials()
        assert {x.id for x in all_mats} == {"m1", "m2", "m3"}
        subset = store.load_materials(["m1", "m3"])
        assert {x.id for x in subset} == {"m1", "m3"}
        by_id = store.load_materials_by_id(["m2"])
        assert by_id["m2"].source_type == SourceType.LOCAL
        assert store.load_materials([]) == []

    def test_material_idempotent_rewrite(self, tmp_path):
        store = ArtifactStore(tmp_path / "artifacts.db")
        store.save_material(_material("m1"))
        m = _material("m1")
        m.quality_score = 0.9
        store.save_material(m)  # INSERT OR REPLACE, no duplicate
        mats = store.load_materials()
        assert len(mats) == 1
        assert mats[0].quality_score == 0.9

    def test_task_roundtrip_and_run_filter(self, tmp_path):
        store = ArtifactStore(tmp_path / "artifacts.db")
        t1 = _task("t1", ["n_x"])
        t1.run_id = "run_a"
        t2 = _task("t2", ["n_y"])
        t2.run_id = "run_b"
        store.save_tasks([t1, t2])
        store.save_task(t1)  # idempotent rewrite
        assert {t.id for t in store.load_tasks()} == {"t1", "t2"}
        assert [t.id for t in store.load_tasks(run_id="run_b")] == ["t2"]
        loaded = store.load_tasks(run_id="run_a")[0]
        assert loaded.sample_bundle.nodes == ["n_x"]

    def test_align_decision_roundtrip(self, tmp_path):
        store = ArtifactStore(tmp_path / "artifacts.db")
        d = AlignDecision(candidate_label="ML", verdict=AlignVerdict.DISTINCT)
        store.save_align_decision(d)
        store.save_align_decision(d)  # no error on rewrite
        import sqlite3

        conn = sqlite3.connect(str(tmp_path / "artifacts.db"))
        n = conn.execute("SELECT COUNT(*) FROM align_verdicts").fetchone()[0]
        conn.close()
        assert n == 1

    def test_run_lifecycle(self, tmp_path):
        store = ArtifactStore(tmp_path / "artifacts.db")
        run = Run(config_hash="abc123")
        store.create_run(run)
        assert store.load_run(run.id).finished_at is None
        run.finished_at = "2026-01-01T00:00:00Z"
        run.stage_stats["n"] = 5
        store.finish_run(run)
        loaded = store.load_run(run.id)
        assert loaded.finished_at == "2026-01-01T00:00:00Z"
        assert loaded.stage_stats == {"n": 5}
        assert [r.id for r in store.list_runs()] == [run.id]
        assert store.load_run("run_missing") is None

    def test_shares_db_file_with_graphstore(self, tmp_path):
        db = tmp_path / "shared.db"
        GraphStore().save(db)  # graph tables
        ArtifactStore(db).save_material(_material("m1"))  # artifact tables
        assert GraphStore.load(db) is not None
        assert len(ArtifactStore(db).load_materials()) == 1


# -------------------------------------------------------------- exporter
class TestExporter:
    def test_export_sft_filters(self):
        tasks = [
            _task("t1", ["n"], verify=VerifyResult.PASS),
            _task("t2", ["n"], verify=VerifyResult.SFT_ONLY, verifier=None),
            _task("t3", ["n"], verify=VerifyResult.FAIL),
            _task("t4", ["n"], verify=VerifyResult.UNVERIFIED),
        ]
        rows = export_sft(tasks)
        assert [r["task_id"] for r in rows] == ["t1", "t2"]
        assert rows[0]["messages"] == [
            {"role": "user", "content": tasks[0].question},
            {"role": "assistant", "content": tasks[0].answer},
        ]
        assert rows[0]["task_type"] == "atomic_qa"

    def test_export_rl_requires_pass_and_verifier(self):
        tasks = [
            _task("t1", ["n"], verify=VerifyResult.PASS, verifier="exact_match"),
            _task("t2", ["n"], verify=VerifyResult.PASS, verifier=None),
            _task("t3", ["n"], verify=VerifyResult.SFT_ONLY, verifier="exact_match"),
        ]
        rows = export_rl(tasks)
        assert [r["task_id"] for r in rows] == ["t1"]
        assert rows[0]["prompt"] == tasks[0].question
        assert rows[0]["rubric"] == ["is correct"]
        assert rows[0]["verifier"] == "exact_match"
        assert rows[0]["env"] is None

    def test_write_export_jsonl(self, tmp_path):
        tasks = [_task("t1", ["n"]), _task("t2", ["n"])]
        out = tmp_path / "out" / "sft.jsonl"
        n = write_export(tasks, "sft", out)
        assert n == 2
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["task_id"] == "t1"
        with pytest.raises(ValueError):
            write_export(tasks, "bogus", tmp_path / "x.jsonl")

    def test_write_manifest_lineage(self, tmp_path, fixture):
        _, tasks, materials, _ = fixture
        run = Run(config_hash="hash_run")
        out = tmp_path / "manifest.json"
        write_manifest(run, tasks, materials, "hash_cfg", out)
        manifest = json.loads(out.read_text())
        assert manifest["run"]["id"] == run.id
        assert manifest["config_hash"] == "hash_cfg"
        assert manifest["counts"]["tasks"] == 5
        assert manifest["counts"]["materials"] == 4
        assert manifest["counts"]["by_task_type"]["atomic_qa"] == 3
        assert manifest["counts"]["by_verify_result"]["fail"] == 1
        assert manifest["counts"]["by_source_type"]["local"] == 4
        lineage = {e["task_id"]: e for e in manifest["lineage"]}
        assert len(lineage) == 5
        entry = lineage["t2"]
        assert entry["material_ids"] == ["m1", "m4"]
        assert entry["nodes"] == tasks[1].sample_bundle.nodes
        assert entry["verify_result"] == "pass"
        assert entry["verifier"] == "exact_match"


# ---------------------------------------------------------------- report
class TestReport:
    def test_coverage(self, fixture):
        store, tasks, materials, ids = fixture
        *_, a1, a2, b1 = ids
        report = generate_report(store, tasks, materials)
        cov = report["coverage"]
        assert cov["n_nodes"] == 6
        assert cov["n_sampled_nodes"] == 2  # a1, a2
        # b1 is atomic and never sampled -> 1/3
        assert cov["long_tail_ratio"] == pytest.approx(1 / 3)
        assert cov["per_level"]["0"] == {"nodes": 1, "sampled": 0}
        assert cov["per_level"]["2"]["nodes"] == 3
        assert cov["per_level"]["2"]["sampled"] == 2

    def test_duplication(self, fixture):
        store, tasks, materials, _ = fixture
        dup = generate_report(store, tasks, materials)["duplication"]
        assert set(dup) >= {"pairwise_jaccard_mean", "material_reuse_rate"}
        # t1 and t2 share the same question -> mean jaccard > 0
        assert 0.0 < dup["pairwise_jaccard_mean"] <= 1.0
        # used materials: m1 (2 tasks), m2 (2 tasks), m4 (2 tasks) -> all reused
        assert dup["material_reuse_rate"] == pytest.approx(1.0)

    def test_diversity_matrix(self, fixture):
        store, tasks, materials, _ = fixture
        div = generate_report(store, tasks, materials)["diversity"]
        assert div["total"] == 5
        assert div["matrix"]["atomic_qa"]["2"] == 3
        assert div["matrix"]["multihop_qa"]["2"] == 1
        assert div["matrix"]["aggregated_qa"]["1"] == 1
        assert sum(sum(m.values()) for m in div["matrix"].values()) == 5

    def test_quality(self, fixture):
        store, tasks, materials, _ = fixture
        qual = generate_report(store, tasks, materials)["quality"]
        assert qual["overall"] == {"pass": 3, "fail": 1, "sft_only": 1, "unverified": 0}
        assert qual["by_task_type"]["multihop_qa"]["fail"] == 1

    def test_provenance_complete(self, fixture):
        store, tasks, materials, _ = fixture
        prov = generate_report(store, tasks, materials)["provenance"]
        assert prov["completeness"] == 1.0
        assert prov["n_broken"] == 0
        assert prov["broken_ids"] == []

    def test_provenance_broken(self, fixture):
        store, tasks, materials, _ = fixture
        broken = _task("t_bad", ["n_missing"], materials=["m_missing"])
        prov = generate_report(store, tasks + [broken], materials)["provenance"]
        assert prov["n_broken"] == 1
        assert prov["broken_ids"] == ["t_bad"]
        assert prov["completeness"] == pytest.approx(5 / 6)

    def test_render_markdown(self, fixture):
        store, tasks, materials, _ = fixture
        report = generate_report(store, tasks, materials)
        md = render_markdown(report)
        for section in ("Coverage", "Duplication", "Diversity", "Quality", "Provenance"):
            assert f"## {section}" in md


def test_write_back_material_stats(tmp_path):
    """Stage C feedback: counts land in node.stats; a material-sufficient
    EXPANDING node is re-judged ATOMIC; graph.db is re-persisted."""
    from kgts.config import Config, RunConfig
    from kgts.models import Material, NodeStatus, SourceType
    from kgts.orchestrate.runner import _write_back_material_stats

    cfg = Config(run=RunConfig(workdir=str(tmp_path)))
    store = GraphStore()
    parent = store.add_node(Node.create("Parent"))
    child = store.add_node(Node.create("Child"))
    store.add_edge(Edge(parent=parent.id, child=child.id))
    store.save(tmp_path / "graph.db")

    mats = [
        Material(source_type=SourceType.LOCAL, text="x", linked_nodes=[child.id])
        for _ in range(5)  # default atomicity.min_materials == 5
    ]
    _write_back_material_stats(cfg, store, mats)

    assert store.get(child.id).stats.n_materials == 5
    assert store.get(child.id).status == NodeStatus.ATOMIC
    assert store.get(parent.id).stats.n_materials == 0
    reloaded = GraphStore.load(tmp_path / "graph.db")
    assert reloaded.get(child.id).stats.n_materials == 5
    assert reloaded.get(child.id).status == NodeStatus.ATOMIC
