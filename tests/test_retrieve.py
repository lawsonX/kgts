"""Offline tests for Stage C (retrieve). No network access."""

import pytest

from kgts.config import LocalSourceConfig, RetrieveConfig, WebSourceConfig
from kgts.graph.store import GraphStore
from kgts.models import Edge, Material, Node, SampleBundle, SampleIntent, SourceType
from kgts.retrieve.postprocess import dedup, filter_license, simhash64
from kgts.retrieve.query import QueryBuilder
from kgts.retrieve.retriever import Retriever
from kgts.retrieve.sources import LocalCorpusSource, WebSearchSource, build_sources

# ------------------------------------------------------------ QueryBuilder


def test_querybuilder_disambiguation_contains_ancestor():
    queries = QueryBuilder.build(
        "kernel", ["CS", "AI", "GPU programming"], SampleIntent.DEPTH, SourceType.WEB
    )
    assert any("GPU programming kernel" in q for q in queries)


def test_querybuilder_web_shape_has_tutorial_variant():
    queries = QueryBuilder.build("kernel", ["GPU programming"], SampleIntent.DEPTH, SourceType.WEB)
    assert 2 <= len(queries) <= 3
    assert any("tutorial" in q for q in queries)


def test_querybuilder_github_shape():
    queries = QueryBuilder.build(
        "kernel", ["CS", "GPU programming"], SampleIntent.DEPTH, SourceType.GITHUB
    )
    assert any("CS" in q and "kernel" in q for q in queries)
    assert any("topic:" in q for q in queries)


def test_querybuilder_arxiv_shape_quoted():
    queries = QueryBuilder.build(
        "kernel", ["GPU programming"], SampleIntent.DEPTH, SourceType.ARXIV
    )
    assert any(q.startswith('"') and q.endswith('"') for q in queries)


def test_querybuilder_local_shape_keyword_list():
    queries = QueryBuilder.build(
        "kernel", ["CS", "GPU programming"], SampleIntent.DEPTH, SourceType.LOCAL
    )
    assert any("CS" in q and "kernel" in q for q in queries)


def test_querybuilder_joint_adds_and_query():
    queries = QueryBuilder.build(
        "kernel", ["CS"], SampleIntent.JOINT, SourceType.WEB, group_labels=["kernel", "CUDA"]
    )
    assert any("AND" in q and "kernel" in q and "CUDA" in q for q in queries)


def test_querybuilder_deterministic():
    args = ("kernel", ["CS", "AI"], SampleIntent.DEPTH, SourceType.WEB)
    assert QueryBuilder.build(*args) == QueryBuilder.build(*args)


# -------------------------------------------------------- LocalCorpusSource


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "gpu.txt").write_text(
        "GPU programming kernels are functions that run on the device. " * 10
    )
    (tmp_path / "cooking.txt").write_text(
        "Sourdough bread needs flour water salt and time. " * 10
    )
    (tmp_path / "notes.md").write_text("CUDA threads are organized into blocks and grids. " * 10)
    return tmp_path


def test_local_source_returns_relevant_chunk(corpus):
    cfg = LocalSourceConfig(paths=[str(corpus)], chunk_size=200, chunk_overlap=50)
    out = LocalCorpusSource(cfg).search(["GPU programming kernel"], budget=2)
    assert out
    top = out[0]
    assert top.source_type == SourceType.LOCAL
    assert top.path and top.path.endswith("gpu.txt")
    assert top.title == "gpu.txt"
    assert top.snippet == top.text[:200]
    assert len(top.snippet) <= 200
    assert top.text
    assert top.quality_score > 0


def test_local_source_tolerates_missing_dir(tmp_path):
    src = LocalCorpusSource(LocalSourceConfig(paths=[str(tmp_path / "nope")]))
    assert src.search(["anything"], budget=3) == []


# ------------------------------------------------------------- postprocess


def _mat(text: str, source: SourceType = SourceType.LOCAL, uri: str | None = None) -> Material:
    return Material(source_type=source, text=text, snippet=text[:200], title="t", uri=uri)


def test_simhash_dedup_collapses_near_identical():
    # simhash needs a realistic number of features, so use paragraph-length texts
    body = (
        "GPU kernels launch many threads in parallel on the device and each "
        "thread runs the same program over its own slice of data " * 4
    )
    a = _mat(body + "final words here")
    b = _mat(body + "final words there")  # one token differs
    c = _mat("Sourdough bread requires flour water salt and patience " * 4)
    out = dedup([a, b, c], mode="simhash", threshold=3)
    assert len(out) == 2
    assert simhash64(a.text) != simhash64(c.text)


def test_url_dedup_normalizes():
    a = _mat("x", SourceType.WEB, uri="https://Example.com/page/?utm_source=news")
    b = _mat("y", SourceType.WEB, uri="https://example.com/page")
    assert len(dedup([a, b], mode="url")) == 1


def test_filter_license_whitelist_drops_all_rights_reserved():
    web_bad = Material(
        source_type=SourceType.WEB, uri="https://x.com", license="all-rights-reserved"
    )
    web_ok = Material(source_type=SourceType.WEB, uri="https://y.com", license="cc-by")
    web_none = Material(source_type=SourceType.WEB, uri="https://z.com")
    local = Material(source_type=SourceType.LOCAL, text="t")
    out = filter_license([web_bad, web_ok, web_none, local])
    assert web_bad not in out
    assert web_ok in out and web_none in out and local in out


def test_filter_license_off_keeps_everything():
    web_bad = Material(source_type=SourceType.WEB, license="all-rights-reserved")
    assert filter_license([web_bad], mode="off") == [web_bad]


# --------------------------------------------------------- build_sources


def test_build_sources_unknown_raises():
    with pytest.raises(ValueError, match="valid sources"):
        build_sources(RetrieveConfig(sources=["local", "nope"]))


def test_build_sources_local_only(tmp_path):
    cfg = RetrieveConfig(sources=["local"])
    cfg.local.paths = [str(tmp_path)]
    assert set(build_sources(cfg)) == {"local"}


def test_web_source_requires_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    src = WebSearchSource(WebSourceConfig())
    with pytest.raises(RuntimeError, match="set TAVILY_API_KEY or disable the web source"):
        src.search(["q"], budget=3)


# --------------------------------------------------------------- Retriever


def _store_with_kernel() -> tuple[GraphStore, Node]:
    store = GraphStore()
    root = store.add_node(
        Node.create("GPU programming", description="parallel programming on GPUs")
    )
    leaf = store.add_node(Node.create("kernel", description="device function for GPU programming"))
    store.add_edge(Edge(parent=root.id, child=leaf.id))
    return store, leaf


def _local_config(corpus, sources=("local",)) -> RetrieveConfig:
    cfg = RetrieveConfig(sources=list(sources), per_node_materials=3)
    cfg.local.paths = [str(corpus)]
    cfg.local.chunk_size = 200
    return cfg


def test_retriever_end_to_end_local(corpus):
    store, leaf = _store_with_kernel()
    bundle = SampleBundle(nodes=[leaf.id], intent=SampleIntent.DEPTH)
    cfg = _local_config(corpus)
    mats = Retriever(build_sources(cfg), cfg).retrieve(store, bundle)
    assert mats
    assert all(leaf.id in m.linked_nodes for m in mats)


def test_retriever_skips_failing_source_when_others_succeed(corpus, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    store, leaf = _store_with_kernel()
    bundle = SampleBundle(nodes=[leaf.id], intent=SampleIntent.DEPTH)
    cfg = _local_config(corpus, sources=("local", "web"))
    mats = Retriever(build_sources(cfg), cfg).retrieve(store, bundle)
    assert mats  # web failed (no API key) but local succeeded


def test_retriever_reraises_when_all_sources_fail(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    store, leaf = _store_with_kernel()
    bundle = SampleBundle(nodes=[leaf.id], intent=SampleIntent.DEPTH)
    cfg = RetrieveConfig(sources=["web"])
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        Retriever(build_sources(cfg), cfg).retrieve(store, bundle)
