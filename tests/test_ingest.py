"""Offline tests for the CorpusAdapterAgent (agentic input compatibility)."""

import json

from kgts.config import LocalSourceConfig
from kgts.llm import MockLLM
from kgts.retrieve.ingest import (
    CorpusAdapterAgent,
    ExtractionSpec,
    analyze_corpus,
    extract_documents,
    extract_text,
)
from kgts.retrieve.sources import LocalCorpusSource

OCR_SPEC = ExtractionSpec(format="jsonl", join_dict_field="ocr", notes="OCR dump")


# ----------------------------------------------------------------- interpreter
def test_extract_documents_jsonl_ocr_dict():
    line = json.dumps({"filename": "a.pdf", "ocr": {"k2": "第二段", "k1": "第一段", "k3": ""}})
    docs = extract_documents(line + "\n{bad json}\n", OCR_SPEC)
    assert docs == ["第一段\n第二段"]


def test_extract_documents_text_fields_and_dotted_path():
    spec = ExtractionSpec(format="jsonl", text_fields=["payload.body", "text"])
    assert extract_text({"payload": {"body": "  正文  "}}, spec) == "正文"
    assert extract_text({"text": "备选"}, spec) == "备选"
    assert extract_text({"other": 1}, spec) is None
    docs = extract_documents('{"payload": {"body": "内容"}}\n', spec)
    assert docs == ["内容"]


def test_extract_documents_csv_json_text():
    csv_spec = ExtractionSpec(format="csv", text_fields=["body"])
    assert extract_documents("id,body\n1,第一条\n2,第二条\n", csv_spec) == ["第一条", "第二条"]
    json_spec = ExtractionSpec(format="json", join_list_field="paras")
    assert extract_documents('{"paras": ["a", "b"]}', json_spec) == ["a\nb"]  # one record
    json_list = ExtractionSpec(format="json", text_fields=["t"])
    assert extract_documents('[{"t": "x"}, {"t": ""}, {"t": "y"}]', json_list) == ["x", "y"]
    text_spec = ExtractionSpec(format="text")
    assert extract_documents("  整篇文档  ", text_spec) == ["整篇文档"]


# ---------------------------------------------------------------------- agent
def _sample_dir(tmp_path):
    (tmp_path / "book.jsonl").write_text(
        json.dumps(
            {"page": 1,
             "ocr": {"a": "犯罪现场保护是侦查工作的首要环节", "b": "提取物证需要遵循法定程序"}},
            ensure_ascii=False,
        )
    )
    return [str(tmp_path)]


def test_agent_infers_and_verifies_spec(tmp_path):
    llm = MockLLM(default=OCR_SPEC.model_dump())
    agent = CorpusAdapterAgent(llm)
    samples = agent.sample_files(_sample_dir(tmp_path))
    spec = agent.infer_spec(samples)
    assert spec is not None and spec.join_dict_field == "ocr"


def test_agent_self_corrects_on_failed_verification(tmp_path):
    # first plan points at a wrong field; verification fails; retry succeeds
    wrong = ExtractionSpec(format="jsonl", text_fields=["nonexistent"]).model_dump()
    right = OCR_SPEC.model_dump()
    llm = MockLLM(script={"failed verification": right}, default=wrong)
    agent = CorpusAdapterAgent(llm, max_attempts=2)
    spec = agent.infer_spec(agent.sample_files(_sample_dir(tmp_path)))
    assert spec is not None and spec.join_dict_field == "ocr"
    assert any("failed verification" in c for c in llm.calls)  # the retry prompt fired


def test_agent_gives_up_and_falls_back(tmp_path):
    llm = MockLLM(default={"bogus": True})  # schema validation fails
    agent = CorpusAdapterAgent(llm)
    assert agent.infer_spec(agent.sample_files(_sample_dir(tmp_path))) is None
    assert analyze_corpus(_sample_dir(tmp_path), None) is None  # no LLM -> heuristics


def test_sample_files_skips_binary_and_groups_by_extension(tmp_path):
    (tmp_path / "a.txt").write_text("文本")
    (tmp_path / "b.bin").write_bytes(b"\x00\x01\x02")
    (tmp_path / "c.txt").write_text("另一篇")
    samples = CorpusAdapterAgent.sample_files([str(tmp_path)])
    assert len(samples) == 1 and next(iter(samples)).endswith("a.txt")


# ------------------------------------------------- LocalCorpusSource integration
def test_local_source_uses_spec_for_arbitrary_extension(tmp_path):
    (tmp_path / "dump.data").write_text(  # extension not in the heuristic allowlist
        json.dumps({"ocr": {"1": "视频侦查追踪嫌疑人轨迹"}}, ensure_ascii=False)
    )
    src = LocalCorpusSource(
        LocalSourceConfig(paths=[str(tmp_path)], chunk_size=500), spec=OCR_SPEC
    )
    hits = src.search(["视频侦查"], budget=3)
    assert hits and "嫌疑人轨迹" in hits[0].text
    # without the spec, the same file is invisible to heuristics
    src_plain = LocalCorpusSource(LocalSourceConfig(paths=[str(tmp_path)]))
    assert src_plain.search(["视频侦查"], budget=3) == []


def test_text_fields_are_container_tolerant():
    """LLMs often write an OCR-dump field into text_fields instead of
    join_dict_field; the interpreter must still extract it."""
    spec = ExtractionSpec(format="jsonl", text_fields=["ocr"])
    assert extract_text({"ocr": {"b": "乙", "a": "甲"}}, spec) == "甲\n乙"
    assert extract_text({"ocr": ["甲", "乙"]}, spec) == "甲\n乙"
