"""Agentic input-data compatibility: CorpusAdapterAgent (design §6.2a).

Instead of hardcoding one adapter per corpus format, an agent inspects raw
samples from the corpus and infers a *declarative* extraction spec. The spec
is executed by a deterministic interpreter (no LLM-generated code is ever
run), verified on real samples with one self-correction round, persisted to
the workdir for reuse, and falls back to built-in heuristics when no LLM is
available or inference fails.

Loop: observe (sample files) -> plan (LLM -> ExtractionSpec) -> execute
(apply_spec) -> verify (stats on real records; retry once with feedback).
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kgts.llm import LLMClient


# --------------------------------------------------------------------- spec
class ExtractionSpec(BaseModel):
    """Declarative record -> text mapping for one corpus format."""

    model_config = ConfigDict(extra="forbid")  # a hallucinated key = an invalid plan

    format: Literal["jsonl", "json", "csv", "text"] = "text"
    text_fields: list[str] = Field(default_factory=list)  # dotted paths, first hit wins
    join_dict_field: str | None = None  # join string values of this dict (key-sorted)
    join_list_field: str | None = None  # join string items of this list
    title_field: str | None = None
    delimiter: str = ","  # csv only
    notes: str = ""  # agent's rationale, kept for audit


def _resolve(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)] if int(part) < len(cur) else None
        else:
            return None
    return cur


def extract_text(record: Any, spec: ExtractionSpec) -> str | None:
    """Apply a spec to one parsed record; None means 'no text here'.

    ``text_fields`` are container-tolerant: a field resolving to a dict/list
    of strings is joined (real LLMs often put an OCR-dump field in
    ``text_fields`` instead of ``join_dict_field`` — the verification loop
    should not have to fail for that).
    """
    if isinstance(record, str):
        return record.strip() or None
    if not isinstance(record, (dict, list)):
        return None
    for field in spec.text_fields:
        value = _resolve(record, field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            joined = "\n".join(
                v.strip() for _, v in sorted(value.items()) if isinstance(v, str) and v.strip()
            )
            if joined:
                return joined
        if isinstance(value, list):
            joined = "\n".join(v.strip() for v in value if isinstance(v, str) and v.strip())
            if joined:
                return joined
    if spec.join_dict_field:
        value = _resolve(record, spec.join_dict_field)
        if isinstance(value, dict):
            joined = "\n".join(
                v.strip() for _, v in sorted(value.items()) if isinstance(v, str) and v.strip()
            )
            if joined:
                return joined
    if spec.join_list_field:
        value = _resolve(record, spec.join_list_field)
        if isinstance(value, list):
            joined = "\n".join(v.strip() for v in value if isinstance(v, str) and v.strip())
            if joined:
                return joined
    return None


def extract_documents(raw: str, spec: ExtractionSpec) -> list[str]:
    """Split a whole file into documents per the spec."""
    if spec.format == "text":
        return [raw.strip()] if raw.strip() else []
    if spec.format == "csv":
        rows = csv.DictReader(io.StringIO(raw), delimiter=spec.delimiter)
        return [t for row in rows if (t := extract_text(dict(row), spec))]
    if spec.format == "json":
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return []
        records = obj if isinstance(obj, list) else [obj]
        return [t for r in records if (t := extract_text(r, spec))]
    # jsonl
    docs: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (t := extract_text(record, spec)):
            docs.append(t)
    return docs


# ------------------------------------------------------------------- agent
_PROMPT = """You are a data-ingestion engineer. Decide how to extract plain-text
documents from the corpus files sampled below.

Rules:
- Reply with ONE JSON object matching this schema (no markdown, no commentary):
  {{"format": "jsonl"|"json"|"csv"|"text",
    "text_fields": ["dotted.path", ...],   // first non-empty string wins
    "join_dict_field": "field"|null,       // join string values of this dict, key-sorted
    "join_list_field": "field"|null,       // join string items of this list
    "title_field": "field"|null,
    "delimiter": ",",                       // csv only
    "notes": "one sentence: what the data is and why this mapping"}}
- "jsonl" = one JSON object per line; "json" = whole file is a JSON array/object.
- text_fields may point at a plain string OR a container of strings: a dict
  (e.g. OCR position->text maps) or list has its string values joined, in
  key/order. Use join_dict_field / join_list_field for the explicit form.
- The extracted text will be chunked and indexed for retrieval; JSON syntax,
  coordinates and ids must NOT leak into it.

Samples (truncated):
{samples}
"""

_RETRY_PROMPT = """Your previous extraction spec failed verification.
Spec: {spec}
Result: extracted {n_ok}/{n_total} records, sample output: {sample!r}
Fix the spec. Reply with ONE corrected JSON object, same schema as before."""


class CorpusAdapterAgent:
    """Infers an ExtractionSpec for an arbitrary corpus via LLM + verification."""

    def __init__(self, llm: LLMClient, max_attempts: int = 2):
        self.llm = llm
        self.max_attempts = max(1, max_attempts)

    @staticmethod
    def sample_files(
        paths: list[str], *, max_files: int = 3, read_bytes: int = 3000
    ) -> dict[str, str]:
        """Pick representative text-readable files (one per extension)."""
        samples: dict[str, str] = {}
        seen_ext: set[str] = set()
        for base in paths:
            root = Path(base)
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or len(samples) >= max_files:
                    continue
                ext = path.suffix.lower()
                if ext in seen_ext:
                    continue
                try:
                    head = path.read_bytes()[:read_bytes]
                except OSError:
                    continue
                if b"\x00" in head:  # binary
                    continue
                try:
                    text = head.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    continue
                seen_ext.add(ext)
                samples[str(path)] = text
        return samples

    def infer_spec(self, samples: dict[str, str]) -> ExtractionSpec | None:
        """observe -> plan -> execute -> verify (one self-correction round)."""
        if not samples:
            return None
        sample_block = "\n\n".join(
            f"--- {name} ---\n{text[:1200]}" for name, text in samples.items()
        )
        prompt = _PROMPT.format(samples=sample_block)
        for attempt in range(self.max_attempts):
            try:
                reply = self.llm.complete_json(prompt, temperature=0.0)
                spec = ExtractionSpec.model_validate(reply)
            except Exception:
                return None  # unparseable plan: caller falls back to heuristics
            ok, stats = self._verify(samples, spec)
            if ok:
                return spec
            if attempt + 1 >= self.max_attempts:
                return None
            prompt = _RETRY_PROMPT.format(spec=spec.model_dump_json(), **stats)
        return None

    @staticmethod
    def _verify(samples: dict[str, str], spec: ExtractionSpec) -> tuple[bool, dict]:
        """A spec passes if it extracts real text from most sampled files."""
        n_ok, n_total, example = 0, 0, ""
        for text in samples.values():
            docs = extract_documents(text, spec)
            n_total += 1
            usable = [d for d in docs if len(d.strip()) >= 20]
            if usable:
                n_ok += 1
                if not example:
                    example = usable[0][:120]
        passed = n_ok > 0 and n_ok >= n_total / 2
        return passed, {"n_ok": n_ok, "n_total": n_total, "sample": example}


def analyze_corpus(paths: list[str], llm: LLMClient | None) -> ExtractionSpec | None:
    """Top-level helper: sample the corpus and infer a spec (None = use heuristics)."""
    if llm is None:
        return None
    agent = CorpusAdapterAgent(llm)
    return agent.infer_spec(agent.sample_files(paths))
