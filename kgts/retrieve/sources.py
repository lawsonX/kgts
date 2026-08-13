"""Stage C material sources (design doc section 6.2).

Pure-python + stdlib implementations only: HTTP via urllib, TF-IDF and XML
parsing implemented here. Every network source raises RuntimeError on failure
instead of silently returning nothing.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Protocol

from kgts.config import LocalSourceConfig, RetrieveConfig, WebSourceConfig
from kgts.models import Material, SourceType
from kgts.retrieve.text import tokenize


def _tokens(text: str) -> list[str]:
    return tokenize(text)


class MaterialSource(Protocol):
    """Pluggable material source contract."""

    def search(self, queries: list[str], budget: int) -> list[Material]:
        """Return up to ``budget`` materials for ``queries``."""
        ...


class LocalCorpusSource:
    """Chunks .txt/.md/.jsonl files under ``config.paths`` and scores chunks
    with a pure-python TF-IDF (log tf * idf, cosine over query terms).

    The index is built lazily on first search; missing directories are
    tolerated (they simply contribute no chunks).
    """

    def __init__(self, config: LocalSourceConfig) -> None:
        self.config = config
        self._chunks: list[dict] | None = None
        self._idf: dict[str, float] = {}

    # --------------------------------------------------------------- indexing
    def _index(self) -> None:
        chunks: list[dict] = []
        for base in self.config.paths:
            root = Path(base)
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".jsonl"}:
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for piece in self._chunk(text):
                    chunks.append({"path": str(path), "title": path.name, "text": piece})
        df: Counter[str] = Counter()
        for chunk in chunks:
            chunk["tf"] = Counter(_tokens(chunk["text"]))
            df.update(set(chunk["tf"]))
        n = max(len(chunks), 1)
        self._idf = {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df.items()}
        self._chunks = chunks

    def _chunk(self, text: str) -> list[str]:
        size, overlap = self.config.chunk_size, self.config.chunk_overlap
        if size <= 0:
            return [text.strip()] if text.strip() else []
        step = max(size - overlap, 1)
        out = []
        for start in range(0, len(text), step):
            piece = text[start : start + size].strip()
            if piece:
                out.append(piece)
            if start + size >= len(text):
                break
        return out

    # --------------------------------------------------------------- search
    def search(self, queries: list[str], budget: int) -> list[Material]:
        if self._chunks is None:
            self._index()
        assert self._chunks is not None
        q_tf = Counter(_tokens(" ".join(queries)))
        if not q_tf or not self._chunks or budget <= 0:
            return []
        q_vec = {t: (1.0 + math.log(c)) * self._idf.get(t, 0.0) for t, c in q_tf.items()}
        q_norm = math.sqrt(sum(w * w for w in q_vec.values())) or 1.0
        scored: list[tuple[float, dict]] = []
        for chunk in self._chunks:
            tf: Counter[str] = chunk["tf"]
            c_vec = {
                t: (1.0 + math.log(tf[t])) * self._idf.get(t, 0.0) for t in q_tf if t in tf
            }
            if not c_vec:
                continue
            dot = sum(c_vec[t] * q_vec[t] for t in c_vec)
            c_norm = math.sqrt(
                sum(((1.0 + math.log(v)) * self._idf.get(t, 0.0)) ** 2 for t, v in tf.items())
            ) or 1.0
            scored.append((dot / (q_norm * c_norm), chunk))
        scored.sort(key=lambda item: -item[0])  # stable: earlier chunks win ties
        return [
            Material(
                source_type=SourceType.LOCAL,
                path=chunk["path"],
                title=chunk["title"],
                snippet=chunk["text"][:200],
                text=chunk["text"],
                quality_score=score,
            )
            for score, chunk in scored[:budget]
        ]


class WebSearchSource:
    """Tavily web search (POST https://api.tavily.com/search) via urllib."""

    ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, config: WebSourceConfig) -> None:
        self.config = config

    def search(self, queries: list[str], budget: int) -> list[Material]:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"web source needs an API key: set {self.config.api_key_env} "
                "or disable the web source (retrieve.sources)"
            )
        materials: list[Material] = []
        for query in queries:
            if len(materials) >= budget:
                break
            payload = json.dumps(
                {"api_key": api_key, "query": query, "max_results": budget - len(materials)}
            ).encode()
            req = urllib.request.Request(
                self.ENDPOINT, data=payload, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"tavily search failed: HTTP {e.code}") from e
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
                raise RuntimeError(f"tavily search failed: {e}") from e
            for item in data.get("results", []):
                materials.append(
                    Material(
                        source_type=SourceType.WEB,
                        uri=item.get("url"),
                        title=item.get("title", ""),
                        snippet=(item.get("content") or "")[:500],
                        license="unknown",
                    )
                )
        return materials[:budget]


class GitHubSource:
    """Unauthenticated GitHub repository search via urllib."""

    ENDPOINT = "https://api.github.com/search/repositories"

    def search(self, queries: list[str], budget: int) -> list[Material]:
        materials: list[Material] = []
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "kgts"}
        for query in queries:
            if len(materials) >= budget:
                break
            params = urllib.parse.urlencode({"q": query, "per_page": budget - len(materials)})
            req = urllib.request.Request(f"{self.ENDPOINT}?{params}", headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"github search failed: HTTP {e.code}") from e
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
                raise RuntimeError(f"github search failed: {e}") from e
            for item in data.get("items", []):
                spdx = (item.get("license") or {}).get("spdx_id")
                materials.append(
                    Material(
                        source_type=SourceType.GITHUB,
                        uri=item.get("html_url"),
                        title=item.get("full_name", ""),
                        snippet=(item.get("description") or "")[:500],
                        license=spdx.lower() if spdx and spdx != "NOASSERTION" else "unknown",
                    )
                )
        return materials[:budget]


class ArxivSource:
    """arXiv export API (GET) parsed with ElementTree."""

    ENDPOINT = "https://export.arxiv.org/api/query"
    _NS = {"atom": "http://www.w3.org/2005/Atom"}

    def search(self, queries: list[str], budget: int) -> list[Material]:
        materials: list[Material] = []
        for query in queries:
            if len(materials) >= budget:
                break
            params = urllib.parse.urlencode(
                {"search_query": f"all:{query}", "start": 0, "max_results": budget - len(materials)}
            )
            try:
                with urllib.request.urlopen(f"{self.ENDPOINT}?{params}", timeout=30) as resp:
                    xml_text = resp.read().decode()
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"arxiv search failed: HTTP {e.code}") from e
            except (urllib.error.URLError, OSError) as e:
                raise RuntimeError(f"arxiv search failed: {e}") from e
            try:
                root = ET.fromstring(xml_text)
            except ET.ParseError as e:
                raise RuntimeError(f"arxiv search returned invalid XML: {e}") from e
            for entry in root.findall("atom:entry", self._NS):
                title = entry.findtext("atom:title", default="", namespaces=self._NS)
                summary = entry.findtext("atom:summary", default="", namespaces=self._NS)
                link = entry.findtext("atom:id", default="", namespaces=self._NS)
                materials.append(
                    Material(
                        source_type=SourceType.ARXIV,
                        uri=link.strip() or None,
                        title=" ".join(title.split()),
                        snippet=" ".join(summary.split())[:500],
                    )
                )
        return materials[:budget]


def build_sources(config: RetrieveConfig) -> dict[str, MaterialSource]:
    """Instantiate the sources named in ``config.sources`` (orchestrator contract)."""
    factories = {
        "local": lambda: LocalCorpusSource(config.local),
        "web": lambda: WebSearchSource(config.web),
        "github": GitHubSource,
        "arxiv": ArxivSource,
    }
    sources: dict[str, MaterialSource] = {}
    for name in config.sources:
        if name not in factories:
            valid = ", ".join(sorted(factories))
            raise ValueError(f"unknown material source {name!r}; valid sources: {valid}")
        sources[name] = factories[name]()
    return sources
