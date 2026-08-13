"""Stage C: material retrieval."""

from kgts.retrieve.query import QueryBuilder
from kgts.retrieve.retriever import Retriever
from kgts.retrieve.sources import (
    ArxivSource,
    GitHubSource,
    LocalCorpusSource,
    MaterialSource,
    WebSearchSource,
    build_sources,
)

__all__ = [
    "ArxivSource",
    "GitHubSource",
    "LocalCorpusSource",
    "MaterialSource",
    "QueryBuilder",
    "Retriever",
    "WebSearchSource",
    "build_sources",
]
