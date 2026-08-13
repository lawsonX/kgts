"""Shared lightweight tokenizer for retrieval scoring.

ASCII text is split into words; CJK text (no whitespace separators) is broken
into character unigrams + bigrams, the standard no-dependency approach for
Chinese/Japanese/Korean retrieval. Used by the local corpus TF-IDF, the
relevance reranker, and simhash dedup — all three must tokenize identically.
"""

from __future__ import annotations

import re

_ASCII_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[一-鿿぀-ヿ가-힯]+")


def tokenize(text: str) -> list[str]:
    """ASCII words + CJK unigrams/bigrams, lowercased."""
    text = text.lower()
    tokens = _ASCII_RE.findall(text)
    for run in _CJK_RE.findall(text):
        tokens.extend(run)  # unigrams: recall
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))  # bigrams: precision
    return tokens
