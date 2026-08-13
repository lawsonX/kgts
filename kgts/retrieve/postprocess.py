"""Stage C post-processing (design doc section 6.3).

License filtering, cross-source dedup (URL normalization or content simhash)
and a pure-python relevance rerank against the node description.
"""

from __future__ import annotations

import hashlib
import urllib.parse

from kgts.models import Material, SourceType
from kgts.retrieve.text import tokenize

_LICENSE_WHITELIST = {None, "unknown", "cc-by", "cc0", "public-domain", "mit", "apache-2.0"}


def _tokens(text: str) -> list[str]:
    return tokenize(text)


def simhash64(text: str) -> int:
    """64-bit simhash over token 3-shingles with md5 fingerprints."""
    tokens = _tokens(text)
    if not tokens:
        return 0
    if len(tokens) < 4:
        shingles = tokens
    else:
        shingles = [" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    acc = [0] * 64
    for shingle in shingles:
        digest = int.from_bytes(hashlib.md5(shingle.encode()).digest()[:8], "big")
        for bit in range(64):
            acc[bit] += 1 if (digest >> bit) & 1 else -1
    out = 0
    for bit in range(64):
        if acc[bit] > 0:
            out |= 1 << bit
    return out


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _normalize_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url.strip())
    host = parts.netloc.lower()
    query = [
        (k, v)
        for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_")
    ]
    path = parts.path.rstrip("/")
    return urllib.parse.urlunsplit(
        (parts.scheme.lower(), host, path, urllib.parse.urlencode(query), "")
    )


def dedup(materials: list[Material], mode: str = "simhash", threshold: int = 3) -> list[Material]:
    """Drop duplicate materials, keeping the first occurrence.

    ``mode="url"`` normalizes URIs (lowercase host, strip trailing slash and
    utm_* params); ``mode="simhash"`` drops materials whose content simhash is
    within ``threshold`` hamming distance of an already-kept one.
    """
    if mode in ("off", "none"):
        return list(materials)
    if mode == "url":
        seen: set[str] = set()
        out = []
        for m in materials:
            if m.uri:
                key = _normalize_url(m.uri)
            elif m.path:
                key = f"path:{m.path}"
            else:
                key = f"id:{m.id}"
            if key in seen:
                continue
            seen.add(key)
            out.append(m)
        return out
    if mode == "simhash":
        kept: list[tuple[int, Material]] = []
        for m in materials:
            h = simhash64(" ".join([m.title, m.text or m.snippet]))
            if any(_hamming(h, kept_h) <= threshold for kept_h, _ in kept):
                continue
            kept.append((h, m))
        return [m for _, m in kept]
    raise ValueError(f"unknown dedup mode {mode!r}; use 'simhash', 'url' or 'off'")


def rerank(materials: list[Material], node_description: str) -> list[Material]:
    """Token-overlap rerank against the node description (stable, descending).

    This is the pure-python stand-in for the cross-encoder / LLM rerankers the
    config names; it keeps ordering deterministic and offline.
    """
    desc = set(_tokens(node_description))
    if not desc:
        return list(materials)

    def score(m: Material) -> float:
        toks = set(_tokens(" ".join([m.title, m.snippet, m.text])))
        return len(toks & desc) / len(desc)

    return sorted(materials, key=lambda m: -score(m))


def filter_license(materials: list[Material], mode: str = "whitelist") -> list[Material]:
    """License filter: local/arxiv/github are always kept; web materials must
    carry a permissive (or unrecorded) license. ``mode="off"`` keeps all."""
    if mode == "off":
        return list(materials)
    if mode != "whitelist":
        raise ValueError(f"unknown license mode {mode!r}; use 'whitelist' or 'off'")
    out = []
    for m in materials:
        if m.source_type != SourceType.WEB:
            out.append(m)
            continue
        lic = m.license.lower() if m.license else None
        if lic in _LICENSE_WHITELIST:
            out.append(m)
    return out
