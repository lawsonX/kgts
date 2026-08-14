"""Core data models for KGTS.

Every artifact in the pipeline (graph nodes, materials, tasks, runs) is a
pydantic model with provenance, so any training example can be traced back to
``Task -> SampleBundle -> Node -> Material -> Run`` (see design doc section 3).
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def slugify(label: str) -> str:
    """Normalize a label into a stable slug used for node ids.

    Keeps unicode word characters so CJK labels stay readable
    (``\\w`` is unicode-aware in Python 3).
    """
    slug = re.sub(r"[^\w]+", "-", label.lower()).strip("-")
    return slug or "node"


def make_node_id(label: str) -> str:
    """Deterministic node id: slug of the canonical label + short hash.

    Deterministic so that re-discovering the same canonical label across runs
    maps to the same id (alias index + cache reuse rely on this).
    """
    digest = hashlib.sha1(label.strip().lower().encode()).hexdigest()[:8]
    return f"n_{slugify(label)[:48]}-{digest}"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class NodeStatus(str, Enum):
    EXPANDING = "expanding"  # queued for / under exploration
    ATOMIC = "atomic"  # terminal: fine-grained enough, stops growing
    MERGED = "merged"  # folded into another node; alias index only
    FROZEN = "frozen"  # manually locked


class Relation(str, Enum):
    IS_SUBCONCEPT = "is_subconcept"  # coarse -> fine (default)
    IS_RELATED = "is_related"  # lateral link; excluded from level computation


class Ref(BaseModel):
    """A provenance pointer: which material / exploration produced something."""

    kind: str  # "material" | "exploration" | "seed" | "human"
    ref_id: str
    note: str = ""


class NodeStats(BaseModel):
    n_materials: int = 0
    synth_success_rate: float | None = None
    times_sampled: int = 0
    ece: float | None = None  # optional blind-spot score (v0.4+ plugin)
    child_material_overlap: float | None = None


class Node(BaseModel):
    id: str
    label: str  # canonical name (product of alignment canonicalization)
    aliases: list[str] = Field(default_factory=list)
    level: int = 0  # shortest-path depth from nearest seed
    description: str = ""
    status: NodeStatus = NodeStatus.EXPANDING
    provenance: list[Ref] = Field(default_factory=list)
    stats: NodeStats = Field(default_factory=NodeStats)
    embedding: list[float] | None = None

    @classmethod
    def create(cls, label: str, **kw: Any) -> Node:
        return cls(id=make_node_id(label), label=label, **kw)


class Edge(BaseModel):
    parent: str  # NodeID, coarser
    child: str  # NodeID, finer
    relation: Relation = Relation.IS_SUBCONCEPT
    confidence: float = 1.0


class SourceType(str, Enum):
    WEB = "web"
    LOCAL = "local"
    GITHUB = "github"
    ARXIV = "arxiv"


class Material(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("m"))
    source_type: SourceType
    uri: str | None = None
    path: str | None = None
    title: str = ""
    snippet: str = ""
    full_text_ref: str | None = None  # pointer to chunk store / blob
    text: str = ""  # inline text for small/local chunks
    license: str | None = None  # mandatory to record for web materials
    fetched_at: str = Field(default_factory=utc_now)
    quality_score: float = 0.0
    linked_nodes: list[str] = Field(default_factory=list)


class SampleIntent(str, Enum):
    BREADTH = "breadth"
    DEPTH = "depth"
    JOINT = "joint"


class SampleBundle(BaseModel):
    """Output of the sampler (Stage B): what to synthesize about, with the full
    ancestor paths the retriever (Stage C) needs for disambiguation."""

    id: str = Field(default_factory=lambda: _new_id("sb"))
    nodes: list[str]  # NodeIDs
    ancestor_paths: dict[str, list[str]] = Field(default_factory=dict)  # node -> path of labels
    level: int = 0
    intent: SampleIntent


class AlignVerdict(str, Enum):
    EQUIVALENT = "equivalent"  # reuse existing node
    RELATED = "related"  # keep both, add is_related edge
    DISTINCT = "distinct"  # create new node


class AlignDecision(BaseModel):
    """An alignment judgment, persisted for human review and prompt iteration."""

    id: str = Field(default_factory=lambda: _new_id("av"))
    candidate_label: str
    matched_node: str | None = None
    verdict: AlignVerdict
    canonical_suggestion: str = ""
    judge_model: str = ""
    prompt_hash: str = ""
    decided_at: str = Field(default_factory=utc_now)


class VerifyResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SFT_ONLY = "sft_only"  # no verifier registered; usable for SFT only
    UNVERIFIED = "unverified"


class Task(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("t"))
    task_type: str
    sample_bundle: SampleBundle
    materials: list[str] = Field(default_factory=list)  # MaterialIDs
    prompt: str = ""
    question: str = ""
    answer: str = ""
    rubric: list[str] = Field(default_factory=list)
    verifier: str | None = None
    verify_result: VerifyResult = VerifyResult.UNVERIFIED
    style: dict[str, Any] = Field(default_factory=dict)
    run_id: str = ""


class Run(BaseModel):
    """Audit unit for one pipeline execution."""

    id: str = Field(default_factory=lambda: _new_id("run"))
    started_at: str = Field(default_factory=utc_now)
    finished_at: str | None = None
    config_hash: str = ""
    stage_stats: dict[str, Any] = Field(default_factory=dict)
    llm_usage: dict[str, Any] = Field(default_factory=dict)
