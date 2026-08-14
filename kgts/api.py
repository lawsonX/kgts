"""Stable consumption API for external data-synthesis agents (design §7a).

KGTS deliberately decouples *graph creation & consumption* from *dataset
generation*. Task synthesis at production quality needs user-specific
requirement gathering, trial production and diversity control — that belongs
to the user's own agent. KGTS provides the knowledge DAG, sampling bundles,
and anchored materials through this small, stable surface.

Three consumption levels (see docs/consuming.md):
- files: workdir checkpoints + exports (language-agnostic)
- CLI: ``kgts build|sample|retrieve|graph --export`` ...
- Python: this module

Everything here is read-only with respect to the pipeline.
"""

from __future__ import annotations

from pathlib import Path

from kgts.config import Config, load_config
from kgts.graph.store import GraphStore
from kgts.models import Material, SampleBundle
from kgts.orchestrate.runner import load_bundles, load_materials_json


def load_graph(workdir: str | Path) -> GraphStore:
    """Load the knowledge DAG checkpoint (``workdir/graph.db``).

    The GraphStore gives you: ``nodes()``, ``edges()``, ``children()``,
    ``parents()``, ``ancestors()``, ``ancestor_path()`` (label crumbs for
    disambiguation), ``atomic_nodes()`` and per-node ``stats``.
    """
    return GraphStore.load(Path(workdir) / "graph.db")


def load_sample_bundles(workdir: str | Path) -> list[SampleBundle]:
    """Load the sampled bundles (``workdir/bundles.json``).

    Each bundle carries node ids, full ancestor paths, level and intent
    (breadth/depth/joint) — everything a generator needs to know *what* to
    synthesize about.
    """
    return load_bundles(_as_config(workdir), "api")


def load_materials(workdir: str | Path) -> list[Material]:
    """Load retrieved materials (``workdir/materials.json``)."""
    return load_materials_json(_as_config(workdir), "api")


def _as_config(workdir: str | Path) -> Config:
    cfg = Config()
    cfg.run.workdir = str(workdir)
    return cfg


__all__ = [
    "Config",
    "GraphStore",
    "Material",
    "SampleBundle",
    "load_config",
    "load_graph",
    "load_materials",
    "load_sample_bundles",
]
