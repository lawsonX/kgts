"""SQLite persistence for pipeline artifacts (design appendix C).

Each artifact class is stored as a full pydantic JSON blob in a ``data``
column, plus a few indexed columns for filtering. Writes are idempotent
(``INSERT OR REPLACE``), so re-running a stage never duplicates rows.

The store may share a db file with :class:`kgts.graph.store.GraphStore`
persistence -- the table names do not overlap.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from kgts.models import AlignDecision, Material, Run, Task

_SCHEMA = """
CREATE TABLE IF NOT EXISTS materials (
  id TEXT PRIMARY KEY, source_type TEXT NOT NULL, license TEXT, data JSON NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY, task_type TEXT NOT NULL, verify_result TEXT NOT NULL,
  run_id TEXT, data JSON NOT NULL
);
CREATE TABLE IF NOT EXISTS align_verdicts (
  id TEXT PRIMARY KEY, verdict TEXT NOT NULL, decided_at TEXT NOT NULL, data JSON NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, config_hash TEXT NOT NULL, started_at TEXT NOT NULL,
  finished_at TEXT, data JSON NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_type ON tasks(task_type);
CREATE INDEX IF NOT EXISTS idx_materials_st ON materials(source_type);
"""


class ArtifactStore:
    """Artifact persistence for materials, tasks, align verdicts, and runs."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        if self.db_path.parent != self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    # ------------------------------------------------------------- materials
    def save_material(self, m: Material) -> None:
        self.save_materials([m])

    def save_materials(self, materials: list[Material]) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO materials (id, source_type, license, data)"
                    " VALUES (?,?,?,?)",
                    [
                        (m.id, m.source_type.value, m.license, m.model_dump_json())
                        for m in materials
                    ],
                )
        finally:
            conn.close()

    def load_materials(self, ids: list[str] | None = None) -> list[Material]:
        conn = self._connect()
        try:
            if ids is None:
                rows = conn.execute("SELECT data FROM materials").fetchall()
            elif not ids:
                return []
            else:
                marks = ",".join("?" for _ in ids)
                rows = conn.execute(
                    f"SELECT data FROM materials WHERE id IN ({marks})", ids
                ).fetchall()
        finally:
            conn.close()
        return [Material.model_validate_json(r[0]) for r in rows]

    def load_materials_by_id(self, ids: list[str]) -> dict[str, Material]:
        return {m.id: m for m in self.load_materials(ids)}

    # ----------------------------------------------------------------- tasks
    def save_task(self, t: Task) -> None:
        self.save_tasks([t])

    def save_tasks(self, tasks: list[Task]) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO tasks (id, task_type, verify_result, run_id, data)"
                    " VALUES (?,?,?,?,?)",
                    [
                        (t.id, t.task_type, t.verify_result.value, t.run_id, t.model_dump_json())
                        for t in tasks
                    ],
                )
        finally:
            conn.close()

    def load_tasks(self, run_id: str | None = None) -> list[Task]:
        conn = self._connect()
        try:
            if run_id is None:
                rows = conn.execute("SELECT data FROM tasks").fetchall()
            else:
                rows = conn.execute(
                    "SELECT data FROM tasks WHERE run_id = ?", (run_id,)
                ).fetchall()
        finally:
            conn.close()
        return [Task.model_validate_json(r[0]) for r in rows]

    # --------------------------------------------------------- align verdicts
    def save_align_decision(self, d: AlignDecision) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO align_verdicts (id, verdict, decided_at, data)"
                    " VALUES (?,?,?,?)",
                    (d.id, d.verdict.value, d.decided_at, d.model_dump_json()),
                )
        finally:
            conn.close()

    def load_align_decisions(self, verdict: str | None = None) -> list[AlignDecision]:
        conn = self._connect()
        try:
            if verdict:
                rows = conn.execute(
                    "SELECT data FROM align_verdicts WHERE verdict=? ORDER BY decided_at",
                    (verdict,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT data FROM align_verdicts ORDER BY decided_at"
                ).fetchall()
        finally:
            conn.close()
        return [AlignDecision.model_validate_json(r[0]) for r in rows]

    # ------------------------------------------------------------------- runs
    def create_run(self, run: Run) -> None:
        conn = self._connect()
        try:
            with conn:
                row = (run.id, run.config_hash, run.started_at, run.finished_at)
                conn.execute(
                    "INSERT OR REPLACE INTO runs (id, config_hash, started_at, finished_at, data)"
                    " VALUES (?,?,?,?,?)",
                    (*row, run.model_dump_json()),
                )
        finally:
            conn.close()

    def finish_run(self, run: Run) -> None:
        """Persist ``finished_at`` plus the full run blob (stage stats, usage)."""
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "UPDATE runs SET finished_at = ?, data = ? WHERE id = ?",
                    (run.finished_at, run.model_dump_json(), run.id),
                )
        finally:
            conn.close()

    def load_run(self, run_id: str) -> Run | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT data FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        finally:
            conn.close()
        return Run.model_validate_json(row[0]) if row else None

    def list_runs(self) -> list[Run]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT data FROM runs ORDER BY started_at").fetchall()
        finally:
            conn.close()
        return [Run.model_validate_json(r[0]) for r in rows]
