"""SQLite persistence.

Two tables carry the whole state:

* ``input_versions`` – append-only versions of each input (baseline,
  samples and the optional handover manifest).  Exactly one row per
  ``kind`` is flagged ``is_current``.  A report names the version ids it
  was computed from, so "is this report stale?" is always a comparison
  against the current versions rather than a flag that could drift.
* ``reports`` – only fully computed reports are ever inserted.  The insert
  re-checks that the snapshot versions are still current *inside* the same
  ``BEGIN IMMEDIATE`` transaction that writes the report, which is what
  makes publishing safe when an upload interleaves with a calculation.

All writes from this process are serialised by a re-entrant lock; SQLite's
own locking covers other processes/workers.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

KINDS = ("baseline", "samples", "manifest")


class StoreError(RuntimeError):
    """A persistence problem surfaced to the UI."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class VersionedInput:
    version_id: int
    kind: str
    filename: str | None
    created_at: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class Snapshot:
    baseline: VersionedInput
    samples: VersionedInput
    manifest: VersionedInput | None = None


@dataclass(frozen=True)
class ReportRecord:
    report_id: int
    baseline_version: int
    samples_version: int
    created_at: str
    payload: dict[str, Any]
    baseline_is_current: bool
    samples_is_current: bool
    manifest_version: int | None = None
    manifest_is_current: bool | None = None

    @property
    def is_stale(self) -> bool:
        if not (self.baseline_is_current and self.samples_is_current):
            return True
        # A report computed without a manifest stays fresh while no
        # manifest exists; once one is uploaded the comparison applies.
        if self.manifest_version is None:
            return self.manifest_is_current is False
        return self.manifest_is_current is not True


class Store:
    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path)
        self._write_lock = threading.RLock()
        self._initialized = False

    # -- connections -----------------------------------------------------

    def _ensure_ready(self) -> None:
        """Create the directory and schema on first use.

        Deferring this keeps merely importing the app (or constructing a
        store) free of filesystem side effects.
        """

        if self._initialized:
            return
        with self._write_lock:
            if self._initialized:
                return
            if self.path.parent and str(self.path.parent) not in ("", "."):
                self.path.parent.mkdir(parents=True, exist_ok=True)
            self._migrate_schema()
            self._initialized = True

    def _migrate_schema(self) -> None:
        """Create the schema, upgrading any pre-manifest database in place.

        The original schema constrained ``input_versions.kind`` to
        ``('baseline', 'samples')`` and ``reports`` had no manifest column.
        A freshly created database uses the current schema directly; an
        existing file is rebuilt transactionally so its versions and
        reports survive the upgrade.
        """

        fresh = not self.path.exists() or self.path.stat().st_size == 0
        conn = self._connect()
        try:
            if fresh:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS input_versions (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        kind        TEXT    NOT NULL
                                    CHECK (kind IN
                                    ('baseline', 'samples', 'manifest')),
                        filename    TEXT,
                        payload     TEXT    NOT NULL,
                        created_at  TEXT    NOT NULL,
                        is_current  INTEGER NOT NULL DEFAULT 0
                                    CHECK (is_current IN (0, 1))
                    );

                    CREATE UNIQUE INDEX IF NOT EXISTS
                        input_versions_one_current
                    ON input_versions (kind)
                    WHERE is_current = 1;

                    CREATE TABLE IF NOT EXISTS reports (
                        id                INTEGER PRIMARY KEY AUTOINCREMENT,
                        baseline_version  INTEGER NOT NULL
                                          REFERENCES input_versions (id),
                        samples_version   INTEGER NOT NULL
                                          REFERENCES input_versions (id),
                        manifest_version  INTEGER
                                          REFERENCES input_versions (id),
                        payload           TEXT    NOT NULL,
                        created_at        TEXT    NOT NULL
                    );
                    """
                )
                return

            conn.execute("PRAGMA journal_mode = WAL")
            existing_tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if not existing_tables:
                conn.executescript(
                    """
                    CREATE TABLE input_versions (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        kind        TEXT    NOT NULL
                                    CHECK (kind IN
                                    ('baseline', 'samples', 'manifest')),
                        filename    TEXT,
                        payload     TEXT    NOT NULL,
                        created_at  TEXT    NOT NULL,
                        is_current  INTEGER NOT NULL DEFAULT 0
                                    CHECK (is_current IN (0, 1))
                    );

                    CREATE UNIQUE INDEX input_versions_one_current
                    ON input_versions (kind)
                    WHERE is_current = 1;

                    CREATE TABLE reports (
                        id                INTEGER PRIMARY KEY AUTOINCREMENT,
                        baseline_version  INTEGER NOT NULL
                                          REFERENCES input_versions (id),
                        samples_version   INTEGER NOT NULL
                                          REFERENCES input_versions (id),
                        manifest_version  INTEGER
                                          REFERENCES input_versions (id),
                        payload           TEXT    NOT NULL,
                        created_at        TEXT    NOT NULL
                    );
                    """
                )
                return
            needs_kind_migration = True
            needs_manifest_column = True
            kind_sql = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'table' AND name = 'input_versions'"
            ).fetchone()
            if kind_sql is not None and "'manifest'" in kind_sql["sql"]:
                needs_kind_migration = False
            if "reports" not in existing_tables:
                raise StoreError(
                    "初始化数据库失败：现有数据库缺少 reports 表，无法安全迁移"
                )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(reports)")
            }
            needs_manifest_column = "manifest_version" not in columns
            if not needs_kind_migration and not needs_manifest_column:
                return

            # The table rebuild is the documented SQLite way of changing a
            # CHECK constraint.  Foreign keys are off for this one
            # connection; the data is copied verbatim inside a transaction.
            # Individual ``execute`` calls are used (not ``executescript``,
            # which would implicitly commit the open transaction).
            conn.execute("PRAGMA foreign_keys = OFF")
            try:
                conn.execute("BEGIN IMMEDIATE")
                if needs_kind_migration:
                    conn.execute(
                        """
                        CREATE TABLE input_versions_new (
                            id          INTEGER PRIMARY KEY AUTOINCREMENT,
                            kind        TEXT    NOT NULL
                                        CHECK (kind IN
                                        ('baseline', 'samples', 'manifest')),
                            filename    TEXT,
                            payload     TEXT    NOT NULL,
                            created_at  TEXT    NOT NULL,
                            is_current  INTEGER NOT NULL DEFAULT 0
                                        CHECK (is_current IN (0, 1))
                        )
                        """
                    )
                    conn.execute(
                        """
                        INSERT INTO input_versions_new
                            (id, kind, filename, payload, created_at, is_current)
                        SELECT id, kind, filename, payload, created_at, is_current
                        FROM input_versions
                        """
                    )
                    conn.execute("DROP TABLE input_versions")
                    conn.execute(
                        "ALTER TABLE input_versions_new RENAME TO input_versions"
                    )
                    conn.execute(
                        """
                        CREATE UNIQUE INDEX input_versions_one_current
                        ON input_versions (kind)
                        WHERE is_current = 1
                        """
                    )
                if needs_manifest_column:
                    conn.execute(
                        "ALTER TABLE reports ADD COLUMN manifest_version INTEGER"
                    )
                integrity = conn.execute("PRAGMA foreign_key_check").fetchall()
                if integrity:
                    raise sqlite3.IntegrityError(
                        "迁移后外键校验失败: " + repr(integrity)
                    )
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.execute("PRAGMA foreign_keys = ON")
        except sqlite3.Error as exc:
            raise StoreError(f"初始化数据库失败：{exc}") from exc
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.path), timeout=30.0, isolation_level=None
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        self._ensure_ready()
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            self._ensure_ready()
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    # -- inputs ----------------------------------------------------------

    def save_inputs(
        self, updates: dict[str, tuple[dict[str, Any], str | None]]
    ) -> dict[str, int]:
        """Replace one or both inputs in a single transaction.

        Whole-file replacement plus atomicity: either every supplied input
        gets a new current version, or none of them change.
        """

        for kind in updates:
            if kind not in KINDS:
                raise ValueError(f"unknown input kind: {kind!r}")
        encoded = {
            kind: json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            for kind, (payload, _filename) in updates.items()
        }
        now = _utcnow()
        new_ids: dict[str, int] = {}
        try:
            with self._write() as conn:
                for kind in sorted(encoded):
                    _payload, filename = updates[kind]
                    conn.execute(
                        "UPDATE input_versions SET is_current = 0 "
                        "WHERE kind = ? AND is_current = 1",
                        (kind,),
                    )
                    cursor = conn.execute(
                        "INSERT INTO input_versions "
                        "(kind, filename, payload, created_at, is_current) "
                        "VALUES (?, ?, ?, ?, 1)",
                        (kind, filename, encoded[kind], now),
                    )
                    new_ids[kind] = int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise StoreError(f"保存输入失败：{exc}") from exc
        return new_ids

    def save_input(
        self, kind: str, payload: dict[str, Any], filename: str | None
    ) -> int:
        return self.save_inputs({kind: (payload, filename)})[kind]

    def _current_row(self, conn: sqlite3.Connection, kind: str):
        return conn.execute(
            "SELECT id, kind, filename, payload, created_at "
            "FROM input_versions WHERE kind = ? AND is_current = 1",
            (kind,),
        ).fetchone()

    @staticmethod
    def _row_to_input(row: sqlite3.Row) -> VersionedInput:
        return VersionedInput(
            version_id=row["id"],
            kind=row["kind"],
            filename=row["filename"],
            created_at=row["created_at"],
            payload=json.loads(row["payload"]),
        )

    def current_inputs(self) -> dict[str, VersionedInput | None]:
        with self._read() as conn:
            result: dict[str, VersionedInput | None] = {}
            for kind in KINDS:
                row = self._current_row(conn, kind)
                result[kind] = self._row_to_input(row) if row else None
            return result

    def snapshot(self) -> Snapshot | None:
        """Consistent read of the current inputs, or ``None`` if incomplete.

        The manifest is optional: the two calculation inputs are still
        mandatory, and the manifest (if uploaded) rides along in the same
        read transaction.
        """

        with self._read() as conn:
            rows = {kind: self._current_row(conn, kind) for kind in KINDS}
            if rows["baseline"] is None or rows["samples"] is None:
                return None
            return Snapshot(
                baseline=self._row_to_input(rows["baseline"]),
                samples=self._row_to_input(rows["samples"]),
                manifest=(
                    self._row_to_input(rows["manifest"])
                    if rows["manifest"] is not None
                    else None
                ),
            )

    # -- reports ---------------------------------------------------------

    @staticmethod
    def _versions_are_current(conn: sqlite3.Connection, snapshot: Snapshot) -> bool:
        rows = {
            kind: conn.execute(
                "SELECT id FROM input_versions WHERE kind = ? AND is_current = 1",
                (kind,),
            ).fetchone()
            for kind in KINDS
        }
        if (
            rows["baseline"] is None
            or rows["samples"] is None
            or rows["baseline"]["id"] != snapshot.baseline.version_id
            or rows["samples"]["id"] != snapshot.samples.version_id
        ):
            return False
        current_manifest = rows["manifest"]
        if snapshot.manifest is None:
            # The snapshot was taken with no manifest present; a manifest
            # uploaded mid-computation is also a version change that must
            # discard this run.
            return current_manifest is None
        return (
            current_manifest is not None
            and current_manifest["id"] == snapshot.manifest.version_id
        )

    def publish_report(
        self,
        snapshot: Snapshot,
        payload: dict[str, Any],
    ) -> int | None:
        """Insert a complete report if, and only if, its inputs are current.

        Returns the new report id, or ``None`` when an upload won the race.
        The check and the insert share one ``BEGIN IMMEDIATE`` transaction,
        so an upload cannot slip in between them.
        """

        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        try:
            with self._write() as conn:
                if not self._versions_are_current(conn, snapshot):
                    return None
                cursor = conn.execute(
                    "INSERT INTO reports "
                    "(baseline_version, samples_version, manifest_version, "
                    " payload, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        snapshot.baseline.version_id,
                        snapshot.samples.version_id,
                        (
                            snapshot.manifest.version_id
                            if snapshot.manifest is not None
                            else None
                        ),
                        encoded,
                        _utcnow(),
                    ),
                )
                return int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise StoreError(f"保存报告失败：{exc}") from exc

    _REPORT_SELECT = (
        "SELECT r.id, r.baseline_version, r.samples_version, "
        "       r.manifest_version, r.payload, r.created_at, "
        "       (SELECT id FROM input_versions "
        "         WHERE kind = 'baseline' AND is_current = 1) "
        "         AS current_baseline, "
        "       (SELECT id FROM input_versions "
        "         WHERE kind = 'samples' AND is_current = 1) "
        "         AS current_samples, "
        "       (SELECT id FROM input_versions "
        "         WHERE kind = 'manifest' AND is_current = 1) "
        "         AS current_manifest "
        "FROM reports r"
    )

    def latest_report(self) -> ReportRecord | None:
        with self._read() as conn:
            row = conn.execute(
                self._REPORT_SELECT + " ORDER BY r.id DESC LIMIT 1"
            ).fetchone()
            return self._row_to_report(row) if row else None

    def get_report(self, report_id: int) -> ReportRecord | None:
        with self._read() as conn:
            row = conn.execute(
                self._REPORT_SELECT + " WHERE r.id = ?",
                (report_id,),
            ).fetchone()
            return self._row_to_report(row) if row else None

    @staticmethod
    def _row_to_report(row: sqlite3.Row) -> ReportRecord:
        current_manifest = row["current_manifest"]
        manifest_version = row["manifest_version"]
        if manifest_version is None:
            manifest_is_current: bool | None = (
                None if current_manifest is None else False
            )
        else:
            manifest_is_current = current_manifest == manifest_version
        return ReportRecord(
            report_id=row["id"],
            baseline_version=row["baseline_version"],
            samples_version=row["samples_version"],
            created_at=row["created_at"],
            payload=json.loads(row["payload"]),
            baseline_is_current=row["current_baseline"]
            == row["baseline_version"],
            samples_is_current=row["current_samples"] == row["samples_version"],
            manifest_version=manifest_version,
            manifest_is_current=manifest_is_current,
        )
