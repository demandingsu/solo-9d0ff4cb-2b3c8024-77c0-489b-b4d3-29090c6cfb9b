"""Tests for the handover manifest: parsing, reconciliation, publish
gating, version staleness and upload/publish interleaving."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.inputs import (
    InputValidationError,
    ManifestInput,
    parse_manifest,
)
from app.main import create_app

BASE_CSV = """ore,Fe,SiO2,Al2O3,CaO
A,62.3,3.9,2.1,1.2
B,68.5,2.5,1.1,0.8
C,55.2,6.2,3.4,2.1
"""

SAMPLES_CSV = """sample_id,Fe,SiO2,Al2O3,CaO
S1,62.74,3.71,2.02,1.13
S2,60,4,2,1
"""


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(str(tmp_path / "test.db"))
    with TestClient(app) as test_client:
        test_client.app_state = app.state  # type: ignore[attr-defined]
        yield test_client


def upload_inputs(client: TestClient) -> None:
    client.post(
        "/inputs/upload",
        files={
            "baseline_file": ("baseline.csv", BASE_CSV.encode(), "text/csv"),
            "samples_file": ("samples.csv", SAMPLES_CSV.encode(), "text/csv"),
        },
        follow_redirects=False,
    )


def upload_manifest(client: TestClient, content: str, name: str = "m.csv"):
    return client.post(
        "/manifest/upload",
        files={"manifest_file": (name, content.encode(), "text/csv")},
        follow_redirects=False,
    )


def compute(client: TestClient):
    return client.post("/compute", follow_redirects=False)


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def test_parse_manifest_ok():
    parsed = parse_manifest("sample_id\nS-1001\nS-1002\n".encode())
    assert parsed == ManifestInput(sample_ids=("S-1001", "S-1002"))
    # Header name is free-form and surrounding whitespace is trimmed.
    parsed = parse_manifest(" 编号 \n A \n".encode())
    assert parsed.sample_ids == ("A",)


def test_manifest_accepts_bom():
    raw = ("﻿sample_id\nS1\n").encode("utf-8")
    assert parse_manifest(raw).sample_ids == ("S1",)


def test_manifest_rejects_duplicate_id():
    with pytest.raises(InputValidationError, match="重复"):
        parse_manifest("sample_id\nS1\nS1\n".encode())


def test_manifest_rejects_empty_id_cell():
    with pytest.raises(InputValidationError, match="不能为空"):
        parse_manifest("sample_id\nS1\n,\n".encode())
    with pytest.raises(InputValidationError, match="不能为空"):
        parse_manifest("sample_id\nS1\n   \n".encode())


def test_manifest_rejects_extra_columns():
    with pytest.raises(InputValidationError, match="只能有 1 列"):
        parse_manifest("sample_id\nS1,extra\n".encode())
    with pytest.raises(InputValidationError, match="只有 1 列"):
        parse_manifest("a,b\nS1\n".encode())


def test_manifest_rejects_empty_and_header_only_files():
    with pytest.raises(InputValidationError):
        parse_manifest(b"")
    with pytest.raises(InputValidationError, match="没有数据行"):
        parse_manifest("sample_id\n".encode())


def test_manifest_payload_roundtrip():
    parsed = parse_manifest("sample_id\nS1\nS2\n".encode())
    again = ManifestInput.from_payload(parsed.to_payload())
    assert again == parsed


# --------------------------------------------------------------------------
# Upload behaviour
# --------------------------------------------------------------------------


def test_manifest_upload_creates_version_and_preview(client: TestClient):
    response = upload_manifest(client, "sample_id\nS1\nS2\n")
    assert response.status_code == 303
    store = client.app_state.store
    current = store.current_inputs()
    assert current["manifest"] is not None
    assert current["manifest"].version_id == 1
    html = client.get("/").text
    assert "当前清单" in html
    assert "S1" in html and "2 个编号" in html


def test_invalid_manifest_is_whole_file_rejected(client: TestClient):
    upload_manifest(client, "sample_id\nS1\nS2\n")
    store = client.app_state.store
    before = store.current_inputs()["manifest"].version_id

    response = client.post(
        "/manifest/upload",
        files={"manifest_file": ("bad.csv", b"sample_id\nS1\nS1\n")},
    )
    assert response.status_code == 400
    assert "重复" in response.text
    # Previous manifest version is untouched.
    after = store.current_inputs()["manifest"]
    assert after.version_id == before
    assert ManifestInput.from_payload(after.payload).sample_ids == ("S1", "S2")


def test_manifest_upload_without_file_is_rejected(client: TestClient):
    response = client.post("/manifest/upload", follow_redirects=False)
    assert response.status_code == 303
    assert client.app_state.store.current_inputs()["manifest"] is None


def test_manifest_replace_marks_existing_report_stale_but_viewable(
    client: TestClient,
):
    upload_inputs(client)
    upload_manifest(client, "sample_id\nS1\nS2\n")
    first_manifest_version = client.app_state.store.current_inputs()[
        "manifest"
    ].version_id
    compute(client)
    assert client.get("/api/latest").json()["report"]["stale"] is False

    upload_manifest(client, "sample_id\nS1\nS2\nS3\n", name="m2.csv")
    data = client.get("/api/latest").json()["report"]
    assert data["stale"] is True
    # The old report remains fully viewable.
    html = client.get("/").text
    assert "已过期" in html and "S1" in html
    # And it still names the manifest version it was published against.
    assert data["manifest_version"] == first_manifest_version


def test_first_manifest_marks_manifest_free_report_stale(client: TestClient):
    upload_inputs(client)
    compute(client)
    assert client.get("/api/latest").json()["report"]["stale"] is False
    upload_manifest(client, "sample_id\nS1\nS2\n")
    assert client.get("/api/latest").json()["report"]["stale"] is True


def test_manifest_write_failure_keeps_previous_manifest_and_report(
    client: TestClient, monkeypatch
):
    from app.db import StoreError

    upload_inputs(client)
    upload_manifest(client, "sample_id\nS1\nS2\n")
    compute(client)
    store = client.app_state.store
    before = store.current_inputs()["manifest"].version_id

    def broken_save(*args, **kwargs):
        raise StoreError("数据库磁盘已满（测试注入）")

    monkeypatch.setattr(store, "save_inputs", broken_save)
    response = client.post(
        "/manifest/upload",
        files={"manifest_file": ("m.csv", b"sample_id\nS1\n")},
    )
    assert response.status_code == 400
    assert "清单写入失败" in response.text
    assert store.current_inputs()["manifest"].version_id == before
    report = client.get("/api/latest").json()["report"]
    assert report["stale"] is False


# --------------------------------------------------------------------------
# Reconciliation display
# --------------------------------------------------------------------------


def test_reconciliation_shows_missing_and_extra(client: TestClient):
    upload_inputs(client)
    # Manifest wants S1, S2, S3; samples only carry S1, S2 plus intruder S9.
    upload_manifest(client, "sample_id\nS1\nS2\nS3\n")
    # Swap samples so S9 replaces S2: missing S2,S3 and extra S9.
    samples = (
        "sample_id,Fe,SiO2,Al2O3,CaO\n"
        "S1,62.74,3.71,2.02,1.13\n"
        "S9,60,4,2,1\n"
    )
    client.post(
        "/inputs/upload",
        files={"samples_file": ("s.csv", samples.encode(), "text/csv")},
        follow_redirects=False,
    )
    html = client.get("/").text
    assert "缺失" in html and "额外" in html
    assert "S3" in html and "S9" in html
    api = client.get("/api/latest").json()["reconcile"]
    assert api["matches"] is False
    assert set(api["missing"]) == {"S2", "S3"}
    assert api["extra"] == ["S9"]


def test_matching_sets_show_match_badge(client: TestClient):
    upload_inputs(client)
    upload_manifest(client, "sample_id\nS2\nS1\n")  # order differs on purpose
    html = client.get("/").text
    assert "编号集合一致" in html
    api = client.get("/api/latest").json()["reconcile"]
    assert api["matches"] is True


def test_no_manifest_means_no_reconciliation_and_old_flow(client: TestClient):
    upload_inputs(client)
    data = client.get("/api/latest").json()
    assert data["reconcile"] is None
    compute(client)
    report = client.get("/api/latest").json()["report"]
    assert report["manifest_version"] is None
    assert report["stale"] is False


# --------------------------------------------------------------------------
# Publish gating
# --------------------------------------------------------------------------


def test_publish_blocked_when_sets_differ(client: TestClient):
    upload_inputs(client)
    upload_manifest(client, "sample_id\nS1\nS2\nS3\n")  # S3 missing
    response = compute(client)
    assert response.status_code == 303
    html = client.get(response.headers["location"]).text
    assert "阻止发布报告" in html and "S3" in html
    assert client.get("/api/latest").json()["report"] is None
    # Button is disabled on the page.
    assert "编号集合与清单一致后才可发布" in client.get("/").text


def test_publish_blocked_for_intruder_sample(client: TestClient):
    upload_inputs(client)
    upload_manifest(client, "sample_id\nS1\n")  # S2 is an intruder
    response = compute(client)
    assert response.status_code == 303
    html = client.get(response.headers["location"]).text
    assert "额外 1 个" in html and "S2" in html
    assert client.get("/api/latest").json()["report"] is None


def test_publish_allowed_when_sets_match_and_links_manifest(client: TestClient):
    upload_inputs(client)
    upload_manifest(client, "sample_id\nS1\nS2\n")
    manifest_version = client.app_state.store.current_inputs()[
        "manifest"
    ].version_id
    assert compute(client).status_code == 303
    report = client.get("/api/latest").json()["report"]
    assert report["stale"] is False
    assert report["manifest_version"] == manifest_version
    assert report["payload"]["manifest_sample_ids"] == ["S1", "S2"]
    html = client.get("/").text
    assert f"清单 v{manifest_version}" in html


def test_report_stale_once_manifest_version_changes_then_refreshable(
    client: TestClient,
):
    upload_inputs(client)
    upload_manifest(client, "sample_id\nS1\nS2\n")
    compute(client)
    first = client.app_state.store.current_inputs()["manifest"].version_id
    # Same identifier set, new version: the report must be stale anyway.
    upload_manifest(client, "sample_id\nS1\nS2\n", name="m2.csv")
    second = client.app_state.store.current_inputs()["manifest"].version_id
    assert second > first
    assert client.get("/api/latest").json()["report"]["stale"] is True
    assert compute(client).status_code == 303
    fresh = client.get("/api/latest").json()["report"]
    assert fresh["stale"] is False
    assert fresh["manifest_version"] == second


# --------------------------------------------------------------------------
# Interleaving: uploads landing while a calculation is in flight
# --------------------------------------------------------------------------


def test_manifest_swap_during_compute_discards_result(client: TestClient, monkeypatch):
    import app.main as main_module
    from app.compute import compute_report as real_compute

    upload_inputs(client)
    upload_manifest(client, "sample_id\nS1\nS2\n")
    store = client.app_state.store

    def compute_and_swap_manifest(baseline, samples):
        result = real_compute(baseline, samples)
        # A new manifest version lands before the publish transaction runs.
        store.save_input(
            "manifest",
            ManifestInput(sample_ids=("S1", "S2")).to_payload(),
            "raced.csv",
        )
        return result

    monkeypatch.setattr(main_module, "compute_report", compute_and_swap_manifest)
    response = compute(client)
    assert response.status_code == 303
    html = client.get(response.headers["location"]).text
    assert "输入已被更新" in html and "已丢弃" in html
    assert client.get("/api/latest").json()["report"] is None


def test_first_manifest_during_manifest_free_compute_discards_result(
    client: TestClient, monkeypatch
):
    import app.main as main_module
    from app.compute import compute_report as real_compute

    upload_inputs(client)
    store = client.app_state.store
    assert store.snapshot().manifest is None

    def compute_and_add_manifest(baseline, samples):
        result = real_compute(baseline, samples)
        store.save_input(
            "manifest",
            ManifestInput(sample_ids=("S1", "S2")).to_payload(),
            "late.csv",
        )
        return result

    monkeypatch.setattr(main_module, "compute_report", compute_and_add_manifest)
    response = compute(client)
    assert response.status_code == 303
    assert client.get("/api/latest").json()["report"] is None


def test_samples_swap_during_compute_with_manifest_discards_result(
    client: TestClient, monkeypatch
):
    import app.main as main_module
    from app.compute import compute_report as real_compute
    from app.inputs import SamplesInput

    upload_inputs(client)
    upload_manifest(client, "sample_id\nS1\nS2\n")
    store = client.app_state.store

    def compute_and_swap_samples(baseline, samples):
        result = real_compute(baseline, samples)
        store.save_input("samples", samples.to_payload(), "raced.csv")
        return result

    monkeypatch.setattr(main_module, "compute_report", compute_and_swap_samples)
    response = compute(client)
    assert response.status_code == 303
    assert client.get("/api/latest").json()["report"] is None
    # The raced upload itself survived.
    assert isinstance(
        SamplesInput.from_payload(
            store.current_inputs()["samples"].payload
        ),
        SamplesInput,
    )


def test_report_publish_failure_keeps_old_report_with_manifest(
    client: TestClient, monkeypatch
):
    from app.db import StoreError

    upload_inputs(client)
    upload_manifest(client, "sample_id\nS1\nS2\n")
    manifest_version = client.app_state.store.current_inputs()[
        "manifest"
    ].version_id
    compute(client)
    old_id = client.get("/api/latest").json()["report"]["id"]

    def broken_publish(*args, **kwargs):
        raise StoreError("磁盘只读（测试注入）")

    monkeypatch.setattr(client.app_state.store, "publish_report", broken_publish)
    response = client.post("/compute", follow_redirects=True)
    assert response.status_code == 200
    assert "磁盘只读" in response.text
    current = client.get("/api/latest").json()["report"]
    assert current["id"] == old_id
    assert current["manifest_version"] == manifest_version


def test_concurrent_manifest_uploads_and_publishes_are_consistent(tmp_path: Path):
    """Hammer manifest replacement against publishing.

    Every published report must name a manifest version that was current at
    publish time; a replaced manifest must either block the publish (report
    absent) or be reflected exactly in the stored report.
    """
    import sqlite3
    import threading
    import time

    from app.db import Store
    from app.inputs import BaselineInput, SamplesInput

    store = Store(str(tmp_path / "race.db"))
    baseline_payload = BaselineInput.from_payload(
        {
            "elements": ["Fe", "SiO2", "Al2O3", "CaO"],
            "ores": ["A", "B", "C"],
            "rows": [["1", "2", "3", "4"], ["5", "6", "7", "8"], ["9", "1", "2", "3"]],
        }
    ).to_payload()
    samples_payload = SamplesInput.from_payload(
        {
            "elements": ["Fe", "SiO2", "Al2O3", "CaO"],
            "sample_ids": ["S1"],
            "rows": [["1", "2", "3", "4"]],
        }
    ).to_payload()
    manifest_payload = ManifestInput(sample_ids=("S1",)).to_payload()
    store.save_inputs(
        {
            "baseline": (baseline_payload, "b.csv"),
            "samples": (samples_payload, "s.csv"),
            "manifest": (manifest_payload, "m0.csv"),
        }
    )

    errors: list[BaseException] = []
    published: list[int] = []

    def publisher():
        try:
            for _ in range(60):
                snapshot = store.snapshot()
                if snapshot is None or snapshot.manifest is None:
                    continue
                report_id = store.publish_report(snapshot, {"marker": "x"})
                if report_id is not None:
                    published.append(report_id)
                    record = store.get_report(report_id)
                    assert record is not None
                    # The stored report always names exactly the snapshot
                    # versions; staleness is recomputed against live state.
                    assert record.manifest_version == snapshot.manifest.version_id
                    assert record.baseline_version == snapshot.baseline.version_id
                    assert record.samples_version == snapshot.samples.version_id
                time.sleep(0.001)
        except BaseException as exc:  # pragma: no cover - re-raised below
            errors.append(exc)

    def manifest_uploader():
        try:
            time.sleep(0.02)
            for i in range(30):
                store.save_input(
                    "manifest",
                    ManifestInput(sample_ids=("S1",)).to_payload(),
                    f"m{i}.csv",
                )
                time.sleep(0.005)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [
        threading.Thread(target=publisher),
        threading.Thread(target=manifest_uploader),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    assert published
    with sqlite3.connect(tmp_path / "race.db") as conn:
        report_count = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        current_kinds = conn.execute(
            "SELECT COUNT(*) FROM input_versions WHERE is_current = 1"
        ).fetchone()[0]
    assert report_count == len(published)
    assert current_kinds == 3


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def test_manifest_export_roundtrip(client: TestClient):
    upload_manifest(client, "sample_id\nS1\nS2\n")
    response = client.get("/inputs/manifest/export.csv")
    assert response.status_code == 200
    lines = response.content.decode("utf-8-sig").strip().splitlines()
    assert lines == ["sample_id", "S1", "S2"]


def test_manifest_export_404_when_absent(client: TestClient):
    assert client.get("/inputs/manifest/export.csv").status_code == 404
    assert client.get("/inputs/unknown/export.csv").status_code == 404


# --------------------------------------------------------------------------
# Schema migration from a pre-manifest database
# --------------------------------------------------------------------------


def test_legacy_database_is_migrated_and_data_kept(tmp_path: Path):
    import sqlite3

    db_path = tmp_path / "legacy.db"
    # Build a database with the original, manifest-free schema.
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE input_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL CHECK (kind IN ('baseline', 'samples')),
            filename TEXT, payload TEXT NOT NULL, created_at TEXT NOT NULL,
            is_current INTEGER NOT NULL DEFAULT 0
        );
        CREATE UNIQUE INDEX input_versions_one_current
            ON input_versions (kind) WHERE is_current = 1;
        CREATE TABLE reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            baseline_version INTEGER NOT NULL REFERENCES input_versions (id),
            samples_version INTEGER NOT NULL REFERENCES input_versions (id),
            payload TEXT NOT NULL, created_at TEXT NOT NULL
        );
        INSERT INTO input_versions (kind, filename, payload, created_at, is_current)
        VALUES ('baseline', 'b.csv', '{}', '2026-01-01T00:00:00+00:00', 1),
               ('samples', 's.csv', '{}', '2026-01-01T00:00:00+00:00', 1);
        INSERT INTO reports (baseline_version, samples_version, payload, created_at)
        VALUES (1, 2, '{"old": true}', '2026-01-01T00:00:01+00:00');
        """
    )
    conn.commit()
    conn.close()

    from app.db import Store

    store = Store(str(db_path))
    current = store.current_inputs()
    assert current["manifest"] is None
    assert current["baseline"].version_id == 1
    record = store.latest_report()
    assert record is not None
    assert record.manifest_version is None
    # No manifest yet: the legacy report is still consistent with inputs.
    assert record.is_stale is False

    # The upgraded schema accepts manifest versions and new reports.
    new_id = store.save_input(
        "manifest",
        ManifestInput(sample_ids=("S1",)).to_payload(),
        "m.csv",
    )
    assert new_id == 3
    assert store.current_inputs()["manifest"].version_id == 3
    # Once a manifest exists, the manifest-free legacy report is stale.
    assert store.latest_report().is_stale is True
    with sqlite3.connect(db_path) as check:
        kinds = {
            row[0]
            for row in check.execute(
                "SELECT DISTINCT kind FROM input_versions"
            )
        }
    assert kinds == {"baseline", "samples", "manifest"}


def test_migration_is_idempotent_on_new_schema(tmp_path: Path):
    from app.db import Store

    db_path = tmp_path / "new.db"
    store = Store(str(db_path))
    store.save_input(
        "manifest",
        ManifestInput(sample_ids=("S1",)).to_payload(),
        "m.csv",
    )
    # A second store over the same (already current-schema) database must
    # not attempt another rebuild or lose data.
    second = Store(str(db_path))
    current = second.current_inputs()
    assert ManifestInput.from_payload(
        current["manifest"].payload
    ).sample_ids == ("S1",)


def test_database_with_wal_but_no_tables_is_initialized(tmp_path: Path):
    # A non-empty file (WAL header) containing no user tables must take the
    # create path rather than crash on missing schema rows.
    import sqlite3

    from app.db import Store

    db_path = tmp_path / "wal.db"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.close()
    assert db_path.stat().st_size > 0
    store = Store(str(db_path))
    assert store.current_inputs() == {
        "baseline": None,
        "samples": None,
        "manifest": None,
    }
    store.save_input(
        "manifest",
        ManifestInput(sample_ids=("S1",)).to_payload(),
        "m.csv",
    )
    assert store.current_inputs()["manifest"] is not None
