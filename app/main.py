"""FastAPI application for ore-blend proportion reconstruction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from . import formatting
from .compute import ComputeError, compute_report
from .db import ReportRecord, Store, StoreError, VersionedInput
from .inputs import (
    MAX_UPLOAD_BYTES,
    BaselineInput,
    InputValidationError,
    ManifestInput,
    SamplesInput,
    parse_baseline,
    parse_manifest,
    parse_samples,
)

BASE_DIR = Path(__file__).resolve().parent
REPORT_ROW_LIMIT = 500
# Reconciliation lists on the page are truncated to keep it readable.
RECONCILE_DISPLAY_LIMIT = 200


def create_app(database_path: str | None = None) -> FastAPI:
    app = FastAPI(title="矿石配料反推与异常定位")
    store = Store(database_path or os.environ.get("ORE_DB_PATH", "data/app.db"))
    app.state.store = store

    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    templates.env.filters["decimal"] = formatting.decimal
    templates.env.filters["percent"] = formatting.percent
    templates.env.filters["signed"] = formatting.signed
    templates.env.filters["ratio"] = formatting.ratio_text
    templates.env.filters["exact"] = formatting.exact

    app.mount(
        "/static",
        StaticFiles(directory=str(BASE_DIR / "static")),
        name="static",
    )

    # -- helpers ---------------------------------------------------------

    def redirect_notice(
        message: str, kind: str = "ok", anchor: str = ""
    ) -> RedirectResponse:
        query = urlencode({"notice": message, "notice_kind": kind})
        url = f"/?{query}"
        if anchor:
            url += f"#{anchor}"
        return RedirectResponse(url, status_code=303)

    def payload_to_inputs(
        snapshot,
    ) -> tuple[BaselineInput, SamplesInput, ManifestInput | None]:
        manifest = (
            ManifestInput.from_payload(snapshot.manifest.payload)
            if snapshot.manifest is not None
            else None
        )
        return (
            BaselineInput.from_payload(snapshot.baseline.payload),
            SamplesInput.from_payload(snapshot.samples.payload),
            manifest,
        )

    def input_summary(value: VersionedInput | None) -> dict[str, Any] | None:
        if value is None:
            return None
        if value.kind == "baseline":
            parsed = BaselineInput.from_payload(value.payload)
            return {
                "version": value.version_id,
                "filename": value.filename,
                "created_at": value.created_at,
                "elements": list(parsed.elements),
                "names": list(parsed.ores),
                "rows": [
                    [formatting.decimal(v, 8) for v in row] for row in parsed.rows
                ],
                "count": len(parsed.rows),
            }
        if value.kind == "manifest":
            parsed_manifest = ManifestInput.from_payload(value.payload)
            preview = list(parsed_manifest.sample_ids[:5])
            return {
                "version": value.version_id,
                "filename": value.filename,
                "created_at": value.created_at,
                "elements": [],
                "names": preview,
                "rows": [],
                "count": len(parsed_manifest.sample_ids),
            }
        parsed_samples = SamplesInput.from_payload(value.payload)
        return {
            "version": value.version_id,
            "filename": value.filename,
            "created_at": value.created_at,
            "elements": list(parsed_samples.elements),
            "names": list(parsed_samples.sample_ids[:5]),
            "rows": [],
            "count": len(parsed_samples.sample_ids),
        }

    def reconcile(
        current: dict[str, VersionedInput | None],
    ) -> dict[str, Any] | None:
        """Diff the current sample batch against the current manifest.

        ``missing`` = manifest identifiers absent from the finished-sample
        input (handover omissions); ``extra`` = sample identifiers that are
        not on the manifest (samples mixed in from another batch).
        """

        manifest_input = current.get("manifest")
        samples_input = current.get("samples")
        if manifest_input is None:
            return None
        manifest_ids = list(
            ManifestInput.from_payload(manifest_input.payload).sample_ids
        )
        sample_ids = (
            list(SamplesInput.from_payload(samples_input.payload).sample_ids)
            if samples_input is not None
            else []
        )
        manifest_set = set(manifest_ids)
        sample_set = set(sample_ids)
        missing = [value for value in manifest_ids if value not in sample_set]
        extra = [value for value in sample_ids if value not in manifest_set]
        return {
            "manifest_count": len(manifest_ids),
            "samples_count": len(sample_ids),
            "matches": not missing and not extra,
            "missing": missing,
            "extra": extra,
            "missing_display": missing[:RECONCILE_DISPLAY_LIMIT],
            "extra_display": extra[:RECONCILE_DISPLAY_LIMIT],
            "missing_hidden": max(0, len(missing) - RECONCILE_DISPLAY_LIMIT),
            "extra_hidden": max(0, len(extra) - RECONCILE_DISPLAY_LIMIT),
        }

    def report_view(record: ReportRecord | None) -> dict[str, Any] | None:
        if record is None:
            return None
        payload = dict(record.payload)
        rows = payload.get("results", [])
        visible = rows[:REPORT_ROW_LIMIT]
        return {
            "id": record.report_id,
            "created_at": record.created_at,
            "baseline_version": record.baseline_version,
            "samples_version": record.samples_version,
            "manifest_version": record.manifest_version,
            "stale": record.is_stale,
            "baseline_current": record.baseline_is_current,
            "samples_current": record.samples_is_current,
            "manifest_current": record.manifest_is_current,
            "payload": payload,
            "rows": visible,
            "hidden_rows": max(0, len(rows) - len(visible)),
        }

    def context(request: Request) -> dict[str, Any]:
        current = store.current_inputs()
        record = store.latest_report()
        reconciliation = reconcile(current)
        return {
            "request": request,
            "current": {
                kind: input_summary(value) for kind, value in current.items()
            },
            "reconcile": reconciliation,
            "publish_blocked": bool(
                reconciliation is not None and not reconciliation["matches"]
            ),
            "report": report_view(record),
            "notice": request.query_params.get("notice"),
            "notice_kind": request.query_params.get("notice_kind", "ok"),
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        }

    # -- pages -----------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Response:
        return templates.TemplateResponse(request, "index.html", context(request))

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/inputs/upload")
    async def upload_inputs(
        request: Request,
        baseline_file: UploadFile | None = File(None),
        samples_file: UploadFile | None = File(None),
    ) -> Response:
        provided = {
            "baseline": baseline_file,
            "samples": samples_file,
        }
        if not any(file is not None and file.filename for file in provided.values()):
            return redirect_notice("请至少选择一个 CSV 文件再上传。", "error", "upload")

        parsed: dict[str, Any] = {}
        filenames: dict[str, str] = {}
        labels = {"baseline": "基准", "samples": "成品"}
        try:
            for kind, file in provided.items():
                if file is None or not file.filename:
                    continue
                raw = await file.read(MAX_UPLOAD_BYTES + 1)
                if len(raw) > MAX_UPLOAD_BYTES:
                    raise _LabeledError(
                        labels[kind],
                        f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB 上限",
                    )
                parser = parse_baseline if kind == "baseline" else parse_samples
                try:
                    parsed[kind] = await run_in_threadpool(parser, raw)
                except InputValidationError as exc:
                    raise _LabeledError(labels[kind], str(exc)) from exc
                filenames[kind] = (
                    file.filename.replace("\\", "/").rsplit("/", 1)[-1][:200]
                    or "未命名.csv"
                )
        except _LabeledError as exc:
            return _render_with_error(request, f"{exc.label}文件：{exc.message}")

        updates: dict[str, tuple[dict[str, Any], str]] = {}
        for kind, value in parsed.items():
            updates[kind] = (value.to_payload(), filenames[kind])

        had_report = await run_in_threadpool(
            lambda: store.latest_report() is not None
        )
        try:
            versions = await run_in_threadpool(store.save_inputs, updates)
        except StoreError as exc:
            return _render_with_error(
                request, f"输入写入失败，原有输入未被替换：{exc}"
            )

        replaced = []
        if "baseline" in versions:
            replaced.append(f"基准 v{versions['baseline']}")
        if "samples" in versions:
            replaced.append(f"成品 v{versions['samples']}")
        suffix = "。旧报告已标记为过期。" if had_report else "。"
        return redirect_notice(
            "已整版替换：" + "，".join(replaced) + suffix,
            "ok",
            "report",
        )

    def _render_with_error(request: Request, message: str) -> HTMLResponse:
        ctx = context(request)
        ctx["notice"] = message
        ctx["notice_kind"] = "error"
        return templates.TemplateResponse(
            request, "index.html", ctx, status_code=400
        )

    @app.post("/manifest/upload")
    async def upload_manifest(
        request: Request,
        manifest_file: UploadFile | None = File(None),
    ) -> Response:
        if manifest_file is None or not manifest_file.filename:
            return redirect_notice(
                "请选择清单 CSV 文件再上传。", "error", "manifest"
            )
        raw = await manifest_file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            return _render_with_error(
                request,
                f"清单文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB 上限",
            )
        try:
            parsed = await run_in_threadpool(parse_manifest, raw)
        except InputValidationError as exc:
            return _render_with_error(request, f"清单文件：{exc}")
        filename = (
            manifest_file.filename.replace("\\", "/").rsplit("/", 1)[-1][:200]
            or "未命名.csv"
        )

        had_report = await run_in_threadpool(
            lambda: store.latest_report() is not None
        )
        try:
            versions = await run_in_threadpool(
                store.save_inputs,
                {"manifest": (parsed.to_payload(), filename)},
            )
        except StoreError as exc:
            return _render_with_error(
                request, f"清单写入失败，原有清单与旧报告均未改动：{exc}"
            )

        suffix = "。旧报告已标记为过期。" if had_report else "。"
        return redirect_notice(
            f"清单已整版替换：v{versions['manifest']}（{len(parsed.sample_ids)} "
            f"个编号）{suffix}",
            "ok",
            "manifest",
        )

    @app.post("/compute")
    async def compute(request: Request) -> Response:
        snapshot = await run_in_threadpool(store.snapshot)
        if snapshot is None:
            return redirect_notice(
                "基准与成品输入都必须先上传，才能计算。", "error", "upload"
            )

        try:
            baseline, samples, manifest = payload_to_inputs(snapshot)
        except InputValidationError as exc:
            return redirect_notice(str(exc), "error", "report")

        # Publishing gate: while a manifest exists, the sample batch must
        # carry exactly the manifest's identifier set.  The check happens
        # before the expensive solve, and is re-checked by version id at
        # publish time against a write transaction.
        if manifest is not None:
            manifest_set = set(manifest.sample_ids)
            sample_set = set(samples.sample_ids)
            missing = [
                value for value in manifest.sample_ids if value not in sample_set
            ]
            extra = [
                value for value in samples.sample_ids if value not in manifest_set
            ]
            if missing or extra:
                details = []
                if missing:
                    details.append(
                        "缺失 "
                        + str(len(missing))
                        + " 个（如 "
                        + "、".join(missing[:5])
                        + "）"
                    )
                if extra:
                    details.append(
                        "额外 "
                        + str(len(extra))
                        + " 个（如 "
                        + "、".join(extra[:5])
                        + "）"
                    )
                return redirect_notice(
                    "清单存在但当前成品编号集合与清单不一致，已阻止发布报告："
                    + "，".join(details)
                    + "。请先核对交接批次或重新上传输入。",
                    "error",
                    "manifest",
                )

        try:
            payload = await run_in_threadpool(
                compute_report, baseline, samples
            )
        except ComputeError as exc:
            return redirect_notice(str(exc), "error", "report")
        except Exception as exc:  # pragma: no cover - defensive
            return redirect_notice(
                f"计算失败，未产生新报告：{type(exc).__name__}: {exc}",
                "error",
                "report",
            )

        # The report payload carries the manifest version and identifiers
        # it was published against, so an exported report stays traceable.
        payload["manifest_version"] = (
            snapshot.manifest.version_id if snapshot.manifest is not None else None
        )
        payload["manifest_sample_ids"] = (
            list(manifest.sample_ids) if manifest is not None else None
        )

        try:
            report_id = await run_in_threadpool(
                store.publish_report, snapshot, payload
            )
        except StoreError as exc:
            return redirect_notice(
                f"报告写入失败，仍显示上一份报告：{exc}", "error", "report"
            )

        if report_id is None:
            return redirect_notice(
                "计算期间输入已被更新，本次结果与当前输入不符，已丢弃，"
                "未发布报告。请基于新输入重新计算。",
                "error",
                "report",
            )

        return redirect_notice(
            f"报告 #{report_id} 已生成并发布（{payload['sample_count']} 个样本，"
            f"{payload['non_unique_count']} 个无法唯一确定）。",
            "ok",
            "report",
        )

    # -- exports ---------------------------------------------------------

    @app.get("/reports/{report_id}/export.csv")
    def export_report(report_id: int) -> Response:
        record = store.get_report(report_id)
        if record is None:
            return PlainTextResponse("报告不存在", status_code=404)
        payload = record.payload
        elements = payload["elements"]
        header = ["sample_id", "unique"]
        for ore in payload["ores"]:
            header.append(f"{ore}_weight")
        for ore in payload["ores"]:
            header.append(f"{ore}_weight_min")
            header.append(f"{ore}_weight_max")
        for element in elements:
            header.append(f"measured_{element}")
        for element in elements:
            header += [
                f"predicted_{element}",
                f"predicted_{element}_min",
                f"predicted_{element}_max",
            ]
        for element in elements:
            header += [
                f"residual_{element}",
                f"residual_{element}_min",
                f"residual_{element}_max",
            ]
        header += ["sse", "note"]

        def triple(point_value, interval):
            """One element's (value, min, max) cells.

            A non-unique optimum has no single value, so the value cell is
            left empty and only the exact attainable range is emitted.
            """

            low, high = interval
            return [
                "" if point_value is None else formatting.decimal(point_value, 10),
                formatting.decimal(low, 10),
                formatting.decimal(high, 10),
            ]

        rows = [",".join(_csv_cell(cell) for cell in header)]
        for result in payload["results"]:
            row = [result["sample_id"], "yes" if result["unique"] else "no"]
            if result["unique"]:
                row += [formatting.percent(v, 8) for v in result["weights"]]
            else:
                row += [""] * len(payload["ores"])
            for low, high in result["weight_ranges"]:
                row += [formatting.percent(low, 8), formatting.percent(high, 8)]
            row += [formatting.decimal(v, 10) for v in result["measured"]]
            for i in range(len(elements)):
                row += triple(
                    None if result["predicted"] is None else result["predicted"][i],
                    result["predicted_ranges"][i],
                )
            for i in range(len(elements)):
                row += triple(
                    None if result["residuals"] is None else result["residuals"][i],
                    result["residual_ranges"][i],
                )
            row.append(formatting.decimal(result["sse"], 12))
            row.append(
                ""
                if result["unique"]
                else "最优占比不唯一：占比、预测值与残差仅以下列 min/max 范围给出，"
                "不存在唯一的配料组合"
            )
            rows.append(",".join(_csv_cell(cell) for cell in row))
        return Response(
            "﻿" + "\r\n".join(rows) + "\r\n",
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="ore-report-{report_id}.csv"'
                )
            },
        )

    @app.get("/inputs/{kind}/export.csv")
    def export_input(kind: str) -> Response:
        if kind not in ("baseline", "samples", "manifest"):
            return PlainTextResponse("输入类型不存在", status_code=404)
        current = store.current_inputs().get(kind)
        if current is None:
            return PlainTextResponse("输入尚未上传", status_code=404)
        if kind == "baseline":
            value = BaselineInput.from_payload(current.payload)
            header = ["ore", *value.elements]
            rows = [
                [ore, *[formatting.decimal(x, 12) for x in row]]
                for ore, row in zip(value.ores, value.rows)
            ]
        elif kind == "manifest":
            value = ManifestInput.from_payload(current.payload)
            header = ["sample_id"]
            rows = [[sample_id] for sample_id in value.sample_ids]
        else:
            value = SamplesInput.from_payload(current.payload)
            header = ["sample_id", *value.elements]
            rows = [
                [sample_id, *[formatting.decimal(x, 12) for x in row]]
                for sample_id, row in zip(value.sample_ids, value.rows)
            ]
        lines = [",".join(_csv_cell(cell) for cell in header)]
        lines += [",".join(_csv_cell(cell) for cell in row) for row in rows]
        return Response(
            "﻿" + "\r\n".join(lines) + "\r\n",
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{kind}.csv"'
            },
        )

    @app.get("/api/latest")
    def api_latest() -> JSONResponse:
        record = store.latest_report()
        current = store.current_inputs()
        if record is None:
            return JSONResponse(
                {
                    "report": None,
                    "inputs": {
                        kind: (
                            {
                                "version": value.version_id,
                                "filename": value.filename,
                                "created_at": value.created_at,
                            }
                            if value
                            else None
                        )
                        for kind, value in current.items()
                    },
                    "reconcile": reconcile(current),
                }
            )
        return JSONResponse(
            {
                "report": {
                    "id": record.report_id,
                    "created_at": record.created_at,
                    "baseline_version": record.baseline_version,
                    "samples_version": record.samples_version,
                    "manifest_version": record.manifest_version,
                    "stale": record.is_stale,
                    "payload": record.payload,
                }
            }
        )

    return app


def _csv_cell(value: Any) -> str:
    text = str(value)
    if any(ch in text for ch in [",", '"', "\r", "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text


class _LabeledError(Exception):
    def __init__(self, label: str, message: str) -> None:
        super().__init__(message)
        self.label = label
        self.message = message


app = create_app()
