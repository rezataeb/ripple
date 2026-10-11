"""Ripple API — bring your own data (spec Section 8).

The demo needs no server: the posterior ships to the browser as JSON. This API
exists only for the BYOD path, where a customer's CSV has to be validated and
fitted somewhere that is not their laptop.

  GET    /health                     status, and whether the model can be imported
  POST   /datasets                   upload CSV(s) -> dataset_id + preview
  POST   /datasets/{id}/validate     a column mapping -> validation report
  POST   /datasets/{id}/fit          start a background fit -> job_id
  GET    /jobs/{id}                  status, step, then posterior + trust
  DELETE /datasets/{id}              the delete button from Section 6

It imports `ripple_model` directly, so the fit, the checks and the export are the
same code that produced the demo — there is no second implementation to drift.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.app.schemas import (
    DatasetPreview,
    FitAccepted,
    FitRequest,
    Health,
    JobStatus,
    ValidateRequest,
    ValidationReport,
)
from api.app.store import MAX_UPLOAD_BYTES, Dataset, Job, Store

VERSION = "0.1.0"
PREVIEW_ROWS = 5
FIT_TIME_LIMIT = 90.0  # seconds, spec Section 3

store = Store()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Uploaded rows live for the session only (spec Section 6)."""
    yield
    store.shutdown()


app = FastAPI(
    title="Ripple API",
    version=VERSION,
    description="Bring your own data: validate a CSV and fit the Ripple demand model.",
    lifespan=lifespan,
)

# the browser calls this directly from the /data page
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("RIPPLE_CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
def _model_available() -> bool:
    try:
        import ripple_model.model  # noqa: F401
    except ImportError:
        return False
    return True


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(
        status="ok",
        demo_mode=os.environ.get("DEMO_MODE", "1") == "1",
        model_available=_model_available(),
        version=VERSION,
    )


# --------------------------------------------------------------------------
def _read_csv(raw: bytes, filename: str) -> pd.DataFrame:
    from io import BytesIO

    try:
        df = pd.read_csv(BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"{filename} could not be read as CSV: {exc}") from exc
    if df.empty:
        raise HTTPException(400, f"{filename} has no rows.")
    return df


def _detect_mapping(columns: list[str]) -> dict[str, str]:
    """Exact and case-insensitive name matches only.

    The real mapping assistant is the Gemini route in the Next.js app; this is the
    free part, so the UI can pre-fill the obvious columns before asking anyone.
    """
    from ripple_model.data import OPTIONAL, REQUIRED

    lower = {c.strip().lower(): c for c in columns}
    return {want: lower[want] for want in (*REQUIRED, *OPTIONAL) if want in lower}


@app.post("/datasets", response_model=DatasetPreview)
async def create_dataset(
    file: Annotated[UploadFile, File(description="The weekly panel CSV.")],
    cities: Annotated[UploadFile | None, File(description="Optional city traits CSV.")] = None,
) -> DatasetPreview:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB.")
    df = _read_csv(raw, file.filename or "upload.csv")

    dataset_id, folder = store.new_dataset_dir()
    path = folder / "data.csv"
    path.write_bytes(raw)

    cities_path: Path | None = None
    if cities is not None:
        craw = await cities.read()
        if craw:
            if len(craw) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Cities file is too large.")
            _read_csv(craw, cities.filename or "cities.csv")
            cities_path = folder / "cities.csv"
            cities_path.write_bytes(craw)

    preview = json.loads(df.head(PREVIEW_ROWS).to_json(orient="records"))
    dataset = Dataset(
        id=dataset_id,
        name=file.filename or "upload.csv",
        path=path,
        cities_path=cities_path,
        rows=int(len(df)),
        columns=[str(c) for c in df.columns],
        preview=preview,
    )
    store.put_dataset(dataset)

    return DatasetPreview(
        dataset_id=dataset.id,
        name=dataset.name,
        rows=dataset.rows,
        columns=dataset.columns,
        preview=dataset.preview,
        has_cities_file=cities_path is not None,
        created_at=dataset.created_at,
        detected_mapping=_detect_mapping(dataset.columns),
    )


def _require(dataset_id: str) -> Dataset:
    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(404, "Unknown dataset. It may have expired or been deleted.")
    return dataset


def _apply_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename the customer's columns to Ripple's names."""
    if not mapping:
        return df
    unknown = [src for src in mapping.values() if src and src not in df.columns]
    if unknown:
        raise HTTPException(400, f"Mapped column(s) not in the file: {', '.join(unknown)}")
    return df.rename(columns={src: want for want, src in mapping.items() if src})


def _load(dataset: Dataset, mapping: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    df = _apply_mapping(pd.read_csv(dataset.path), mapping)
    cities = pd.read_csv(dataset.cities_path) if dataset.cities_path else None
    return df, cities


@app.post("/datasets/{dataset_id}/validate", response_model=ValidationReport)
def validate_dataset(dataset_id: str, body: ValidateRequest) -> ValidationReport:
    from ripple_model.data import validate

    dataset = _require(dataset_id)
    df, cities = _load(dataset, body.mapping)
    report = validate(df, cities)
    return ValidationReport(dataset_id=dataset_id, **report)


# --------------------------------------------------------------------------
def _run_fit(job_id: str, dataset_id: str, body: FitRequest) -> None:
    """The background fit. Every failure becomes a job error, never a crash."""
    from ripple_model.checks import build_trust
    from ripple_model.data import DataError, prepare
    from ripple_model.export import build_posterior
    from ripple_model.model import draws_dict, fit

    try:
        dataset = store.get_dataset(dataset_id)
        if dataset is None:
            store.update_job(job_id, status="error", step="failed",
                             error="dataset was deleted before the fit started")
            return

        store.update_job(job_id, status="running", step="reading the file", progress=0.1)
        df, cities = _load(dataset, body.mapping)

        store.update_job(job_id, step="checking the data", progress=0.2)
        data = prepare(
            df, cities, base_price=body.base_price, unit_cost=body.unit_cost,
            dataset_name=dataset.name,
        )

        store.update_job(job_id, step="fitting the model", progress=0.35)
        result = fit(
            data, draws=body.draws, tune=body.draws, chains=body.chains,
            time_limit=FIT_TIME_LIMIT,
        )
        arrays = draws_dict(result.idata)

        store.update_job(job_id, step="running the trust checks", progress=0.8)
        # no planted truth on a customer upload, and no holdout refit inside the
        # time budget: build_trust marks both as skipped rather than passing them
        trust = build_trust(arrays, result, data)

        store.update_job(job_id, step="exporting", progress=0.95)
        posterior = build_posterior(arrays, result, data)

        store.update_job(
            job_id, status="done", step="done", progress=1.0,
            posterior=posterior, trust=trust,
            finished_at=_iso_now(),
        )
    except DataError as exc:
        store.update_job(job_id, status="error", step="failed", error=str(exc),
                         finished_at=_iso_now())
    except Exception as exc:  # noqa: BLE001 - a failed fit is a job state, not a 500
        store.update_job(job_id, status="error", step="failed",
                         error=f"{type(exc).__name__}: {exc}", finished_at=_iso_now())


def _iso_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


@app.post("/datasets/{dataset_id}/fit", response_model=FitAccepted, status_code=202)
def start_fit(dataset_id: str, body: FitRequest) -> FitAccepted:
    from ripple_model.data import validate

    dataset = _require(dataset_id)
    df, cities = _load(dataset, body.mapping)
    report = validate(df, cities)
    if report["errors"]:
        raise HTTPException(422, {"message": "The data cannot be fitted.",
                                  "errors": report["errors"]})

    job = Job(id=uuid.uuid4().hex[:12], dataset_id=dataset_id)
    store.put_job(job)
    threading.Thread(target=_run_fit, args=(job.id, dataset_id, body), daemon=True).start()
    return FitAccepted(
        job_id=job.id, dataset_id=dataset_id, status=job.status, poll=f"/jobs/{job.id}"
    )


@app.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    return JobStatus(
        job_id=job.id, dataset_id=job.dataset_id, status=job.status, step=job.step,
        progress=job.progress, error=job.error, created_at=job.created_at,
        finished_at=job.finished_at, posterior=job.posterior, trust=job.trust,
    )


@app.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict[str, Any]:
    if not store.delete_dataset(dataset_id):
        raise HTTPException(404, "Unknown dataset.")
    return {"deleted": dataset_id, "message": "The rows were removed from temp storage."}


@app.get("/datasets")
def list_datasets() -> dict[str, Any]:
    return {
        "datasets": [
            {"dataset_id": d.id, "name": d.name, "rows": d.rows, "created_at": d.created_at}
            for d in store.list_datasets()
        ]
    }
