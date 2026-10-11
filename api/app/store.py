"""Session-scoped storage for uploaded datasets and fit jobs.

Spec Section 6: a customer's file stays in the API's temp storage for the session
and there is a delete button. So: a temp directory that is removed with the
process, an in-memory index, and no database.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DATASETS = 32


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Dataset:
    id: str
    name: str
    path: Path
    cities_path: Path | None
    rows: int
    columns: list[str]
    preview: list[dict[str, Any]]
    created_at: str = field(default_factory=_now)


@dataclass
class Job:
    id: str
    dataset_id: str
    status: str = "queued"          # queued | running | done | error
    step: str = "queued"
    progress: float = 0.0
    error: str | None = None
    posterior: dict | None = None
    trust: dict | None = None
    created_at: str = field(default_factory=_now)
    finished_at: str | None = None


class Store:
    """Thread-safe because fits run on a worker thread while requests keep coming."""

    def __init__(self) -> None:
        self._root = Path(tempfile.mkdtemp(prefix="ripple-api-"))
        self._datasets: dict[str, Dataset] = {}
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    # -- datasets ---------------------------------------------------------
    def new_dataset_dir(self) -> tuple[str, Path]:
        dataset_id = uuid.uuid4().hex[:12]
        d = self._root / dataset_id
        d.mkdir(parents=True, exist_ok=True)
        return dataset_id, d

    def put_dataset(self, dataset: Dataset) -> None:
        with self._lock:
            if len(self._datasets) >= MAX_DATASETS:
                oldest = min(self._datasets.values(), key=lambda x: x.created_at)
                self._drop(oldest.id)
            self._datasets[dataset.id] = dataset

    def get_dataset(self, dataset_id: str) -> Dataset | None:
        with self._lock:
            return self._datasets.get(dataset_id)

    def list_datasets(self) -> list[Dataset]:
        with self._lock:
            return sorted(self._datasets.values(), key=lambda x: x.created_at, reverse=True)

    def delete_dataset(self, dataset_id: str) -> bool:
        with self._lock:
            return self._drop(dataset_id)

    def _drop(self, dataset_id: str) -> bool:
        """Caller holds the lock. Removes the rows from disk, not just the index."""
        dataset = self._datasets.pop(dataset_id, None)
        if dataset is None:
            return False
        shutil.rmtree(dataset.path.parent, ignore_errors=True)
        for job in list(self._jobs.values()):
            if job.dataset_id == dataset_id:
                self._jobs.pop(job.id, None)
        return True

    # -- jobs -------------------------------------------------------------
    def put_job(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update_job(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)

    def shutdown(self) -> None:
        shutil.rmtree(self._root, ignore_errors=True)
