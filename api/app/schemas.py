"""Request and response models. Pydantic here is the server-side twin of the
zod validation in the browser, both generated from contract/*.schema.json by hand.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str = "ok"
    demo_mode: bool
    model_available: bool
    version: str


class DatasetPreview(BaseModel):
    dataset_id: str
    name: str
    rows: int
    columns: list[str]
    preview: list[dict[str, Any]]
    has_cities_file: bool
    created_at: str
    detected_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Best-guess column mapping from an exact/lowercase name match. "
        "The UI shows it for confirmation; the LLM mapping assistant runs in the "
        "Next.js route, not here.",
    )


class ValidateRequest(BaseModel):
    mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Ripple column -> the uploaded file's column, e.g. "
        '{"units": "qty_sold", "week": "week_start"}.',
    )


class ValidationReport(BaseModel):
    dataset_id: str
    ok: bool
    rows: int
    columns: list[str]
    missing_required: list[str]
    errors: list[str]
    warnings: list[str]
    info: list[str]
    weeks: int | None = None
    week_range: list[str] | None = None
    cities: int | None = None
    segments: int | None = None
    missing_values: int | None = None
    negative_units: int | None = None
    price_cv: float | None = None


class FitRequest(BaseModel):
    mapping: dict[str, str] = Field(default_factory=dict)
    base_price: float | None = None
    unit_cost: float = 0.0
    draws: int = Field(500, ge=50, le=2000)
    chains: int = Field(2, ge=1, le=4)


class FitAccepted(BaseModel):
    job_id: str
    dataset_id: str
    status: str
    poll: str


class JobStatus(BaseModel):
    job_id: str
    dataset_id: str
    status: str
    step: str
    progress: float
    error: str | None = None
    created_at: str
    finished_at: str | None = None
    posterior: dict | None = None
    trust: dict | None = None
