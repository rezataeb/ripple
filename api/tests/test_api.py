"""The BYOD flow end to end: upload -> validate -> fit -> poll.

The fit test is slow (it really fits a model), but it is the only way to prove
the customer path uses the same code as the demo rather than a second one.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.app.main import app

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "model" / "data" / "brightside_weekly.csv"
CITIES = ROOT / "model" / "data" / "brightside_cities.csv"


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def demo_csv() -> bytes:
    if not DEMO.exists():
        pytest.skip("run `uv run python -m ripple_model.synth` first")
    return DEMO.read_bytes()


def small_panel(weeks: int = 40, cities: int = 2, price_varies: bool = True) -> bytes:
    """A tiny but fittable panel, so the fit test finishes in seconds."""
    rows = []
    for t in range(weeks):
        for ci in range(cities):
            promo = 1 if (t % 7) in (0, 1) else 0
            price = 20.0 - (3.0 if (promo and price_varies) else 0.0)
            rows.append(
                {
                    "week": str(pd.Timestamp("2025-01-06") + pd.Timedelta(days=7 * t)),
                    "city": f"City {ci}",
                    "units": 100 + 20 * promo + ci * 10 + (t % 5),
                    "price": price,
                    "promo": promo,
                }
            )
    buf = io.BytesIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue()


# --------------------------------------------------------------------------
def test_health(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_available"] is True, "the API must be able to import ripple_model"


def test_upload_returns_a_preview_and_a_detected_mapping(client, demo_csv) -> None:
    r = client.post(
        "/datasets",
        files={
            "file": ("brightside.csv", demo_csv, "text/csv"),
            "cities": ("cities.csv", CITIES.read_bytes(), "text/csv"),
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rows"] == 2496
    assert body["has_cities_file"] is True
    assert len(body["preview"]) == 5
    # the demo CSV already uses Ripple's names, so every column should be detected
    for want in ("week", "units", "price", "city", "segment", "promo", "feature"):
        assert body["detected_mapping"].get(want) == want


def test_validate_accepts_the_demo_data(client, demo_csv) -> None:
    did = client.post(
        "/datasets", files={"file": ("d.csv", demo_csv, "text/csv")}
    ).json()["dataset_id"]
    body = client.post(f"/datasets/{did}/validate", json={"mapping": {}}).json()
    assert body["ok"] is True
    assert body["errors"] == []
    assert body["weeks"] == 104
    assert body["price_cv"] > 0.01


def test_validate_applies_a_column_mapping(client) -> None:
    """A customer's own column names must work once mapped."""
    df = pd.read_csv(io.BytesIO(small_panel()))
    df = df.rename(columns={"units": "qty_sold", "week": "week_start", "price": "unit_price"})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)

    did = client.post(
        "/datasets", files={"file": ("theirs.csv", buf.getvalue(), "text/csv")}
    ).json()["dataset_id"]

    unmapped = client.post(f"/datasets/{did}/validate", json={"mapping": {}}).json()
    assert unmapped["ok"] is False
    assert set(unmapped["missing_required"]) == {"week", "units", "price"}

    mapped = client.post(
        f"/datasets/{did}/validate",
        json={"mapping": {"units": "qty_sold", "week": "week_start", "price": "unit_price"}},
    ).json()
    assert mapped["ok"] is True, mapped["errors"]


def test_validate_warns_when_price_never_changes(client) -> None:
    """The warning that saves a user from a meaningless elasticity."""
    did = client.post(
        "/datasets", files={"file": ("flat.csv", small_panel(price_varies=False), "text/csv")}
    ).json()["dataset_id"]
    body = client.post(f"/datasets/{did}/validate", json={}).json()
    assert body["ok"] is True, "flat price is a warning, not an error"
    assert any("price never changes" in w.lower() for w in body["warnings"])


def test_validate_rejects_negative_units(client) -> None:
    df = pd.read_csv(io.BytesIO(small_panel()))
    df.loc[0, "units"] = -5
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    did = client.post(
        "/datasets", files={"file": ("bad.csv", buf.getvalue(), "text/csv")}
    ).json()["dataset_id"]
    body = client.post(f"/datasets/{did}/validate", json={}).json()
    assert body["ok"] is False
    assert body["negative_units"] == 1
    assert any("negative units" in e.lower() for e in body["errors"])


def test_fit_is_refused_when_the_data_cannot_be_fitted(client) -> None:
    bad = b"a,b,c\n1,2,3\n4,5,6\n"
    did = client.post(
        "/datasets", files={"file": ("junk.csv", bad, "text/csv")}
    ).json()["dataset_id"]
    r = client.post(f"/datasets/{did}/fit", json={})
    assert r.status_code == 422
    assert "errors" in r.json()["detail"]


def test_unknown_ids_are_404(client) -> None:
    assert client.post("/datasets/nope/validate", json={}).status_code == 404
    assert client.get("/jobs/nope").status_code == 404
    assert client.delete("/datasets/nope").status_code == 404


def test_delete_removes_the_rows_from_disk(client, demo_csv) -> None:
    """Spec Section 6 promises a delete button that actually deletes."""
    from api.app.main import store

    did = client.post(
        "/datasets", files={"file": ("d.csv", demo_csv, "text/csv")}
    ).json()["dataset_id"]
    path = store.get_dataset(did).path
    assert path.exists()

    assert client.delete(f"/datasets/{did}").status_code == 200
    assert not path.exists(), "the CSV is still on disk after delete"
    assert client.post(f"/datasets/{did}/validate", json={}).status_code == 404


@pytest.mark.slow
def test_full_fit_flow_produces_a_valid_posterior(client) -> None:
    """Upload -> fit -> poll -> a posterior that satisfies the contract."""
    import json

    import jsonschema

    did = client.post(
        "/datasets", files={"file": ("small.csv", small_panel(), "text/csv")}
    ).json()["dataset_id"]

    r = client.post(f"/datasets/{did}/fit", json={"draws": 150, "chains": 2, "unit_cost": 8.0})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    deadline = time.time() + 240
    steps = set()
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        steps.add(body["step"])
        if body["status"] in ("done", "error"):
            break
        time.sleep(1.0)

    assert body["status"] == "done", f"fit failed: {body.get('error')}"
    assert body["progress"] == 1.0
    assert len(steps) > 1, f"the UI needs progress steps, only saw {steps}"

    schema = json.loads((ROOT / "contract" / "posterior.schema.json").read_text())
    jsonschema.validate(body["posterior"], schema)
    trust_schema = json.loads((ROOT / "contract" / "trust.schema.json").read_text())
    jsonschema.validate(body["trust"], trust_schema)

    meta = body["posterior"]["meta"]
    assert meta["has_promo"] is True
    assert meta["has_segments"] is False, "this upload has no segment column"
    assert meta["has_traits"] is False, "no cities file was uploaded"
    assert meta["segments"] == ["All"]

    # recovery cannot run without planted truth, and it must say so rather than pass
    recovery = next(c for c in body["trust"]["checks"] if c["name"] == "Parameter recovery")
    assert recovery.get("skipped_reason"), "recovery must be marked skipped on BYOD"
