"""The fixture and every real export must satisfy the schemas in contract/.

This is the cheapest guard we have against the two sessions drifting apart:
if Session A changes a shape without changing the schema, this fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contract"
PUBLIC = ROOT / "web" / "public" / "model"

SCHEMAS = [
    "posterior.schema.json",
    "trust.schema.json",
    "scenario.schema.json",
    "result.schema.json",
    "launch.schema.json",
    "update.schema.json",
    "calibration.schema.json",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.mark.parametrize("name", SCHEMAS)
def test_schema_is_a_valid_schema(name: str) -> None:
    schema = load(CONTRACT / name)
    jsonschema.Draft202012Validator.check_schema(schema)


def test_fixture_matches_posterior_schema() -> None:
    jsonschema.validate(
        load(CONTRACT / "posterior.fixture.json"), load(CONTRACT / "posterior.schema.json")
    )


def test_fixture_shapes_are_self_consistent() -> None:
    f = load(CONTRACT / "posterior.fixture.json")
    m, d = f["meta"], f["draws"]
    D, C, S = m["n_draws"], len(m["cities"]), len(m["segments"])
    P, K = len(m["trait_names"]), 2 * m["n_fourier"]

    for key in d:
        assert len(d[key]) == D, f"draws.{key} has {len(d[key])} rows, want n_draws={D}"
    widths = [("g", P), ("a_city", C), ("a_seg", S), ("e", S), ("b_promo", S), ("season", K)]
    for key, width in widths:
        assert all(len(r) == width for r in d[key]), f"draws.{key} rows must be length {width}"

    assert len(m["shares"]) == C and all(len(r) == S for r in m["shares"])
    assert all(abs(sum(r) - 1.0) < 1e-9 for r in m["shares"]), "segment shares must sum to 1"
    for key in ("trait_mean", "trait_std", "trait_min", "trait_max"):
        assert len(m[key]) == P, f"meta.{key} must have one entry per trait"

    h = f["history"]
    lengths = {k: len(v) for k, v in h.items()}
    assert len(set(lengths.values())) == 1, f"history columns must share one length, got {lengths}"
    assert max(h["t"]) == m["weeks"] - 1, "history week index must run 0..weeks-1"
    assert all(0 <= i < C for i in h["city_idx"])
    assert all(0 <= i < S for i in h["segment_idx"])

    for cand in f["candidates"]:
        assert len(cand["traits"]) == P and len(cand["traits_z"]) == P
        assert len(cand["shares"]) == S
        out = [
            m["trait_names"][i]
            for i in range(P)
            if cand["traits_z"][i] < m["trait_min"][i] - 1e-9
            or cand["traits_z"][i] > m["trait_max"][i] + 1e-9
        ]
        assert cand["extrapolation"] is bool(out), f"{cand['name']}: extrapolation flag is wrong"
        assert cand["out_of_range_traits"] == out


def test_fixture_exercises_the_interesting_paths() -> None:
    """Session B builds the UI on this file, so it has to contain the edge cases."""
    f = load(CONTRACT / "posterior.fixture.json")
    assert any(c["extrapolation"] for c in f["candidates"]), "need an out-of-range candidate"
    assert any(c for c in f["candidates"] if not c["extrapolation"]), "need one in-range candidate"
    assert any(p > 0 for p in f["history"]["promo"]), "need promo weeks so the dip is visible"
    assert any(p > 0 for p in f["history"]["promo_last_week"])


@pytest.mark.parametrize(
    ("export", "schema"),
    [
        ("posterior.json", "posterior.schema.json"),
        ("trust.json", "trust.schema.json"),
        ("launches.json", "launch.schema.json"),
        ("calibration.json", "calibration.schema.json"),
    ],
)
def test_real_export_matches_schema(export: str, schema: str) -> None:
    path = PUBLIC / export
    if not path.exists():
        pytest.skip(f"{export} not exported yet")
    jsonschema.validate(load(path), load(CONTRACT / schema))
