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


def test_fixture_a_city_contains_g0() -> None:
    """Regression guard.

    contract/README.md says g0 is already inside the fitted a_city and is added
    only for a NEW city. A fixture whose a_city omits g0 still validates against
    the schema but makes every mu ~400x too small, and the UI built on it looks
    fine while being wrong. So: check the identity, and check the magnitude.
    """
    import numpy as np
    from ripple_model.simulate import mu_for_cells

    f = load(CONTRACT / "posterior.fixture.json")
    m, d = f["meta"], f["draws"]

    # a_city - g0 - traits_z.g = tau_city * z_city, so dividing by tau_city must
    # give the same z per city in every draw.
    a_city = np.array(d["a_city"])
    g0 = np.array(d["g0"])
    g = np.array(d["g"])
    tau = np.array(d["tau_city"])
    traits_z = np.array(m["traits_z"])
    implied_z = (a_city - g0[:, None] - g @ traits_z.T) / tau[:, None]
    assert implied_z.std(axis=0).max() < 1e-9, "a_city does not follow g0 + Xz.g + tau*z"

    # magnitude: mu on the history rows must be in the same world as the observed units
    h = f["history"]
    seg_idx = np.array(h["segment_idx"])
    city_idx = np.array(h["city_idx"])
    shares = np.array(m["shares"])
    mu = mu_for_cells(
        f,
        a_city_draws=a_city[:, city_idx],
        seg_idx=seg_idx,
        log_share=np.log(shares[city_idx, seg_idx]),
        t=np.array([0]),
        price=np.array(h["price"])[None, :],
        promo=np.array(h["promo"])[None, :],
        promo_last=np.array(h["promo_last_week"])[None, :],
        feature=np.array(h["feature"])[None, :],
    )[:, 0, :]
    ratio = mu.mean() / np.mean(h["units"])
    assert 0.5 < ratio < 2.0, f"fixture mu is {ratio:.3g}x the observed units — g0 lost?"


def test_exported_candidate_shares_are_in_meta_segment_order() -> None:
    """Regression guard, found by Session B.

    candidates[].shares is positional against meta.segments. synth.py declares
    segments in its own order and data.py sorts them alphabetically, so passing a
    plain list gave every candidate city the wrong customer mix — a new-city
    forecast built on the wrong segment weights, with nothing to flag it.
    """
    path = PUBLIC / "posterior.json"
    truth_path = ROOT / "model" / "data" / "truth.json"
    if not (path.exists() and truth_path.exists()):
        pytest.skip("export the demo model first")
    post, truth = load(path), load(truth_path)
    segments = post["meta"]["segments"]

    for cand in post["candidates"]:
        planted = dict(zip(truth["segments"], truth["candidates"][cand["name"]]["shares"],
                           strict=True))
        for i, segment in enumerate(segments):
            assert abs(cand["shares"][i] - planted[segment]) < 1e-9, (
                f"{cand['name']}: shares[{i}] is for {segment} but does not match the "
                f"planted mix — the list is in the wrong order"
            )
        assert abs(sum(cand["shares"]) - 1.0) < 1e-9


def test_build_candidates_rejects_positional_shares() -> None:
    """A list cannot be checked against meta.segments, so it is refused outright."""
    from ripple_model.data import load_demo
    from ripple_model.export import build_candidates

    if not (ROOT / "model" / "data" / "brightside_weekly.csv").exists():
        pytest.skip("generate the demo data first")
    d = load_demo()
    with pytest.raises(TypeError, match="keyed by segment name"):
        build_candidates(d, {"Nowhere": {"traits": [13.0, 80.0, 0.3], "shares": [0.25] * 4}})
    with pytest.raises(ValueError, match="no share given"):
        build_candidates(
            d, {"Nowhere": {"traits": [13.0, 80.0, 0.3], "shares": {"students": 1.0}}}
        )
