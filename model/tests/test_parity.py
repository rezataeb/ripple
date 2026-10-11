"""simulate.py must reproduce contract/parity.fixture.json exactly.

This is the Python half of the parity pair. web/tests runs the same assertions
against web/lib/simulate.ts and the same fixture, which is what makes "the
browser simulator and the reference simulator agree" a tested claim rather than
an intention. Tolerance: 1e-6 on mu (spec Section 5).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ripple_model.simulate import mu_for_cells, new_city_a, new_city_z, simulate

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contract"

PARITY = json.loads((CONTRACT / "parity.fixture.json").read_text())
POSTERIOR = json.loads((CONTRACT / "posterior.fixture.json").read_text())
MU_TOL = PARITY["tolerance"]["mu"]
SUM_TOL = PARITY["tolerance"]["summary"]
CASES = {c["name"]: c for c in PARITY["cases"]}


def _walk(node, path=""):
    """Yield (dotted_path, value) for every leaf number in a nested result dict."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")
    elif isinstance(node, bool) or node is None or isinstance(node, str):
        yield path, node
    else:
        yield path, float(node)


@pytest.mark.parametrize("name", list(CASES))
def test_result_matches_parity_fixture(name: str) -> None:
    case = CASES[name]
    got = simulate(POSTERIOR, case["scenario"])
    ignore = set(PARITY["ignore_fields"])

    want_leaves = dict(_walk(case["expected"]))
    got_leaves = dict(_walk(got))

    missing = [k for k in want_leaves if k not in got_leaves and k.split("[")[0] not in ignore]
    assert not missing, f"{name}: result is missing {missing[:5]}"

    for key, want in want_leaves.items():
        if any(key.startswith(i) for i in ignore) or key.startswith("scenario."):
            continue
        have = got_leaves[key]
        if isinstance(want, (str, bool)) or want is None:
            assert have == want, f"{name}.{key}: {have!r} != {want!r}"
        else:
            assert abs(have - want) <= SUM_TOL + SUM_TOL * abs(want), (
                f"{name}.{key}: {have!r} != {want!r}"
            )


@pytest.mark.parametrize("name", list(CASES))
def test_mu_matches_parity_fixture(name: str) -> None:
    """Recompute mu cell by cell at the recorded weeks and compare to 1e-6."""
    case = CASES[name]
    mu = case["mu"]
    scenario = case["scenario"]
    meta, draws = POSTERIOR["meta"], POSTERIOR["draws"]
    base_price = float(meta["base_price"])
    cities, segments = list(meta["cities"]), list(meta["segments"])
    t = np.array(mu["t"])
    assert list(t) == [int(meta["weeks"]) + w for w in mu["weeks"]], (
        "horizon week w must use absolute t = meta.weeks + w"
    )
    want_bl = np.array(mu["baseline_flat"]).reshape(mu["shape"])
    want_sc = np.array(mu["scenario_flat"]).reshape(mu["shape"])

    if scenario["type"] == "new_city":
        cand = next(c for c in POSTERIOR["candidates"] if c["name"] == scenario["candidate"])
        sel = np.arange(len(segments))
        a_new = new_city_a(POSTERIOR, np.array(cand["traits_z"]), int(scenario.get("seed", 7)))
        a_city_draws = np.repeat(a_new[:, None], len(sel), axis=1)
        price = np.ones((len(t), len(sel))) * base_price
        zeros = np.zeros_like(price)
        got_sc = mu_for_cells(
            POSTERIOR, a_city_draws=a_city_draws, seg_idx=sel,
            log_share=np.log(np.array(cand["shares"])[sel]), t=t,
            price=price, promo=zeros, promo_last=zeros, feature=zeros,
        )
        got_bl = np.zeros_like(got_sc)
    else:
        gc, gs = np.meshgrid(np.arange(len(cities)), np.arange(len(segments)), indexing="ij")
        city_idx, seg_idx = gc.ravel(), gs.ravel()
        shares = np.array(meta["shares"])
        city_sel = set(scenario.get("cities") or cities)
        seg_sel = set(scenario.get("segments") or segments)
        affected = np.array(
            [cities[c] in city_sel and segments[s] in seg_sel
             for c, s in zip(city_idx, seg_idx, strict=True)]
        )
        base = np.ones((len(t), len(city_idx))) * base_price
        zeros = np.zeros_like(base)
        price_sc, promo_sc, promo_last_sc = base.copy(), zeros.copy(), zeros.copy()
        if scenario["type"] == "promo":
            n_promo = min(int(scenario["weeks"]), int(scenario["horizon"]))
            for i, w in enumerate(mu["weeks"]):
                if w < n_promo:
                    price_sc[i, affected] = base_price * (1.0 - float(scenario["discount"]))
                    promo_sc[i, affected] = 1.0
                if w == n_promo:
                    promo_last_sc[i, affected] = 1.0
        else:
            price_sc[:, affected] = base_price * (1.0 + float(scenario["price_change"]))

        common = dict(
            a_city_draws=np.array(draws["a_city"])[:, city_idx],
            seg_idx=seg_idx,
            log_share=np.log(shares[city_idx, seg_idx]),
            t=t,
        )
        got_bl = mu_for_cells(
            POSTERIOR, **common, price=base, promo=zeros, promo_last=zeros, feature=zeros
        )
        got_sc = mu_for_cells(
            POSTERIOR, **common, price=price_sc, promo=promo_sc,
            promo_last=promo_last_sc, feature=zeros,
        )

    assert got_sc.shape == want_sc.shape, f"{name}: mu shape {got_sc.shape} != {want_sc.shape}"
    assert np.max(np.abs(got_bl - want_bl)) <= MU_TOL, f"{name}: baseline mu drifted"
    assert np.max(np.abs(got_sc - want_sc)) <= MU_TOL, f"{name}: scenario mu drifted"


def test_cases_cover_all_three_scenario_types() -> None:
    assert {c["scenario"]["type"] for c in PARITY["cases"]} == {"promo", "price", "new_city"}


def test_new_city_case_is_flagged_as_extrapolation() -> None:
    """The out-of-range candidate must actually raise the flag the UI renders."""
    case = CASES["new-city-miami"]
    assert case["expected"]["flags"]["extrapolation"] is True
    assert case["expected"]["flags"]["out_of_range_traits"]
    assert case["expected"]["flags"]["warnings"]


def test_unselected_cells_cancel_out_of_the_incremental() -> None:
    """A promo on one cell must leave the other cells' incremental at exactly zero."""
    case = CASES["promo-students-boston"]
    got = simulate(POSTERIOR, case["scenario"])
    others = [c for c in got["cities"] if c["city"] != "Boston"]
    assert others, "expected other cities in the result"
    for c in others:
        assert abs(c["incremental"]["units"]["mean"]) < 1e-12, f"{c['city']} should not move"


def test_new_city_z_comes_from_the_export_not_an_rng() -> None:
    """Regression guard for a cross-language bug that a green test would hide.

    numpy's RNG and a browser PRNG cannot produce the same standard normals, so
    if the z draws were generated at simulation time the browser would disagree
    with this file on every new-city scenario — while a parity test that backs z
    out of the fixture would still pass. The draws therefore ship in the export.
    """
    assert "z_new" in POSTERIOR["draws"], "the posterior must carry the new-city z draws"
    z = np.array(POSTERIOR["draws"]["z_new"])
    assert len(z) == POSTERIOR["meta"]["n_draws"]
    assert abs(z.mean()) < 0.2 and 0.7 < z.std() < 1.3, "z_new should look standard normal"

    # the seed rotates the shipped array, which any language can reproduce exactly
    assert np.array_equal(new_city_z(POSTERIOR, 0), z)
    assert np.array_equal(new_city_z(POSTERIOR, 3), np.roll(z, -3))
    n = len(z)
    assert np.array_equal(new_city_z(POSTERIOR, n), z), "rotation must wrap"

    case = CASES["new-city-miami"]
    assert "z" in case, "the fixture must publish the z vector it used"
    seed = int(case["scenario"].get("seed", 7))
    assert np.allclose(case["z"], new_city_z(POSTERIOR, seed), atol=1e-12)


def test_new_city_results_are_reproducible_and_seed_dependent() -> None:
    scenario = dict(CASES["new-city-miami"]["scenario"])
    a = simulate(POSTERIOR, scenario)["incremental"]["profit"]["mean"]
    b = simulate(POSTERIOR, scenario)["incremental"]["profit"]["mean"]
    assert a == b, "the same scenario must always give the same number"
    other = simulate(POSTERIOR, {**scenario, "seed": 99})["incremental"]["profit"]["mean"]
    assert other != a, "changing the seed must change the new-city draw"
