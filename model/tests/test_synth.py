"""The demo data has to be fit-able, and the planted truth has to be self-consistent.

These checks are cheap and they catch the failure that would waste the most time
later: a dataset where a parameter is not identified (e.g. price never varies, so
no elasticity can be learned).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ripple_model import synth

DATA = Path(synth.__file__).resolve().parents[1] / "data"
CUSTOMER_COLUMNS = ["week", "city", "segment", "units", "price", "promo", "feature"]


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not (DATA / "brightside_weekly.csv").exists():
        synth.main()
    return pd.read_csv(DATA / "brightside_weekly.csv")


@pytest.fixture(scope="module")
def truth() -> dict:
    if not (DATA / "truth.json").exists():
        synth.main()
    return json.loads((DATA / "truth.json").read_text())


def test_csv_is_exactly_the_customer_upload_format(panel: pd.DataFrame) -> None:
    """Demo and BYOD share one code path, so the demo CSV must carry no extra columns."""
    assert list(panel.columns) == CUSTOMER_COLUMNS


def test_panel_is_complete_and_clean(panel: pd.DataFrame) -> None:
    assert len(panel) == len(synth.CITIES) * len(synth.SEGMENTS) * synth.WEEKS
    assert not panel.isna().any().any()
    assert (panel.units >= 0).all()
    assert (panel.price > 0).all()
    assert panel.promo.isin([0, 1]).all()
    assert panel.feature.isin([0, 1]).all()
    assert panel.week.nunique() == synth.WEEKS
    # one row per week x city x segment, no duplicates
    assert not panel.duplicated(["week", "city", "segment"]).any()


def test_price_varies_within_every_city_so_elasticity_is_identified(panel: pd.DataFrame) -> None:
    for city, grp in panel.groupby("city"):
        assert grp.price.nunique() > 1, f"{city}: price never changes — no elasticity"
        assert grp.price.std() / grp.price.mean() > 0.02, f"{city}: price barely moves"


def test_promotions_and_their_dip_are_both_observable(panel: pd.DataFrame) -> None:
    rate = panel.promo.mean()
    assert 0.08 < rate < 0.40, f"promo rate {rate:.1%} is too extreme to learn a promo effect"
    # at least some promos must be followed by a non-promo week, or b_dip is unidentified
    wide = panel.pivot_table(index="week", columns=["city", "segment"], values="promo")
    followed = ((wide.shift(1) == 1) & (wide == 0)).to_numpy().sum()
    assert followed > 50, f"only {followed} post-promo weeks — b_dip would be unidentified"


def test_targeted_promotions_exist(panel: pd.DataFrame) -> None:
    """Some promotions hit one segment only; the segment-level promo effect needs that."""
    per_week = panel.groupby(["week", "city"]).promo.nunique()
    assert (per_week > 1).sum() > 10, "no segment-targeted promotions in the data"


def test_truth_is_self_consistent(truth: dict) -> None:
    p = truth["params"]
    C, S, P = len(truth["cities"]), len(truth["segments"]), len(truth["trait_names"])
    assert len(p["g"]) == P
    assert len(p["z_city"]) == C and len(p["a_city"]) == C
    for key in ("a_seg", "e", "b_promo"):
        assert len(p[key]) == S, f"params.{key} must have one entry per segment"
    assert len(p["season"]) == 2 * truth["n_fourier"]
    assert p["phi"] > 0 and p["tau_city"] > 0

    assert np.array(truth["candidates"]["Austin"]["traits_z"]).shape == (P,)
    # a_city must equal g0 + X_z . g + tau * z, or recovery will compare the wrong thing
    implied = p["g0"] + synth.traits_z() @ np.array(p["g"]) + p["tau_city"] * np.array(p["z_city"])
    assert np.allclose(implied, p["a_city"], atol=1e-9)


def test_planted_elasticities_are_ordered_and_negative(truth: dict) -> None:
    """Students most price-sensitive, retirees least — the story the demo tells."""
    e = truth["params"]["e"]
    assert all(x < 0 for x in e)
    assert e == sorted(e), "elasticities must increase from students to retirees"
    assert truth["params"]["b_dip"] < 0, "the post-promo dip must be a dip"
    assert all(b > 0 for b in truth["params"]["b_promo"])


def test_shares_sum_to_one(truth: dict) -> None:
    for row in truth["shares"]:
        assert abs(sum(row) - 1.0) < 1e-9
    for cand in truth["candidates"].values():
        assert abs(sum(cand["shares"]) - 1.0) < 1e-9


def test_one_candidate_is_out_of_range_and_two_are_not(truth: dict) -> None:
    lo, hi = np.array(truth["trait_min"]), np.array(truth["trait_max"])
    flags = {}
    for name, cand in truth["candidates"].items():
        z = np.array(cand["traits_z"])
        flags[name] = bool(((z < lo - 1e-9) | (z > hi + 1e-9)).any())
    assert flags["Miami"] is True, "Miami must trigger the extrapolation warning"
    assert flags["Austin"] is False and flags["Portland"] is False


def test_launches_have_truth_and_the_right_unknowns() -> None:
    if not (DATA / "launches.json").exists():
        synth.main()
    doc = json.loads((DATA / "launches.json").read_text())
    ids = {L["id"]: L for L in doc["launches"]}
    assert set(ids) == {"austin-launch", "student-coupon", "gift-boxes"}
    for L in doc["launches"]:
        obs = L["observations"]
        n = len(obs["t"])
        assert n > 0 and all(len(v) == n for v in obs.values()), f"{L['id']}: ragged observations"
        assert max(obs["week"]) == L["weeks_observed"] == 8
        assert L["horizon"] >= L["weeks_observed"]
        assert min(obs["t"]) == L["t0"], f"{L['id']}: t0 must be the first observed week index"
        assert "truth" in L and "unknown_value" in L["truth"]
        assert all(u >= 0 for u in obs["units"])
    assert ids["gift-boxes"]["has_holdout"] is True
    assert set(ids["gift-boxes"]["observations"]["holdout"]) == {0, 1}
    assert ids["austin-launch"]["observations"]["city_idx"][0] == -1, "new city uses city_idx -1"


def test_synth_is_deterministic() -> None:
    """Fixed seed: two runs must produce identical data."""
    rng1 = np.random.default_rng(synth.SEED)
    rng2 = np.random.default_rng(synth.SEED)
    t1, t2 = synth.make_truth(rng1), synth.make_truth(rng2)
    assert np.array_equal(t1.z_city, t2.z_city)
    p1 = synth.generate_panel(rng1, t1)
    p2 = synth.generate_panel(rng2, t2)
    pd.testing.assert_frame_equal(p1, p2)
