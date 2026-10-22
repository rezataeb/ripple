"""The launch tracker has to actually learn, and has to know when it cannot.

These tests run the real update against the real export, so they also catch the
two bugs that bit during the build: launch indices written in declaration order
instead of export order, and a launch that reused the existing `feature` column
so b_feature absorbed the lift the unknown was supposed to measure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ripple_model.update import (
    nb_loglik,
    normalize_log_weights,
    update_with_launch,
    weight_ess,
    weighted_quantile,
)

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "web" / "public" / "model"

pytestmark = pytest.mark.skipif(
    not (PUBLIC / "posterior.json").exists(),
    reason="run model/scripts/run_fits.py and export_demo.py first",
)


@pytest.fixture(scope="module")
def post() -> dict:
    return json.loads((PUBLIC / "posterior.json").read_text())


@pytest.fixture(scope="module")
def launches() -> dict[str, dict]:
    doc = json.loads((PUBLIC / "launches.json").read_text())
    return {launch["id"]: launch for launch in doc["launches"]}


LAUNCH_IDS = ["austin-launch", "student-coupon", "gift-boxes"]


# --------------------------------------------------------------------------
# the maths underneath
# --------------------------------------------------------------------------
def test_weighted_quantile_matches_numpy_on_equal_weights() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    w = np.full(5000, 1.0 / 5000)
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        assert weighted_quantile(x, q, w)[0] == pytest.approx(np.quantile(x, q), abs=0.02)


def test_weighted_quantile_respects_weights() -> None:
    """All weight on the small values must pull every quantile down."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    assert weighted_quantile(x, 0.5, np.array([1.0, 1.0, 0.0, 0.0]))[0] < 1.01
    assert weighted_quantile(x, 0.5, np.array([0.0, 0.0, 1.0, 1.0]))[0] > 1.99


def test_normalize_log_weights_is_shift_invariant() -> None:
    logw = np.array([-1000.5, -1001.0, -1002.0])
    a = normalize_log_weights(logw)
    b = normalize_log_weights(logw + 937.0)
    assert np.allclose(a, b)
    assert a.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(a)), "a large constant offset must not underflow"


def test_nb_loglik_matches_scipy() -> None:
    from scipy.stats import nbinom

    rng = np.random.default_rng(1)
    y = rng.poisson(200, 12).astype(float)
    mu, phi = 180.0, 22.0
    want = nbinom.logpmf(y, phi, phi / (phi + mu)).sum()
    got = nb_loglik(y[None, :], np.full((1, 12), mu), np.array([[phi]]))[0]
    assert got == pytest.approx(want, rel=1e-10)


def test_weight_ess_bounds() -> None:
    assert weight_ess(np.ones(500)) == pytest.approx(500.0)
    spike = np.zeros(500)
    spike[3] = 1.0
    assert weight_ess(spike) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# the update itself
# --------------------------------------------------------------------------
@pytest.mark.parametrize("launch_id", LAUNCH_IDS)
def test_week_zero_is_exactly_the_prior(post, launches, launch_id) -> None:
    r = update_with_launch(post, launches[launch_id], 0)
    assert r["weeks_used"] == 0
    assert r["forecast"]["total_units"] == r["prior_forecast"]["total_units"]
    assert r["gate"]["ok"] is True
    assert r["meta"]["n_obs_cells"] == 0


@pytest.mark.parametrize("launch_id", LAUNCH_IDS)
def test_the_interval_narrows_as_weeks_are_revealed(post, launches, launch_id) -> None:
    """The whole promise of the launch tracker: more data, less uncertainty."""
    widths = []
    for weeks in (0, 2, 4, 8):
        f = update_with_launch(post, launches[launch_id], weeks)["forecast"]["total_units"]
        widths.append(f["p95"] - f["p05"])
    assert widths[-1] < widths[0] * 0.75, f"interval barely narrowed: {widths}"
    assert widths[0] > widths[2] > widths[3], f"not monotone: {widths}"


@pytest.mark.parametrize("launch_id", LAUNCH_IDS)
def test_the_truth_stays_inside_the_interval_at_every_week(post, launches, launch_id) -> None:
    """A single realization, so this is a smoke test, not calibration — but a
    90% interval that misses the planted value at every week means a bug."""
    launch = launches[launch_id]
    truth = launch["truth"]["unknown_value"]
    misses = []
    for weeks in range(0, launch["weeks_observed"] + 1):
        u = update_with_launch(post, launch, weeks)["unknown"]["posterior"]
        if not (u["p05"] <= truth <= u["p95"]):
            misses.append(weeks)
    assert not misses, f"{launch_id}: truth {truth:+.3f} outside the 90% interval at {misses}"


@pytest.mark.parametrize("launch_id", LAUNCH_IDS)
def test_the_unknown_moves_away_from_its_prior(post, launches, launch_id) -> None:
    """If the posterior equals the prior, the likelihood is not being applied.

    How much it narrows depends on how much the data can say: a city-level effect
    seen through 8 weeks of count noise moves less than a per-cell offer lift. So
    this asserts that it both narrows and shifts, not a particular factor.
    """
    launch = launches[launch_id]
    r = update_with_launch(post, launch, launch["weeks_observed"])
    prior, posterior = r["unknown"]["prior"], r["unknown"]["posterior"]
    prior_w = prior["p95"] - prior["p05"]
    post_w = posterior["p95"] - posterior["p05"]
    assert post_w < prior_w / 1.2, f"{launch_id}: barely narrowed ({prior_w:.3f} -> {post_w:.3f})"
    shift = abs(posterior["mean"] - prior["mean"])
    assert shift > 0.25 * post_w, f"{launch_id}: posterior did not move off the prior"
    assert r["meta"]["n_obs_cells"] > 0


def test_austin_learns_a_city_bigger_than_its_traits_predict(post, launches) -> None:
    """Austin's planted z is +0.85, so the forecast must move UP, not down."""
    launch = launches["austin-launch"]
    prior = update_with_launch(post, launch, 0)["forecast"]["total_units"]["mean"]
    after = update_with_launch(post, launch, 8)["forecast"]["total_units"]["mean"]
    assert after > prior, f"forecast fell from {prior:,.0f} to {after:,.0f}"
    true_total = launch["truth"]["total_units"]
    final = update_with_launch(post, launch, 8)["forecast"]["total_units"]
    assert final["p05"] <= true_total <= final["p95"], "true 12-week total outside the interval"


def test_gift_boxes_is_labelled_an_experiment(post, launches) -> None:
    """It has a random holdout column, so the lift is measured, not inferred."""
    r = update_with_launch(post, launches["gift-boxes"], 8)
    assert r["causal"]["mode"] == "experiment"
    assert "holdout" in r["causal"]["recommendation"].lower()


def test_austin_is_labelled_model_based(post, launches) -> None:
    r = update_with_launch(post, launches["austin-launch"], 8)
    assert r["causal"]["mode"] == "model_based"
    assert len(r["causal"]["assumptions"]) == 3, "the 3 causal assumptions must be shown"


def test_percent_lift_is_computed_for_the_ui(post, launches) -> None:
    """The UI must never convert a log lift into a percent itself."""
    for launch_id in LAUNCH_IDS:
        r = update_with_launch(post, launches[launch_id], 4)
        if r["unknown"]["display"] == "percent_lift":
            assert "posterior_percent" in r["unknown"]
            assert "prior_percent" in r["unknown"]


def test_observed_points_appear_only_up_to_the_slider(post, launches) -> None:
    launch = launches["austin-launch"]
    r = update_with_launch(post, launch, 3)
    seen = [w["week"] for w in r["forecast"]["weekly"] if w["observed_units"] is not None]
    assert seen == [1, 2, 3]
    assert len(r["forecast"]["weekly"]) == launch["horizon"]


def test_a_skeptical_prior_scale_can_be_changed(post, launches) -> None:
    """The gift-boxes prior is editable in the UI, so the engine must honour it."""
    launch = launches["gift-boxes"]
    tight = update_with_launch(post, launch, 0, prior_sigma=0.05)["unknown"]["prior"]
    wide = update_with_launch(post, launch, 0, prior_sigma=0.40)["unknown"]["prior"]
    assert (wide["p95"] - wide["p05"]) > 3 * (tight["p95"] - tight["p05"])


def test_a_wildly_wrong_launch_trips_the_gate(post, launches) -> None:
    """The gate has to fire on something. Multiply the observed units by 50 and
    the launch becomes impossible under the prior."""
    launch = json.loads(json.dumps(launches["austin-launch"]))
    launch["observations"]["units"] = [u * 50 for u in launch["observations"]["units"]]
    r = update_with_launch(post, launch, 8)
    assert r["gate"]["ok"] is False, f"gate stayed open: {r['gate']}"
    assert "refit" in r["gate"]["message"].lower()


@pytest.mark.parametrize("launch_id", LAUNCH_IDS)
def test_result_matches_the_update_schema(post, launches, launch_id) -> None:
    import jsonschema

    schema = json.loads((ROOT / "contract" / "update.schema.json").read_text())
    r = update_with_launch(post, launches[launch_id], 4)
    r.pop("_net_price", None)
    jsonschema.validate(r, schema)


# --------------------------------------------------------------------------
# the incremental verdict (found while building the web app)
# --------------------------------------------------------------------------
def test_the_headline_is_incremental_not_gross(post, launches) -> None:
    """P(pays) must come from incremental profit, or it is trivially 1.00.

    For an offer or a feature launch the treated cells are an existing business,
    so gross profit over the horizon carries the baseline margin and is nowhere
    near zero. Reporting that as "chance it pays" says a coupon pays whatever
    the coupon actually does.
    """
    for launch_id in ("student-coupon", "gift-boxes"):
        r = update_with_launch(post, launches[launch_id], 0)["forecast"]
        assert r["total_profit"]["p05"] > 0, "setup: gross profit should be clearly positive"
        assert abs(r["incremental_profit"]["mean"]) < r["total_profit"]["mean"], (
            f"{launch_id}: incremental should be much smaller than gross"
        )
        # the verdict must be the incremental one
        assert r["p_profit_positive"] < 0.999, (
            f"{launch_id}: P(pays) is {r['p_profit_positive']:.3f} before any data — "
            "that is the gross-profit bug"
        )


def test_new_city_incremental_equals_gross(post, launches) -> None:
    """A city that does not exist has no baseline, so the two must coincide."""
    for weeks in (0, 4, 8):
        f = update_with_launch(post, launches["austin-launch"], weeks)["forecast"]
        for key in ("mean", "p05", "p50", "p95"):
            assert f["incremental_profit"][key] == pytest.approx(
                f["total_profit"][key], rel=1e-12
            )


def test_p_profit_positive_agrees_with_the_incremental_interval(post, launches) -> None:
    """Internal consistency: a mostly-negative interval cannot be a high chance."""
    for launch_id in LAUNCH_IDS:
        for weeks in (0, 2, 8):
            f = update_with_launch(post, launches[launch_id], weeks)["forecast"]
            p, lo, hi = f["p_profit_positive"], f["incremental_profit"]["p05"], (
                f["incremental_profit"]["p95"]
            )
            if lo > 0:
                assert p > 0.95, f"{launch_id} wk{weeks}: interval all positive but p={p:.2f}"
            if hi < 0:
                assert p < 0.05, f"{launch_id} wk{weeks}: interval all negative but p={p:.2f}"


def test_the_coupon_verdict_moves_as_data_arrives(post, launches) -> None:
    """The spec's story: 'Chance it pays: 61% -> 78%'. It has to actually move."""
    before = update_with_launch(post, launches["student-coupon"], 0)
    after = update_with_launch(post, launches["student-coupon"], 8)
    assert before["forecast"]["p_profit_positive"] < 0.8, "week 0 should be genuinely unsure"
    assert after["forecast"]["p_profit_positive"] > before["forecast"]["p_profit_positive"] + 0.2
    assert after["prior_forecast"]["p_profit_positive"] == pytest.approx(
        before["forecast"]["p_profit_positive"]
    ), "prior_forecast must equal the week-0 forecast"


def test_weighted_quantile_is_tie_stable(post, launches) -> None:
    """K samples per draw tie exactly; an unstable sort made TS differ by ~5e-5."""
    import numpy as np

    x = np.repeat(np.array([3.0, 1.0, 2.0, 1.0]), 5)
    w = np.full(x.size, 1.0 / x.size)
    first = [weighted_quantile(x, q, w)[0] for q in (0.05, 0.5, 0.95)]
    for _ in range(5):
        again = [weighted_quantile(x, q, w)[0] for q in (0.05, 0.5, 0.95)]
        assert again == first, "weighted_quantile is not deterministic across calls"


def test_update_parity_fixture_is_reproducible() -> None:
    """Every case in the shipped fixture must still come out of update.py."""
    import json as _json

    path = ROOT / "contract" / "update_parity.fixture.json"
    if not path.exists():
        pytest.skip("run contract/make_update_parity_fixture.py first")
    fixture = _json.loads(path.read_text())
    post_doc = _json.loads((PUBLIC / "posterior.json").read_text())
    launch_docs = {
        x["id"]: x for x in _json.loads((PUBLIC / "launches.json").read_text())["launches"]
    }
    k, seed = fixture["k_per_draw"], fixture["seed"]

    for case in fixture["cases"]:
        got = update_with_launch(
            post_doc, launch_docs[case["launch_id"]], case["weeks_used"],
            k_per_draw=k, seed=seed,
        )
        want = case["expected"]
        label = f"{case['launch_id']} wk{case['weeks_used']}"
        assert got["gate"]["ok"] == want["gate"]["ok"], label
        for field in ("incremental_profit", "total_units"):
            for q in ("mean", "p05", "p95"):
                assert got["forecast"][field][q] == pytest.approx(
                    want["forecast"][field][q], rel=1e-9
                ), f"{label}.{field}.{q}"
        assert got["forecast"]["p_profit_positive"] == pytest.approx(
            want["forecast"]["p_profit_positive"], rel=1e-9
        ), label


def test_update_parity_fixture_ships_the_prior_samples() -> None:
    """update.ts cannot generate these, so they must be in the fixture."""
    import json as _json

    import numpy as np

    path = ROOT / "contract" / "update_parity.fixture.json"
    if not path.exists():
        pytest.skip("run contract/make_update_parity_fixture.py first")
    fixture = _json.loads(path.read_text())
    post_doc = _json.loads((PUBLIC / "posterior.json").read_text())
    k = fixture["k_per_draw"]
    expected_n = int(post_doc["meta"]["n_draws"]) * k

    assert set(fixture["priors"]) == set(LAUNCH_IDS)
    for launch_id, prior in fixture["priors"].items():
        assert len(prior["u"]) == expected_n, launch_id
        assert prior["prior_sigma"] > 0, launch_id
        assert np.all(np.isfinite(prior["u"])), launch_id
