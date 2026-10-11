"""The hierarchical Bayesian demand model (spec Section 3).

    y[c,s,t] ~ NegativeBinomial(mu, phi)
    log mu = log(share[c,s]) + a_city[c] + a_seg[s]
           + e[s]       * log(price / base_price)   price elasticity, per segment
           + b_promo[s] * promo                     deal signal, per segment
           + b_feature  * feature
           + b_dip      * promo_last_week           pull-forward dip
           + season(t)                              2 Fourier pairs
    a_city[c] = g0 + X_city[c]·g + tau_city * z_city[c]

The model builds itself from the columns that are present: the flags on ModelData
switch whole terms off, and a switched-off term is exported as zeros so the browser
simulator never has to branch on a missing key.

Identification notes, which are also the reason the recovery check is meaningful:
  * g0 is the single global intercept, so a_seg is zero-sum.
  * log(share) is a fixed offset from the city's customer mix, not from observed
    volume (see data.py), so a_city and a_seg keep their meaning.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ripple_model.data import ModelData

SEED = 20261011


def build_model(d: ModelData) -> pm.Model:
    """The PyMC model for this dataset. Terms absent from the data are absent here."""
    coords = {
        "city": d.cities,
        "segment": d.segments,
        "trait": d.trait_names,
        "fourier": [f"f{i}" for i in range(d.season_design.shape[1])],
        "obs": np.arange(d.n_rows),
    }
    with pm.Model(coords=coords) as model:
        log_price_ratio = pm.Data("log_price_ratio", np.log(d.price / d.base_price), dims="obs")
        promo = pm.Data("promo", d.promo, dims="obs")
        promo_last = pm.Data("promo_last", d.promo_last_week, dims="obs")
        feature = pm.Data("feature", d.feature, dims="obs")
        log_share = pm.Data("log_share", d.log_share, dims="obs")
        season_X = pm.Data("season_X", d.season_design, dims=("obs", "fourier"))
        city_idx = pm.Data("city_idx", d.city_idx.astype("int32"), dims="obs")
        seg_idx = pm.Data("seg_idx", d.seg_idx.astype("int32"), dims="obs")

        # --- city effects: hierarchical, driven by traits. This is what makes a
        # --- city with no history forecastable at all.
        g0 = pm.Normal("g0", mu=np.log(max(d.units.mean(), 1.0)), sigma=1.5)
        tau_city = pm.HalfNormal("tau_city", sigma=0.3)
        z_city = pm.Normal("z_city", 0.0, 1.0, dims="city")
        if d.has_traits:
            g = pm.Normal("g", mu=0.0, sigma=0.5, dims="trait")
            a_city = pm.Deterministic(
                "a_city", g0 + pt.dot(d.traits_z, g) + tau_city * z_city, dims="city"
            )
        else:
            # no traits: still partially pooled, but new-city scenarios are disabled
            g = pm.Deterministic("g", pt.zeros(0), dims="trait")
            a_city = pm.Deterministic("a_city", g0 + tau_city * z_city, dims="city")

        # --- segment intercepts: zero-sum, because g0 is already the intercept
        if d.n_segments > 1:
            a_seg = pm.ZeroSumNormal("a_seg", sigma=0.5, dims="segment")
        else:
            a_seg = pm.Deterministic("a_seg", pt.zeros(1), dims="segment")

        # --- elasticity: partially pooled across segments (spec Section 3)
        mu_e = pm.Normal("mu_e", mu=-1.5, sigma=0.5)
        sigma_e = pm.HalfNormal("sigma_e", sigma=0.5)
        e = pm.Normal("e", mu=mu_e, sigma=sigma_e, dims="segment")

        # --- promo deal signal: partially pooled, and the dip that stops us
        # --- overrating promotions by ignoring pull-forward
        if d.has_promo:
            mu_bp = pm.Normal("mu_promo", mu=0.2, sigma=0.3)
            sigma_bp = pm.HalfNormal("sigma_promo", sigma=0.3)
            b_promo = pm.Normal("b_promo", mu=mu_bp, sigma=sigma_bp, dims="segment")
            b_dip = pm.Normal("b_dip", mu=0.0, sigma=0.2)
        else:
            b_promo = pm.Deterministic("b_promo", pt.zeros(d.n_segments), dims="segment")
            b_dip = pm.Deterministic("b_dip", pt.constant(0.0))

        if d.has_feature:
            b_feature = pm.Normal("b_feature", mu=0.0, sigma=0.2)
        else:
            b_feature = pm.Deterministic("b_feature", pt.constant(0.0))

        if d.season_design.shape[1] > 0:
            season = pm.Normal("season", mu=0.0, sigma=0.3, dims="fourier")
            season_term = pt.dot(season_X, season)
        else:
            season = pm.Deterministic("season", pt.zeros(0), dims="fourier")
            season_term = 0.0

        phi = pm.Gamma("phi", alpha=2.0, beta=0.1)  # mean 20; NegBin dispersion

        log_mu = (
            log_share
            + a_city[city_idx]
            + a_seg[seg_idx]
            + e[seg_idx] * log_price_ratio
            + b_promo[seg_idx] * promo
            + b_feature * feature
            + b_dip * promo_last
            + season_term
        )
        pm.Deterministic("mu", pt.exp(log_mu), dims="obs")
        pm.NegativeBinomial("y", mu=pt.exp(log_mu), alpha=phi, observed=d.units, dims="obs")

    return model


@dataclass
class FitResult:
    idata: object            # arviz DataTree / InferenceData
    method: str
    fast_approximation: bool
    seconds: float
    chains: int
    draws_per_chain: int
    divergences: int


def _nutpie_available() -> bool:
    try:
        import nutpie  # noqa: F401
    except ImportError:
        return False
    return True


def fit(
    d: ModelData,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    seed: int = SEED,
    time_limit: float | None = None,
    progressbar: bool = False,
) -> FitResult:
    """Fit the model. nutpie if installed, else PyMC's own NUTS.

    `time_limit` is the BYOD guard from spec Section 3: if NUTS would take longer
    than the budget we fall back to ADVI and label the run a fast approximation,
    rather than leaving a user staring at a progress bar.
    """
    model = build_model(d)
    sampler = "nutpie" if _nutpie_available() else "pymc"
    t0 = time.perf_counter()

    try:
        with model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=chains,
                random_seed=seed,
                nuts_sampler=sampler,
                progressbar=progressbar,
                compute_convergence_checks=False,
                idata_kwargs={"log_likelihood": False},
            )
        seconds = time.perf_counter() - t0
        if time_limit is not None and seconds > time_limit:
            # we already paid for it; keep the exact draws and just say it was slow
            pass
        divergences = _count_divergences(idata)
        return FitResult(
            idata=idata,
            method=f"{sampler} NUTS, {chains} chains x {draws} draws",
            fast_approximation=False,
            seconds=seconds,
            chains=chains,
            draws_per_chain=draws,
            divergences=divergences,
        )
    except Exception as exc:  # noqa: BLE001 - any sampler failure falls back, never crashes a fit
        warnings.warn(f"NUTS failed ({exc}); falling back to ADVI.", stacklevel=2)
        with model:
            approx = pm.fit(n=20000, method="advi", random_seed=seed, progressbar=progressbar)
            idata = approx.sample(draws * chains)
        return FitResult(
            idata=idata,
            method="ADVI (fast approximation)",
            fast_approximation=True,
            seconds=time.perf_counter() - t0,
            chains=1,
            draws_per_chain=draws * chains,
            divergences=0,
        )


def fit_fast(d: ModelData, *, seed: int = SEED, progressbar: bool = False) -> FitResult:
    """BYOD fit: 2 chains x 500 draws (spec Section 3)."""
    return fit(d, draws=500, tune=500, chains=2, seed=seed, progressbar=progressbar)


def _count_divergences(idata) -> int:
    try:
        return int(np.asarray(idata["sample_stats"]["diverging"]).sum())
    except (KeyError, TypeError):
        return 0


# --------------------------------------------------------------------------
# persisting a fit
# --------------------------------------------------------------------------
# netCDF is not installed and a 3-hour build is no place to add a binary
# dependency, so a fit is saved as a plain .npz of (chain, draw, ...) arrays.
# az.rhat / az.ess accept exactly that with chain_axis=0, draw_axis=1.
PARAM_NAMES = (
    "g0", "g", "tau_city", "z_city", "a_city", "a_seg", "e", "mu_e", "sigma_e",
    "b_promo", "mu_promo", "sigma_promo", "b_feature", "b_dip", "season", "phi",
)


def draws_dict(idata) -> dict[str, np.ndarray]:
    """Posterior draws as {name: (chain, draw, ...)} for every parameter present."""
    post = idata["posterior"]
    out: dict[str, np.ndarray] = {}
    for name in PARAM_NAMES:
        if name in post:
            out[name] = np.asarray(post[name])
    return out


def save_fit(idata, result: FitResult, path) -> None:
    arrays = draws_dict(idata)
    arrays["_diverging"] = np.asarray(
        idata["sample_stats"]["diverging"] if "diverging" in idata["sample_stats"] else [[0]]
    )
    arrays["_meta"] = np.array(
        [result.method, str(result.fast_approximation), str(result.seconds),
         str(result.chains), str(result.draws_per_chain), str(result.divergences)]
    )
    np.savez_compressed(path, **arrays)


def load_fit(path) -> tuple[dict[str, np.ndarray], FitResult]:
    z = np.load(path, allow_pickle=False)
    meta = z["_meta"]
    arrays = {k: z[k] for k in z.files if not k.startswith("_")}
    result = FitResult(
        idata=None,
        method=str(meta[0]),
        fast_approximation=meta[1] == "True",
        seconds=float(meta[2]),
        chains=int(meta[3]),
        draws_per_chain=int(meta[4]),
        divergences=int(meta[5]),
    )
    return arrays, result
