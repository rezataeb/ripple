"""Export a fit for the browser (contract/posterior.schema.json).

The browser gets the whole posterior and does the simulation itself, so sliders
move without a round trip. That means this file decides exactly what crosses the
wire: thinned draws, the history in columnar form, the candidate cities with
their extrapolation flags already computed, and nothing else.

Terms the data did not support are exported as ZEROS rather than omitted, so
web/lib/simulate.ts never branches on a missing key.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from ripple_model.data import ModelData
from ripple_model.model import FitResult

EXPORT_DRAWS = 1000  # what the browser downloads; the fit keeps more
SEED = 20261011


def _flat(x: np.ndarray) -> np.ndarray:
    """(chain, draw, ...) -> (chain*draw, ...). See the note in export.py about -1."""
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def thin(arrays: dict[str, np.ndarray], n: int = EXPORT_DRAWS) -> dict[str, np.ndarray]:
    """Evenly spaced draws across the pooled chains — not the first n, which
    would come from one chain and understate the spread."""
    any_key = next(iter(arrays))
    total = _flat(arrays[any_key]).shape[0]
    if total <= n:
        idx = np.arange(total)
    else:
        idx = np.linspace(0, total - 1, n).round().astype(int)
    return {k: _flat(v)[idx] for k, v in arrays.items()}


def _round(a, nd: int = 6):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def build_posterior(
    arrays: dict[str, np.ndarray],
    result: FitResult,
    d: ModelData,
    *,
    candidates: dict[str, dict] | None = None,
    n_draws: int = EXPORT_DRAWS,
    seed: int = SEED,
) -> dict:
    """The posterior export dict. Validated against the schema by the tests."""
    td = thin(arrays, n_draws)
    D = len(np.atleast_1d(td["g0"]))
    S, C, P = d.n_segments, d.n_cities, d.n_traits
    K = d.season_design.shape[1]

    def col(name: str, width: int) -> list:
        """A [D][width] block, zero-filled when the term is not in the model."""
        if name in td and np.asarray(td[name]).size:
            a = np.asarray(td[name], dtype=float).reshape(D, -1)
            if a.shape[1] == width:
                return _round(a)
        return _round(np.zeros((D, width)))

    def scalar(name: str) -> list:
        if name in td and np.asarray(td[name]).size:
            return _round(np.asarray(td[name], dtype=float).reshape(D))
        return _round(np.zeros(D))

    # price variation drives whether the elasticity is evidence or prior
    price_cv = float(np.std(d.price) / np.mean(d.price)) if len(d.price) else 0.0

    meta = {
        "dataset_name": d.dataset_name,
        "cities": list(d.cities),
        "segments": list(d.segments),
        "weeks": int(d.n_weeks),
        "n_draws": int(D),
        "base_price": float(d.base_price),
        "unit_cost": float(d.unit_cost),
        "shares": _round(d.shares, 8),
        "trait_names": list(d.trait_names),
        "traits_z": _round(d.traits_z, 8),
        "trait_mean": _round(d.trait_mean, 8),
        "trait_std": _round(d.trait_std, 8),
        "trait_min": _round(d.traits_z.min(axis=0) if P else np.zeros(0), 8),
        "trait_max": _round(d.traits_z.max(axis=0) if P else np.zeros(0), 8),
        "has_promo": bool(d.has_promo),
        "has_feature": bool(d.has_feature),
        "has_traits": bool(d.has_traits),
        "has_segments": bool(d.has_segments),
        "season_period": 52.0,
        "n_fourier": int(K // 2),
        "method": result.method,
        "fast_approximation": bool(result.fast_approximation),
        "seed": int(seed),
        "fitted_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    if price_cv < 0.01:
        meta["no_price_variation"] = True

    draws = {
        "g0": scalar("g0"),
        "g": col("g", P),
        "tau_city": scalar("tau_city"),
        "a_city": col("a_city", C),
        "a_seg": col("a_seg", S),
        "e": col("e", S),
        "b_promo": col("b_promo", S) if d.has_promo else _round(np.zeros((D, S))),
        "b_feature": scalar("b_feature") if d.has_feature else _round(np.zeros(D)),
        "b_dip": scalar("b_dip") if d.has_promo else _round(np.zeros(D)),
        "season": col("season", K),
        "phi": scalar("phi"),
        # Drawn here, once, with the export seed, and shipped with the posterior.
        # Generating them at simulation time would mean numpy and the browser's
        # PRNG disagreeing on every new-city scenario — see posterior.schema.json.
        "z_new": _round(np.random.default_rng(seed).standard_normal(D)),
    }

    history = {
        "week_start": list(d.week_start),
        "t": [int(x) for x in d.t],
        "city_idx": [int(x) for x in d.city_idx],
        "segment_idx": [int(x) for x in d.seg_idx],
        "units": _round(d.units, 1),
        "price": _round(d.price, 2),
        "promo": _round(d.promo, 1),
        "promo_last_week": _round(d.promo_last_week, 1),
        "feature": _round(d.feature, 1),
    }

    return {
        "meta": meta,
        "draws": draws,
        "history": history,
        "candidates": build_candidates(d, candidates),
    }


def build_candidates(d: ModelData, candidates: dict[str, dict] | None) -> list[dict]:
    """Candidate cities with the extrapolation flag already decided.

    The flag is computed here, once, so the UI and the agent cannot disagree about
    whether a city is inside the training range.
    """
    if not candidates or not d.has_traits:
        return []
    lo, hi = d.traits_z.min(axis=0), d.traits_z.max(axis=0)
    out = []
    for name, spec in candidates.items():
        raw = np.asarray(spec["traits"], dtype=float)
        z = (raw - d.trait_mean) / d.trait_std
        bad = [
            d.trait_names[i]
            for i in range(d.n_traits)
            if z[i] < lo[i] - 1e-12 or z[i] > hi[i] + 1e-12
        ]
        shares = np.asarray(
            spec.get("shares") or d.shares.mean(axis=0), dtype=float
        )
        shares = shares / shares.sum()
        entry = {
            "name": name,
            "traits": _round(raw, 6),
            "traits_z": _round(z, 8),
            "shares": _round(shares, 8),
            "extrapolation": bool(bad),
            "out_of_range_traits": bad,
        }
        if bad:
            entry["note"] = (
                f"{name} is outside the training range on {', '.join(bad)}. "
                "The forecast extrapolates."
            )
        out.append(entry)
    return out


def write_json(doc: dict, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":")))
    return path.stat().st_size
