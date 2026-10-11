"""Fit the demo model twice: the full fit, and a time-holdout fit.

Saved as .npz so checks.py and export.py can be developed without refitting.
Run: uv run python model/scripts/run_fits.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pymc as pm
from ripple_model.data import load_demo
from ripple_model.model import build_model, fit, save_fit

OUT = Path(__file__).resolve().parents[1] / "data"
HOLDOUT_WEEKS = 12


def main() -> None:
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    chains = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    target = float(sys.argv[3]) if len(sys.argv) > 3 else 0.95
    d = load_demo()
    print(f"data: {d.n_rows} rows, {d.n_cities} cities, {d.n_segments} segments, "
          f"{d.n_weeks} weeks, shares from {d.shares_source}", flush=True)

    t0 = time.perf_counter()
    print(f"[1/3] full fit: {chains} chains x {draws} draws, "
          f"target_accept {target} ...", flush=True)
    res = fit(d, draws=draws, tune=draws, chains=chains, target_accept=target)
    save_fit(res.idata, res, OUT / "fit_full.npz")
    print(f"      {res.method} in {res.seconds:.1f}s, {res.divergences} divergences", flush=True)

    # posterior predictive on the training rows, for the coverage check
    print("[2/3] posterior predictive ...", flush=True)
    with build_model(d):
        ppc = pm.sample_posterior_predictive(
            res.idata, var_names=["y"], random_seed=20261011, progressbar=False
        )
    y_rep = np.asarray(ppc["posterior_predictive"]["y"])  # (chain, draw, obs)
    np.savez_compressed(OUT / "ppc_full.npz", y_rep=y_rep.reshape(-1, y_rep.shape[-1]))
    print(f"      y_rep {y_rep.shape}", flush=True)

    # time holdout: refit without the last 12 weeks, then predict them
    print(f"[3/3] holdout fit (last {HOLDOUT_WEEKS} weeks removed) ...", flush=True)
    cutoff = d.n_weeks - HOLDOUT_WEEKS
    train = d.subset(d.t < cutoff)
    res_h = fit(train, draws=draws, tune=draws, chains=chains, target_accept=target)
    save_fit(res_h.idata, res_h, OUT / "fit_holdout.npz")
    print(f"      {res_h.method} in {res_h.seconds:.1f}s, "
          f"{res_h.divergences} divergences", flush=True)

    print(f"done in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
