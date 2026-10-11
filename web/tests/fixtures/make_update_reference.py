"""Regenerate update.reference.json: Python reference output for web/lib/update.ts parity.

Run from the repo root:  .venv/bin/python web/tests/fixtures/make_update_reference.py

Why a web-side snapshot: numpy's RNG cannot be reproduced in the browser, so the unknown's prior
samples `u` are exported and fed to update.ts. k_per_draw=5 keeps the file small. The quantile sort
is made stable here because update.py's weighted_quantile uses numpy's default (unstable) argsort,
and ties (K samples share one posterior draw) make the interpolated quantile order-dependent. Once
update.py uses kind="stable" this patch is a no-op.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "model")
import numpy as np
import ripple_model.update as U

K = 5
WEEKS = (0, 3, 8)


def stable_weighted_quantile(x, q, w):
    x = np.asarray(x, float).ravel()
    w = np.asarray(w, float).ravel()
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], w[order]
    cum = np.cumsum(ws)
    cdf = (cum - 0.5 * ws) / cum[-1]
    return np.interp(np.atleast_1d(q), cdf, xs)


U.weighted_quantile = stable_weighted_quantile

post = json.load(open("web/public/model/posterior.json"))
launches = json.load(open("web/public/model/launches.json"))["launches"]
out = {
    "about": "Python reference for web/lib/update.ts. Compare every field except meta and causal.",
    "k_per_draw": K,
    "posterior_fitted_at": post["meta"].get("fitted_at"),
    "n_draws": post["meta"]["n_draws"],
    "u": {},
    "cases": [],
}
for launch in launches:
    u, _, _ = U.draw_unknown(post, launch, k_per_draw=K)
    out["u"][launch["id"]] = [round(float(v), 12) for v in u]
    for wk in WEEKS:
        r = U.update_with_launch(post, launch, wk, k_per_draw=K, smooth=False)
        r.pop("_net_price", None)
        r.pop("meta")
        r.pop("causal")
        out["cases"].append(
            {"launch_id": launch["id"], "weeks_used": wk, "expected": r}
        )
Path("web/tests/fixtures/update.reference.json").write_text(json.dumps(out))
print("wrote", len(out["cases"]), "cases")
