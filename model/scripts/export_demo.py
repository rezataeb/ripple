"""Turn the saved demo fits into the three files the browser loads.

  web/public/model/posterior.json    the fitted model (contract/posterior.schema.json)
  web/public/model/trust.json        the checks       (contract/trust.schema.json)
  web/public/model/launches.json     the 3 launches   (contract/launch.schema.json)

web/ belongs to Session B, but spec Section 12 makes this directory the handoff
point, so Session A writes data files here and no code.

Run after model/scripts/run_fits.py:
    uv run python model/scripts/export_demo.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import jsonschema
import numpy as np
from ripple_model.checks import build_trust, write_trust
from ripple_model.data import load_demo
from ripple_model.export import build_posterior, write_json
from ripple_model.model import load_fit
from ripple_model.synth import CANDIDATE_SHARES, CANDIDATES, SEGMENTS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "model" / "data"
CONTRACT = ROOT / "contract"
PUBLIC = ROOT / "web" / "public" / "model"


def main() -> None:
    d = load_demo()
    truth = json.loads((DATA / "truth.json").read_text())

    arrays, result = load_fit(DATA / "fit_full.npz")
    y_rep = np.load(DATA / "ppc_full.npz")["y_rep"]
    holdout_arrays, _ = load_fit(DATA / "fit_holdout.npz")

    # keyed by segment name: synth.py declares segments in its own order and
    # data.py sorts them alphabetically, so a positional list mis-orders the mix
    candidates = {
        name: {
            "traits": raw,
            "shares": dict(zip(SEGMENTS, CANDIDATE_SHARES[name], strict=True)),
        }
        for name, raw in CANDIDATES.items()
    }

    post = build_posterior(arrays, result, d, candidates=candidates)
    trust = build_trust(
        arrays, result, d, y_rep=y_rep, holdout_arrays=holdout_arrays, truth=truth
    )

    PUBLIC.mkdir(parents=True, exist_ok=True)
    size_p = write_json(post, PUBLIC / "posterior.json")
    write_trust(trust, PUBLIC / "trust.json")
    shutil.copy(DATA / "launches.json", PUBLIC / "launches.json")

    # the export is worthless if it does not satisfy the contract
    for name, schema in [
        ("posterior.json", "posterior.schema.json"),
        ("trust.json", "trust.schema.json"),
        ("launches.json", "launch.schema.json"),
    ]:
        jsonschema.validate(
            json.loads((PUBLIC / name).read_text()),
            json.loads((CONTRACT / schema).read_text()),
        )
        kb = (PUBLIC / name).stat().st_size / 1024
        print(f"  {name:<16} {kb:>7.0f} KB  valid against {schema}")

    print(f"\nposterior: {post['meta']['n_draws']} draws, {size_p / 1024:.0f} KB")
    print(f"method:    {trust['method']}")
    print(f"\n{trust['summary']['headline']}")
    for c in trust["checks"]:
        mark = "PASS" if c["passed"] else ("SKIP" if c.get("skipped_reason") else "FAIL")
        val = "" if c["value"] is None else f"{c['value']}"
        print(f"  [{mark}] {c['name']:<30} {val:>10}   {c.get('detail', '')}")

    rec = trust.get("recovery", [])
    inside = sum(r["inside"] for r in rec)
    print(f"\nrecovery: {inside}/{len(rec)} planted values inside the 94% HDI")
    for r in rec:
        if not r["inside"]:
            print(f"  MISS {r['param']:<22} truth {r['truth']:+.4f} "
                  f"not in [{r['hdi_low']:+.4f}, {r['hdi_high']:+.4f}]")
    h = trust.get("holdout")
    if h:
        print(f"holdout:  coverage {h['coverage_90']:.0%}, MAPE {h['mape_model']:.1%} "
              f"vs naive {h['mape_naive']:.1%} -> beats naive: {h['beats_naive']}")


if __name__ == "__main__":
    main()
