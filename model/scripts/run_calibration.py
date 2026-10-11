"""Run the calibration study and write web/public/model/calibration.json.

Run: uv run python model/scripts/run_calibration.py [n_launches]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
from ripple_model.calibration import run_calibration

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "web" / "public" / "model"
CONTRACT = ROOT / "contract"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    post = json.loads((PUBLIC / "posterior.json").read_text())
    print(f"calibration: {n} launches per arm, 8 revealed weeks, 3 arms ...", flush=True)
    doc = run_calibration(post, n_launches=n)

    jsonschema.validate(doc, json.loads((CONTRACT / "calibration.schema.json").read_text()))
    out = PUBLIC / "calibration.json"
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"\nwrote {out} ({out.stat().st_size / 1024:.0f} KB) "
          f"in {doc['runtime_seconds']:.0f}s\n")
    print(doc["headline"])
    for arm in doc["arms"]:
        flag = "" if arm["assumptions_hold"] else "   [STRESS ARM]"
        print(f"\n{arm['label']}{flag}")
        print(f"  {'week':>4} {'coverage':>9} {'+-mcse':>7} {'cov(unknown)':>13} "
              f"{'width':>10} {'bias':>9} {'refit':>6} {'khat':>7}")
        for b in arm["by_week"]:
            print(f"  {b['week']:>4} {b['coverage_total']:>9.0%} "
                  f"{b['coverage_total_mcse']:>7.1%} {b['coverage_unknown']:>13.0%} "
                  f"{b['mean_width_total']:>10,.0f} {b['bias_total']:>9,.0f} "
                  f"{b['refit_rate']:>6.0%} {b['mean_pareto_k']:>7.2f}")
        print(f"  -> {arm['verdict']}")


if __name__ == "__main__":
    main()
