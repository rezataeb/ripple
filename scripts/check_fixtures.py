"""Fail if a committed fixture no longer matches what its generator produces.

This gate exists because a fixture edited by hand — or a change to simulate.py that
was not followed by regenerating parity.fixture.json — would leave the parity test
comparing new code against expectations derived from new code: green and worthless.

It cannot be a byte comparison. float64 arithmetic differs in the last one or two
digits between CPUs and BLAS builds, so `git diff --exit-code` fails on CI for
0.4233074266243379 vs 0.42330742662433785 — a relative difference of 1e-16 that
means nothing. With ~8000 numbers per fixture, rounding before writing does not fix
it either: a value sitting near a rounding boundary still flips.

So: regenerate, compare structurally and with a relative tolerance, and restore the
committed bytes afterwards so the working tree is left exactly as it was found.

Run: uv run python scripts/check_fixtures.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# relative tolerance for numbers; anything larger is a real change
RTOL = 1e-9
ATOL = 1e-12

GENERATORS: list[tuple[list[str], list[str]]] = [
    (["python", "contract/make_fixture.py"], ["contract/posterior.fixture.json"]),
    (["python", "contract/make_parity_fixture.py"], ["contract/parity.fixture.json"]),
    (
        ["python", "model/scripts/make_demo_scan.py"],
        ["web/public/model/market_scan.demo.json"],
    ),
]


def compare(a, b, path: str, out: list[str]) -> None:
    """Collect every meaningful difference between two loaded JSON documents."""
    if len(out) > 25:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                out.append(f"{path}.{key}: added by the generator")
            elif key not in b:
                out.append(f"{path}.{key}: missing from the generator's output")
            else:
                compare(a[key], b[key], f"{path}.{key}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} committed vs {len(b)} generated")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            compare(x, y, f"{path}[{i}]", out)
    elif isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        if a is not b:
            out.append(f"{path}: {a!r} committed vs {b!r} generated")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(a - b) > ATOL + RTOL * abs(b):
            out.append(f"{path}: {a!r} committed vs {b!r} generated")
    elif a != b:
        out.append(f"{path}: {a!r} committed vs {b!r} generated")


def main() -> int:
    problems: list[str] = []
    for cmd, targets in GENERATORS:
        paths = [ROOT / t for t in targets]
        missing = [p for p in paths if not p.exists()]
        if missing:
            problems.append(f"{missing[0].relative_to(ROOT)} does not exist — run `make fixtures`")
            continue

        committed = {p: p.read_bytes() for p in paths}
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        try:
            for p in paths:
                diffs: list[str] = []
                compare(
                    json.loads(committed[p]),
                    json.loads(p.read_text()),
                    p.relative_to(ROOT).as_posix(),
                    diffs,
                )
                problems += diffs
        finally:
            # leave the working tree exactly as we found it
            for p, raw in committed.items():
                p.write_bytes(raw)

    if problems:
        print("Committed fixtures do not match their generators:\n", file=sys.stderr)
        for line in problems[:25]:
            print(f"  {line}", file=sys.stderr)
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more", file=sys.stderr)
        print("\nRun `make fixtures` and commit the result.", file=sys.stderr)
        return 1

    print(f"fixtures are current (relative tolerance {RTOL:g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
